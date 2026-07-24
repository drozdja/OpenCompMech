#!/usr/bin/env python3
"""Phase-J pilot: conditional DDPM over the mechanism cache, @64x64, on the
R9700. Answers "can we emerge usable designs from this dataset" empirically.

Runs INSIDE the rocm/pytorch container under hard cgroup caps (see
run_pilot_docker.sh). Keep --threads / --num-workers tiny: the box's cores
belong to the CPU generation.

Periodically renders target-vs-generation grids for held-out conditions
(out/samples/step_*.png) so progress is eyeballable, and checkpoints EMA weights.
"""

import argparse
import copy
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ml.dataset import PilotCache
from src.ml.unet import ConditionalUNet
from src.ml.diffusion import Diffusion
from src.ml.flow import RectifiedFlow
from src.ml.optim_guard import check_optimizer, sanitize_optimizer
from src.ml.physics import physics_loss
from src.ml.tensor_spec import COND_DIM, SCALAR_DIM


def save_grid(path, targets, gens):
    """targets/gens: (K,64,64) in [-1,1]. Two rows (GT top, gen bottom)."""
    def to255(a):
        return (np.clip((a + 1) / 2, 0, 1) * 255).astype(np.uint8)
    K = targets.shape[0]
    R = targets.shape[-1]
    pad = 2
    row_t = np.full((R, K * (R + pad) - pad), 255, np.uint8)
    row_g = np.full((R, K * (R + pad) - pad), 255, np.uint8)
    for k in range(K):
        x = k * (R + pad)
        row_t[:, x:x + R] = to255(targets[k])
        row_g[:, x:x + R] = to255(gens[k])
    img = np.full((2 * R + pad, row_t.shape[1]), 255, np.uint8)
    img[:R] = row_t
    img[R + pad:] = row_g
    try:
        from PIL import Image
        Image.fromarray(img).save(path)
    except Exception:
        np.save(path.replace(".png", ".npy"), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="accumulate this many micro-batches per optimizer step; "
                         "effective batch = batch * grad_accum. Lets a memory-safe "
                         "micro-batch match a large effective batch at high res.")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=0,
                    help="linear LR warmup steps; stabilizes early high-res "
                         "training where the first large gradients can diverge")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed model init + sampling for a reproducible run")
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--sample-every", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--objective", choices=["ddpm", "flow"], default="ddpm")
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16",
                    help="autocast dtype; fp32 disables autocast. NOTE: gfx1201/"
                         "RDNA4 (R9700) has broken half-precision conv kernels that "
                         "produce intermittent NaNs at 128px (fine at 64px) -> use "
                         "fp32 for 128px on that GPU.")
    ap.add_argument("--reset-optimizer", action="store_true",
                    help="resume weights but start Adam state fresh")
    ap.add_argument("--opt-max-exp-avg", type=float, default=1.0,
                    help="reset Adam moments whose |exp_avg| exceeds this; "
                         "gradient clipping bounds it by the clip norm, so a "
                         "larger value means corruption (0 = off)")
    ap.add_argument("--opt-check-every", type=int, default=200,
                    help="check+repair corrupted Adam state this often (0 = off)")
    ap.add_argument("--snapshot-every", type=int, default=10000,
                    help="also keep a numbered checkpoint this often (0 = off)")
    ap.add_argument("--resume", default=None,
                    help="checkpoint to resume model/EMA/optimizer/step from")
    ap.add_argument("--nan-consec", type=int, default=50,
                    help="abort after this many CONSECUTIVE non-finite micro-steps "
                         "(catches a broken recipe)")
    ap.add_argument("--nan-window", type=int, default=2000,
                    help="length in steps of the non-finite rate window")
    ap.add_argument("--nan-window-max", type=int, default=200,
                    help="abort if more than this many non-finite micro-steps "
                         "occur within one --nan-window (catches slow poisoning)")
    ap.add_argument("--physics-weight", type=float, default=0.0,
                    help="weight of the port-attach + connectivity aux loss")
    ap.add_argument("--physics-iters", type=int, default=24)
    ap.add_argument("--cfg-dropout", type=float, default=0.0,
                    help="prob. of ZEROing conditioning (channels+scalars) per "
                         "sample so the model learns the unconditional field too "
                         "-> enables classifier-free guidance + a real "
                         "conditioning-strength dial at sampling")
    ap.add_argument("--split-mode", choices=["lineage", "spec", "family", "type", "iid"],
                    default="lineage",
                    help="grouped holdout; iid is retained only for legacy comparison")
    ap.add_argument("--holdout-value", default=None,
                    help="explicit group to hold out for --split-mode family/type; "
                         "use this for a reproducible OOD run")
    ap.add_argument("--val-frac", type=float, default=0.04)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--split-plan", default=None,
                    help="frozen split plan from scripts/build_eval_split.py; "
                         "use for claim-bearing runs")
    args = ap.parse_args()

    import contextlib

    def autocast_ctx():
        # fp32 disables autocast entirely; gfx1201/RDNA4 half-precision conv
        # kernels NaN at 128px, so fp32 is the stable choice there.
        if args.precision == "fp32":
            return contextlib.nullcontext()
        dt = torch.float16 if args.precision == "fp16" else torch.bfloat16
        return torch.autocast(args.device, dtype=dt)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)
    stop_file = os.path.join(args.out, "STOP")

    split_kwargs = dict(split_mode=args.split_mode, val_frac=args.val_frac,
                        seed=args.split_seed, holdout_value=args.holdout_value,
                        split_plan=args.split_plan)
    train = PilotCache(args.cache, split="train", **split_kwargs)
    val = PilotCache(args.cache, split="val", **split_kwargs)
    print(f"[train] {len(train)} train / {len(val)} val samples "
          f"split={train.split_info}", flush=True)
    dl = DataLoader(train, batch_size=args.batch, shuffle=True,
                    num_workers=args.num_workers, pin_memory=True,
                    drop_last=True, persistent_workers=(args.num_workers > 0))

    # Fixed held-out conditioning batch for progress grids.
    K = min(8, len(val))
    vb = [val[i] for i in range(K)]
    v_cond = torch.stack([b["cond"] for b in vb]).to(args.device)
    v_scal = torch.stack([b["scalars"] for b in vb]).to(args.device)
    v_tgt = torch.stack([b["target"] for b in vb]).cpu().numpy()[:, 0]

    model = ConditionalUNet(1, COND_DIM, SCALAR_DIM, base=args.base).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params/1e6:.1f}M", flush=True)
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    obj = (RectifiedFlow(device=args.device) if args.objective == "flow"
           else Diffusion(T=args.timesteps, device=args.device))
    print(f"[train] objective={args.objective} physics_weight={args.physics_weight}",
          flush=True)

    step = 0
    loss_ema = None
    nan_skips = 0        # lifetime count (reported only)
    nan_consec = 0       # consecutive -> a genuinely broken recipe
    nan_window = 0       # count inside the current window -> slow poisoning
    window_start = 0
    opt_repairs = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        if "opt" in ck and not args.reset_optimizer:
            opt.load_state_dict(ck["opt"])
            chk = check_optimizer(opt)
            if not chk["ok"]:
                # Corrupt Adam state is fatal on the first step (sqrt of a
                # negative second moment is NaN), and it is invisible until the
                # weights are already dead.  Repair it before training resumes.
                print(f"[train] corrupt optimizer state in checkpoint: {chk}",
                      flush=True)
                print(f"[train] repaired: "
                      f"{sanitize_optimizer(opt, max_exp_avg=1.0)}", flush=True)
        step = int(ck.get("step", 0))
        window_start = step
        print(f"[train] resumed from {args.resume} @ step {step}", flush=True)
    start_step = step
    t0 = time.time()
    data_iter = iter(dl)
    accum = max(1, args.grad_accum)
    while step < args.steps:
        if os.path.exists(stop_file):
            print("[train] STOP file -> checkpoint and exit", flush=True)
            break

        # Linear LR warmup: the first high-resolution gradients are large, and
        # at full LR they can kick the model into a region that diverges to NaN.
        if args.warmup > 0:
            lr_now = args.lr * min(1.0, (step + 1) / args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr_now

        # Gradient accumulation: `accum` micro-batches make one optimizer step,
        # so the effective batch = args.batch * accum. `step` counts optimizer
        # steps (warmup/sample/ckpt all use it), matching the baseline recipe.
        opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        n_ok = 0
        for _ in range(accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dl)
                batch = next(data_iter)
            x0 = batch["target"].to(args.device, non_blocking=True)
            cond = batch["cond"].to(args.device, non_blocking=True)
            scal = batch["scalars"].to(args.device, non_blocking=True)

            # Classifier-free guidance: null out conditioning on a random subset
            # so the model also learns p(x) (all-zero cond = the null token).
            # Real conditioning is never exactly zero, so it is unambiguous.
            physics_cond = cond
            physics_keep = None
            if args.cfg_dropout > 0:
                keep = (torch.rand(x0.shape[0], device=args.device)
                        >= args.cfg_dropout).float()
                physics_keep = keep.bool()
                cond = cond * keep.view(-1, 1, 1, 1)
                scal = scal * keep.view(-1, 1)

            with autocast_ctx():
                base, x0_pred = obj.loss_and_x0(model, x0, cond, scal)
                if args.physics_weight > 0:
                    pl, _ = physics_loss(x0_pred.float(), physics_cond,
                                         iters=args.physics_iters,
                                         sample_mask=physics_keep)
                    loss = base + args.physics_weight * pl
                else:
                    loss = base
                loss = loss / accum

            if not torch.isfinite(loss):
                # One NaN gradient survives grad-norm clipping and permanently
                # poisons the weights; drop this micro-batch's contribution.
                #
                # gfx1201 (RDNA4) emits sporadic non-finite conv outputs even in
                # fp32 at 128px (~0.5% of steps; see docs/HARDWARE_NOTES_RDNA4.md), so
                # a *lifetime* counter guarantees a false abort on any long run.
                # Guard on rate instead: consecutive failures catch a genuinely
                # broken recipe, a windowed rate catches slow poisoning.
                nan_skips += 1
                nan_consec += 1
                nan_window += 1
                if nan_consec >= args.nan_consec:
                    raise SystemExit(
                        f"[train] aborting: {nan_consec} consecutive non-finite "
                        f"micro-steps at step {step}")
                if nan_window > args.nan_window_max:
                    raise SystemExit(
                        f"[train] aborting: {nan_window} non-finite micro-steps "
                        f"within {step - window_start} steps (rate too high)")
                continue
            nan_consec = 0
            loss.backward()
            accum_loss += loss.item() * accum
            n_ok += 1

        if n_ok == 0:
            step += 1
            continue

        # A finite loss does NOT imply finite gradients: gfx1201 emits sporadic
        # non-finite values in the conv BACKWARD pass too.  clip_grad_norm_ then
        # rescales *every* gradient by a non-finite total norm, so a single bad
        # micro-batch poisons the whole model permanently on the next opt.step().
        # Checking the loss alone (above) cannot catch this — the norm must be
        # verified before the optimizer is allowed to touch the weights.
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gnorm):
            opt.zero_grad(set_to_none=True)
            nan_skips += 1
            nan_consec += 1
            nan_window += 1
            if nan_consec >= args.nan_consec:
                raise SystemExit(
                    f"[train] aborting: {nan_consec} consecutive non-finite "
                    f"gradient steps at step {step}")
            if nan_window > args.nan_window_max:
                raise SystemExit(
                    f"[train] aborting: {nan_window} non-finite steps within "
                    f"{step - window_start} steps (rate too high)")
            step += 1
            continue
        nan_consec = 0
        opt.step()

        # gfx1201 sporadically corrupts the fused optimizer kernels, leaving a
        # NEGATIVE exp_avg_sq.  That is undetectable from the loss until the
        # next step has already written NaN into every weight, so check the
        # invariant directly and repair it.
        if args.opt_check_every and step % args.opt_check_every == 0:
            rep = sanitize_optimizer(opt, max_exp_avg=args.opt_max_exp_avg)
            if rep["repaired"]:
                opt_repairs += 1
                print(f"[train] repaired optimizer state @ step {step}: {rep}",
                      flush=True)

        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(args.ema).add_(pm, alpha=1 - args.ema)

        lv = accum_loss / n_ok
        loss_ema = lv if loss_ema is None else 0.98 * loss_ema + 0.02 * lv
        step += 1

        if step - window_start >= args.nan_window:
            window_start, nan_window = step, 0

        if step % 100 == 0:
            rate = (step - start_step) / (time.time() - t0)
            print(f"[train] step {step}/{args.steps} loss {loss_ema:.4f} "
                  f"({rate:.1f} it/s)"
                  + (f" [nan={nan_skips}]" if nan_skips else "")
                  + (f" [optfix={opt_repairs}]" if opt_repairs else ""),
                  flush=True)

        if step % args.sample_every == 0 or step == args.steps:
            ema.eval()
            with autocast_ctx():
                gen = obj.sample(ema, v_cond, v_scal, steps=50)
            gen = gen.float().cpu().numpy()[:, 0]
            save_grid(os.path.join(args.out, "samples", f"step_{step:06d}.png"),
                      v_tgt, gen)
            print(f"[train] wrote sample grid @ step {step}", flush=True)

        if step % args.ckpt_every == 0 or step == args.steps:
            # Never overwrite a good checkpoint with a poisoned one: a NaN that
            # reaches the weights is unrecoverable, so the last clean state is
            # the only way back.
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise SystemExit(
                    f"[train] aborting at step {step}: model weights are "
                    f"non-finite; keeping the previous checkpoint")
            torch.save({"step": step, "model": model.state_dict(),
                        "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "args": vars(args), "split": train.split_info},
                       os.path.join(args.out, "ckpt.pt"))
            if args.snapshot_every and step % args.snapshot_every == 0:
                torch.save({"step": step, "model": model.state_dict(),
                            "ema": ema.state_dict(), "args": vars(args),
                            "split": train.split_info},
                           os.path.join(args.out, f"ckpt_{step:06d}.pt"))

    torch.save({"step": step, "model": model.state_dict(),
                "ema": ema.state_dict(), "args": vars(args),
                "split": train.split_info},
               os.path.join(args.out, "ckpt_final.pt"))
    print(f"[train] done at step {step} in {(time.time()-t0)/60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
