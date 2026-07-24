# Training diffusion models on RDNA4 (gfx1201): two failure modes

Operational notes from training conditional generative models at 128×128 on a
Radeon AI PRO R9700 (gfx1201 / RDNA4) under ROCm 7.2 + PyTorch 2.12.1.

Both bugs below cost days to diagnose, produce **silent** corruption rather than
a clean error, and are not specific to this project — anyone training on RDNA4
may hit them. Neither is a claim about model quality; they are reproducible
engineering observations.

---

## 1. Half-precision convolution kernels emit intermittent NaNs

**Symptom.** Training at 128px produced NaN loss almost immediately. The same
code at 64px trained fine.

**What it is not.** It is *not* the rocBLAS large-GEMM issue: bf16 GEMM was
verified correct on torch 2.12.1 for M up to ~4M elements, and forcing the
rocBLAS backend (`TORCH_BLAS_PREFER_HIPBLASLT=0`) does not help.

**What it is.** The half-precision **convolution** kernels on gfx1201. They emit
intermittent non-finite values at 128px under **both bf16 and fp16**,
independent of input magnitude and non-deterministically — re-running the
identical forward pass on identical weights sometimes returns finite values.
This has the signature of an immature MIOpen/driver kernel, not numerical
overflow. Disabling MIOpen implicit-GEMM does not help.

**Fix.** Train and sample in fp32:

```bash
python scripts/train_pilot.py --precision fp32     # disables autocast
COMP2D_SAMPLE_PRECISION=fp32 python scripts/eval_graph.py ...
```

64px was unaffected and can stay in bf16.

**Warning about a tempting dead end.** A hand-built "patched rocBLAS" is both
irrelevant here (rocBLAS is not the culprit) and unusable: it is ABI-incompatible
with the PyTorch wheel's bundled copy and aborts the process with
`free(): invalid pointer`.

## 2. Fused AdamW corrupts optimizer state (the expensive one)

fp32 is **necessary but not sufficient.** Long runs still died — around step
10k — despite `--precision fp32`.

**Root cause.** Inspecting the checkpoint showed the optimizer state contained
**negative `exp_avg_sq` entries** (119,564 of them in one case). `exp_avg_sq` is
an exponential moving average of *squared* gradients and cannot be negative. The
fused multi-tensor AdamW kernels corrupt it. AdamW then evaluates
`sqrt(negative) = NaN`, and the next `opt.step()` writes NaN into every weight.

**Why it is so hard to catch.** The model weights, the EMA weights, the
optimizer's finiteness check, and the entire dataset all pass an `isfinite`
test. The loss only goes non-finite *after* the weights are already destroyed,
so any loss-based guard reports the failure one step too late, and the saved
checkpoint looks perfectly clean.

**Fix.** [`src/ml/optim_guard.py`](../src/ml/optim_guard.py) checks the
invariant directly and repairs it, wired in via `--opt-check-every`.

```python
from src.ml.optim_guard import check_optimizer, sanitize_optimizer
chk = check_optimizer(opt)          # counts negative / non-finite entries
if not chk["ok"]:
    sanitize_optimizer(opt, max_exp_avg=1.0)
```

**Critical detail — reset both moments together.** Zeroing only `exp_avg_sq` is
*worse* than the corruption it repairs. AdamW divides by
`sqrt(exp_avg_sq) + eps`, so an entry with a live `exp_avg` and a zeroed second
moment takes a step of order `lr * exp_avg / eps` — roughly 1e8 times too large.
Measured: that mistake drove training loss from 0.046 to 0.97 in 100 steps.
Resetting the *pair* makes the parameter behave as if it had no gradient
history, which Adam rebuilds within a few steps.

Since gradient clipping bounds `|exp_avg|` by the clip norm, an `exp_avg` larger
than that is itself evidence of corruption — hence the `max_exp_avg` threshold.

## 3. Related guards this motivated

- **A finite loss does not imply finite gradients.** `clip_grad_norm_` rescales
  *every* gradient by a non-finite total norm, so one bad micro-batch poisons the
  whole model. The optimizer step is now gated on `torch.isfinite(gnorm)`.
- **Guard on NaN *rate*, not lifetime count.** With a sporadic hardware fault at
  roughly 0.5% of steps, a lifetime counter guarantees a false abort on any long
  run. The trainer uses a consecutive-failure limit plus a windowed rate.
- **Never overwrite a good checkpoint with poisoned weights.** Checkpointing
  refuses to write if any parameter is non-finite.
- **`sqrt` at zero has infinite derivative.** Unrelated to the hardware, but the
  same class of silent NaN: in the EGNN coordinate update, `d2.sqrt()` is exactly
  zero on the diagonal, so masked-out self-distances still produce NaN gradients
  (`0 * inf`). The epsilon must go *inside* the sqrt.
