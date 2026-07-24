"""Detect and repair corrupted Adam/AdamW state.

On gfx1201 (RDNA4) this project sees sporadic numerical corruption in the fused
multi-tensor optimizer kernels: ``exp_avg_sq`` — an exponential moving average
of *squared* gradients, which is non-negative by definition — comes back with
negative entries.  AdamW then evaluates ``sqrt(exp_avg_sq)``, gets NaN, and the
very next step writes NaN into every weight.  The failure is silent and
unrecoverable: the loss only goes non-finite *after* the model is already dead,
so a loss-based guard always reports it one step too late.

The corruption is rare per step but certain over a long run, so the practical
answer is to check the invariant and repair it rather than to abort.  Clamping a
negative second moment to zero is the conservative repair: it makes that
parameter behave as if it had no gradient history, which Adam recovers from
within a few steps.

See docs/HARDWARE_NOTES_RDNA4.md for how this was diagnosed.
"""
from __future__ import annotations

import torch


def check_optimizer(opt) -> dict:
    """Count invalid entries in an optimizer's state without modifying it."""
    bad_sq = bad_finite = total = 0
    max_avg = 0.0
    for st in opt.state.values():
        sq = st.get("exp_avg_sq")
        if sq is not None:
            total += sq.numel()
            bad_sq += int((sq < 0).sum())
            bad_finite += int((~torch.isfinite(sq)).sum())
        av = st.get("exp_avg")
        if av is not None:
            bad_finite += int((~torch.isfinite(av)).sum())
            max_avg = max(max_avg, float(av.abs().max()))
    return {"negative_exp_avg_sq": bad_sq, "non_finite": bad_finite,
            "total": total, "max_abs_exp_avg": max_avg,
            "ok": bad_sq == 0 and bad_finite == 0}


def sanitize_optimizer(opt, max_exp_avg: float = 0.0) -> dict:
    """Repair corrupted state in place.  Returns what was fixed.

    **Both** moments of an offending entry are reset together.  Zeroing only
    ``exp_avg_sq`` would be worse than the corruption it repairs: AdamW divides
    by ``sqrt(exp_avg_sq) + eps``, so an entry with a live ``exp_avg`` and a
    zeroed second moment takes a step of order ``lr * exp_avg / eps`` — about
    1e8 times too large — and destroys the weight on the spot.  Resetting the
    pair makes the parameter behave as if it had no history, which contributes
    no update at all and rebuilds within a few steps.

    ``max_exp_avg`` optionally also resets implausibly large first moments.
    With gradient clipping at norm ``c`` no element of ``exp_avg`` can exceed
    ``c``, so a larger value is itself evidence of corruption.
    """
    reset = 0
    for st in opt.state.values():
        sq, av = st.get("exp_avg_sq"), st.get("exp_avg")
        if sq is None or av is None:
            continue
        bad = (sq < 0) | ~torch.isfinite(sq) | ~torch.isfinite(av)
        if max_exp_avg > 0:
            bad = bad | (av.abs() > max_exp_avg)
        n = int(bad.sum())
        if n:
            sq[bad] = 0.0
            av[bad] = 0.0
            reset += n
    return {"entries_reset": reset, "repaired": reset > 0}
