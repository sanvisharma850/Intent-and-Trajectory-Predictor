"""
scheduler.py
Learning-rate scheduler: linear warmup followed by cosine decay.
"""

from __future__ import annotations

import math
from torch.optim.lr_scheduler import LambdaLR


def get_warmup_cosine_scheduler(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Create a LambdaLR scheduler with:
      • Linear warmup from 0 → peak LR over *warmup_steps* steps.
      • Cosine decay from peak LR → min_lr_ratio * peak_LR over the remaining steps.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    warmup_steps : int
        Number of steps for the linear warmup phase.
    total_steps : int
        Total number of training steps (warmup + decay).
    min_lr_ratio : float
        Fraction of the base LR to use as the minimum (floor) value.
        0.0 means decay to zero; 0.01 means decay to 1 % of peak.

    Returns
    -------
    LambdaLR scheduler (call .step() each optimiser step, not each epoch).
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear ramp-up
            return float(current_step) / max(float(warmup_steps), 1)

        # Cosine decay phase
        progress = float(current_step - warmup_steps) / max(
            float(total_steps - warmup_steps), 1
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio + (1.0 - min_lr_ratio) * cosine, min_lr_ratio)

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ──────────────────────────────────────────────────────────────────────────────
# Epoch-based convenience wrapper
# ──────────────────────────────────────────────────────────────────────────────

def get_warmup_cosine_scheduler_epochs(
    optimizer,
    warmup_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """
    Same as :func:`get_warmup_cosine_scheduler` but specified in epochs.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    warmup_epochs : int
    total_epochs : int
    steps_per_epoch : int
        Number of optimizer steps per epoch.
    min_lr_ratio : float

    Returns
    -------
    LambdaLR scheduler
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    return get_warmup_cosine_scheduler(
        optimizer, warmup_steps, total_steps, min_lr_ratio
    )


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch

    dummy_param = [torch.nn.Parameter(torch.zeros(1))]
    opt = torch.optim.Adam(dummy_param, lr=1e-3)
    sched = get_warmup_cosine_scheduler(opt, warmup_steps=10, total_steps=100)

    lrs = []
    for step in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()

    print("LR at step  0 (warm-up start):", round(lrs[0], 6))
    print("LR at step 10 (warm-up end)  :", round(lrs[10], 6))
    print("LR at step 55 (mid cosine)   :", round(lrs[55], 6))
    print("LR at step 99 (end)          :", round(lrs[99], 6))
