"""
metrics.py
Evaluation metrics for multi-modal trajectory prediction.

Implements
----------
minADE@K  — minimum Average Displacement Error over K predicted modes
minFDE@K  — minimum Final Displacement Error over K predicted modes
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Displacement error helpers
# ──────────────────────────────────────────────────────────────────────────────

def ade(
    pred: torch.Tensor,   # (T_fut, 2) — a single predicted trajectory
    gt: torch.Tensor,     # (T_fut, 2) — ground truth
) -> torch.Tensor:
    """Average Displacement Error for a single trajectory."""
    return torch.norm(pred - gt, dim=-1).mean()


def fde(
    pred: torch.Tensor,   # (T_fut, 2)
    gt: torch.Tensor,     # (T_fut, 2)
) -> torch.Tensor:
    """Final Displacement Error for a single trajectory."""
    return torch.norm(pred[-1] - gt[-1], dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# minADE@K and minFDE@K (batched)
# ──────────────────────────────────────────────────────────────────────────────

def min_ade_k(
    predictions: torch.Tensor,   # (B, K, T_fut, 2) — K candidate trajectories
    ground_truth: torch.Tensor,  # (B, T_fut, 2)
    k: Optional[int] = None,     # use all K if None
) -> torch.Tensor:
    """
    Compute minADE@K.

    For each sample, pick the *k* modes with the highest predicted probability
    (or the first *k* if no ordering is available) and return the ADE of the
    best one.

    Parameters
    ----------
    predictions : (B, K, T_fut, 2)
    ground_truth : (B, T_fut, 2)
    k : int or None — number of modes to consider

    Returns
    -------
    scalar tensor — mean minADE@k over the batch
    """
    B, K, T, _ = predictions.shape
    if k is not None:
        K = min(k, K)
        predictions = predictions[:, :K]              # (B, k, T, 2)

    gt_exp = ground_truth.unsqueeze(1).expand_as(predictions)   # (B, k, T, 2)
    errors = torch.norm(predictions - gt_exp, dim=-1).mean(dim=-1)  # (B, k)
    min_errors = errors.min(dim=-1).values                           # (B,)
    return min_errors.mean()


def min_fde_k(
    predictions: torch.Tensor,   # (B, K, T_fut, 2)
    ground_truth: torch.Tensor,  # (B, T_fut, 2)
    k: Optional[int] = None,
) -> torch.Tensor:
    """
    Compute minFDE@K.

    Parameters
    ----------
    predictions : (B, K, T_fut, 2)
    ground_truth : (B, T_fut, 2)
    k : int or None

    Returns
    -------
    scalar tensor — mean minFDE@k over the batch
    """
    B, K, T, _ = predictions.shape
    if k is not None:
        K = min(k, K)
        predictions = predictions[:, :K]

    final_pred = predictions[:, :, -1, :]                     # (B, k, 2)
    final_gt   = ground_truth[:, -1, :].unsqueeze(1).expand_as(final_pred)  # (B, k, 2)
    errors = torch.norm(final_pred - final_gt, dim=-1)        # (B, k)
    min_errors = errors.min(dim=-1).values                    # (B,)
    return min_errors.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Intent accuracy
# ──────────────────────────────────────────────────────────────────────────────

def intent_accuracy(
    intent_logits: torch.Tensor,   # (B, num_classes)
    intent_labels: torch.Tensor,   # (B,)
) -> torch.Tensor:
    """Top-1 classification accuracy for intent."""
    preds = intent_logits.argmax(dim=-1)
    return (preds == intent_labels).float().mean()


# ──────────────────────────────────────────────────────────────────────────────
# Aggregated metric computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    predictions: torch.Tensor,      # (B, K, T_fut, 2)
    ground_truth: torch.Tensor,     # (B, T_fut, 2)
    intent_logits: torch.Tensor,    # (B, num_classes)
    intent_labels: torch.Tensor,    # (B,)
    k_values: tuple[int, ...] = (1, 3, 6),
) -> dict:
    """
    Compute all evaluation metrics for a batch.

    Returns
    -------
    dict with keys:
        "minADE@{k}"  for k in k_values
        "minFDE@{k}"  for k in k_values
        "intent_acc"
    """
    results: dict = {}
    for k in k_values:
        results[f"minADE@{k}"] = min_ade_k(predictions, ground_truth, k=k).item()
        results[f"minFDE@{k}"] = min_fde_k(predictions, ground_truth, k=k).item()
    results["intent_acc"] = intent_accuracy(intent_logits, intent_labels).item()
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, K, T = 8, 3, 6
    preds = torch.randn(B, K, T, 2)
    gt = torch.randn(B, T, 2)
    int_logits = torch.randn(B, 4)
    int_labels = torch.randint(0, 4, (B,))

    m = compute_metrics(preds, gt, int_logits, int_labels)
    for k, v in m.items():
        print(f"  {k}: {v:.4f}")
