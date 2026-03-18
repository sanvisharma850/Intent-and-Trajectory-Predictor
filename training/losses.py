"""
losses.py
Loss functions for IntentFormer-3D.

Components
----------
gmm_nll_loss   — negative log-likelihood of a bivariate Gaussian mixture
intent_ce_loss — cross-entropy intent classification loss
total_loss     — weighted combination of both
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Bivariate Gaussian NLL helpers
# ──────────────────────────────────────────────────────────────────────────────

LOG_2PI = torch.log(torch.tensor(2.0 * 3.141592653589793))


def _bivariate_gaussian_log_prob(
    y: torch.Tensor,      # (B, T, 2)  — ground-truth positions
    mu: torch.Tensor,     # (B, K, T, 2)
    sigma: torch.Tensor,  # (B, K, T, 2) — strictly positive
    rho: torch.Tensor,    # (B, K, T)    — ∈ (-1, 1)
) -> torch.Tensor:
    """
    Log-probability of *y* under a bivariate Gaussian for each mixture
    component and timestep.

    Returns
    -------
    log_prob : torch.Tensor, shape (B, K, T)
    """
    # Expand y to match the K-components dimension
    y_exp = y.unsqueeze(1).expand_as(mu)          # (B, K, T, 2)

    dx = (y_exp[..., 0] - mu[..., 0]) / (sigma[..., 0] + 1e-8)
    dy = (y_exp[..., 1] - mu[..., 1]) / (sigma[..., 1] + 1e-8)

    z = dx ** 2 - 2.0 * rho * dx * dy + dy ** 2

    one_minus_rho2 = 1.0 - rho ** 2 + 1e-8

    log_norm = (
        -LOG_2PI.to(y.device)
        - torch.log(sigma[..., 0] + 1e-8)
        - torch.log(sigma[..., 1] + 1e-8)
        - 0.5 * torch.log(one_minus_rho2)
    )

    log_prob = log_norm - 0.5 * z / one_minus_rho2  # (B, K, T)
    return log_prob


# ──────────────────────────────────────────────────────────────────────────────
# GMM NLL loss
# ──────────────────────────────────────────────────────────────────────────────

def gmm_nll_loss(
    future_gt: torch.Tensor,   # (B, T_fut, 2)
    mu: torch.Tensor,          # (B, K, T_fut, 2)
    sigma: torch.Tensor,       # (B, K, T_fut, 2)
    rho: torch.Tensor,         # (B, K, T_fut)
    pi_logits: torch.Tensor,   # (B, K)
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of *future_gt* under the GMM.

    Uses the log-sum-exp trick for numerical stability.

    Returns
    -------
    scalar tensor — mean NLL over the batch.
    """
    log_pi = F.log_softmax(pi_logits, dim=-1)      # (B, K)

    # Log-prob of ground truth under each component: (B, K, T)
    log_p = _bivariate_gaussian_log_prob(future_gt, mu, sigma, rho)

    # Sum over time steps (assuming temporal independence within a mode)
    log_p_sum = log_p.sum(dim=-1)                  # (B, K)

    # Log mixture weights + component log-probs
    log_mix = log_pi + log_p_sum                   # (B, K)

    # Log-sum-exp over K modes
    nll = -torch.logsumexp(log_mix, dim=-1)        # (B,)

    return nll.mean()


# ──────────────────────────────────────────────────────────────────────────────
# Intent classification loss
# ──────────────────────────────────────────────────────────────────────────────

def intent_ce_loss(
    intent_logits: torch.Tensor,   # (B, num_classes)
    intent_labels: torch.Tensor,   # (B,) long
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Cross-entropy loss for intent classification.

    Parameters
    ----------
    intent_logits : (B, num_classes)
    intent_labels : (B,) — integer class indices
    class_weights : (num_classes,) or None — per-class weights

    Returns
    -------
    scalar tensor
    """
    return F.cross_entropy(intent_logits, intent_labels, weight=class_weights)


# ──────────────────────────────────────────────────────────────────────────────
# Combined loss
# ──────────────────────────────────────────────────────────────────────────────

def total_loss(
    future_gt: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    rho: torch.Tensor,
    pi_logits: torch.Tensor,
    intent_logits: torch.Tensor,
    intent_labels: torch.Tensor,
    lambda_traj: float = 1.0,
    lambda_intent: float = 1.0,
    class_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Weighted sum of GMM NLL + Intent CE.

    Parameters
    ----------
    future_gt      : (B, T_fut, 2)
    mu             : (B, K, T_fut, 2)
    sigma          : (B, K, T_fut, 2)
    rho            : (B, K, T_fut)
    pi_logits      : (B, K)
    intent_logits  : (B, num_classes)
    intent_labels  : (B,)
    lambda_traj    : weight for GMM NLL
    lambda_intent  : weight for intent CE
    class_weights  : optional per-class weights for intent CE

    Returns
    -------
    loss_total  : scalar
    loss_traj   : scalar (GMM NLL, unweighted)
    loss_intent : scalar (intent CE, unweighted)
    """
    loss_traj = gmm_nll_loss(future_gt, mu, sigma, rho, pi_logits)
    loss_intent = intent_ce_loss(intent_logits, intent_labels, class_weights)
    loss_total = lambda_traj * loss_traj + lambda_intent * loss_intent
    return loss_total, loss_traj, loss_intent


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, K, T = 4, 3, 6

    future_gt = torch.randn(B, T, 2)
    mu = torch.randn(B, K, T, 2)
    sigma = torch.rand(B, K, T, 2) + 0.1
    rho = torch.tanh(torch.randn(B, K, T)) * 0.99
    pi_logits = torch.randn(B, K)
    intent_logits = torch.randn(B, 4)
    intent_labels = torch.randint(0, 4, (B,))

    traj_loss = gmm_nll_loss(future_gt, mu, sigma, rho, pi_logits)
    cls_loss = intent_ce_loss(intent_logits, intent_labels)
    tot, l_t, l_i = total_loss(
        future_gt, mu, sigma, rho, pi_logits, intent_logits, intent_labels
    )
    print(f"GMM NLL    : {traj_loss.item():.4f}")
    print(f"Intent CE  : {cls_loss.item():.4f}")
    print(f"Total      : {tot.item():.4f}")
