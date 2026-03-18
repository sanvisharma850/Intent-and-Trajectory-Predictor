"""
gmm_head.py
Gaussian Mixture Model trajectory prediction head.

Outputs per-mixture-component parameters for K bivariate Gaussians:
  μ  — mean trajectory  (K, T_fut, 2)
  σ  — log-std          (K, T_fut, 2)  → exp() to get std
  ρ  — correlation      (K, T_fut)     → tanh() to get ρ ∈ (-1, 1)
  π  — mixture weights  (K,)           → softmax

At inference, the most-likely mode or top-K samples can be drawn.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GMMHead(nn.Module):
    """
    Gaussian Mixture Model head for multi-modal trajectory prediction.

    Architecture
    ------------
    Shared MLP(d_model → d_hidden → d_hidden)
    ↓
    ┌── μ  branch: Linear → (K, T_fut, 2)
    ├── σ  branch: Linear → (K, T_fut, 2)  [log-scale, use softplus/exp later]
    ├── ρ  branch: Linear → (K, T_fut)     [pre-tanh]
    └── π  branch: Linear → (K,)           [pre-softmax logits]

    Parameters
    ----------
    d_model : int
        Fused agent representation dimension.
    T_fut : int
        Number of future prediction steps.
    K : int
        Number of mixture components (modes).
    d_hidden : int
        MLP hidden dimension.
    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 256,
        T_fut: int = 6,
        K: int = 3,
        d_hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.K = K
        self.T_fut = T_fut

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output heads
        self.mu_head = nn.Linear(d_hidden, K * T_fut * 2)      # mean (x, y)
        self.log_sigma_head = nn.Linear(d_hidden, K * T_fut * 2)  # log std
        self.rho_head = nn.Linear(d_hidden, K * T_fut)         # correlation (pre-tanh)
        self.pi_head = nn.Linear(d_hidden, K)                  # mixture weights

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialise log_sigma bias to log(1) = 0 → σ starts at 1
        nn.init.zeros_(self.log_sigma_head.bias)

    def forward(
        self, fused: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        fused : torch.Tensor, shape (B, d_model)
            Fused agent context.

        Returns
        -------
        mu        : (B, K, T_fut, 2)   — predicted mean trajectory per mode
        sigma     : (B, K, T_fut, 2)   — predicted std  (always > 0, softplus)
        rho       : (B, K, T_fut)      — predicted correlation ∈ (-1, 1)
        pi_logits : (B, K)             — mixture weight logits (pre-softmax)
        """
        B = fused.shape[0]
        h = self.trunk(fused)                            # (B, d_hidden)

        # ── μ ─────────────────────────────────────────────────────────────────
        mu = self.mu_head(h).view(B, self.K, self.T_fut, 2)

        # ── σ  (softplus for numerical stability, ensures σ > 0) ─────────────
        log_sigma = self.log_sigma_head(h).view(B, self.K, self.T_fut, 2)
        sigma = torch.nn.functional.softplus(log_sigma) + 1e-5

        # ── ρ  (tanh to ensure ρ ∈ (-1, 1)) ──────────────────────────────────
        rho_raw = self.rho_head(h).view(B, self.K, self.T_fut)
        rho = torch.tanh(rho_raw) * 0.99                # avoid ±1 exactly

        # ── π  (raw logits; apply softmax in loss / sampling) ─────────────────
        pi_logits = self.pi_head(h)                     # (B, K)

        return mu, sigma, rho, pi_logits

    # ──────────────────────────────────────────────────────────────────────
    # Sampling utilities
    # ──────────────────────────────────────────────────────────────────────

    def sample_trajectories(
        self,
        fused: torch.Tensor,
        n_samples: int = 6,
    ) -> torch.Tensor:
        """
        Draw *n_samples* trajectory samples by:
        1. Sampling a mode index from π.
        2. Drawing from the corresponding bivariate Gaussian at each step.

        Returns
        -------
        samples : torch.Tensor, shape (B, n_samples, T_fut, 2)
        """
        mu, sigma, rho, pi_logits = self.forward(fused)
        B = fused.shape[0]
        pi = torch.softmax(pi_logits, dim=-1)           # (B, K)

        # Sample mode indices
        mode_idx = torch.multinomial(pi, n_samples, replacement=True)  # (B, n_samples)

        trajectories = []
        for s in range(n_samples):
            idx = mode_idx[:, s]                        # (B,)
            mu_s = mu[torch.arange(B), idx]             # (B, T_fut, 2)
            sx = sigma[torch.arange(B), idx, :, 0]     # (B, T_fut)
            sy = sigma[torch.arange(B), idx, :, 1]     # (B, T_fut)
            r = rho[torch.arange(B), idx]               # (B, T_fut)

            # Sample from bivariate Gaussian using Cholesky parametrisation
            eps1 = torch.randn_like(sx)
            eps2 = torch.randn_like(sy)
            x = mu_s[:, :, 0] + sx * eps1
            y = mu_s[:, :, 1] + sy * (r * eps1 + torch.sqrt(1 - r ** 2) * eps2)
            traj = torch.stack([x, y], dim=-1)          # (B, T_fut, 2)
            trajectories.append(traj)

        return torch.stack(trajectories, dim=1)         # (B, n_samples, T_fut, 2)

    def best_mode(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Return the mean trajectory of the most-likely mixture component.

        Returns
        -------
        torch.Tensor, shape (B, T_fut, 2)
        """
        mu, _, _, pi_logits = self.forward(fused)
        best_idx = pi_logits.argmax(dim=-1)             # (B,)
        B = fused.shape[0]
        return mu[torch.arange(B), best_idx]            # (B, T_fut, 2)


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B = 4
    head = GMMHead(d_model=256, T_fut=6, K=3)
    fused = torch.randn(B, 256)
    mu, sigma, rho, pi = head(fused)
    print("mu     :", mu.shape)        # (4, 3, 6, 2)
    print("sigma  :", sigma.shape)     # (4, 3, 6, 2)
    print("rho    :", rho.shape)       # (4, 3, 6)
    print("pi     :", pi.shape)        # (4, 3)
    samples = head.sample_trajectories(fused, n_samples=6)
    print("samples:", samples.shape)   # (4, 6, 6, 2)
