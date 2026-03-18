"""
intentformer.py
Full IntentFormer-3D model: wires all modules together.

Forward pass
------------
Input
  history    (B, T_hist, 6)    — agent-centric motion history
  neighbours (B, N, T_hist, 6) — neighbour histories
  nbr_mask   (B, N)            — True = valid neighbour
  lidar_feat (B, 6)            — LiDAR context vector
  nbr_dists  (B, N) [optional] — distances to neighbours

Output  (IntentFormerOutput namedtuple)
  intent_logits (B, 4)
  mu            (B, K, T_fut, 2)
  sigma         (B, K, T_fut, 2)
  rho           (B, K, T_fut)
  pi_logits     (B, K)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .temporal_encoder import TemporalEncoder
from .social_attention import SocialAttention
from .intent_head import IntentHead
from .gmm_head import GMMHead


# ──────────────────────────────────────────────────────────────────────────────
# Output container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class IntentFormerOutput:
    intent_logits: torch.Tensor   # (B, num_intent_classes)
    mu: torch.Tensor              # (B, K, T_fut, 2)
    sigma: torch.Tensor           # (B, K, T_fut, 2)
    rho: torch.Tensor             # (B, K, T_fut)
    pi_logits: torch.Tensor       # (B, K)


# ──────────────────────────────────────────────────────────────────────────────
# IntentFormer-3D
# ──────────────────────────────────────────────────────────────────────────────

class IntentFormer(nn.Module):
    """
    Unified Intent + Trajectory Prediction model.

    Parameters
    ----------
    d_input : int
        Input feature dimension per history step (default 6).
    d_model : int
        Transformer / social-attention model width (default 128).
    n_heads : int
        Number of attention heads (default 4).
    n_temporal_layers : int
        Stacked encoder layers in the temporal branch (default 2).
    d_ff : int
        Feed-forward hidden size inside transformers (default 256).
    dropout : float
        Dropout rate (default 0.1).
    d_lidar : int
        Dimension of LiDAR context features (default 6).
    T_fut : int
        Number of future prediction steps (default 6).
    K : int
        Number of GMM mixture components (default 3).
    num_intent_classes : int
        Number of intent classes (default 4).
    max_neighbours : int
        Max number of neighbours to attend over (default 10).
    """

    def __init__(
        self,
        d_input: int = 6,
        d_model: int = 128,
        n_heads: int = 4,
        n_temporal_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        d_lidar: int = 6,
        T_fut: int = 6,
        K: int = 3,
        num_intent_classes: int = 4,
        max_neighbours: int = 10,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.max_neighbours = max_neighbours

        # ── 1. Temporal encoder (focal agent) ─────────────────────────────────
        self.temporal_enc = TemporalEncoder(
            d_input=d_input,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_temporal_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        # ── 2. Temporal encoder (neighbours — shared weights) ─────────────────
        self.neighbour_enc = self.temporal_enc          # weight sharing

        # ── 3. Social attention ────────────────────────────────────────────────
        self.social_attn = SocialAttention(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            use_distance_bias=True,
        )

        # ── 4. Fusion: concat focal summary + social context → project ────────
        d_fused = d_model * 2                          # temporal + social
        self.fusion_proj = nn.Sequential(
            nn.Linear(d_fused, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── 5. Intent head ─────────────────────────────────────────────────────
        self.intent_head = IntentHead(
            d_model=d_model * 2,
            d_lidar=d_lidar,
            d_hidden=d_model,
            num_classes=num_intent_classes,
            dropout=dropout,
        )

        # ── 6. GMM trajectory head ────────────────────────────────────────────
        self.gmm_head = GMMHead(
            d_model=d_model * 2,
            T_fut=T_fut,
            K=K,
            d_hidden=d_model * 2,
            dropout=dropout,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        history: torch.Tensor,
        neighbours: torch.Tensor,
        nbr_mask: torch.Tensor,
        lidar_feat: torch.Tensor,
        nbr_dists: Optional[torch.Tensor] = None,
    ) -> IntentFormerOutput:
        """
        Parameters
        ----------
        history    : (B, T_hist, d_input)
        neighbours : (B, N, T_hist, d_input)
        nbr_mask   : (B, N)  — bool, True = valid
        lidar_feat : (B, d_lidar)
        nbr_dists  : (B, N) or None

        Returns
        -------
        IntentFormerOutput
        """
        B, T_hist, _ = history.shape
        N = neighbours.shape[1]

        # ── Encode focal agent history ──────────────────────────────────────
        _, focal_summary = self.temporal_enc(history)   # (B, d_model)

        # ── Encode neighbours ───────────────────────────────────────────────
        # Flatten (B, N, T, d) → (B*N, T, d), encode, reshape back
        nbr_flat = neighbours.view(B * N, T_hist, -1)
        _, nbr_summary_flat = self.neighbour_enc(nbr_flat)  # (B*N, d_model)
        nbr_summaries = nbr_summary_flat.view(B, N, self.d_model)   # (B, N, d_model)

        # ── Social attention ────────────────────────────────────────────────
        social_ctx = self.social_attn(
            focal=focal_summary,
            neighbours=nbr_summaries,
            nbr_mask=nbr_mask,
            nbr_distances=nbr_dists,
        )                                               # (B, d_model)

        # ── Fuse temporal + social ──────────────────────────────────────────
        fused = self.fusion_proj(
            torch.cat([focal_summary, social_ctx], dim=-1)
        )                                               # (B, d_model*2)

        # ── Intent head ─────────────────────────────────────────────────────
        intent_logits = self.intent_head(fused, lidar_feat)  # (B, num_classes)

        # ── GMM trajectory head ─────────────────────────────────────────────
        mu, sigma, rho, pi_logits = self.gmm_head(fused)

        return IntentFormerOutput(
            intent_logits=intent_logits,
            mu=mu,
            sigma=sigma,
            rho=rho,
            pi_logits=pi_logits,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Convenience
    # ──────────────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, T_hist, N, d_input = 4, 5, 10, 6

    model = IntentFormer(
        d_input=d_input, d_model=128, n_heads=4,
        n_temporal_layers=2, K=3, T_fut=6,
    )
    print(f"Parameters: {model.count_parameters():,}")

    history = torch.randn(B, T_hist, d_input)
    neighbours = torch.randn(B, N, T_hist, d_input)
    nbr_mask = torch.ones(B, N, dtype=torch.bool)
    nbr_mask[:, 5:] = False
    lidar_feat = torch.randn(B, 6)
    nbr_dists = torch.rand(B, N) * 30.0

    out = model(history, neighbours, nbr_mask, lidar_feat, nbr_dists)
    print("intent_logits:", out.intent_logits.shape)   # (4, 4)
    print("mu           :", out.mu.shape)              # (4, 3, 6, 2)
    print("sigma        :", out.sigma.shape)           # (4, 3, 6, 2)
    print("rho          :", out.rho.shape)             # (4, 3, 6)
    print("pi_logits    :", out.pi_logits.shape)       # (4, 3)
