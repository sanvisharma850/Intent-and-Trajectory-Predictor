"""
intent_head.py
Classification head that predicts agent intent.

Intent classes
--------------
0 — STRAIGHT
1 — TURNING
2 — CROSSING
3 — WAITING
"""

from __future__ import annotations

import torch
import torch.nn as nn


NUM_INTENT_CLASSES = 4
INTENT_NAMES = ["STRAIGHT", "TURNING", "CROSSING", "WAITING"]


class IntentHead(nn.Module):
    """
    MLP classification head for intent prediction.

    Takes the fused agent representation and outputs a probability distribution
    over intent classes via a softmax.

    Architecture
    ------------
    Linear(d_model + d_lidar, d_hidden) → GELU → Dropout
    → Linear(d_hidden, d_hidden // 2) → GELU → Dropout
    → Linear(d_hidden // 2, num_classes)

    Parameters
    ----------
    d_model : int
        Dimension of the fused agent context (temporal + social).
    d_lidar : int
        Dimension of the LiDAR feature vector (default 6).
    d_hidden : int
        Hidden layer width (default 128).
    num_classes : int
        Number of intent classes (default 4).
    dropout : float
        Dropout probability (default 0.1).
    """

    def __init__(
        self,
        d_model: int = 256,
        d_lidar: int = 6,
        d_hidden: int = 128,
        num_classes: int = NUM_INTENT_CLASSES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        d_in = d_model + d_lidar

        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden // 2, num_classes),
        )

        self.num_classes = num_classes
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        fused: torch.Tensor,
        lidar_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        fused : torch.Tensor, shape (B, d_model)
            Fused agent context from temporal + social branches.
        lidar_feat : torch.Tensor, shape (B, d_lidar)
            LiDAR context features.

        Returns
        -------
        logits : torch.Tensor, shape (B, num_classes)
            Raw (un-normalised) classification scores.
            Apply softmax for probabilities or pass directly to
            ``F.cross_entropy``.
        """
        x = torch.cat([fused, lidar_feat], dim=-1)     # (B, d_model + d_lidar)
        return self.mlp(x)                              # (B, num_classes)

    def predict(
        self,
        fused: torch.Tensor,
        lidar_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method that returns both probabilities and predicted class.

        Returns
        -------
        probs : torch.Tensor, shape (B, num_classes)
        pred  : torch.Tensor, shape (B,)  — argmax class index
        """
        logits = self.forward(fused, lidar_feat)
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        return probs, pred


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B = 8
    head = IntentHead(d_model=256, d_lidar=6, d_hidden=128)
    fused = torch.randn(B, 256)
    lidar = torch.randn(B, 6)
    logits = head(fused, lidar)
    print("logits:", logits.shape)    # (8, 4)
    probs, pred = head.predict(fused, lidar)
    print("probs:", probs.shape)      # (8, 4)
    print("pred:", pred)              # (8,)
