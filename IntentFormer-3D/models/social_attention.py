"""
social_attention.py
Structured cross-attention over neighbour agents.

The focal agent's representation attends to its neighbours' representations,
producing a socially-aware context vector that captures inter-agent interactions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Social Attention Module
# ──────────────────────────────────────────────────────────────────────────────

class SocialAttention(nn.Module):
    """
    Cross-attention block: focal agent queries neighbour agents.

    Architecture
    ------------
    • Multi-head cross-attention (query = focal, key/value = neighbours)
    • Additive residual connection + LayerNorm
    • Two-layer feed-forward network with residual + LayerNorm
    • Optional distance-based attention bias to give closer agents higher weight

    Parameters
    ----------
    d_model : int
        Model embedding dimension (must match TemporalEncoder output).
    n_heads : int
        Number of attention heads.
    d_ff : int
        Hidden dimension of the feed-forward block.
    dropout : float
        Dropout probability.
    use_distance_bias : bool
        If True, adds a distance-based attention logit bias so that
        spatially closer neighbours receive higher attention weights.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_distance_bias: bool = True,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.use_distance_bias = use_distance_bias

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward block
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # Optional distance bias MLP: scalar bias per neighbour
        if use_distance_bias:
            self.dist_bias_mlp = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, n_heads),
            )

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        focal: torch.Tensor,
        neighbours: torch.Tensor,
        nbr_mask: torch.Tensor | None = None,
        nbr_distances: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        focal : torch.Tensor, shape (B, d_model)
            Summary embedding of the focal agent.
        neighbours : torch.Tensor, shape (B, N, d_model)
            Summary embeddings of neighbour agents.
        nbr_mask : torch.Tensor or None, shape (B, N)
            Boolean mask: True = **valid** neighbour, False = padding.
            Passed as *key_padding_mask* (inverted) to MultiheadAttention.
        nbr_distances : torch.Tensor or None, shape (B, N)
            Euclidean distance from focal agent to each neighbour (metres).
            Used to compute the optional distance-based attention bias.

        Returns
        -------
        out : torch.Tensor, shape (B, d_model)
            Socially-aware representation of the focal agent.
        """
        B, d = focal.shape
        N = neighbours.shape[1]

        # Reshape focal to (B, 1, d_model) for attention query
        query = focal.unsqueeze(1)                       # (B, 1, d_model)

        # Build attention bias from distances
        attn_bias: torch.Tensor | None = None
        if self.use_distance_bias and nbr_distances is not None:
            # dist: (B, N) → (B, N, 1) → MLP → (B, N, n_heads)
            dist_in = nbr_distances.unsqueeze(-1).float()
            dist_bias = self.dist_bias_mlp(dist_in)      # (B, N, n_heads)
            # Rearrange to (B * n_heads, 1, N) for attn_mask in MHA
            dist_bias = dist_bias.permute(0, 2, 1)       # (B, n_heads, N)
            attn_bias = dist_bias.reshape(B * self.n_heads, 1, N)

        # Key-padding mask for MHA: shape (B, N), True = ignore
        key_padding_mask: torch.Tensor | None = None
        if nbr_mask is not None:
            key_padding_mask = ~nbr_mask                 # invert: True = pad

        # Cross-attention (pre-norm style)
        residual = query
        query_normed = self.norm1(query)

        attn_out, _ = self.cross_attn(
            query=query_normed,
            key=neighbours,
            value=neighbours,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_bias,
            need_weights=False,
        )
        query = residual + attn_out                      # (B, 1, d_model)

        # Feed-forward (pre-norm style)
        residual = query
        query = residual + self.ff(self.norm2(query))    # (B, 1, d_model)

        return query.squeeze(1)                          # (B, d_model)


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, N, d = 4, 10, 128
    focal = torch.randn(B, d)
    nbrs = torch.randn(B, N, d)
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[:, 5:] = False  # last 5 are padding
    dists = torch.rand(B, N) * 30.0

    sa = SocialAttention(d_model=d, n_heads=4)
    out = sa(focal, nbrs, nbr_mask=mask, nbr_distances=dists)
    print("social_out:", out.shape)   # (4, 128)
