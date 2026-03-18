"""
temporal_encoder.py
Transformer encoder that processes an agent's 2-second motion history.

Input  : (B, T, d_input)  — batch of history feature sequences
Output : (B, T, d_model)  — contextualised representations
         (B, d_model)      — sequence summary (mean-pooled CLS token)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# Positional encoding (sinusoidal)
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding from *Attention Is All You Need*."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)               # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ──────────────────────────────────────────────────────────────────────────────
# Temporal Encoder
# ──────────────────────────────────────────────────────────────────────────────

class TemporalEncoder(nn.Module):
    """
    Transformer encoder for temporal motion history.

    Architecture
    ------------
    1. Linear projection: d_input → d_model
    2. Sinusoidal positional encoding
    3. N stacked Transformer encoder layers
    4. Output: per-timestep features + pooled summary

    Parameters
    ----------
    d_input : int
        Dimensionality of each input feature vector (default 6).
    d_model : int
        Internal model dimension (default 128).
    n_heads : int
        Number of multi-head attention heads (default 4).
    n_layers : int
        Number of stacked encoder layers (default 2).
    d_ff : int
        Feed-forward hidden dimension (default 256).
    dropout : float
        Dropout probability (default 0.1).
    max_seq_len : int
        Maximum expected sequence length (default 64).
    """

    def __init__(
        self,
        d_input: int = 6,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 64,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(d_input, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len=max_seq_len, dropout=dropout
        )

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # pre-norm for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Learnable CLS token appended at the end of each sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    # ──────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, T, d_input)
            Batch of agent history sequences.
        src_key_padding_mask : torch.Tensor or None, shape (B, T+1)
            Boolean mask: True = position is padding (ignored).
            If None, no masking is applied.

        Returns
        -------
        seq_out : torch.Tensor, shape (B, T, d_model)
            Per-timestep encoder representations (CLS token excluded).
        summary : torch.Tensor, shape (B, d_model)
            Summary vector from the CLS token position.
        """
        B, T, _ = x.shape

        # Project input features to model dimension
        x = self.input_proj(x)                          # (B, T, d_model)

        # Append CLS token
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, d_model)
        x = torch.cat([x, cls], dim=1)                  # (B, T+1, d_model)

        # Positional encoding
        x = self.pos_enc(x)                             # (B, T+1, d_model)

        # Transformer encoder
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Split back
        seq_out = x[:, :T, :]                           # (B, T,   d_model)
        summary = x[:, T, :]                            # (B,      d_model)

        return seq_out, summary


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    enc = TemporalEncoder(d_input=6, d_model=128, n_heads=4, n_layers=2)
    dummy = torch.randn(4, 5, 6)  # batch=4, T=5, d_input=6
    seq, summary = enc(dummy)
    print("seq_out  :", seq.shape)    # (4, 5, 128)
    print("summary  :", summary.shape)  # (4, 128)
