"""
visualize.py
Visualisation utilities for IntentFormer-3D.

Provides
--------
plot_gmm_trajectories  — draws GMM ellipses and per-mode confidence bars
plot_intent_distribution — bar chart of intent class probabilities
save_visualization       — convenience wrapper that saves the figure to disk
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

INTENT_NAMES = ["STRAIGHT", "TURNING", "CROSSING", "WAITING"]
MODE_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
GROUND_TRUTH_COLOR = "black"
HISTORY_COLOR = "gray"

# Number of std-deviations for the ellipse radius
ELLIPSE_N_STD = 2.0


# ──────────────────────────────────────────────────────────────────────────────
# Ellipse helper
# ──────────────────────────────────────────────────────────────────────────────

def _confidence_ellipse(
    ax: plt.Axes,
    mu_x: float,
    mu_y: float,
    sigma_x: float,
    sigma_y: float,
    rho: float,
    n_std: float = ELLIPSE_N_STD,
    color: str = "tab:blue",
    alpha: float = 0.25,
    linewidth: float = 1.0,
) -> None:
    """
    Draw a 2-D confidence ellipse for a bivariate Gaussian.

    Parameters
    ----------
    mu_x, mu_y   — mean
    sigma_x, sigma_y — standard deviations (> 0)
    rho          — correlation coefficient ∈ (-1, 1)
    n_std        — radius multiplier in units of standard deviation
    """
    # Covariance matrix
    cov = np.array([
        [sigma_x ** 2,              rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y,   sigma_y ** 2],
    ])
    # Eigendecomposition for principal axes
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 0.0)      # numerical safety
    angle_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

    width  = 2.0 * n_std * np.sqrt(eigenvalues[0])
    height = 2.0 * n_std * np.sqrt(eigenvalues[1])

    ellipse = Ellipse(
        xy=(mu_x, mu_y),
        width=width,
        height=height,
        angle=np.degrees(angle_rad),
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=linewidth,
        linestyle="--",
        zorder=2,
    )
    ax.add_patch(ellipse)


# ──────────────────────────────────────────────────────────────────────────────
# Main visualisation
# ──────────────────────────────────────────────────────────────────────────────

def plot_gmm_trajectories(
    history: np.ndarray,         # (T_hist, 2) — agent-centric [x, y]
    future_gt: np.ndarray,       # (T_fut, 2)  — ground truth
    mu: np.ndarray,              # (K, T_fut, 2)
    sigma: np.ndarray,           # (K, T_fut, 2)
    rho: np.ndarray,             # (K, T_fut)
    pi: np.ndarray,              # (K,) — mixture weights (after softmax)
    intent_probs: Optional[np.ndarray] = None,   # (num_classes,)
    title: str = "IntentFormer-3D Prediction",
    ax: Optional[plt.Axes] = None,
    show_ellipses: bool = True,
    show_confidence_bars: bool = True,
) -> plt.Figure:
    """
    Plot GMM trajectory predictions with:
    • History path (grey)
    • Ground-truth future path (black)
    • K predicted mean trajectories (coloured lines)
    • Per-timestep 2σ confidence ellipses
    • Per-mode confidence bar chart (inset)
    • Optional intent probability bar chart

    Parameters
    ----------
    history     : (T_hist, 2)
    future_gt   : (T_fut, 2)
    mu          : (K, T_fut, 2)
    sigma       : (K, T_fut, 2)
    rho         : (K, T_fut)
    pi          : (K,) — already softmax-normalised
    intent_probs: (num_classes,) or None
    title       : figure title
    ax          : existing Axes or None (creates new figure)
    show_ellipses : bool
    show_confidence_bars : bool

    Returns
    -------
    matplotlib.figure.Figure
    """
    K = mu.shape[0]
    colors = [MODE_COLORS[k % len(MODE_COLORS)] for k in range(K)]

    if ax is None:
        n_cols = 2 if intent_probs is not None else 1
        fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6))
        if n_cols == 1:
            axes = [axes]
    else:
        fig = ax.get_figure()
        axes = [ax]

    traj_ax = axes[0]

    # ── History ───────────────────────────────────────────────────────────────
    traj_ax.plot(
        history[:, 0], history[:, 1],
        color=HISTORY_COLOR, alpha=0.5, linewidth=1.5,
        linestyle=":", marker="o", markersize=4, label="History",
    )

    # ── Ground truth future ───────────────────────────────────────────────────
    # Prepend last history point for visual continuity
    gt_path = np.concatenate([history[-1:], future_gt], axis=0)
    traj_ax.plot(
        gt_path[:, 0], gt_path[:, 1],
        color=GROUND_TRUTH_COLOR, linewidth=2.5,
        marker="x", markersize=6, label="Ground truth",
    )

    # ── Predicted modes ───────────────────────────────────────────────────────
    for k in range(K):
        c = colors[k]
        mode_path = np.concatenate([history[-1:], mu[k]], axis=0)
        alpha_line = float(np.clip(pi[k] * 2.0, 0.3, 1.0))

        traj_ax.plot(
            mode_path[:, 0], mode_path[:, 1],
            color=c, linewidth=2.0, alpha=alpha_line,
            label=f"Mode {k+1} (π={pi[k]:.2f})",
        )

        # Confidence ellipses at each future step
        if show_ellipses:
            for t in range(mu.shape[1]):
                _confidence_ellipse(
                    traj_ax,
                    mu_x=mu[k, t, 0], mu_y=mu[k, t, 1],
                    sigma_x=sigma[k, t, 0], sigma_y=sigma[k, t, 1],
                    rho=rho[k, t],
                    color=c, alpha=0.15,
                )

    traj_ax.set_aspect("equal")
    traj_ax.set_xlabel("x (m)")
    traj_ax.set_ylabel("y (m)")
    traj_ax.legend(loc="upper left", fontsize=8)
    traj_ax.set_title(title)
    traj_ax.grid(True, alpha=0.3)

    # ── Per-mode confidence inset bar ─────────────────────────────────────────
    if show_confidence_bars:
        inset_ax = traj_ax.inset_axes([0.65, 0.0, 0.34, 0.28])
        inset_ax.bar(range(K), pi, color=colors, alpha=0.8)
        inset_ax.set_xticks(range(K))
        inset_ax.set_xticklabels([f"M{k+1}" for k in range(K)], fontsize=7)
        inset_ax.set_ylim(0, 1)
        inset_ax.set_ylabel("π", fontsize=7)
        inset_ax.set_title("Mode weights", fontsize=7)
        inset_ax.tick_params(axis="both", labelsize=6)

    # ── Intent probability bar chart ──────────────────────────────────────────
    if intent_probs is not None and len(axes) > 1:
        intent_ax = axes[1]
        bars = intent_ax.bar(
            INTENT_NAMES[: len(intent_probs)],
            intent_probs,
            color=["tab:blue", "tab:orange", "tab:green", "tab:red"][: len(intent_probs)],
            alpha=0.8,
        )
        intent_ax.set_ylim(0, 1)
        intent_ax.set_ylabel("Probability")
        intent_ax.set_title("Intent Distribution")
        intent_ax.grid(axis="y", alpha=0.3)
        for bar, prob in zip(bars, intent_probs):
            intent_ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{prob:.2f}",
                ha="center", va="bottom", fontsize=9,
            )

    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Convenience save wrapper
# ──────────────────────────────────────────────────────────────────────────────

def save_visualization(
    save_path: str,
    history: np.ndarray,
    future_gt: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    rho: np.ndarray,
    pi: np.ndarray,
    intent_probs: Optional[np.ndarray] = None,
    title: str = "IntentFormer-3D Prediction",
    dpi: int = 150,
) -> None:
    """
    Generate and save the visualisation to *save_path*.

    Supported formats: .png, .pdf, .svg (determined by the file extension).
    """
    fig = plot_gmm_trajectories(
        history=history,
        future_gt=future_gt,
        mu=mu,
        sigma=sigma,
        rho=rho,
        pi=pi,
        intent_probs=intent_probs,
        title=title,
    )
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] Saved → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile
    import os

    T_hist, T_fut, K = 5, 6, 3
    np.random.seed(0)

    history = np.cumsum(np.random.randn(T_hist, 2) * 0.5, axis=0)
    future_gt = history[-1] + np.cumsum(np.random.randn(T_fut, 2) * 0.5, axis=0)

    mu = np.stack([
        future_gt + np.random.randn(T_fut, 2) * (k + 1) * 0.3
        for k in range(K)
    ])                                                         # (K, T_fut, 2)
    sigma = np.abs(np.random.randn(K, T_fut, 2)) * 0.4 + 0.1  # (K, T_fut, 2)
    rho = np.tanh(np.random.randn(K, T_fut)) * 0.5             # (K, T_fut)
    raw_pi = np.array([0.6, 0.3, 0.1])
    pi = raw_pi / raw_pi.sum()

    intent_probs = np.array([0.1, 0.7, 0.15, 0.05])

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name

    save_visualization(
        save_path=path,
        history=history,
        future_gt=future_gt,
        mu=mu,
        sigma=sigma,
        rho=rho,
        pi=pi,
        intent_probs=intent_probs,
        title="Smoke-test visualisation",
    )
    print("Saved to:", path)
    os.unlink(path)
