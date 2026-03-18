"""
preprocess.py
Normalise raw nuScenes samples to agent-centric coordinates and compute
per-step velocity features.

Agent-centric convention:
  • The *current* agent position is the origin.
  • The *current* agent heading aligns with the positive x-axis.
  • All coordinates are in metres; all angles are in radians.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Rotation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rotation_matrix_2d(yaw: float) -> np.ndarray:
    """2-D counter-clockwise rotation matrix for *yaw* radians."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s], [-s, c]], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Core normalisation
# ──────────────────────────────────────────────────────────────────────────────

def to_agent_centric(
    history: np.ndarray,
    future: np.ndarray,
    neighbours: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[List[np.ndarray]]]:
    """
    Transform *history* and *future* into agent-centric coordinates.

    Parameters
    ----------
    history : np.ndarray, shape (T_hist, 4)
        Columns: [x, y, yaw, speed] in **global** frame.
        The *last* row is the current (most-recent) state.
    future : np.ndarray, shape (T_fut, 2)
        Columns: [x, y] in **global** frame.
    neighbours : optional list of np.ndarray, each shape (T_hist, 4)
        Neighbour histories in global frame.

    Returns
    -------
    hist_local : np.ndarray, shape (T_hist, 6)
        Columns: [x_rel, y_rel, cos_yaw, sin_yaw, vx, vy] in agent-centric frame.
    fut_local : np.ndarray, shape (T_fut, 2)  — [x_rel, y_rel]
    nbrs_local : list of np.ndarray or None   — same layout as hist_local
    """
    # Reference frame: last history step
    ref_x, ref_y, ref_yaw, _ = history[-1]
    R = _rotation_matrix_2d(ref_yaw)  # rotates global → agent-centric

    # ── History ──────────────────────────────────────────────────────────────
    positions_global = history[:, :2]                # (T, 2)
    positions_rel = (positions_global - np.array([ref_x, ref_y])) @ R.T

    # Compute velocities in local frame from consecutive positions
    velocities = np.zeros((len(history), 2), dtype=np.float32)
    if len(history) > 1:
        # Forward differences; the first step gets zero velocity
        dxy_global = np.diff(positions_global, axis=0)           # (T-1, 2)
        dxy_local = dxy_global @ R.T                              # (T-1, 2)
        # Approximate dt ≈ 0.5 s (2 Hz)
        dt = 0.5
        velocities[1:] = dxy_local / dt

    # Heading relative to reference
    global_yaws = history[:, 2]
    rel_yaws = global_yaws - ref_yaw
    cos_yaw = np.cos(rel_yaws).astype(np.float32)
    sin_yaw = np.sin(rel_yaws).astype(np.float32)

    hist_local = np.stack(
        [positions_rel[:, 0], positions_rel[:, 1],
         cos_yaw, sin_yaw,
         velocities[:, 0], velocities[:, 1]],
        axis=-1,
    ).astype(np.float32)  # (T_hist, 6)

    # ── Future ────────────────────────────────────────────────────────────────
    fut_global = future[:, :2]
    fut_local = ((fut_global - np.array([ref_x, ref_y])) @ R.T).astype(np.float32)

    # ── Neighbours ────────────────────────────────────────────────────────────
    nbrs_local: Optional[List[np.ndarray]] = None
    if neighbours is not None:
        nbrs_local = []
        for nbr in neighbours:
            nbr_pos_global = nbr[:, :2]
            nbr_pos_rel = (nbr_pos_global - np.array([ref_x, ref_y])) @ R.T
            nbr_vel = np.zeros((len(nbr), 2), dtype=np.float32)
            if len(nbr) > 1:
                d_nbr = np.diff(nbr_pos_global, axis=0) @ R.T
                nbr_vel[1:] = d_nbr / 0.5
            nbr_yaws = nbr[:, 2] - ref_yaw
            nbr_cos = np.cos(nbr_yaws).astype(np.float32)
            nbr_sin = np.sin(nbr_yaws).astype(np.float32)
            nbr_feat = np.stack(
                [nbr_pos_rel[:, 0], nbr_pos_rel[:, 1],
                 nbr_cos, nbr_sin,
                 nbr_vel[:, 0], nbr_vel[:, 1]],
                axis=-1,
            ).astype(np.float32)
            nbrs_local.append(nbr_feat)

    return hist_local, fut_local, nbrs_local


# ──────────────────────────────────────────────────────────────────────────────
# Sample-level preprocessor
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_sample(
    sample: dict,
    neighbour_samples: Optional[List[dict]] = None,
    max_neighbours: int = 10,
) -> dict:
    """
    Preprocess a raw sample dict (as returned by NuScenesLoader) into
    normalised tensors ready for batching.

    Parameters
    ----------
    sample : dict
        Raw sample from ``NuScenesLoader.get_samples()``.
    neighbour_samples : list of dict, optional
        Other agents' raw samples taken at the *same* current timestep.
    max_neighbours : int
        Maximum number of neighbours to include (nearest first).

    Returns
    -------
    dict with keys:
        "history"      — np.ndarray (T_hist, 6) agent-centric
        "future"       — np.ndarray (T_fut, 2) agent-centric
        "neighbours"   — np.ndarray (N, T_hist, 6) padded, zero = absent
        "nbr_mask"     — np.ndarray (N,) bool, True = valid neighbour
        "category"     — str
        "instance_token" — str
        "scene_token"  — str
    """
    history = sample["history"]   # (T_hist, 4)  global
    future = sample["future"]     # (T_fut, 2)   global

    # Collect neighbour histories if provided
    raw_neighbours: Optional[List[np.ndarray]] = None
    if neighbour_samples:
        ref_xy = history[-1, :2]
        # Sort neighbours by distance to focal agent
        def dist_to_focal(s: dict) -> float:
            return float(np.linalg.norm(s["history"][-1, :2] - ref_xy))

        sorted_nbrs = sorted(neighbour_samples, key=dist_to_focal)
        raw_neighbours = [n["history"] for n in sorted_nbrs[:max_neighbours]]

    hist_local, fut_local, nbrs_local = to_agent_centric(
        history, future, raw_neighbours
    )

    T_hist = hist_local.shape[0]
    feat_dim = hist_local.shape[1]  # 6

    # Pad neighbours to max_neighbours
    nbr_array = np.zeros((max_neighbours, T_hist, feat_dim), dtype=np.float32)
    nbr_mask = np.zeros(max_neighbours, dtype=bool)
    if nbrs_local is not None:
        for i, nf in enumerate(nbrs_local[:max_neighbours]):
            t_avail = min(nf.shape[0], T_hist)
            nbr_array[i, :t_avail] = nf[:t_avail]
            nbr_mask[i] = True

    return {
        "history": hist_local,
        "future": fut_local,
        "neighbours": nbr_array,
        "nbr_mask": nbr_mask,
        "category": sample["category"],
        "instance_token": sample["instance_token"],
        "scene_token": sample["scene_token"],
        "lidar_token": sample.get("lidar_token", ""),
    }


def preprocess_dataset(
    raw_samples: List[dict],
    max_neighbours: int = 10,
) -> List[dict]:
    """
    Preprocess all samples returned by NuScenesLoader.

    Groups samples by scene and current-timestep so that neighbour
    context is correctly assembled.

    Parameters
    ----------
    raw_samples : list of dict
        All samples from NuScenesLoader.get_samples().
    max_neighbours : int
        Passed through to preprocess_sample().

    Returns
    -------
    List of preprocessed sample dicts.
    """
    # Group by (scene_token, last-history-timestamp) for neighbour lookup
    from collections import defaultdict

    groups: Dict[Tuple[str, float], List[dict]] = defaultdict(list)
    for s in raw_samples:
        key = (s["scene_token"], s["history_timestamps"][-1])
        groups[key].append(s)

    processed: List[dict] = []
    for key, group in groups.items():
        for focal in group:
            neighbours = [s for s in group
                          if s["instance_token"] != focal["instance_token"]]
            processed.append(
                preprocess_sample(focal, neighbours, max_neighbours)
            )

    print(f"[preprocess] Processed {len(processed)} samples.")
    return processed


# ──────────────────────────────────────────────────────────────────────────────
# CLI helper
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, "..")
    from data.nuscenes_loader import NuScenesLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--split", default="mini_train")
    args = parser.parse_args()

    loader = NuScenesLoader(dataroot=args.dataroot, version=args.version, split=args.split)
    raw = loader.get_samples()
    processed = preprocess_dataset(raw)
    if processed:
        p = processed[0]
        print("history shape:", p["history"].shape)
        print("future shape:", p["future"].shape)
        print("neighbours shape:", p["neighbours"].shape)
