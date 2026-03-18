"""
intent_labeler.py
Rule-based intent labels derived from ground-truth future trajectories.

Intent classes (int)
--------------------
0 — STRAIGHT   agent moves roughly straight ahead
1 — TURNING    lateral deviation > threshold (left or right turn)
2 — CROSSING   pedestrian/cyclist crossing the roadway
3 — WAITING    near-zero speed throughout the future window
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Label constants
# ──────────────────────────────────────────────────────────────────────────────

INTENT_STRAIGHT = 0
INTENT_TURNING = 1
INTENT_CROSSING = 2
INTENT_WAITING = 3

INTENT_NAMES: Dict[int, str] = {
    INTENT_STRAIGHT: "STRAIGHT",
    INTENT_TURNING:  "TURNING",
    INTENT_CROSSING: "CROSSING",
    INTENT_WAITING:  "WAITING",
}

# Vulnerable-road-user categories that can perform a "crossing" manoeuvre
CROSSING_CATEGORIES = {
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.wheelchair",
    "vehicle.bicycle",
    "vehicle.motorcycle",
}

# ──────────────────────────────────────────────────────────────────────────────
# Thresholds (tunable)
# ──────────────────────────────────────────────────────────────────────────────

# m/s — below this average future speed → WAITING
WAITING_SPEED_THRESHOLD = 0.5

# metres — lateral displacement in agent-centric frame → TURNING
TURNING_LATERAL_THRESHOLD = 1.5

# degrees — absolute heading change → also flag as TURNING
TURNING_HEADING_THRESHOLD_DEG = 20.0

# metres — cross-track lateral displacement relative to initial heading → CROSSING
CROSSING_LATERAL_THRESHOLD = 2.0


# ──────────────────────────────────────────────────────────────────────────────
# Label computation
# ──────────────────────────────────────────────────────────────────────────────

def label_intent(
    future_local: np.ndarray,
    category: str,
    history_local: Optional[np.ndarray] = None,
    future_yaws_local: Optional[np.ndarray] = None,
) -> int:
    """
    Assign a single intent label to an agent from its future trajectory.

    Parameters
    ----------
    future_local : np.ndarray, shape (T_fut, 2)
        Future [x_rel, y_rel] in **agent-centric** coordinates.
        x points forward (heading), y points left.
    category : str
        nuScenes category name of the agent.
    history_local : np.ndarray or None, shape (T_hist, 6)
        Agent-centric history (columns: x, y, cos_yaw, sin_yaw, vx, vy).
        Used to estimate current speed for the WAITING check.
    future_yaws_local : np.ndarray or None, shape (T_fut,)
        Future heading angles in agent-centric frame (radians).
        If provided, used to improve TURNING detection.

    Returns
    -------
    int  — one of INTENT_STRAIGHT / TURNING / CROSSING / WAITING
    """
    if len(future_local) == 0:
        return INTENT_STRAIGHT

    # ── 1. WAITING ────────────────────────────────────────────────────────────
    # Estimate average speed from position differences
    if len(future_local) >= 2:
        dxy = np.diff(future_local, axis=0)           # (T-1, 2)
        dists = np.linalg.norm(dxy, axis=-1)          # (T-1,)
        avg_speed = float(dists.mean() / 0.5)         # 0.5 s ≈ inter-frame dt
    else:
        avg_speed = float(np.linalg.norm(future_local[0]) / 0.5)

    if avg_speed < WAITING_SPEED_THRESHOLD:
        return INTENT_WAITING

    # ── 2. CROSSING (VRU only) ────────────────────────────────────────────────
    if category in CROSSING_CATEGORIES:
        # Max lateral displacement (y in agent-centric = left/right)
        max_lateral = float(np.abs(future_local[:, 1]).max())
        if max_lateral > CROSSING_LATERAL_THRESHOLD:
            return INTENT_CROSSING

    # ── 3. TURNING ────────────────────────────────────────────────────────────
    # Method A: lateral displacement
    max_lateral = float(np.abs(future_local[:, 1]).max())
    if max_lateral > TURNING_LATERAL_THRESHOLD:
        return INTENT_TURNING

    # Method B: heading change (if heading sequence is available)
    if future_yaws_local is not None and len(future_yaws_local) >= 2:
        total_heading_change = float(
            np.abs(np.diff(future_yaws_local)).sum()
        ) * 180.0 / np.pi
        if total_heading_change > TURNING_HEADING_THRESHOLD_DEG:
            return INTENT_TURNING

    # ── 4. STRAIGHT ───────────────────────────────────────────────────────────
    return INTENT_STRAIGHT


# ──────────────────────────────────────────────────────────────────────────────
# Dataset-level labelling
# ──────────────────────────────────────────────────────────────────────────────

def label_dataset(processed_samples: List[dict]) -> List[dict]:
    """
    Add an ``"intent"`` key (int) to every preprocessed sample dict in-place.

    Parameters
    ----------
    processed_samples : list of dict
        Output of ``preprocess.preprocess_dataset()``.

    Returns
    -------
    The same list with ``"intent"`` populated on each entry.
    """
    label_counts: Dict[int, int] = {k: 0 for k in INTENT_NAMES}

    for s in processed_samples:
        intent = label_intent(
            future_local=s["future"],
            category=s["category"],
            history_local=s["history"],
        )
        s["intent"] = intent
        label_counts[intent] += 1

    total = len(processed_samples)
    print("[intent_labeler] Label distribution:")
    for k, name in INTENT_NAMES.items():
        pct = 100.0 * label_counts[k] / max(total, 1)
        print(f"  {name:10s}: {label_counts[k]:5d}  ({pct:.1f} %)")

    return processed_samples


# ──────────────────────────────────────────────────────────────────────────────
# CLI helper
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, "..")
    from data.nuscenes_loader import NuScenesLoader
    from data.preprocess import preprocess_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--split", default="mini_train")
    args = parser.parse_args()

    loader = NuScenesLoader(dataroot=args.dataroot, version=args.version, split=args.split)
    raw = loader.get_samples()
    processed = preprocess_dataset(raw)
    labelled = label_dataset(processed)
    print(f"Labelled {len(labelled)} samples.")
