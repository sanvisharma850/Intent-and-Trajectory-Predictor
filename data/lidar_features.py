"""
lidar_features.py
Extract a compact 6-dimensional LiDAR context vector per agent per frame.

Feature vector layout (6 dims)
-------------------------------
[0] n_pts_near    — number of LiDAR points within 2 m of agent centre
[1] n_pts_box     — number of LiDAR points inside the agent bounding-box
[2] mean_height   — mean z-height of in-box points (relative to ground)
[3] height_range  — z_max − z_min of in-box points
[4] point_density — n_pts_box / bounding-box volume  (points / m³)
[5] occupancy     — binary: 1 if any points in bounding box else 0

All distances are in metres.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

LIDAR_FEATURE_DIM = 6
NEAR_RADIUS_M = 2.0   # metres around agent centre for n_pts_near


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def _rotation_matrix_3d_z(yaw: float) -> np.ndarray:
    """Rotation matrix that rotates points by *yaw* radians around the z-axis."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c,  s, 0.],
        [-s, c, 0.],
        [0., 0., 1.],
    ], dtype=np.float32)


def _points_in_box(
    points_xyz: np.ndarray,  # (N, 3)
    center_xyz: np.ndarray,  # (3,)
    wlh: np.ndarray,         # (3,) — width, length, height
    yaw: float,
) -> np.ndarray:
    """
    Return a boolean mask of shape (N,) indicating which *points_xyz* fall
    inside the oriented 3-D bounding box.

    The box is defined by its centre, half-extents (wlh/2), and a yaw rotation
    around the z-axis.
    """
    R = _rotation_matrix_3d_z(yaw)                        # (3, 3)
    pts_shifted = points_xyz - center_xyz                  # (N, 3)
    pts_local = pts_shifted @ R.T                          # rotate to box frame
    half = wlh / 2.0
    inside = (
        (np.abs(pts_local[:, 0]) <= half[1]) &            # along length axis
        (np.abs(pts_local[:, 1]) <= half[0]) &            # along width axis
        (np.abs(pts_local[:, 2]) <= half[2])              # along height axis
    )
    return inside


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_lidar_features(
    points_xyz: np.ndarray,  # (N, 3) — full LiDAR sweep in sensor/global frame
    center_xyz: np.ndarray,  # (3,)   — agent position in the same frame
    wlh: np.ndarray,         # (3,)   — bounding-box width, length, height
    yaw: float,              # radians — agent heading in the same frame
) -> np.ndarray:
    """
    Compute the 6-dim LiDAR feature vector for a single agent at one frame.

    Parameters
    ----------
    points_xyz : np.ndarray, shape (N, 3)
        LiDAR point cloud in the global/sensor frame.
    center_xyz : np.ndarray, shape (3,)
        Agent bounding-box centre in the same frame.
    wlh : np.ndarray, shape (3,)
        Agent bounding-box dimensions [width, length, height].
    yaw : float
        Agent heading in radians.

    Returns
    -------
    np.ndarray, shape (6,) — dtype float32
    """
    if len(points_xyz) == 0:
        return np.zeros(LIDAR_FEATURE_DIM, dtype=np.float32)

    points_xyz = np.asarray(points_xyz, dtype=np.float32)

    # ── n_pts_near ─────────────────────────────────────────────────────────
    dist_xy = np.linalg.norm(points_xyz[:, :2] - center_xyz[:2], axis=-1)
    n_pts_near = int((dist_xy < NEAR_RADIUS_M).sum())

    # ── in-box mask ─────────────────────────────────────────────────────────
    in_box_mask = _points_in_box(points_xyz, center_xyz, wlh, yaw)
    n_pts_box = int(in_box_mask.sum())

    # ── height statistics ──────────────────────────────────────────────────
    if n_pts_box > 0:
        z_vals = points_xyz[in_box_mask, 2] - center_xyz[2] + wlh[2] / 2.0
        mean_height = float(z_vals.mean())
        height_range = float(z_vals.max() - z_vals.min())
    else:
        mean_height = 0.0
        height_range = 0.0

    # ── point density ──────────────────────────────────────────────────────
    vol = max(float(wlh[0] * wlh[1] * wlh[2]), 1e-6)
    point_density = n_pts_box / vol

    # ── occupancy ──────────────────────────────────────────────────────────
    occupancy = 1.0 if n_pts_box > 0 else 0.0

    return np.array(
        [n_pts_near, n_pts_box, mean_height, height_range, point_density, occupancy],
        dtype=np.float32,
    )


# ──────────────────────────────────────────────────────────────────────────────
# nuScenes integration
# ──────────────────────────────────────────────────────────────────────────────

def load_lidar_sweep(
    nusc,                    # NuScenes instance
    lidar_token: str,
    max_dist_m: float = 50.0,
) -> np.ndarray:
    """
    Load a LiDAR sweep from nuScenes and return points in the *global* frame.

    Parameters
    ----------
    nusc : NuScenes
        Loaded NuScenes object.
    lidar_token : str
        sample_data token for a LIDAR_TOP sweep.
    max_dist_m : float
        Ignore points farther than this distance from the sensor.

    Returns
    -------
    np.ndarray, shape (N, 3) — x, y, z in global metres.
    """
    from nuscenes.utils.data_classes import LidarPointCloud
    from pyquaternion import Quaternion

    sd_record = nusc.get("sample_data", lidar_token)
    pcl_path = os.path.join(nusc.dataroot, sd_record["filename"])

    pc = LidarPointCloud.from_file(pcl_path)

    # 1. Sensor → ego frame
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
    pc.translate(np.array(cs_record["translation"]))

    # 2. Ego → global frame
    ego_pose = nusc.get("ego_pose", sd_record["ego_pose_token"])
    pc.rotate(Quaternion(ego_pose["rotation"]).rotation_matrix)
    pc.translate(np.array(ego_pose["translation"]))

    points = pc.points[:3].T  # (N, 3)

    # Optional range filter
    dists = np.linalg.norm(points[:, :2], axis=-1)
    points = points[dists < max_dist_m]

    return points.astype(np.float32)


def add_lidar_features_to_samples(
    nusc,
    processed_samples: List[dict],
    raw_samples: List[dict],
) -> List[dict]:
    """
    Attach a ``"lidar_feat"`` key (np.ndarray, shape (6,)) to each processed
    sample by extracting features from the corresponding LIDAR sweep.

    Caches loaded sweeps to avoid redundant disk I/O.

    Parameters
    ----------
    nusc : NuScenes
        Loaded NuScenes object.
    processed_samples : list of dict
        Preprocessed samples (must contain ``"lidar_token"`` and
        ``"instance_token"``).
    raw_samples : list of dict
        Raw samples from NuScenesLoader (parallel list, same order).

    Returns
    -------
    Same list with ``"lidar_feat"`` added in-place.
    """
    sweep_cache: Dict[str, np.ndarray] = {}

    # Build a mapping instance_token → annotation for fast bbox lookup
    inst_to_raw: Dict[str, dict] = {
        s["instance_token"]: s for s in raw_samples
    }

    for proc_s, raw_s in zip(processed_samples, raw_samples):
        lidar_token = proc_s.get("lidar_token", "")
        if not lidar_token:
            proc_s["lidar_feat"] = np.zeros(LIDAR_FEATURE_DIM, dtype=np.float32)
            continue

        # Load and cache sweep
        if lidar_token not in sweep_cache:
            try:
                sweep_cache[lidar_token] = load_lidar_sweep(nusc, lidar_token)
            except Exception:
                sweep_cache[lidar_token] = np.zeros((0, 3), dtype=np.float32)
        points = sweep_cache[lidar_token]

        # Retrieve bounding box info from the raw annotation
        ann_history = raw_s["history"]
        center = ann_history[-1, :2]  # (x, y) from last history step
        center_xyz = np.array([center[0], center[1], 0.0], dtype=np.float32)

        # Get wlh from the nuScenes annotation record
        inst_tok = proc_s["instance_token"]
        instance = nusc.get("instance", inst_tok)
        ann_tok = instance["last_annotation_token"]
        ann = nusc.get("sample_annotation", ann_tok)
        wlh = np.array(ann["size"], dtype=np.float32)  # [width, length, height]
        yaw = raw_s["history"][-1, 2]

        proc_s["lidar_feat"] = extract_lidar_features(
            points, center_xyz, wlh, yaw
        )

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

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    loader = NuScenesLoader(dataroot=args.dataroot, version=args.version, split=args.split)
    raw = loader.get_samples()
    processed = preprocess_dataset(raw)
    processed = add_lidar_features_to_samples(nusc, processed, raw)
    if processed:
        print("lidar_feat:", processed[0]["lidar_feat"])
