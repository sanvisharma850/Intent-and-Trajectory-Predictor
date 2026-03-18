"""
nuscenes_loader.py
Loads raw nuScenes data, filters agents, and builds per-sample dicts ready
for downstream preprocessing and model training.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# nuScenes devkit — installed via requirements.txt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CATEGORY_WHITELIST = {
    "vehicle.car",
    "vehicle.truck",
    "vehicle.bus.rigid",
    "vehicle.bus.bendy",
    "vehicle.motorcycle",
    "vehicle.bicycle",
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.wheelchair",
}

# 2 s history @ 2 Hz annotation rate → 4 past keyframes (+ current = 5 total)
HISTORY_SECONDS = 2.0
# 3 s future horizon for trajectory prediction
FUTURE_SECONDS = 3.0
# nuScenes annotation frequency
ANNOTATION_HZ = 2.0

HISTORY_STEPS = int(HISTORY_SECONDS * ANNOTATION_HZ)  # 4
FUTURE_STEPS = int(FUTURE_SECONDS * ANNOTATION_HZ)    # 6


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

def _quaternion_to_yaw(q: List[float]) -> float:
    """Convert a nuScenes quaternion [w, x, y, z] to a yaw angle in radians."""
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _get_agent_category(nusc: NuScenes, instance_token: str) -> str:
    """Return the category name for an instance token."""
    instance = nusc.get("instance", instance_token)
    category = nusc.get("category", instance["category_token"])
    return category["name"]


# ──────────────────────────────────────────────────────────────────────────────
# Core loader
# ──────────────────────────────────────────────────────────────────────────────

class NuScenesLoader:
    """
    Iterates over a nuScenes split and yields per-agent samples.

    Each sample is a dict::

        {
            "scene_token":   str,
            "instance_token": str,
            "category":      str,
            "history":       np.ndarray  # (HISTORY_STEPS+1, 4) — [x, y, yaw, v]
            "future":        np.ndarray  # (FUTURE_STEPS, 2)    — [x, y]
            "history_timestamps": List[float],
            "future_timestamps":  List[float],
            "ego_pose":      np.ndarray  # (3,) — [x, y, yaw] at current frame
            "lidar_token":   str,        # sample_data token for the current frame
        }
    """

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-trainval",
        split: str = "train",
        min_history_steps: int = HISTORY_STEPS,
        min_future_steps: int = FUTURE_STEPS,
        category_whitelist: Optional[set] = None,
    ) -> None:
        self.dataroot = dataroot
        self.version = version
        self.split = split
        self.min_history_steps = min_history_steps
        self.min_future_steps = min_future_steps
        self.category_whitelist = category_whitelist or CATEGORY_WHITELIST

        print(f"[NuScenesLoader] Loading nuScenes {version} from {dataroot} …")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)

        splits = create_splits_scenes()
        self.scene_names: List[str] = splits[split]

        # Build scene-token → scene-record lookup
        self._scene_map: Dict[str, dict] = {
            s["name"]: s for s in self.nusc.scene
        }

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.scene_names)

    def get_samples(self) -> List[dict]:
        """Return all valid agent samples across every scene in the split."""
        all_samples: List[dict] = []
        for scene_name in self.scene_names:
            if scene_name not in self._scene_map:
                continue
            scene_record = self._scene_map[scene_name]
            all_samples.extend(self._extract_scene_samples(scene_record))
        print(f"[NuScenesLoader] Extracted {len(all_samples)} samples "
              f"from {len(self.scene_names)} scenes.")
        return all_samples

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _extract_scene_samples(self, scene: dict) -> List[dict]:
        """Walk a scene timeline and extract per-agent samples."""
        samples: List[dict] = []

        # Collect ordered sample tokens for this scene
        sample_tokens: List[str] = []
        tok = scene["first_sample_token"]
        while tok:
            sample_tokens.append(tok)
            sample_rec = self.nusc.get("sample", tok)
            tok = sample_rec["next"]

        # For each annotated agent instance in this scene walk through time
        instance_tokens = self._scene_instance_tokens(scene)

        for inst_tok in instance_tokens:
            category = _get_agent_category(self.nusc, inst_tok)
            if category not in self.category_whitelist:
                continue

            # Build ordered list of annotation records for this instance
            ann_by_sample = self._annotations_by_sample(inst_tok, sample_tokens)

            # Slide a window over the timeline
            for t in range(self.min_history_steps,
                           len(sample_tokens) - self.min_future_steps + 1):
                sample_tok = sample_tokens[t]

                # Require full history and future windows
                history_toks = sample_tokens[t - self.min_history_steps: t + 1]
                future_toks = sample_tokens[t + 1: t + 1 + self.min_future_steps]

                if not all(ht in ann_by_sample for ht in history_toks):
                    continue
                if not all(ft in ann_by_sample for ft in future_toks):
                    continue

                history, h_times = self._build_history(
                    ann_by_sample, history_toks)
                future, f_times = self._build_future(
                    ann_by_sample, future_toks)

                ego_pose = self._get_ego_pose(sample_tok)
                lidar_token = self._get_lidar_token(sample_tok)

                samples.append({
                    "scene_token": scene["token"],
                    "instance_token": inst_tok,
                    "category": category,
                    "history": history,
                    "future": future,
                    "history_timestamps": h_times,
                    "future_timestamps": f_times,
                    "ego_pose": ego_pose,
                    "lidar_token": lidar_token,
                })

        return samples

    def _scene_instance_tokens(self, scene: dict) -> List[str]:
        """Return all unique instance tokens that appear in this scene."""
        instances: set = set()
        tok = scene["first_sample_token"]
        while tok:
            sample_rec = self.nusc.get("sample", tok)
            for ann_tok in sample_rec["anns"]:
                ann = self.nusc.get("sample_annotation", ann_tok)
                instances.add(ann["instance_token"])
            tok = sample_rec["next"]
        return list(instances)

    def _annotations_by_sample(
        self, instance_token: str, sample_tokens: List[str]
    ) -> Dict[str, dict]:
        """Map sample_token → annotation record for a given instance."""
        mapping: Dict[str, dict] = {}
        # Walk the instance annotation linked list
        instance = self.nusc.get("instance", instance_token)
        ann_tok = instance["first_annotation_token"]
        while ann_tok:
            ann = self.nusc.get("sample_annotation", ann_tok)
            if ann["sample_token"] in set(sample_tokens):
                mapping[ann["sample_token"]] = ann
            ann_tok = ann["next"]
        return mapping

    def _build_history(
        self,
        ann_by_sample: Dict[str, dict],
        history_toks: List[str],
    ) -> Tuple[np.ndarray, List[float]]:
        """
        Return history array (T, 4) — [x, y, yaw, speed] and timestamps.
        history_toks is ordered oldest → current.
        """
        rows = []
        times = []
        prev_xy: Optional[np.ndarray] = None
        prev_t: Optional[float] = None

        for s_tok in history_toks:
            ann = ann_by_sample[s_tok]
            x, y, _ = ann["translation"]
            yaw = _quaternion_to_yaw(ann["rotation"])
            sample_rec = self.nusc.get("sample", s_tok)
            t = sample_rec["timestamp"] * 1e-6  # μs → s

            if prev_xy is not None and prev_t is not None:
                dt = t - prev_t
                speed = np.linalg.norm([x - prev_xy[0], y - prev_xy[1]]) / max(dt, 1e-6)
            else:
                speed = 0.0

            rows.append([x, y, yaw, speed])
            times.append(t)
            prev_xy = np.array([x, y])
            prev_t = t

        return np.array(rows, dtype=np.float32), times

    def _build_future(
        self,
        ann_by_sample: Dict[str, dict],
        future_toks: List[str],
    ) -> Tuple[np.ndarray, List[float]]:
        """Return future positions (T, 2) — [x, y] and timestamps."""
        rows = []
        times = []
        for s_tok in future_toks:
            ann = ann_by_sample[s_tok]
            x, y, _ = ann["translation"]
            sample_rec = self.nusc.get("sample", s_tok)
            times.append(sample_rec["timestamp"] * 1e-6)
            rows.append([x, y])
        return np.array(rows, dtype=np.float32), times

    def _get_ego_pose(self, sample_token: str) -> np.ndarray:
        """Return ego-vehicle pose [x, y, yaw] at a given sample."""
        sample_rec = self.nusc.get("sample", sample_token)
        # Use the LIDAR_TOP sensor as the reference
        sd_rec = self.nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
        ep = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])
        x, y, _ = ep["translation"]
        yaw = _quaternion_to_yaw(ep["rotation"])
        return np.array([x, y, yaw], dtype=np.float32)

    def _get_lidar_token(self, sample_token: str) -> str:
        """Return the LIDAR_TOP sample_data token for a given sample."""
        sample_rec = self.nusc.get("sample", sample_token)
        return sample_rec["data"]["LIDAR_TOP"]


# ──────────────────────────────────────────────────────────────────────────────
# CLI helper
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test NuScenesLoader")
    parser.add_argument("--dataroot", required=True, help="Path to nuScenes dataset root")
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--split", default="mini_train")
    args = parser.parse_args()

    loader = NuScenesLoader(
        dataroot=args.dataroot,
        version=args.version,
        split=args.split,
    )
    samples = loader.get_samples()
    if samples:
        s = samples[0]
        print("Example sample keys:", list(s.keys()))
        print("History shape:", s["history"].shape)
        print("Future shape:", s["future"].shape)
