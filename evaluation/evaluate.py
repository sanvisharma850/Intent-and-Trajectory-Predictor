"""
evaluate.py
Evaluation loop for IntentFormer-3D.

Usage
-----
python -m evaluation.evaluate \\
    --checkpoint checkpoints/best_model.pt \\
    --config     training/config.yaml \\
    --split      val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

# Allow running from the IntentFormer-3D root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.intentformer import IntentFormer
from evaluation.metrics import compute_metrics
from training.train import IntentFormerDataset, collate_fn, _compute_nbr_distances


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(
    model: IntentFormer,
    loader: DataLoader,
    device: torch.device,
    k_values: tuple[int, ...] = (1, 3, 6),
) -> Dict[str, float]:
    """
    Run the full evaluation loop over *loader* and aggregate metrics.

    Returns
    -------
    dict mapping metric name → mean value over the dataset.
    """
    model.eval()
    all_metrics: Dict[str, List[float]] = {}

    for batch in loader:
        history    = batch["history"].to(device)
        neighbours = batch["neighbours"].to(device)
        nbr_mask   = batch["nbr_mask"].to(device)
        lidar_feat = batch["lidar_feat"].to(device)
        future_gt  = batch["future"].to(device)
        intent_lbl = batch["intent"].to(device)

        nbr_dists = _compute_nbr_distances(neighbours, nbr_mask)

        out = model(history, neighbours, nbr_mask, lidar_feat, nbr_dists)

        # GMM mean trajectories serve as the K candidate predictions
        # shape: (B, K, T_fut, 2)
        predictions = out.mu

        batch_metrics = compute_metrics(
            predictions=predictions,
            ground_truth=future_gt,
            intent_logits=out.intent_logits,
            intent_labels=intent_lbl,
            k_values=k_values,
        )

        for k, v in batch_metrics.items():
            all_metrics.setdefault(k, []).append(v)

    return {k: float(np.mean(v)) for k, v in all_metrics.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Results table printer
# ──────────────────────────────────────────────────────────────────────────────

def print_results_table(metrics: Dict[str, float], split: str = "val") -> None:
    """Pretty-print an evaluation results table."""
    border = "─" * 44
    print(f"\n{'='*44}")
    print(f"  IntentFormer-3D  │  Evaluation: {split}")
    print(f"{'='*44}")
    for name, val in sorted(metrics.items()):
        print(f"  {name:<20s} │  {val:>8.4f}")
    print(f"{'='*44}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IntentFormer-3D")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument(
        "--config", default=str(Path(__file__).parents[1] / "training" / "config.yaml"),
        help="Path to YAML config"
    )
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.device:
        cfg["device"] = args.device

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[evaluate] Device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────────
    from data.nuscenes_loader import NuScenesLoader
    from data.preprocess import preprocess_dataset
    from data.intent_labeler import label_dataset
    from data.lidar_features import add_lidar_features_to_samples
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(
        version=cfg["nuscenes_version"],
        dataroot=cfg["nuscenes_dataroot"],
        verbose=False,
    )

    loader_obj = NuScenesLoader(
        dataroot=cfg["nuscenes_dataroot"],
        version=cfg["nuscenes_version"],
        split=args.split,
    )
    raw = loader_obj.get_samples()
    processed = preprocess_dataset(raw, max_neighbours=cfg.get("max_neighbours", 10))
    processed = add_lidar_features_to_samples(nusc, processed, raw)
    processed = label_dataset(processed)

    ds = IntentFormerDataset(processed)
    data_loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ── Load model ─────────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_cfg = ckpt.get("cfg", cfg)

    model = IntentFormer(
        d_input=model_cfg.get("d_input", 6),
        d_model=model_cfg.get("d_model", 128),
        n_heads=model_cfg.get("n_heads", 4),
        n_temporal_layers=model_cfg.get("n_temporal_layers", 2),
        d_ff=model_cfg.get("d_ff", 256),
        dropout=0.0,         # disable dropout at eval
        d_lidar=model_cfg.get("d_lidar", 6),
        T_fut=model_cfg.get("T_fut", 6),
        K=model_cfg.get("K", 3),
        num_intent_classes=model_cfg.get("num_intent_classes", 4),
        max_neighbours=model_cfg.get("max_neighbours", 10),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[evaluate] Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # ── Run evaluation ─────────────────────────────────────────────────────────
    results = run_evaluation(model, data_loader, device)
    print_results_table(results, split=args.split)


if __name__ == "__main__":
    main()
