"""
train.py
Training loop for IntentFormer-3D.

Usage
-----
python -m training.train --config training/config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Allow running from the IntentFormer-3D root directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.intentformer import IntentFormer
from training.losses import total_loss
from training.scheduler import get_warmup_cosine_scheduler_epochs


# ──────────────────────────────────────────────────────────────────────────────
# Dataset wrapper
# ──────────────────────────────────────────────────────────────────────────────

class IntentFormerDataset(Dataset):
    """Wraps a list of preprocessed+labelled sample dicts."""

    def __init__(self, samples: List[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "history":    torch.from_numpy(s["history"]).float(),
            "neighbours": torch.from_numpy(s["neighbours"]).float(),
            "nbr_mask":   torch.from_numpy(s["nbr_mask"]).bool(),
            "lidar_feat": torch.from_numpy(s["lidar_feat"]).float()
                          if "lidar_feat" in s
                          else torch.zeros(6),
            "future":     torch.from_numpy(s["future"]).float(),
            "intent":     torch.tensor(s["intent"], dtype=torch.long),
        }


def collate_fn(batch: List[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


# ──────────────────────────────────────────────────────────────────────────────
# Training utilities
# ──────────────────────────────────────────────────────────────────────────────

def _compute_nbr_distances(neighbours: torch.Tensor, nbr_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute L2 distance from the focal agent (origin) to each neighbour's
    position at the last (most-recent) history step, using the x/y channels.

    neighbours : (B, N, T, 6)  — agent-centric neighbour history sequences
    nbr_mask   : (B, N) bool   — True = valid neighbour
    returns    : (B, N) float  — distance in metres (large value for padding)
    """
    # Last history step holds the most-recent position (x, y)
    nbr_pos = neighbours[:, :, -1, :2]                   # (B, N, 2)
    dists = torch.norm(nbr_pos, dim=-1)                  # (B, N)
    # Set distance to large value for invalid neighbours
    dists = dists.masked_fill(~nbr_mask, 1e6)
    return dists


def train_one_epoch(
    model: IntentFormer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    lambda_traj: float = 1.0,
    lambda_intent: float = 1.0,
    grad_clip: float = 1.0,
    log_every: int = 50,
) -> Dict[str, float]:
    model.train()
    metrics: Dict[str, list] = {"loss": [], "loss_traj": [], "loss_intent": []}

    for step, batch in enumerate(loader):
        history    = batch["history"].to(device)
        neighbours = batch["neighbours"].to(device)
        nbr_mask   = batch["nbr_mask"].to(device)
        lidar_feat = batch["lidar_feat"].to(device)
        future_gt  = batch["future"].to(device)
        intent_lbl = batch["intent"].to(device)

        nbr_dists = _compute_nbr_distances(neighbours, nbr_mask)

        optimizer.zero_grad(set_to_none=True)

        out = model(history, neighbours, nbr_mask, lidar_feat, nbr_dists)

        loss, l_traj, l_intent = total_loss(
            future_gt=future_gt,
            mu=out.mu,
            sigma=out.sigma,
            rho=out.rho,
            pi_logits=out.pi_logits,
            intent_logits=out.intent_logits,
            intent_labels=intent_lbl,
            lambda_traj=lambda_traj,
            lambda_intent=lambda_intent,
        )

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        metrics["loss"].append(loss.item())
        metrics["loss_traj"].append(l_traj.item())
        metrics["loss_intent"].append(l_intent.item())

        if (step + 1) % log_every == 0:
            avg = {k: float(np.mean(v[-log_every:])) for k, v in metrics.items()}
            lr = scheduler.get_last_lr()[0]
            print(
                f"  step {step+1:5d}  "
                f"loss={avg['loss']:.4f}  "
                f"traj={avg['loss_traj']:.4f}  "
                f"intent={avg['loss_intent']:.4f}  "
                f"lr={lr:.2e}"
            )

    return {k: float(np.mean(v)) for k, v in metrics.items()}


@torch.no_grad()
def evaluate_epoch(
    model: IntentFormer,
    loader: DataLoader,
    device: torch.device,
    lambda_traj: float = 1.0,
    lambda_intent: float = 1.0,
) -> Dict[str, float]:
    model.eval()
    metrics: Dict[str, list] = {"loss": [], "loss_traj": [], "loss_intent": []}

    for batch in loader:
        history    = batch["history"].to(device)
        neighbours = batch["neighbours"].to(device)
        nbr_mask   = batch["nbr_mask"].to(device)
        lidar_feat = batch["lidar_feat"].to(device)
        future_gt  = batch["future"].to(device)
        intent_lbl = batch["intent"].to(device)

        nbr_dists = _compute_nbr_distances(neighbours, nbr_mask)

        out = model(history, neighbours, nbr_mask, lidar_feat, nbr_dists)

        loss, l_traj, l_intent = total_loss(
            future_gt=future_gt,
            mu=out.mu,
            sigma=out.sigma,
            rho=out.rho,
            pi_logits=out.pi_logits,
            intent_logits=out.intent_logits,
            intent_labels=intent_lbl,
            lambda_traj=lambda_traj,
            lambda_intent=lambda_intent,
        )

        metrics["loss"].append(loss.item())
        metrics["loss_traj"].append(l_traj.item())
        metrics["loss_intent"].append(l_intent.item())

    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────

def main(cfg: dict) -> None:
    # ── Reproducibility ────────────────────────────────────────────────────────
    torch.manual_seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[train] Using device: {device}")

    # ── Data ───────────────────────────────────────────────────────────────────
    print("[train] Loading and preprocessing data …")
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

    def load_split(split: str) -> List[dict]:
        loader = NuScenesLoader(
            dataroot=cfg["nuscenes_dataroot"],
            version=cfg["nuscenes_version"],
            split=split,
        )
        raw = loader.get_samples()
        processed = preprocess_dataset(raw, max_neighbours=cfg.get("max_neighbours", 10))
        processed = add_lidar_features_to_samples(nusc, processed, raw)
        return label_dataset(processed)

    train_samples = load_split(cfg.get("train_split", "train"))
    val_samples   = load_split(cfg.get("val_split", "val"))

    train_ds = IntentFormerDataset(train_samples)
    val_ds   = IntentFormerDataset(val_samples)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg.get("num_workers", 4), collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg.get("num_workers", 4), collate_fn=collate_fn,
        pin_memory=True,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = IntentFormer(
        d_input=cfg.get("d_input", 6),
        d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 4),
        n_temporal_layers=cfg.get("n_temporal_layers", 2),
        d_ff=cfg.get("d_ff", 256),
        dropout=cfg.get("dropout", 0.1),
        d_lidar=cfg.get("d_lidar", 6),
        T_fut=cfg.get("T_fut", 6),
        K=cfg.get("K", 3),
        num_intent_classes=cfg.get("num_intent_classes", 4),
        max_neighbours=cfg.get("max_neighbours", 10),
    ).to(device)
    print(f"[train] Model parameters: {model.count_parameters():,}")

    # ── Optimiser & scheduler ──────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    scheduler = get_warmup_cosine_scheduler_epochs(
        optimizer,
        warmup_epochs=cfg.get("warmup_epochs", 5),
        total_epochs=cfg.get("epochs", 50),
        steps_per_epoch=len(train_loader),
        min_lr_ratio=cfg.get("min_lr_ratio", 0.01),
    )

    # ── Checkpointing ──────────────────────────────────────────────────────────
    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    lambda_traj   = cfg.get("lambda_traj", 1.0)
    lambda_intent = cfg.get("lambda_intent", 1.0)

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(1, cfg.get("epochs", 50) + 1):
        t0 = time.time()
        train_m = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            lambda_traj=lambda_traj,
            lambda_intent=lambda_intent,
            grad_clip=cfg.get("grad_clip", 1.0),
            log_every=cfg.get("log_every", 50),
        )
        val_m = evaluate_epoch(
            model, val_loader, device,
            lambda_traj=lambda_traj,
            lambda_intent=lambda_intent,
        )

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{cfg.get('epochs', 50)}  "
            f"train_loss={train_m['loss']:.4f}  "
            f"val_loss={val_m['loss']:.4f}  "
            f"({elapsed:.1f}s)"
        )

        # Save best checkpoint
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            ckpt_path = ckpt_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "cfg": cfg,
                },
                ckpt_path,
            )
            print(f"  ✓ Saved best checkpoint  (val_loss={best_val_loss:.4f})")

        # Periodic checkpoint every N epochs
        save_every = cfg.get("save_every_epochs", 10)
        if epoch % save_every == 0:
            periodic_path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_m["loss"],
                    "cfg": cfg,
                },
                periodic_path,
            )

    print(f"[train] Training complete. Best val_loss: {best_val_loss:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml

    parser = argparse.ArgumentParser(description="Train IntentFormer-3D")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to YAML config file",
    )
    # Allow individual overrides, e.g. --lr 5e-4 --epochs 30
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    for key in ("lr", "epochs", "batch_size", "device"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    main(cfg)
