# IntentFormer-3D

**Intent and 3-D Trajectory Predictor for Autonomous Driving**

IntentFormer-3D is a deep-learning framework for simultaneously predicting
the *intent* (crossing / waiting / turning / straight) and the *multi-modal
future trajectory* of traffic agents from nuScenes sensor data.

---

## Architecture Overview

```
History (2 s)         Neighbours              LiDAR context
     │                     │                       │
     ▼                     ▼                       │
TemporalEncoder    TemporalEncoder (shared)        │
     │                     │                       │
     └──── Social Attention ─────┘                 │
                   │                               │
              Fusion MLP                           │
                   │──────────────────────────────►│
                   │                               ▼
          ┌────────┴────────┐              IntentHead
          │                 │              (4-class CE)
          ▼                 ▼
       GMMHead           intent_logits
  (μ, σ, ρ, π × K=3)
```

| Module | File | Description |
|--------|------|-------------|
| Data loader | `data/nuscenes_loader.py` | Loads nuScenes, filters agents, builds samples |
| Preprocessor | `data/preprocess.py` | Agent-centric normalisation + velocity features |
| Intent labeler | `data/intent_labeler.py` | Rule-based GT intent labels |
| LiDAR features | `data/lidar_features.py` | 6-dim LiDAR context per agent |
| Temporal encoder | `models/temporal_encoder.py` | Transformer encoder for motion history |
| Social attention | `models/social_attention.py` | Cross-attention over neighbour agents |
| Intent head | `models/intent_head.py` | 4-class MLP classification head |
| GMM head | `models/gmm_head.py` | Bivariate Gaussian Mixture trajectory head |
| Full model | `models/intentformer.py` | Wires all modules together |
| Losses | `training/losses.py` | GMM NLL + intent CE combined loss |
| Training loop | `training/train.py` | Training with logging and checkpointing |
| LR scheduler | `training/scheduler.py` | Linear warmup + cosine decay |
| Config | `training/config.yaml` | All hyperparameters |
| Metrics | `evaluation/metrics.py` | minADE@K and minFDE@K |
| Evaluator | `evaluation/evaluate.py` | Eval loop + results table |
| Visualizer | `evaluation/visualize.py` | GMM ellipse plots + intent bars |

---

## Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- nuScenes devkit

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Download

1. Register at [nuscenes.org](https://www.nuscenes.org/nuscenes) and download
   the **nuScenes** dataset (Full dataset v1.0 or the mini split for quick tests).

2. Set `nuscenes_dataroot` in `training/config.yaml` to your dataset root, e.g.:

```yaml
nuscenes_dataroot: "/data/nuscenes"
nuscenes_version:  "v1.0-trainval"   # or "v1.0-mini"
```

---

## Training

```bash
cd IntentFormer-3D

# Default config (edit training/config.yaml first)
python -m training.train --config training/config.yaml

# Quick CLI overrides
python -m training.train --config training/config.yaml \
    --lr 5e-4 --epochs 30 --batch_size 32 --device cuda
```

Checkpoints are saved to `checkpoints/` (configurable via `checkpoint_dir`).

---

## Evaluation

```bash
python -m evaluation.evaluate \
    --checkpoint checkpoints/best_model.pt \
    --config     training/config.yaml \
    --split      val
```

Example output:

```
============================================
  IntentFormer-3D  │  Evaluation: val
============================================
  intent_acc           │    0.7812
  minADE@1             │    1.2340
  minADE@3             │    0.8910
  minADE@6             │    0.7120
  minFDE@1             │    2.8760
  minFDE@3             │    1.9450
  minFDE@6             │    1.5300
============================================
```

---

## Visualisation

```python
from evaluation.visualize import save_visualization
import numpy as np

save_visualization(
    save_path="prediction.png",
    history=history_np,     # (T_hist, 2)
    future_gt=future_np,    # (T_fut, 2)
    mu=mu_np,               # (K, T_fut, 2)
    sigma=sigma_np,         # (K, T_fut, 2)
    rho=rho_np,             # (K, T_fut)
    pi=pi_np,               # (K,)
    intent_probs=intent_np, # (4,)
)
```

The figure shows:
- **Grey dotted line** — 2 s motion history
- **Black solid line** — ground-truth future trajectory
- **Coloured lines** — K predicted mean trajectories
- **Shaded ellipses** — 2σ confidence regions per mode per step
- **Inset bar chart** — mixture weight (π) per mode
- **Right panel** — intent probability distribution

---

## Configuration Reference

All hyperparameters live in `training/config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 128 | Transformer model width |
| `n_heads` | 4 | Attention heads |
| `n_temporal_layers` | 2 | Encoder depth |
| `K` | 3 | GMM mixture components |
| `T_fut` | 6 | Future steps (≈ 3 s @ 2 Hz) |
| `lr` | 1e-3 | Peak learning rate |
| `warmup_epochs` | 5 | LR warmup duration |
| `epochs` | 50 | Total training epochs |
| `lambda_traj` | 1.0 | GMM NLL loss weight |
| `lambda_intent` | 1.0 | Intent CE loss weight |

---

## Project Structure

```
IntentFormer-3D/
├── data/
│   ├── nuscenes_loader.py
│   ├── preprocess.py
│   ├── intent_labeler.py
│   └── lidar_features.py
├── models/
│   ├── __init__.py
│   ├── temporal_encoder.py
│   ├── social_attention.py
│   ├── intent_head.py
│   ├── gmm_head.py
│   └── intentformer.py
├── training/
│   ├── losses.py
│   ├── train.py
│   ├── scheduler.py
│   └── config.yaml
├── evaluation/
│   ├── metrics.py
│   ├── evaluate.py
│   └── visualize.py
├── README.md
└── requirements.txt
```
