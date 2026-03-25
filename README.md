# IntentFormer — Pedestrian & Cyclist Trajectory Prediction

> **Track:** Intent & Trajectory Prediction · **Focus:** Behavioral AI & Temporal Modeling  
> **Dataset:** nuScenes v1.0-trainval · **Platform:** Kaggle (GPU T4)

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Model Architecture](#2-model-architecture)
3. [Dataset](#3-dataset)
4. [Setup & Installation](#4-setup--installation)
5. [How to Run](#5-how-to-run)
6. [Results](#6-results)
7. [Ablation Study](#7-ablation-study)
8. [Example Outputs](#8-example-outputs)
9. [Known Limitations](#9-known-limitations)
10. [Repository Structure](#10-repository-structure)

---

## 1. Problem Statement

In an L4 urban autonomous driving environment, reacting to where a pedestrian *is* is insufficient — the vehicle must predict where they *will be*. This project builds a model that:

- **Input:** 2 seconds of observed (x, y) positions for a pedestrian or cyclist (4 frames at 2 Hz), plus agent-centric neighbour context and LiDAR-derived features
- **Output:** K = 5 probabilistic future trajectory hypotheses covering the next 3 seconds (6 frames at 2 Hz), each represented as a sequence of bivariate Gaussian distributions

The model must account for **multimodal intent** (an agent at an intersection may go straight, turn left, or stop), **social context** (agents avoid each other), and **deployment realism** (it must commit to one trajectory without oracle knowledge of which mode is correct).

**Primary metric:** minADE@5 — the average displacement error (metres) of the best of 5 predicted modes against ground truth, averaged over all future time steps.

---

## 2. Model Architecture

The model is called **IntentFormer**. It is a multi-module neural network built around a Transformer encoder backbone.

```
Past trajectory (4 × 4)
        │
        ▼
┌─────────────────────┐
│   TemporalEncoder   │  Transformer encoder, 3 layers, 4 heads, d_model=256
│   (last-token out)  │  Output: hidden vector h ∈ ℝ²⁵⁶
└────────┬────────────┘
         │
         │◄──── SocialAttention (cross-attention over ≤5 neighbours)
         │
         │◄──── LiDAR Fusion (6 LiDAR scalars + validity flag → ℝ²⁵⁶, fused via LayerNorm)
         │
         ▼
┌─────────────────────┐    ┌───────────────────┐
│      GMMHead        │    │    IntentHead     │
│  K=5 bivariate      │    │  5-class kinematic│
│  Gaussian modes     │    │  intent (CE loss, │
│  + mixture weights  │    │  λ=0.15, detached)│
└─────────────────────┘    └───────────────────┘
         │
         ▼
  (μ, σ, ρ, π) per mode    → minADE@5 / greedy deployment
```

### 2.1 TemporalEncoder

A standard `nn.TransformerEncoder` with sinusoidal positional encoding. Input is a 4-frame sequence of `[x_norm, y_norm, vx_norm, vy_norm]` (position + velocity in the agent-centric normalised frame). The **last token's output** is used as the hidden state — not the mean — because the last token is conditioned on all prior tokens via self-attention and specifically represents the agent's current state and recent dynamics.

| Hyperparameter | Value |
|---|---|
| `d_model` | 256 |
| `nhead` | 4 |
| `n_layers` | 3 |
| `dim_feedforward` | 1024 (4 × d_model) |
| `dropout` | 0.1 |
| Readout | Last token `out[:, -1, :]` |

### 2.2 SocialAttention

Cross-attention over up to `N_MAX_NEIGHBORS = 5` neighbouring agents. Each neighbour's 4-frame past is encoded by a separate (shared-weight) encoder into a key/value pair. The focal agent's hidden state is the query. Slots beyond the actual neighbour count are zero-padded and masked out via a key padding mask so they contribute zero attention weight.

### 2.3 LiDAR Fusion

Six LiDAR-derived scalar features (point density, height statistics, etc.) plus one validity flag are projected into `d_model` space via a 2-layer MLP (`6 → 64 → 256`). The validity flag gates the projection multiplicatively so samples with no LiDAR coverage contribute zero fusion signal. The result is added to the social-attention output and normalised with `nn.LayerNorm`.

### 2.4 GMMHead

Produces K = 5 trajectory hypotheses. For each mode k and each future timestep t, it outputs:
- `μ_k,t ∈ ℝ²` — predicted mean position
- `σ_k,t ∈ ℝ²` — per-axis standard deviation (softplus-activated)
- `ρ_k,t ∈ (-1, 1)` — correlation coefficient (tanh-activated)
- `π_k ∈ (0, 1)` — mixture weight (softmax over K)

### 2.5 IntentHead

A shallow MLP that classifies the agent's coarse kinematic intent (stationary, slow walk, fast walk/run, curved, reversing) from `h.detach()`. The detach is critical: intent gradients do not flow back through the trunk, so the intent head is a diagnostic auxiliary and cannot distort the trajectory representations. At deployment the intent head output is unused.

### 2.6 Total Parameters

~**3.2M parameters** (D=256, L=3, K=5).

---

## 3. Dataset

**nuScenes v1.0-trainval** (full split, accessed via Kaggle).

### 3.1 Target Categories

- `human.pedestrian.*` (all pedestrian subtypes)
- `vehicle.bicycle`

### 3.2 Temporal Configuration

| Setting | Value |
|---|---|
| Annotation frequency | 2 Hz |
| Past history | 4 frames = **2 seconds** |
| Future horizon | 6 frames = **3 seconds** |
| `Δt` | 0.5 s |

### 3.3 Scene-Level Split

Splits are assigned at the **scene level** to prevent any agent's trajectory from appearing in both train and test.

| Split | Source | Scenes | Test samples |
|---|---|---|---|
| Train | nuScenes official `train` | ~700 | — |
| Val | First 100 of official `val` | 100 | — |
| Test | Remaining `val` scenes | ~50 | **3,844** |

A three-way disjointness assertion is enforced at data-prep time and will crash early if any scene leaks across splits.

### 3.4 Agent-Centric Coordinate Frame

All positions and velocities are transformed into the focal agent's local frame at the first observed timestep. The heading direction is aligned with the +Y axis. This makes the representation **rotation-invariant**: the model does not need to learn compass directions.

### 3.5 Normalisation

Displacement statistics (not raw positions) are computed on the **train split only** and applied to all splits.

```
std_x = std(Δx  across all train frames)   # ≠ std_y
std_y = std(Δy  across all train frames)
x_norm = x / std_x
y_norm = y / std_y
```

`std_x` and `std_y` are saved as **separate scalars** in `norm_stats.csv`. They must never be merged or averaged because the agent-centric frame is anisotropic (agents mostly move forward, so `std_y >> std_x`). All ADE/FDE computations denormalise each axis independently before computing Euclidean distance.

### 3.6 Data Augmentation (train split only)

| Augmentation | Detail |
|---|---|
| Lateral flip | `x → -x` for entire scene (agent + all neighbours) |
| Speed scaling | Multiply all positions and velocities by `U[0.8, 1.2]` |
| Neighbour dropout | Each neighbour slot independently zeroed with `p = 0.3` |

Augmentations are applied on-the-fly in `TrajectoryDataset.__getitem__`. Val and test sets use no augmentation.

---

## 4. Setup & Installation

### 4.1 Requirements

```
python >= 3.9
torch >= 2.0
numpy
pandas
nuscenes-devkit (installed with --no-deps)
pyquaternion
cachetools
fire
```

### 4.2 Kaggle Setup

This project runs on Kaggle notebooks with a **GPU T4 accelerator**. Before running:

1. Go to **Settings → Accelerator → GPU T4 x2** and restart the kernel.
2. Attach the following Kaggle datasets to your notebook:
   - `nuscenes-dataset` (the official nuScenes v1.0-trainval blobs + metadata TGZs)
3. The data prep notebook (`01-dataprep.ipynb`) will install `nuscenes-devkit` automatically and extract the TGZ archives on first run.

### 4.3 Local Setup (optional)

```bash
git clone https://github.com/<your-username>/intentformer-trajectory
cd intentformer-trajectory
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install nuscenes-devkit --no-deps
pip install pyquaternion cachetools fire numpy pandas matplotlib
```

You will also need a local nuScenes v1.0-trainval installation. Set `NUSCENES_ROOT` and `NUSCENES_VERSION` in the config block at the top of `01-dataprep.ipynb`.

---

## 5. How to Run

Run the three notebooks **in order**. Each notebook saves its outputs to `/kaggle/working/`, which the next notebook reads as a Kaggle dataset input.

### Step 1 — Data Preparation (`01-dataprep.ipynb`)

```
Run all cells top to bottom.
```

What it does:
- Extracts all nuScenes LiDAR `.pcd.bin` files to a flat on-disk folder (avoids O(archive) per-sample cost)
- Builds per-instance global tracks from annotations
- Applies agent-centric transform and sliding-window extraction
- Computes neighbour features `(N_MAX_NEIGHBORS, PAST_LEN, 4)` — `[x_ac, y_ac, vx_ac, vy_ac]`
- Computes displacement normalisation statistics on the train split
- Runs scene-leakage and normalisation round-trip sanity checks

**Outputs saved to `/kaggle/working/`:**

| File | Contents |
|---|---|
| `trajectories.csv` | One row per (agent, frame) with positions, velocities, split label |
| `norm_stats.csv` | `std_x` and `std_y` scalars |
| `neighbors.pkl` | Dict mapping `sample_id → (5, 4, 4)` float32 array |
| `lidar_feats.pkl` | Dict mapping `sample_id → (7,)` float32 vector |

### Step 2 — Model Training (`02-model-train.ipynb`)

```
Settings → Accelerator → GPU T4 x2 → Run all cells.
```

What it does:
- Instantiates IntentFormer (~3.2M parameters)
- Trains for up to 200 epochs with early stopping (patience = 25)
- Uses AdamW (`lr=3e-4`, `weight_decay=1e-4`) with cosine warmup (10-epoch linear warmup then cosine annealing)
- Composite loss: `L = L_GMM + λ_div · L_diversity + λ_int · L_intent_CE`
- Checkpoints on **greedy pi-ADE** (the deployment metric), not oracle minADE
- Trains the `IntentFormerNoSocial` ablation variant after the main run

**Outputs saved to `/kaggle/working/`:**

| File | Contents |
|---|---|
| `best_intentformer.pt` | Best checkpoint (by val pi-ADE) |
| `best_nosocial.pt` | Best checkpoint for the no-social ablation |
| `training_history.csv` | Per-epoch loss, ADE, collapse rate, intent accuracy |
| `cv_baseline.json` | Constant-velocity baseline ADE on the test set |
| `ablation_results.json` | No-social ablation test ADE |

### Step 3 — Evaluation (`03-eval.ipynb`)

```
Run all cells. GPU recommended but not required (CPU is slow but correct).
```

What it does:
- Loads `best_intentformer.pt` and runs full test-set inference
- Computes minADE@5 (oracle), minFDE@5, and four greedy mode-selection strategies
- Grid-searches combined-strategy weights on the **validation set**, then locks them before test evaluation
- Reports 95% bootstrap confidence intervals (2,000 resamples)
- Computes per-category (pedestrian vs bicycle) breakdown
- Computes per-timestep ADE curve (0.5 s → 3.0 s)
- Generates trajectory visualisation plots (best / worst / mode-collapse cases)
- Saves `results_summary.csv`

---

## 6. Results

All metrics are on the held-out **test set (3,844 samples)**. Distance in metres.

### 6.1 Primary Metrics

| Metric | Value | 95% CI |
|---|---|---|
| **minADE@5** (oracle) | **0.237 m** | [0.229, 0.246] |
| minFDE@5 (oracle) | 0.433 m | — |
| Greedy pi-ADE | 0.298 m | — |
| Greedy CV-ADE | 0.311 m | — |
| Greedy uncertainty-ADE | 0.456 m | — |
| **Greedy combined-ADE** | **0.298 m** | — |
| CV baseline ADE | 0.333 m | — |

**Best greedy strategy:** combined (rank fusion of pi, CV, uncertainty scores)  
**Oracle / greedy ratio:** 1.25 — the greedy mode selector is 25% worse than the oracle best

### 6.2 Baseline Comparison

| Model | minADE (m) | Δ vs CV |
|---|---|---|
| Constant velocity (CV) | 0.333 | — |
| IntentFormer K=5 (greedy) | 0.298 | −10.5% |
| IntentFormer K=5 (oracle) | **0.237** | **−28.8%** |

### 6.3 Per-Timestep Oracle ADE

| Horizon | Oracle ADE |
|---|---|
| 0.5 s | 0.07 m |
| 1.0 s | 0.13 m |
| 1.5 s | 0.19 m |
| 2.0 s | 0.26 m |
| 2.5 s | 0.35 m |
| 3.0 s | 0.44 m |

Error grows approximately linearly with horizon, with no sudden spikes. This indicates stable prediction quality across the full 3-second window.

### 6.4 Greedy Mode Selection

Four strategies were evaluated for single-mode deployment selection:

- **pi** — pick the highest-weight mixture component
- **CV** — pick the mode whose endpoint is closest to the constant-velocity prediction
- **uncertainty** — pick the least-uncertain mode (smallest mean σ)
- **combined** — rank-fusion of all three (weights tuned on validation set)

The combined strategy ties with pi (both 0.298 m), confirming that the learned mixture weights `π` are the dominant signal and auxiliary heuristics add only marginal value.

---

## 7. Ablation Study

An `IntentFormerNoSocial` variant (identical in all respects except the `SocialAttention` module is removed) was trained from scratch under the same hyperparameters and evaluated on the test set.

| Model | Greedy pi-ADE |
|---|---|
| IntentFormer (full) | 0.298 m |
| IntentFormerNoSocial | 0.299 m |
| **Delta** | **+0.36%** (no significant benefit) |

**Interpretation:** Social context modelling does not improve performance on this dataset split. The most likely explanations are: (1) nuScenes contains relatively few crowded pedestrian scenes where avoidance dynamics are critical; (2) the neighbour radius of 20 m is wide enough to include agents that are too far away to be relevant, diluting the attention signal; (3) 4-frame history (2 s) may be insufficient for the model to reliably learn avoidance patterns. This is reported as a finding, not a failure — it motivates tighter neighbourhood radius tuning or a longer history window in future work.

---

## 8. Example Outputs

### Training curves

Four panels logged per epoch: GMM-NLL + diversity + intent CE losses, validation ADE (oracle vs greedy vs CV), mode-collapse rate, intent accuracy. Collapse rate target: < 20%.

### Trajectory visualisations

The evaluation notebook renders 9 panels (3 rows × 3 columns):

- **Row 1 — Best cases:** oracle error ≈ 0.00 m; all K modes converge on the ground-truth path; weight bars show healthy mode diversity
- **Row 2 — Worst cases:** oracle error 2.77–2.84 m; agent makes a sharp turn not anticipated by the past motion; modes spread wide but miss
- **Row 3 — Mode collapse cases:** one mode captures ~60–80% of the weight; remaining modes are near-zero; greedy selection is correct but diversity is lost

![Temporal ADE curve](temporal_ade_curve.png)  
![Trajectory predictions](trajectory_predictions.png)

---

## 9. Known Limitations

| Issue | Detail |
|---|---|
| Mode collapse | ~15–20% of test samples show one mode with > 60% weight. Mitigations: winner-takes-all loss, repulsion between modes, or increased diversity loss weight |
| Social attention null result | 0.36% ablation delta — social context provides no measurable benefit at the current neighbourhood radius and history length |
| Oracle / greedy gap | Ratio of 1.25 means the model generates good hypotheses but the intent weights do not reliably identify the correct mode; better weight calibration is needed |
| LiDAR validity | Samples without LiDAR coverage receive zero fusion contribution — performance on those samples is essentially the no-LiDAR baseline |
| Category imbalance | Bicycles are significantly underrepresented in nuScenes; per-category ADE may differ meaningfully |

---

## 10. Repository Structure

```
intentformer-trajectory/
├── 01-dataprep.ipynb         # nuScenes → trajectories.csv + neighbors.pkl + lidar_feats.pkl
├── 02-model-train.ipynb      # IntentFormer definition, training loop, ablation
├── 03-eval.ipynb             # Full evaluation, visualisation, results_summary.csv
├── results_summary.csv       # Final numeric results (auto-generated by NB3)
├── temporal_ade_curve.png    # Per-timestep oracle ADE plot
├── trajectory_predictions.png# Best / worst / collapse visualisation grid
└── README.md
```

All intermediate artefacts (`trajectories.csv`, `norm_stats.csv`, `neighbors.pkl`, `lidar_feats.pkl`, `best_intentformer.pt`, `best_nosocial.pt`, `training_history.csv`) are produced by the notebooks and are not tracked in the repository. They must be regenerated by running the notebooks in order, or mounted as a Kaggle dataset output.

---

## Acknowledgements

Dataset: [nuScenes](https://www.nuscenes.org/) by Motional (formerly nuTonomy).  
Devkit: [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit).
