# IntentFormer — Pedestrian & Cyclist Trajectory Prediction

> **Track:** Intent & Trajectory Prediction · **Focus:** Behavioral AI & Temporal Modeling  
> **Dataset:** nuScenes v1.0-trainval · **Platform:** Kaggle (GPU T4)

A multimodal trajectory prediction model for pedestrians and cyclists using the nuScenes dataset. IntentFormer combines Transformer-based temporal encoding, social context (neighboring agents), LiDAR features, and intent classification to generate diverse, probabilistic future trajectory predictions.

---

## Project Overview

IntentFormer predicts the future paths of pedestrians and cyclists in autonomous driving scenarios. Given a short observation window (2 seconds / 4 frames), the model outputs **K multimodal trajectory hypotheses** as a Gaussian Mixture Model (GMM), along with an **intent classification** (e.g., walking, waiting, turning). The model is evaluated on the [nuScenes v1.0-trainval](https://www.nuscenes.org/) dataset.

**Key capabilities:**
- Multimodal future prediction (K hypotheses with learned mixture weights π)
- Intent classification (waiting, walking, turning left/right)
- Social context via neighbor trajectory encoding
- LiDAR-based scene features
- Evaluated with minADE, minFDE, and greedy π-ADE metrics

---

## Model Architecture

IntentFormer is composed of three main components:

### 1. TemporalEncoder
A Transformer encoder with sinusoidal positional encoding that processes the agent's past trajectory (4 frames × 4 features: normalized Δx, Δy, speed, heading).

### 2. SocialEncoder
Encodes up to 5 neighboring agents' past trajectories (within a 20m radius) using a shared Transformer encoder, then mean-pools across neighbors to produce a fixed-size social context vector.

### 3. IntentFormer (Full Model)
Combines the temporal encoding, social context, and LiDAR features to:
- Predict **K trajectory modes** (each as a sequence of bivariate Gaussian parameters: μ, σ, ρ)
- Predict **mixture weights π** per mode
- Classify **agent intent** (4 classes)

**Training losses:**
- `GMM-NLL` — Bivariate Gaussian negative log-likelihood for trajectory regression
- `Diversity loss` — Penalizes mode collapse
- `Intent CE` — Cross-entropy loss for intent classification

**Hyperparameters (defaults):**
| Parameter | Value |
|-----------|-------|
| d_model | 256 |
| Attention heads | 4 |
| Transformer layers | 3 |
| K (modes) | 5 |
| Past frames | 4 (2 s) |
| Future frames | 6 (3 s) |
| Δt | 0.5 s |

---

## Dataset

**nuScenes v1.0-trainval** — a large-scale autonomous driving dataset with:
- 1000 scenes (~20s each) with 2Hz annotations
- Target categories: `human.pedestrian.*` and `vehicle.bicycle`
- Neighbor radius: 20 m, max 5 neighbors per agent

**Data split:**
| Split | Scenes |
|-------|--------|
| Train | nuScenes official train split |
| Val | First 100 val scenes |
| Test | Remaining val scenes |

**Preprocessed outputs from Notebook 1:**
- `trajectories.csv` — agent-centric normalized tracks
- `neighbors.pkl` — neighbor trajectory tensors per sample
- `lidar_feats.pkl` — LiDAR occupancy features per sample
- `norm_stats.csv` — normalization statistics (std_x, std_y)

---

## Setup & Installation

### Requirements

```
Python 3.9+
PyTorch (CUDA-enabled)
nuscenes-devkit
pyquaternion
cachetools
fire
tqdm
pandas
numpy
matplotlib
```

### Install dependencies

```bash
pip install nuscenes-devkit --no-deps
pip install pyquaternion cachetools fire tqdm pandas numpy matplotlib torch
```

### Kaggle setup (recommended — GPU T4 ×2)

1. Add datasets to your Kaggle notebook:
   - [`mariaamm/nuscenes-trainval-metadata`](https://www.kaggle.com/datasets/mariaamm/nuscenes-trainval-metadata)
   - [`adecours/nuscenes-fulldataset-1-510-trainval`](https://www.kaggle.com/datasets/adecours/nuscenes-fulldataset-1-510-trainval)
2. Settings → Accelerator → **GPU T4 ×2**
3. Run all cells in each notebook in order (NB1 → NB2 → NB3)
4. After NB1 finishes: Save Version → Output tab → "New Dataset from Output" → name it `intentformer-nb1-out`
5. Attach `intentformer-nb1-out` as input to NB2; repeat for NB2 output before running NB3

### Colab setup

1. Mount Google Drive in Cell 1 of each notebook
2. Run Cell 1 first in NB1 — it installs `kagglehub` and downloads both datasets (~150 GB total)
3. Ensure `kaggle.json` is present at `~/.kaggle/` or set `KAGGLE_USERNAME` / `KAGGLE_KEY` env vars
4. All outputs are saved to `MyDrive/intentformer/` and shared across notebooks automatically

---

## How to Run

The project is structured as three sequential Jupyter notebooks:

### Notebook 1 — Data Preparation (`01-dataprep-v2.ipynb`)

Processes raw nuScenes data into training-ready tensors.

- Downloads/locates nuScenes metadata and LiDAR blobs
- Extracts global agent tracks across all scenes
- Builds sliding windows (past 4 frames + future 6 frames)
- Converts to agent-centric coordinates and infers intent labels
- Normalizes trajectories and saves all outputs

> ⚠️ Cell 2 takes ~15 min; Cell 7 takes ~2 hrs on Kaggle GPU T4 ×2.

```
Run all cells → Save outputs as dataset `intentformer-nb1-out`
```

### Notebook 2 — Model Training (`02-model-train-v2.ipynb`)

Trains IntentFormer on the preprocessed data.

- Loads `trajectories.csv`, `neighbors.pkl`, `lidar_feats.pkl`
- Trains with mixed-precision (AMP), `DataParallel` on multi-GPU
- Saves best checkpoint (`best_intentformer.pt`) based on val π-ADE
- Saves training history (`training_history.csv`)
- Evaluates a Constant Velocity (CV) baseline on the test set

```
Update INPUT_DIR to point to NB1 output → Run all cells
```

### Notebook 3 — Evaluation (`03-eval-v2.ipynb`)

Comprehensive evaluation and visualization.

- Loads best model checkpoint
- Runs mode selection strategies: π-greedy, CV-closest, min-uncertainty, and combined weighted ranking
- Grid-searches optimal combination weights on the validation set
- Reports test metrics with 95% bootstrap confidence intervals
- Generates visualizations: trajectory plots, temporal ADE curve, mode weight distribution

```
Update INPUT_DIR and MODEL_PATH → Run all cells
```

---

## Example Outputs / Results

### Evaluation Metrics (test set)

| Metric | Description |
|--------|-------------|
| **minADE@K** | Min average displacement error across K modes (oracle) |
| **minFDE@K** | Min final displacement error across K modes (oracle) |
| **π-ADE** | ADE using the highest-weight mode (deployment metric) |
| **cv-ADE** | ADE using the mode closest to constant-velocity prediction |
| **combined-ADE** | ADE using weighted rank combination of π, CV, uncertainty |

Metrics are reported per category (Pedestrian / Bicycle) and overall, with 95% bootstrap CIs.

### Visualizations generated by NB3

- **Trajectory plots** — best, worst, and mode-collapse examples showing past (blue), ground truth (green), and K predicted modes (colored)
- **Temporal ADE curve** — oracle ADE at each future timestep (0.5 s to 3.0 s)
- **Mode weight distribution** — histogram of max π weight and entropy across test samples
- **Training curves** — GMM-NLL, diversity loss, intent CE, and val ADE over epochs

---

## Repository Structure

```
Intent-and-Trajectory-Predictor/
├── 01-dataprep-v2.ipynb      # Data preparation pipeline
├── 02-model-train-v2.ipynb   # Model definition + training
├── 03-eval-v2.ipynb          # Evaluation + visualization
└── README.md
```

> 📌 The repository must be **publicly accessible** during evaluation.

---

## Notes

- All notebooks are self-contained and detect the environment (Kaggle vs. Colab) automatically.
- Checkpoints are saved without `module.` prefix (DataParallel-safe) for clean reloading.
- A `no-social` ablation model (`best_nosocial.pt`) can be trained and compared in NB3 to measure the contribution of social context.
