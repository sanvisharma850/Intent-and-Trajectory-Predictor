
# IntentFormer — Pedestrian & Cyclist Trajectory Prediction

<p align="center">
  <img src="assets/hero.png" width="90%" />
</p>

**Track:** Intent & Trajectory Prediction · **Focus:** Behavioral AI & Temporal Modeling  
**Dataset:** nuScenes v1.0-trainval · **Platform:** Kaggle (GPU T4)

A multimodal trajectory prediction model for pedestrians and cyclists using the nuScenes dataset. IntentFormer combines Transformer-based temporal encoding, social context (neighboring agents), LiDAR features, and intent classification to generate diverse, probabilistic future trajectory predictions.

---

## Project Overview

IntentFormer predicts the future paths of pedestrians and cyclists in autonomous driving scenarios. Given a short observation window (**2 seconds / 4 frames**), the model outputs **K multimodal trajectory hypotheses** as a **Gaussian Mixture Model (GMM)**, along with an **intent classification** (e.g., walking, waiting, turning).

<p align="center">
  <img src="assets/trajectory.png" width="75%" />
</p>

The model is evaluated on the [nuScenes v1.0-trainval](https://www.nuscenes.org/) dataset.

### Key Capabilities

- Multimodal future prediction (K hypotheses with learned mixture weights π)
- Intent classification (waiting, walking, turning left/right)
- Social context via neighbor trajectory encoding
- LiDAR-based scene features
- Evaluated with minADE, minFDE, and greedy π-ADE metrics

---

## Model Architecture

<p align="center">
  <img src="intentformer_architecture.png" width="85%" />
</p>

IntentFormer is composed of three main components:

### 1. TemporalEncoder
Transformer encoder with sinusoidal positional encoding processing past trajectory:
- 4 frames × 4 features (Δx, Δy, speed, heading)

### 2. SocialEncoder
- Encodes up to **5 neighboring agents**
- Uses shared Transformer encoder
- Mean-pooling to generate social context vector

<p align="center">
  <img src="assets/social_context.png" width="70%" />
</p>

### 3. IntentFormer (Full Model)
Combines temporal encoding + social context + LiDAR features to:

- Predict **K trajectory modes** (Gaussian parameters: μ, σ, ρ)
- Predict **mixture weights π**
- Classify **agent intent (4 classes)**

---

### Training Losses

- **GMM-NLL** — trajectory regression  
- **Diversity Loss** — prevents mode collapse  
- **Intent CE** — intent classification  

---

### Hyperparameters

| Parameter | Value |
|----------|------|
| d_model | 256 |
| Attention heads | 4 |
| Transformer layers | 3 |
| K (modes) | 5 |
| Past frames | 4 (2 s) |
| Future frames | 6 (3 s) |
| Δt | 0.5 s |

---

## Dataset

**nuScenes v1.0-trainval**

- 1000 scenes (~20s each, 2Hz)
- Categories:
  - `human.pedestrian.*`
  - `vehicle.bicycle`
- Neighbor radius: **20 m (max 5 neighbors)**

### Data Split

| Split | Scenes |
|------|--------|
| Train | Official train split |
| Val | First 100 val scenes |
| Test | Remaining val scenes |

---

### Preprocessed Outputs (Notebook 1)

- `trajectories.csv`
- `neighbors.pkl`
- `lidar_feats.pkl`
- `norm_stats.csv`

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

````

### Install

```bash
pip install nuscenes-devkit --no-deps
pip install pyquaternion cachetools fire tqdm pandas numpy matplotlib torch
````



---

## Results

### Evaluation Metrics

| Metric       | Description            |
| ------------ | ---------------------- |
| minADE@K     | Oracle ADE             |
| minFDE@K     | Oracle FDE             |
| π-ADE        | Top-weighted mode      |
| cv-ADE       | Closest to CV baseline |
| combined-ADE | Weighted ranking       |

---

### Visual Outputs

<p align="center">
  <img src="Training_Output/trajectory_predictions.png" width="80%" />
</p>

<p align="center">
  <img src="Training_Output/temporal_ade_curve.png" width="70%" />
</p>

<p align="center">
  <img src="Training_Output/mode_weight_distribution.png" width="70%" />
</p>

---

## Repository Structure

```
Intent-and-Trajectory-Predictor/
│
├── 01-dataprep-v2.ipynb
├── 02-model-train-v2.ipynb
├── 03-eval-v2.ipynb
│
├── intentformer_architecture.png
│
├── Training_Output/
│   ├── evaluation_summary.md
│   ├── training_curves.png
│   ├── trajectory_predictions.png
│   ├── temporal_ade_curve.png
│   ├── mode_weight_distribution.png
│
├── assets/                     
│   ├── hero.png
│   ├── trajectory.png
│   ├── social_context.png
│
├── .gitignore
└── README.md
```

---

## Notes

* Notebooks auto-detect Kaggle / Colab environment
* Checkpoints saved without `module.` prefix (DataParallel-safe)
* Supports **no-social ablation model** (`best_nosocial.pt`)

---

## Future Work

* Add Transformer vs LSTM baseline comparison
* Graph-based multi-agent interaction modeling
* Real-time inference demo (Streamlit)
* Improved uncertainty calibration

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

