# Model Evaluation Summary

## Model Setup

| Item | Value |
|---|---|
| Architecture | Defined |
| Validation samples | 16,180 |
| Test samples | 3,844 |
| Model parameters | 2,710,497 |
| Best weights | w_cv = 0.1, w_pi = 0.6, w_unc = 0.3 |
| Validation combined ADE | 0.2896 m |

---

## Problem Statement Metrics (Test Set)

### Core Metrics

| Metric | Value | Confidence Interval |
|---|---|---|
| minADE@5 (oracle) | 0.2444 m | [0.2359, 0.2532] |
| minFDE@5 (oracle) | 0.4430 m | [0.4264, 0.4603] |
| Best greedy (pi) | 0.2988 m | Ratio: 1.22× |
| CV baseline ADE | nan | — |

### Method Comparison

| Method | ADE | Confidence Interval | Ratio | Note |
|---|---|---|---|---|
| pi argmax | 0.2988 m | [0.2890, 0.3085] | 1.22 | **Best** |
| CV-guided | 0.3125 m | [0.3021, 0.3228] | 1.28 | — |
| Uncertainty | 1.5131 m | [1.4780, 1.5472] | 6.19 | Worst |
| Combined | 0.2994 m | [0.2895, 0.3092] | 1.23 | Close to best |

---

## Category-wise Performance

| Category | Samples | minADE | minFDE | Best Greedy |
|---|---|---|---|---|
| Pedestrian | 3,696 | 0.2500 m | 0.4538 m | 0.3060 m |
| Pedestrian | 5 | 0.0233 m | 0.0580 m | 0.0521 m |
| Bicycle | 143 | 0.1056 m | 0.1755 m | 0.1208 m |
| **ALL** | **3,844** | **0.2444 m** | **0.4430 m** | **0.2988 m** |

---

## ADE over Prediction Horizon

| Time Horizon | ADE |
|---|---|
| 0.5 s | 0.0636 m |
| 1.0 s | 0.1260 m |
| 1.5 s | 0.1933 m |
| 2.0 s | 0.2678 m |
| 2.5 s | 0.3607 m |
| 3.0 s | 0.4548 m |

---

## Key Takeaways

- **Best performing method:** pi argmax
- **Oracle ADE:** 0.2444 m
- **Best greedy ADE:** 0.2988 m
