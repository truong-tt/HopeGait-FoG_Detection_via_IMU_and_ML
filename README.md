# HopeGait

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-TCN%20%2B%20AMP-orange.svg)](https://pytorch.org/)
[![Edge AI](https://img.shields.io/badge/Edge-TFLite%20int8-green.svg)](https://ai.google.dev/edge/litert)
[![Tests](https://img.shields.io/badge/Tests-pytest-lightgrey.svg)](tests/)

Real-time Freezing of Gait (FoG) detection from wearable IMU data using a causal TCN, targeting edge deployment on microcontrollers.

FoG — sudden gait arrest mid-step — is a hallmark Parkinson's symptom and a primary fall risk. This project trains a low-latency on-device classifier so cueing systems can intervene in real time.

> **Scope disclaimer:** Student research project. Not a medical device. Not clinically validated. Do not use for safety-critical clinical decisions.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Problem Setup](#2-problem-setup)
3. [Modeling and Training](#3-modeling-and-training)
4. [Data Pipeline and Leakage Controls](#4-data-pipeline-and-leakage-controls)
5. [Inference and Post-processing](#5-inference-and-post-processing)
6. [Repository Layout](#6-repository-layout)
7. [Dataset Setup](#7-dataset-setup)
8. [Installation and Quick Start](#8-installation-and-quick-start)
9. [Cloud and Docker Runs](#9-cloud-and-docker-runs)
10. [Evaluation and Tests](#10-evaluation-and-tests)
11. [Next Steps](#11-next-steps)
12. [Data and Credit](#data-and-credit)
13. [License](#license)

---

## 1. System Architecture

| Step | Module | Operation |
|------|--------|-----------|
| 1 | `data_pipeline/preprocess.py` | Resample → bandpass filter → gravity split → windowing |
| 2 | `data_pipeline/dsp.py` + `dataset.py` | 9-channel windows → LOSO subject splits → augmentation |
| 3 | `models/tcn_model.py` | Causal TCN — last-step head + dense auxiliary head |
| 4 | `training/train.py` + `evaluate.py` | Focal loss + AdamW + EMA + MCC-based model selection |
| 5 | `inference/postprocess.py` | Causal smoothing + hysteresis trigger |
| 6 | `edge_conversion/quantize_model.py` | Quantization + export for edge deployment |

---

## 2. Problem Setup

| | |
|---|---|
| **Input** | 9 channels per timestep: linear acceleration (3-axis), gravity component (3-axis), gyroscope (3-axis) |
| **Output** | Per-window FoG probability → post-processed binary decision for cueing logic |
| **Phase 1 — Brain** | Train causal TCN with leave-one-subject-out (LOSO) cross-validation |
| **Phase 2 — Edge** | Compress and export model artifacts for microcontroller-class deployment |

---

## 3. Modeling and Training

### 3.1 Why a causal TCN

| Property | Detail |
|---|---|
| Receptive field | Large, via dilated convolutions — low compute cost |
| Causality | Output at step `t` never depends on future samples |
| Deployment fit | Viable for streaming inference and MCU export |

### 3.2 Heads and losses

| Head | Role |
|---|---|
| Last-step head | Real-time deployment inference |
| Dense per-timestep head | Auxiliary training supervision — stronger gradient signal |

Both heads use focal loss to handle class imbalance. Total loss = last-step loss + (`dense_loss_weight` × dense loss). `dense_loss_weight` is set in `config.py`.

### 3.3 Optimization

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Mixed precision | Enabled on CUDA (optional) |
| EMA shadow weights | Used for validation and checkpoint selection |
| Best epoch criterion | MCC on inner validation subject |

---

## 4. Data Pipeline and Leakage Controls

### Leakage prevention

| Control | Implementation |
|---|---|
| LOSO splitting | Strict split by true subject identity |
| Inner validation | Subject selected from training pool only — never from test fold |
| Threshold selection | Youden-J on inner validation split, frozen before test fold |
| Scaler | `RobustScaler` fit on training data only; serialized for inference |

### Bug fixes integrated

| Fix | Detail |
|---|---|
| Freeze Index correction | PSD computed per axis then summed — prevents magnitude-rectification frequency doubling |
| Filter startup transients | Removed via `lfilter_zi` warm starts |
| Same-subject split leakage | Fixed by grouping recordings under the same person ID |

---

## 5. Inference and Post-processing

| Stage | Operation |
|---|---|
| 1 — Smoothing | Causal moving average over recent classifier probabilities |
| 2 — Decision | Schmitt-trigger hysteresis around fold-specific threshold |

Eliminates near-threshold flicker that would otherwise produce unstable cueing behavior.

---

## 6. Repository Layout

```
HopeGait-FoG_Detection_via_IMU_and_ML/
├── Dockerfile
├── README.md
├── requirements.txt          # Training deps
├── requirements-edge.txt     # Edge conversion deps (install separately)
├── scripts/
│   ├── launch_cloud.sh
│   └── smoke_train.py
├── src/
│   ├── config.py
│   ├── main.py
│   ├── data_pipeline/
│   │   ├── dataset.py
│   │   ├── dsp.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── tcn_model.py
│   │   └── focal_loss.py
│   ├── training/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── ema.py
│   ├── inference/
│   │   └── postprocess.py
│   └── edge_conversion/
│       └── quantize_model.py
└── tests/
    ├── test_dataset.py
    ├── test_dsp.py
    ├── test_focal_loss.py
    ├── test_postprocess.py
    └── test_tcn_causality.py
```

---

## 7. Dataset Setup

Data source: [Stanford NMBL IMU FoG Detection Repository](https://github.com/stanfordnmbl/imu-fog-detection).

**Step 1 — Clone the Stanford dataset:**

```bash
git clone https://github.com/stanfordnmbl/imu-fog-detection.git
```

**Step 2 — Place raw files under `data/raw/`:**

```
data/
└── raw/
    ├── subject_01/
    │   └── *.csv      # 100 Hz, 6-axis IMU recordings
    ├── subject_02/
    └── ...
```

The pipeline expects lumbar-mounted IMU recordings at 100 Hz with 6-axis output (3-axis accelerometer + 3-axis gyroscope). Subject subdirectory naming must be consistent — the LOSO split groups by directory name as the subject ID.

**Step 3 — Verify placement before running:**

```bash
ls data/raw/
```

> Use of raw data is subject to Stanford NMBL's original license and terms. See their repo for details.

---

## 8. Installation and Quick Start

### Requirements

- Python 3.11+
- CUDA-capable GPU recommended for full LOSO training (CPU works for smoke tests)

### Local setup (recommended)

```bash
# 1. Clone
git clone https://github.com/truong-tt/HopeGait-FoG_Detection_via_IMU_and_ML.git
cd HopeGait-FoG_Detection_via_IMU_and_ML

# 2. Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install PyTorch with the correct CUDA wheel FIRST
# CUDA 12.1 (most cloud GPU boxes):
pip install torch --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8:
# pip install torch --index-url https://download.pytorch.org/whl/cu118
# CPU only (CI / smoke tests):
# pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4. Install remaining training dependencies
pip install -r requirements.txt

# 5. Set PYTHONPATH (required for local runs — src/ is the package root)
export PYTHONPATH=$(pwd)/src      # Windows: set PYTHONPATH=%cd%\src

# 6. Place dataset (see Section 7), then run
python src/main.py
```

> **Why install PyTorch first?** `requirements.txt` pins `torch>=2.1` without a wheel URL. Installing PyTorch separately with the correct CUDA index prevents pip from pulling the CPU wheel and silently breaking GPU training.

### CLI flags

```bash
# Skip training, run evaluation only
python src/main.py --skip-train

# Skip both training and evaluation (pipeline dry-run)
python src/main.py --skip-train --skip-eval

# Force re-run of preprocessing even if cache exists
python src/main.py --force-preprocess
```

### Edge conversion (optional)

Only needed when running `edge_conversion/quantize_model.py`. Install on top of the base environment:

```bash
pip install -r requirements-edge.txt
```

Includes `ai-edge-torch>=0.2.0` (primary path) and the legacy ONNX/TF stack (`tensorflow==2.15.0`, `onnx>=1.15`, `onnx-tf>=1.10`, `protobuf>=3.20.3,<5`). Do not install edge deps into the same environment used for cloud training — protobuf and TF version conflicts will break the PyTorch stack.

### Smoke test (no GPU, no data required)

```bash
python scripts/smoke_train.py
```

---

## 9. Cloud and Docker Runs

### Cloud GPU (bare metal)

```bash
bash scripts/launch_cloud.sh

# Override window size
WINDOW=200 bash scripts/launch_cloud.sh

# Override subject and epoch count
SUBJECT=03 EPOCHS=5 bash scripts/launch_cloud.sh
```

### Docker (CUDA GPU)

```bash
# Build
docker build -t hopegait:latest .

# Run with GPU and volume mounts
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  hopegait:latest python src/main.py
```

### Docker (CPU smoke test)

```bash
docker build -t hopegait:cpu --build-arg BASE=python:3.11-slim .
docker run --rm hopegait:cpu python scripts/smoke_train.py
```

`PYTHONPATH=/app/src` is set inside the container by the Dockerfile — no manual export needed in Docker runs.

---

## 10. Evaluation and Tests

### Run all tests

```bash
pytest
```

### Metrics tracked

| Metric | Role |
|---|---|
| MCC | Primary selection criterion (imbalanced-safe) |
| Post-processed operating point | Primary deployment comparison |
| Sensitivity / Specificity | Clinical interpretability |
| F1 | Standard binary classification |
| PR-AUC / ROC-AUC | Threshold-agnostic ranking |
| Accuracy | Reported only — not prioritized (imbalanced classes) |

Three operating points reported per fold: fixed-threshold, fold-optimized threshold, and post-processed (hysteresis) output.

---

## 11. Next Steps

- Run full LOSO on cloud GPU and publish final per-subject results table.
- Complete edge path with robust int8 conversion flow via `ai-edge-torch`.
- Add streaming parity checks between offline and deployed inference.
- Benchmark latency, memory footprint, and compute on target MCU.

---

## Data and Credit

Data: [Stanford NMBL IMU FoG Detection Repository](https://github.com/stanfordnmbl/imu-fog-detection)

All credit for data collection and release goes to Stanford Neuromuscular Biomechanics Laboratory. Use of raw data is subject to their original license and terms.

---

## License

TBD.

> **Note:** The raw IMU dataset (Stanford NMBL) is governed by its own separate license.