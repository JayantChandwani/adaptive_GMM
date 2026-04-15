# Adaptive GMM

This project explores Gaussian mixture based anomaly detection, starting with a simple sklearn-backed GMM baseline and extending toward drift-aware and adaptive mixture models.

## Project Structure

- `src/models/` – model implementations
	- `gmm.py` – sklearn wrapper used as the baseline model
	- `dpmm.py` – custom Dirichlet process mixture model
	- `dpmm_da.py` – drift-aware DP mixture model
	- `basemodel.py` – shared model interface
- `src/utils/` – dataset generation and helper scripts
- `src/eval/` – training and online testing pipeline
- `data/` – generated datasets and saved artifacts (including NSL-KDD)
- `experiments/` – per-run experiment outputs and metrics
- `baseline_experiments/` – optional location for NSL-KDD baseline runs (if you set `exp_root` when calling `run_nsl_kdd`; see below)

## NSL-KDD dataset

### Download

From the project root (with dependencies installed, including `kagglehub` and `pandas`):

```bash
python -m src.utils.download_dataset
```

This downloads the [Hassan NSL-KDD bundle](https://www.kaggle.com/datasets/hassan06/nslkdd) via Kaggle Hub (cached under `~/.cache/kagglehub/`), parses `KDDTrain+.arff` and `KDDTest+.arff`, and writes a single processed file.

### What the script does

- Reads both ARFF files (train and test are separate sources; they are merged into one `.npz` for convenience).
- Encodes labels: `normal` → `0`, `anomaly` → `1`.
- Drops the label column from features.
- One-hot encodes categoricals `protocol_type`, `service`, and `flag` (encoder fit on **train** only, then applied to test).
- Concatenates numeric features with the one-hot columns.
- **Does not** apply `StandardScaler` here; scaling is done during training/evaluation so train statistics are not leaked into test preprocessing.

### Where it is stored

| Path | Contents |
|------|----------|
| `data/nsl_kdd/nsl_kdd.npz` | `X_train`, `y_train`, `X_test`, `y_test` (NumPy arrays) |

If the file already exists, `load_nsl_kdd()` skips re-download unless you use `force_redownload=True` in code.

### How to use the data in code

Load with NumPy:

```python
import numpy as np
data = np.load("data/nsl_kdd/nsl_kdd.npz", allow_pickle=True)
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]
```

Or call `from src.utils.download_dataset import load_nsl_kdd` and use the returned dict.

## NSL-KDD baseline (GMM)

The baseline treats anomaly detection as **density estimation on normal traffic only**:

1. **Train:** `StandardScaler` is fit on **normal training rows only** (`y_train == 0`). A Gaussian mixture (`GMM`) is fit on that scaled normal data (number of components `K` is chosen separately, e.g. via a BIC sweep).
2. **Test:** The same scaler transforms `X_test`. The **frozen** model scores every test point with `score_samples` and flags anomalies with `predict_anomaly` (threshold from training scores). The model is **not** updated during testing for this baseline.

**Commands** (from project root):

```bash
# Optional: sweep K and write BIC plot (default output: experiments/nsl_kdd/bic_sweep.png)
python -m src.eval.train_and_evaluate bic

# Train/test for default K values and write metrics + plots
python -m src.eval.train_and_evaluate nsl
```

By default, outputs go under `experiments/nsl_kdd/` (per-`K` folders `k8`, `k12`, etc., plus `summary.json`, ROC/PR plots, and related figures). To write results under `baseline_experiments/` instead, call from Python:

```python
from src.eval.train_and_evaluate import run_nsl_kdd
run_nsl_kdd(exp_root="baseline_experiments/nsl_kdd")
```

That layout is suitable for versioning baseline figures and metrics alongside the repo without mixing them with the simulated-drift runs under `experiments/`.

## Environment Setup

Requirements:
- Python 3.11+

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Or with Conda:

```bash
conda create -n adaptive_gmm python=3.11 -y
conda activate adaptive_gmm
pip install -U pip
pip install -e .
```



## Run the Pipeline

Generate datasets, train the baseline model, and run the online test pipeline:

```bash
python -m src.eval.train_and_evaluate
```

This writes datasets to `data/simulated/` and experiment outputs to `experiments/`.