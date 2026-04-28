# Adaptive GMM

This project explores Gaussian mixture based anomaly detection, starting with a simple sklearn-backed GMM baseline and extending toward drift-aware and adaptive mixture models.

## Overview

The codebase contains three closely related model families:

- `gmm` as the baseline density estimator
- `dpmm` as the online Dirichlet process mixture model
- `dpmm_da` as the drift-aware variant

The repository is organized around two experiment tracks:

- simulated drift experiments for controlled comparison
- NSL-KDD experiments for a real benchmark dataset

For a full end-to-end batch run, use `run_and_plot_nsl_exp.sh`, which executes the simulated and NSL-KDD experiments and then generates comparison plots.

## Repository Layout

- `src/models/` – model implementations
	- `gmm.py` – sklearn wrapper used as the baseline model
	- `dpmm.py` – custom Dirichlet process mixture model
	- `dpmm_da.py` – drift-aware DP mixture model
	- `basemodel.py` – shared model interface
- `src/eval/` – training and online testing pipeline
- `src/utils/` – dataset generation and plotting helpers
- `data/` – generated datasets and saved artifacts
- `experiments/` – per-run experiment outputs, summaries, and plots
- `baseline_experiments/` – optional location for NSL-KDD baseline runs

## Environment Setup

Requirements:
- Python 3.10+

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

Or with Conda:

```bash
conda create -n adaptive_gmm python=3.10 -y
conda activate adaptive_gmm
pip install -U pip
pip install -e .
```

## Quick Start

If you want the full workflow in one command, run:

```bash
bash run_and_plot_nsl_exp.sh
```

This script checks model availability, downloads the NSL-KDD dataset if needed, runs the simulated drift suite, runs the NSL-KDD experiments, and generates comparison plots.

## Running Experiments

The CLI in `src/eval/train_and_evaluate.py` covers two pipelines:

1. **Simulated drift suite** (default): generates a family of simulated streams with increasing drift levels and evaluates a model on each.
2. **NSL-KDD** (`--nsl`): trains on normal traffic only, then either runs a frozen one-shot test or streams the test set in arrival order with `partial_fit` (`--online`).

### CLI flags

| Flag | Default | Notes |
|------|---------|-------|
| `--model {gmm,dpmm,dpmm_da,all}` | `gmm` | Model to use. `dpmm_da` requires `src/models/dpmm_da.py` to be implemented. |
| `--k <int>` | model default | Number of components / DP-GMM truncation level. Overrides the per-model default in `_build_model`. |
| `--batch-size <int>` | `32` | Streaming batch size for the simulated online test or for `--nsl --online`. |
| `--nsl` | off | Switch to the NSL-KDD pipeline. |
| `--online` | off | With `--nsl`: stream test in arrival order with `partial_fit`. Without it: frozen one-shot scoring. |
| `--dataset <path>` | `data/nsl_kdd/nsl_kdd.npz` | NSL-KDD `.npz` path. |
| `--exp-root <path>` | `experiments/nsl_kdd_dpmm` | Output root for `--nsl` runs. |

### Common commands

Simulated drift suite (each model writes per-dataset metrics + a `summary.json`):

```bash
python -m src.eval.train_and_evaluate --model gmm
python -m src.eval.train_and_evaluate --model dpmm
python -m src.eval.train_and_evaluate --model all          # gmm + dpmm (+ dpmm_da if implemented)
```

NSL-KDD frozen baseline (one-shot scoring, no model updates during test):

```bash
python -m src.eval.train_and_evaluate --nsl --model dpmm --batch-size 256 --k 12
```

NSL-KDD streaming evaluation (arrival-ordered batches, `partial_fit` per batch):

```bash
python -m src.eval.train_and_evaluate --nsl --model dpmm --online --batch-size 256 --k 12
```

### Output Structure

- Datasets: `data/simulated/simulated_drift_*.npz`, `data/nsl_kdd/nsl_kdd.npz`
- Simulated runs: `experiments/<model_name>/<dataset_stem>/{metrics.json, ...}`, plus `experiments/<model_name>/summary.json`
- Combined summary (`--model all`): `experiments/summary_all_models.json`
- NSL-KDD runs: `<exp-root>/<config_name>/{metrics.json, scores.npz, *.png}` and a cumulative `<exp-root>/summary.json`. Config name encodes the run, e.g.:
  - `dpmm_online_bs256_k12_normalonly_fixthr/`
  - `dpmm_frozen_bs256_k12_normalonly_fixthr/`

## Plotting Results

The utility script for plotting a single model's simulated-drift results lives in `src/utils/plot_single_model.py`.

It reads the per-dataset `metrics.json` files from:

- `experiments/gmm/simulated_drift_*`
- `experiments/dpmm/simulated_drift_*`
- `experiments/dpmm_da/simulated_drift_*`

and writes plots to `experiments/<model>/plots/` by default.

### Example usage

```bash
python -m src.utils.plot_single_model --model gmm
python -m src.utils.plot_single_model --model dpmm
python -m src.utils.plot_single_model --model dpmm_da
```

### Generated files

For each model, the script creates:

- `<model>/plots/precision_vs_drift.png`
- `<model>/plots/recall_vs_drift.png`
- `<model>/plots/f1_vs_drift.png`
- `<model>/plots/roc_auc_vs_drift.png`
- `<model>/plots/pr_auc_vs_drift.png`
- `<model>/plots/average_metrics.png`
- `<model>/simulated_summary.json`

The plots use the slide-inspired color palette from the presentation screenshot.

## NSL-KDD Dataset

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
- Saves explicit arrival-time vectors `t_train` and `t_test` (= row index in the ARFF file). The streaming evaluator uses `t_test` to sort batches by time of arrival.

### Where it is stored

| Path | Contents |
|------|----------|
| `data/nsl_kdd/nsl_kdd.npz` | `X_train`, `y_train`, `X_test`, `y_test`, `t_train`, `t_test` (NumPy arrays) |

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

The dedicated baseline driver lives in `src/eval/train_and_evaluate_baselineGMM.py` (`bic` / `nsl` subcommands) and writes per-`K` folders (`k8`, `k12`, ...) plus `summary.json` and ROC/PR plots under `baseline_experiments/nsl_kdd/` (or wherever you point `exp_root`). The unified CLI in `src/eval/train_and_evaluate.py` can also reproduce the frozen baseline directly:

```bash
python -m src.eval.train_and_evaluate --nsl --model gmm --k 12
```

## Notes on the Batch Script

`run_and_plot_nsl_exp.sh` is the convenience wrapper for reproducing the whole experiment pipeline end to end. It runs the simulated experiments, then the NSL-KDD experiments, and finally creates the comparison plots under `experiments/comparisons/`.

