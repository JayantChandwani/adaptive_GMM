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
- `data/` – generated datasets and saved artifacts
- `experiments/` – per-run experiment outputs and metrics
- `baseline_experiments/` – optional location for NSL-KDD baseline runs (if you set `exp_root` when calling `run_nsl_kdd`; see below)

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

## Run the Pipeline

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

### Outputs

- Datasets: `data/simulated/simulated_drift_*.npz`, `data/nsl_kdd/nsl_kdd.npz`
- Simulated runs: `experiments/<model_name>/<dataset_stem>/{metrics.json, ...}`, plus `experiments/<model_name>/summary.json`
- Combined summary (`--model all`): `experiments/summary_all_models.json`
- NSL-KDD runs: `<exp-root>/<config_name>/{metrics.json, scores.npz, *.png}` and a cumulative `<exp-root>/summary.json`. Config name encodes the run, e.g.:
  - `dpmm_online_bs256_k12_normalonly_fixthr/`
  - `dpmm_frozen_bs256_k12_normalonly_fixthr/`



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

