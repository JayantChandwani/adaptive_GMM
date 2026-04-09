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

Generate datasets, train a model, and run the online test pipeline:

```bash
python -m src.eval.train_and_evaluate --model gmm
```

Run the DP-GMM model:

```bash
python -m src.eval.train_and_evaluate --model dpmm
```

Run the drift-aware DP-GMM model:

```bash
python -m src.eval.train_and_evaluate --model dpmm_da
```

Run all models in one command:

```bash
python -m src.eval.train_and_evaluate --model all
```

Optional batch size for online test updates:

```bash
python -m src.eval.train_and_evaluate --model dpmm_da --batch-size 128
```

Outputs:
- Datasets: `data/simulated/simulated_drift_*.npz`
- GMM experiments: `experiments/gmm/`
- DP-GMM experiments: `experiments/dpmm/`
- Drift-aware DP-GMM experiments: `experiments/dpmm_da/`
- Combined summary (when `--model all`): `experiments/summary_all_models.json`