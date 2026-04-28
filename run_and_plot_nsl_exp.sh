#!/usr/bin/env bash
set -euo pipefail

# Runs simulated + NSL-KDD experiments for available models and then
# generates cross-model comparison plots.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
BATCH_SIZE="${BATCH_SIZE:-32}"
K_COMPONENTS="${K_COMPONENTS:-12}"
EXP_ROOT="${EXP_ROOT:-experiments}"

log() {
  printf '\n[%s] %s\n' "$(date +"%H:%M:%S")" "$1"
}

run_cmd() {
  echo "> $*"
  "$@"
}

log "Checking model availability"
MODELS=("gmm" "dpmm")
if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
from src.models.dpmm_da import DPGMMDriftAware
print(DPGMMDriftAware.__name__)
PY
then
  MODELS+=("dpmm_da")
fi

echo "Models to run: ${MODELS[*]}"

log "Ensuring NSL-KDD dataset is present"
run_cmd "$PYTHON_BIN" -m src.utils.download_dataset

log "Running simulated drift experiments"
for model in "${MODELS[@]}"; do
  run_cmd "$PYTHON_BIN" -m src.eval.train_and_evaluate --model "$model" --batch-size "$BATCH_SIZE"
done

log "Running NSL-KDD frozen + online experiments"
for model in "${MODELS[@]}"; do
  run_cmd "$PYTHON_BIN" -m src.eval.train_and_evaluate --nsl --model "$model" --k "$K_COMPONENTS" --batch-size "$BATCH_SIZE"
  if [[ "$model" != "gmm" ]]; then
    run_cmd "$PYTHON_BIN" -m src.eval.train_and_evaluate --nsl --model "$model" --k "$K_COMPONENTS" --batch-size "$BATCH_SIZE" --online
  fi
done

log "Generating comparison plots"
run_cmd "$PYTHON_BIN" -m src.utils.plot_model_comparisons --exp-root "$EXP_ROOT" --out-dir "$EXP_ROOT/comparisons"

log "Done"
echo "Generated outputs in: $EXP_ROOT"
echo "Key plots in: $EXP_ROOT/comparisons"
