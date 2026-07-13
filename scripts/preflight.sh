#!/usr/bin/env bash
set -euo pipefail

: "${PUSHT_DATASET_PATH:?set PUSHT_DATASET_PATH}"
: "${PUSHT_LEWM_WEIGHTS:?set PUSHT_LEWM_WEIGHTS}"
: "${ACT_LEWM_CACHE_ROOT:?set ACT_LEWM_CACHE_ROOT}"
: "${ACT_LEWM_RUN_ROOT:?set ACT_LEWM_RUN_ROOT}"

export ACT_LEWM_CODE_REVISION="${ACT_LEWM_CODE_REVISION:-$(git rev-parse HEAD)}"
export PYTHONDONTWRITEBYTECODE=1

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -m ruff check --no-cache .
python -m ruff format --check --no-cache .
python -m pytest -q
git diff --check

python -m train.cache_latents --cfg job >/dev/null
python -m train.train_world_model --cfg job >/dev/null
python -m train.train_gc_idm --cfg job >/dev/null
python -m train.train_larc --cfg job >/dev/null
python -m eval.open_loop_curve --cfg job >/dev/null

for specification in "25 50 5" "35 70 7" "50 100 10"; do
  read -r goal_offset eval_budget cem_horizon <<<"${specification}"
  for method in cem gc_idm larc; do
    overrides=(
      "method=${method}"
      "protocol.goal_offset=${goal_offset}"
      "protocol.eval_budget=${eval_budget}"
      "cem.horizon=${cem_horizon}"
    )
    python -m eval.closed_loop --cfg job "${overrides[@]}" >/dev/null
  done
done

echo "H50 static checks, tests, and Hydra composition passed."
