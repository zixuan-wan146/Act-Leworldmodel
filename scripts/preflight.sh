#!/usr/bin/env bash
set -euo pipefail

task="${1:-pusht}"
case "${task}" in
  pusht)
    required_task_variables=(PUSHT_DATASET_PATH PUSHT_LEWM_WEIGHTS)
    ;;
  tworoom)
    required_task_variables=(TWOROOM_DATASET_PATH TWOROOM_LEWM_WEIGHTS)
    ;;
  *)
    echo "unknown task: ${task}; expected pusht or tworoom" >&2
    exit 2
    ;;
esac

required_variables=(ACT_LEWM_CACHE_ROOT ACT_LEWM_RUN_ROOT "${required_task_variables[@]}")
for variable in "${required_variables[@]}"; do
  if [[ -z "${!variable:-}" ]]; then
    echo "set ${variable}" >&2
    exit 2
  fi
done

export ACT_LEWM_CODE_REVISION="${ACT_LEWM_CODE_REVISION:-$(git rev-parse HEAD)}"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python -m ruff check --no-cache .
python -m ruff format --check --no-cache .
python -m pytest -q
git diff --check

python -m train.cache_latents --cfg job "task=${task}" >/dev/null
python -m train.train_world_model --cfg job "task=${task}" >/dev/null
python -m train.train_gc_idm --cfg job "task=${task}" >/dev/null
python -m train.train_larc --cfg job "task=${task}" >/dev/null
python -m eval.open_loop_curve --cfg job "task=${task}" >/dev/null

for specification in "25 50 5" "35 70 7" "50 100 10"; do
  read -r goal_offset eval_budget cem_horizon <<<"${specification}"
  for method in cem gc_idm larc; do
    overrides=(
      "task=${task}"
      "method=${method}"
      "protocol.goal_offset=${goal_offset}"
      "protocol.eval_budget=${eval_budget}"
      "cem.horizon=${cem_horizon}"
    )
    python -m eval.closed_loop --cfg job "${overrides[@]}" >/dev/null
  done
done

echo "${task} H50 static checks, tests, and Hydra composition passed."
