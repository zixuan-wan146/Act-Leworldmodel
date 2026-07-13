#!/usr/bin/env bash
set -euo pipefail

task="${1:-pusht}"

if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "refusing to run a production experiment from a dirty worktree" >&2
  exit 1
fi
export ACT_LEWM_CODE_REVISION="$(git rev-parse HEAD)"
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

scripts/preflight.sh "${task}"
python -m train.cache_latents "task=${task}"
python -m train.train_world_model "task=${task}"
python -m eval.open_loop_curve "task=${task}"
python -m train.train_gc_idm "task=${task}"
python -m train.train_larc "task=${task}"

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
    python -m eval.closed_loop "${overrides[@]}"
  done
done

python -m eval.summarize "${ACT_LEWM_RUN_ROOT}/${task}/horizon_h50/eval" results \
  --seed 42 \
  --open-loop-dir "${ACT_LEWM_RUN_ROOT}/${task}/horizon_h50/open_loop"
