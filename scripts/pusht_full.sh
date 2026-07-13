#!/usr/bin/env bash
set -euo pipefail

: "${PUSHT_DATASET_PATH:?set PUSHT_DATASET_PATH}"
: "${PUSHT_LEWM_WEIGHTS:?set PUSHT_LEWM_WEIGHTS}"
: "${ACT_LEWM_CACHE_ROOT:?set ACT_LEWM_CACHE_ROOT}"
: "${ACT_LEWM_RUN_ROOT:?set ACT_LEWM_RUN_ROOT}"

if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  echo "refusing to run a production experiment from a dirty worktree" >&2
  exit 1
fi
export ACT_LEWM_CODE_REVISION="${ACT_LEWM_CODE_REVISION:-$(git rev-parse HEAD)}"

python -m train.cache_latents
python -m train.train_world_model
python -m eval.open_loop_curve
python -m train.train_gc_idm
python -m train.train_larc
python -m eval.closed_loop method=cem
python -m eval.closed_loop method=gc_idm
python -m eval.closed_loop method=larc
python -m eval.summarize "${ACT_LEWM_RUN_ROOT}/pusht/eval" results --seed 42
