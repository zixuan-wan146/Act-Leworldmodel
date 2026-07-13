#!/usr/bin/env bash
set -euo pipefail

python -m pytest -q
python -m train.cache_latents --cfg job >/dev/null
python -m train.train_world_model --cfg job >/dev/null
python -m train.train_gc_idm --cfg job >/dev/null
python -m train.train_larc --cfg job >/dev/null
python -m eval.closed_loop --cfg job >/dev/null

echo "Static configuration and unit-test smoke checks passed."
