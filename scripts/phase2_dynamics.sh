#!/usr/bin/env bash
set -euo pipefail

python -m train.cache_latents
python -m train.train_world_model
python -m eval.open_loop_curve
