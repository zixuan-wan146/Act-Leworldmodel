#!/usr/bin/env bash
set -euo pipefail

python -m train.train_gc_idm
python -m train.train_larc
python -m eval.closed_loop method=cem
python -m eval.closed_loop method=gc_idm
python -m eval.closed_loop method=larc
