# Project Structure

```text
configs/                    Experiment YAML files and all changeable parameters.
data/dataset.py             Offline trajectory loading and skip-5 segment sampling.
models/encoder/             Reused LeWM ViT encoder and SIGReg components.
models/dynamics/            One-step LeWM and parallel prefix dynamics models.
models/strategies/          CEM, IDM, and ARC strategy implementations.
losses/                     Rollout-consistency and behavior-cloning objectives.
planning/frozen_wm.py       Frozen differentiable world-model adapter.
train/                      Dynamics and policy training entry points.
eval/                       Closed-loop, open-loop, and protocol evaluation.
scripts/                    Reproducible phase-level experiment commands.
tests/                      Unit and integration tests.
results/                    Versioned result reports and figures.
third_party/                Pinned external source dependencies.
```

## Boundaries

- `data/` contains loading code, not datasets.
- `configs/` owns paths, model choices, hyperparameters, and run modes.
- `models/`, `losses/`, and `planning/` contain reusable implementation.
- `train/` and `eval/` orchestrate workflows without duplicating core logic.
- `scripts/` remain thin and call Python implementations.
- `results/` stores reviewable reports and figures, not raw logs or checkpoints.
