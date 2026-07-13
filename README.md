# Act-LeWorldModel

Act-LeWorldModel compares three goal-conditioned Push-T controllers under one
fixed evaluation protocol:

- **CEM**: the released LeWM model with online cross-entropy planning.
- **GC-IDM**: a one-action amortized inverse-dynamics policy, replanned every
  environment step.
- **LARC**: a learned action-block policy trained with behavior cloning and a
  differentiable frozen-world-model rollout-consistency loss.

The visual encoder and projector are reused from the released LeWM checkpoint.
Every dataset frame is encoded once, then Fast-LeWM dynamics and both learned
policies train from the same episode-separated latent cache.

## Environment and storage

The project runtime is self-contained and does not import the reference
submodule, `stable-worldmodel`, or `stable-pretraining`. Datasets, tensor-only
weights, caches, logs, videos, and raw metrics belong outside the repository.
Configure their roots before running an experiment:

```bash
export PUSHT_DATASET_PATH=/data/datasets/pusht_expert_train.h5
export PUSHT_LEWM_WEIGHTS=/data/checkpoints/pusht/released_lewm_state.pt
export ACT_LEWM_CACHE_ROOT=/data/act-lewm-cache
export ACT_LEWM_RUN_ROOT=/data/act-lewm-runs
export HF_HOME=/data/huggingface-cache
export UV_CACHE_DIR=/data/uv-cache
```

Install the locked project environment:

```bash
uv sync --frozen --extra test
```

The first four variables are required; `HF_HOME` and `UV_CACHE_DIR` keep
library/build caches off the system disk. No project config contains a personal absolute path.

`PUSHT_LEWM_WEIGHTS` must point to the project portable artifact described in
[`docs/checkpoint_format.md`](docs/checkpoint_format.md). Legacy Python-object
checkpoints are deliberately rejected: models are reconstructed from project
YAML and only tensor state dictionaries are loaded with `weights_only=True`.

## Push-T action protocol

Push-T exposes a two-dimensional action at each environment step. Fast-LeWM
uses `frameskip=5`, so one dynamics action is a 10-dimensional block formed by
normalizing and concatenating five raw actions.

```text
raw environment actions [B, 25, 2]
    -> normalize with train-episode statistics
    -> pack five consecutive actions
world-model blocks      [B,  5, 10]
```

LARC predicts five such blocks, covering 25 environment steps, and executes
five raw actions before replanning. GC-IDM predicts one normalized raw action
and replans immediately. The conversion is centralized in
`ActionBlockTransform`; block-shaped actions are never sent directly to the
environment.

## Reproducible workflow

The complete workflow is:

```bash
python -m train.cache_latents
python -m train.train_world_model
python -m eval.open_loop_curve
python -m train.train_gc_idm
python -m train.train_larc
export ACT_LEWM_CODE_REVISION="$(git rev-parse HEAD)"
python -m eval.closed_loop method=cem
python -m eval.closed_loop method=gc_idm
python -m eval.closed_loop method=larc
python -m eval.summarize "$ACT_LEWM_RUN_ROOT/pusht/eval" results --seed 42
```

The same sequence is available as `scripts/pusht_full.sh`. Every stage is
configuration-driven and writes resumable checkpoints or deterministic cache
metadata under the external roots.
The full script refuses a dirty Git worktree and derives
ACT_LEWM_CODE_REVISION automatically. Direct closed-loop commands must set it
to the full evaluated commit. Result JSON files record that commit plus the
SHA-256 of every loaded config, weight, and metadata artifact.


New dynamics and policy training use seed `3072`. Whole episodes are split
90/10 before their normalization or clip creation, so overlapping trajectory
windows cannot cross the new train/validation boundary. The reused released
LeWM representation and CEM checkpoint retain their upstream clip-level split
provenance; see `docs/pusht_protocol.md`. Closed-loop evaluation uses seed
`42`, 50 validation episodes, goal offset 25, and one shared manifest stored
with the external raw metrics.

## Tests

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
git diff --check
```

Tests cover action packing, episode separation, cache indexing, causal masks,
AdaLN-zero initialization, frozen parameter behavior, action gradients, and
variable-horizon LARC losses. `scripts/phase0_smoke.sh` also validates that all
Hydra configurations compose.
Additional regression tests lock the nonzero-angle Push-T physics/rendering
trajectory, terminated-row CEM random stream, checkpoint formats, evaluation
artifact hashes, and cross-result provenance.

## Backbone API

```python
latents = model.encode_observations(pixels)
future_latents = model.predict_latents(latents, action_blocks)
```

`predict_latents` accepts arbitrary leading batch dimensions and returns one
latent prediction for every causal action prefix. It is independent of CEM,
goals, and controller state.

Project design documents live under `docs/`. Versionable result summaries and
figures live under `results/`; large artifacts remain outside the repository.

The completed Push-T run, quantitative outcomes, interpretation, and
limitations are documented in
[`docs/experiment_outcome.md`](docs/experiment_outcome.md).
