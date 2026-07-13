# Act-LeWorldModel

Act-LeWorldModel compares three goal-conditioned controllers under paired
temporal-horizon stress tests on Push-T and Two-Room:

- **CEM** uses the released LeWM model with online cross-entropy planning.
- **GC-IDM** predicts one raw action and replans every environment step.
- **LARC** predicts action blocks with behavior cloning plus frozen-world-model
  rollout consistency, then replans after one five-action block.

Each task uses one H50 Fast-LeWM, one H50 GC-IDM, and one H50 LARC checkpoint
trained with seed `3072`. The same task-specific learned checkpoints are reused
at goal offsets `25`, `35`, and `50` on 50 paired held-out tasks with evaluation
seed `42`.

The Push-T production run is complete:

| Goal offset | CEM | GC-IDM | LARC |
|---:|---:|---:|---:|
| 25 | 18% | 44% | 80% |
| 35 | 14% | 28% | 82% |
| 50 | 8% | 12% | 50% |

See the [Push-T result report](results/RESULTS_pusht_horizon.md),
[outcome](docs/horizon_stress_outcome.md), and
[protocol](docs/pusht_horizon_stress_test.md).

The Two-Room production run is also complete:

| Goal offset | CEM | GC-IDM | LARC |
|---:|---:|---:|---:|
| 25 | 82% | 98% | 100% |
| 35 | 52% | 94% | 96% |
| 50 | 38% | 84% | 96% |

See the [Two-Room result report](results/RESULTS_tworoom_horizon.md),
[outcome](docs/tworoom_horizon_stress_outcome.md), and
[protocol](docs/tworoom_horizon_stress_test.md). The implementation shares the
task-configured pipeline while using its own released representation, cache,
paired manifest, and project-owned environment.

Both experiments are temporal-offset goal-conditioned benchmarks, not classic
fixed-target full-episode benchmarks. The reports contain paired differences,
Wilson intervals, timing, open-loop error, provenance, and limitations.

## Runtime boundary

Project runtime code is self-contained. It does not import the reference
submodule, `stable-worldmodel`, `stable-pretraining`, `jepa`, or the
upstream `module` package. The reference checkout is read-only material for
implementation comparison. Its virtual environment may be used as a Python
environment; that does not make its source code a runtime dependency. A fresh
subprocess test explicitly blocks all reference-package imports while loading
every project entry point.

When reusing an existing interpreter, only its base dependencies are reused.
The `stable-worldmodel` and `stable-pretraining` distributions must not be
installed. Training also rejects every external Lightning callback factory, so
an environment plugin cannot silently inject reference code into a project
run.

The released LeWM checkpoint must use the portable tensor-only format described
in [docs/checkpoint_format.md](docs/checkpoint_format.md). Models are rebuilt
from project YAML and weights are loaded with `weights_only=True`.

## Environment and external storage

Set all large-artifact roots on the data disk:

```bash
export PUSHT_DATASET_PATH=/data/datasets/pusht_expert_train.h5
export PUSHT_LEWM_WEIGHTS=/data/checkpoints/pusht/released_lewm_state.pt
export TWOROOM_DATASET_PATH=/data/datasets/tworoom.h5
export TWOROOM_LEWM_WEIGHTS=/data/checkpoints/tworoom/released_lewm_state.pt
export ACT_LEWM_CACHE_ROOT=/data/act-lewm-cache
export ACT_LEWM_RUN_ROOT=/data/act-lewm-runs
export HF_HOME=/data/huggingface-cache
export UV_CACHE_DIR=/data/uv-cache
```

Install the exact locked environment:

```bash
uv sync --frozen --extra test
```

No project config contains a personal absolute path. Datasets, caches,
checkpoints, Lightning state, manifests, videos, and raw JSON metrics stay
outside Git.

## H50 data and action protocol

Every dataset frame is encoded once by the frozen released-LeWM
encoder/projector. The frame cache stores only immutable dataset, split,
representation, normalization, and latent lineage. Horizon settings are
training views and are never persisted in the frame cache.

Both active tasks have a two-dimensional raw action. With `frameskip=5`, five
normalized raw actions form one 10-dimensional model action block:

```text
raw actions       [B, 50, 2]
  -> normalize with train-episode statistics
  -> pack consecutive groups of five
model action      [B, 10, 10]
dense targets     at raw offsets 5, 10, ..., 50
```

GC-IDM receives strictly balanced deterministic targets at every raw offset
from 1 through 50. LARC receives strictly balanced deterministic targets at
block offsets 5 through 50; behavior loss is masked after the selected target
offset.

## Paired evaluation

One version-2 manifest chooses 50 unique validation episodes and one start step
per episode, all eligible for the maximum offset 50. Every method uses the same
starts. Goals are taken from the same expert trajectory at `t+25`, `t+35`,
and `t+50`.

| Goal offset | CEM blocks | Raw-step budget |
|---:|---:|---:|
| 25 | 5 | 50 |
| 35 | 7 | 70 |
| 50 | 10 | 100 |

CEM and LARC execute five raw actions before replanning. GC-IDM replans every
raw action. Terminated rows never enter a planner. Results record per-task
success, environment steps, planner calls, synchronized planning time, full
closed-loop wall time, the evaluated commit, and SHA-256 records for every
loaded artifact.

## Reproducible workflow

Run all static checks and config composition for the selected task first:

```bash
scripts/preflight.sh pusht
scripts/preflight.sh tworoom
```

After the relevant pre-training gate is complete, run one production workflow
from a clean committed worktree:

```bash
scripts/horizon_full.sh pusht
scripts/horizon_full.sh tworoom
```

Before the first Two-Room cache build, publish the official bare tensor state
dict into the project artifact schema:

```bash
python scripts/publish_released_lewm.py \
  configs/released_lewm/pusht.yaml \
  /data/source/tworoom/weights.pt \
  "$TWOROOM_LEWM_WEIGHTS"
```

The script performs cache validation/reuse, H50 Fast-LeWM training, ten-prefix
open-loop evaluation, H50 GC-IDM and LARC training, all nine closed-loop runs,
and the joint result summary. It refuses a dirty worktree and records
`ACT_LEWM_CODE_REVISION`.

## Resource tuning

`loader.batch_size`, `loader.num_workers`, `loader.prefetch_factor`,
precision, and related trainer values are operational settings in the YAML
files. They are not benchmark variables, and no code enforces a GPU-memory
minimum. The checked-in values are starting points for the current hardware.

Before a production run, adjust these YAML values or pass Hydra overrides and
repeat a one-step smoke. The resolved configuration and selected batch size are
recorded with each trained artifact.

Equivalent individual training commands are:

```bash
python -m train.cache_latents task=tworoom
python -m train.train_world_model task=tworoom
python -m eval.open_loop_curve task=tworoom
python -m train.train_gc_idm task=tworoom
python -m train.train_larc task=tworoom
```

For a closed-loop run, override the selected offset, proportional budget, and
CEM horizon together. For example, O35 CEM is:

```bash
export ACT_LEWM_CODE_REVISION="$(git rev-parse HEAD)"
python -m eval.closed_loop \
  task=pusht \
  method=cem \
  protocol.goal_offset=35 \
  protocol.eval_budget=70 \
  cem.horizon=7
```

## Verification

```bash
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
git diff --check
```

Tests cover tensor-only checkpoints, episode-safe data views, strictly
balanced H50 target coverage, dense-prefix shapes, masked LARC losses,
terminated-row planning, Push-T physics/rendering, exact Two-Room dataset
rendering and collisions, task-config composition, paired manifest integrity,
timing fields, artifact hashes, cross-offset result consistency, and
reference-package import blocking.

The final versioned reports are:

- `results/RESULTS_pusht_horizon.md`, interpreted in
  `docs/horizon_stress_outcome.md`;
- `results/RESULTS_tworoom_horizon.md`, interpreted in
  `docs/tworoom_horizon_stress_outcome.md`.

Minimal raw artifacts remain under each task's
`$ACT_LEWM_RUN_ROOT/<task>/horizon_h50/` directory.

## Current state

The frozen Push-T and Two-Room production protocols are complete. The Two-Room
completion record is in [docs/tworoom_progress.md](docs/tworoom_progress.md).
Architecture and optimizer-budget ablations are outside these frozen production
runs and must use separate checkpoints, revisions, and reports.
