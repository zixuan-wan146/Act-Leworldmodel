# Push-T temporal-horizon stress-test protocol

## Status

This document defines the sole production Push-T protocol in the repository.
It replaces the earlier 25-step-only workflow and freezes the experiment before
production training.

The experiment measures how released-LeWM CEM, GC-IDM, and LARC change as the
expert-trajectory goal moves from 25 to 35 and 50 raw environment steps into
the future. It remains a LeWorldModel-style goal-conditioned Push-T benchmark;
it is not the classic fixed-target, full-episode Push-T protocol.

## Research question

Can one controller trained for a maximum 50-step horizon retain success as the
goal distance grows, while avoiding the autoregressive rollout and online
search cost paid by CEM?

The primary independent variable is the temporal goal offset. The initial
states, learned checkpoints, visual representation, action protocol, success
criterion, and evaluation sampling remain fixed across offsets.

This stage uses one training seed and one paired evaluation manifest. It does
not estimate training-seed variance, and the final report must not claim that
it does.

## Fixed experiment matrix

| Evaluation | Goal offset | CEM planning horizon | Evaluation budget |
|---|---:|---:|---:|
| O25 | 25 raw steps | 5 action blocks | 50 raw steps |
| O35 | 35 raw steps | 7 action blocks | 70 raw steps |
| O50 | 50 raw steps | 10 action blocks | 100 raw steps |

Common settings:

- frameskip: 5 raw environment actions per model action block;
- raw action dimension: 2;
- model action-block dimension: 10;
- training and data-split seed: 3072;
- evaluation seed: 42;
- evaluation tasks: 50;
- visual representation: the same frozen released-LeWM encoder/projector;
- all runtime code: project-owned; reference packages are not imported;
- success criterion: the existing project Push-T goal-state criterion, unchanged.

## Data split and released-checkpoint provenance

The frame cache uses seed 3072 to split complete episodes 90/10 before any new
training clip is constructed. Action statistics for Fast-LeWM, GC-IDM, and LARC
come only from finite rows in the new training episodes. No new training window
crosses the episode split.

The frozen released-LeWM representation and the CEM baseline retain separate
upstream provenance. Their source training procedure performed a 90/10 random
split over already constructed clips and fitted normalization before that
split. Some upstream clips may therefore originate from episodes assigned to
this project's validation set. The 50 evaluation tasks are held out from the
new Fast-LeWM and learned-policy training, but they cannot be claimed unseen by
the released representation or released CEM dynamics.

CEM uses full-source action statistics to match its released checkpoint.
Fast-LeWM, GC-IDM, and LARC use the new train-episode statistics stored in the
frame-cache lineage. Sharing one scaler across these differently trained
artifacts would be incorrect.

The proportional budgets preserve the released LeWM ratio of two allowed
environment steps per expert goal-offset step. A fixed-budget stress test is
out of scope for this stage.

## One H50 model family

The primary experiment trains one H50 artifact for each learned component and
evaluates the same artifacts at all three offsets. Separate H25 and H35 models
would confound goal distance with model capacity, training data, and checkpoint
selection.

Earlier H25-only artifacts are not valid inputs to this experiment because the
training horizon, paired manifest, and CEM replanning schedule differ. They are
removed only after verified H50 replacements exist, so cleanup cannot destroy
the sole recoverable checkpoint mid-migration.

### Fast-LeWM-H50

- maximum horizon: 10 action blocks / 50 raw steps;
- dense latent targets at raw offsets
  `{5, 10, 15, 20, 25, 30, 35, 40, 45, 50}`;
- released-LeWM encoder and projector frozen;
- existing per-frame latents reused without image re-encoding;
- validation reports MSE separately for every prefix and as a fixed
  equal-weight mean across all ten prefixes.

### GC-IDM-H50

- deterministic training targets cover every raw goal offset from 1 through 50;
- output: one normalized two-dimensional raw action;
- inference replans after every raw environment step;
- the same checkpoint is evaluated at O25, O35, and O50.

### LARC-H50

- output: ten normalized action blocks / 50 raw actions;
- training goal offsets:
  `{5, 10, 15, 20, 25, 30, 35, 40, 45, 50}`;
- behavior loss includes only blocks enabled by the sampled goal mask;
- rollout consistency selects the Fast-LeWM prefix corresponding to the
  sampled remaining horizon;
- inference executes one action block (five raw actions) before replanning;
- the same checkpoint is evaluated at O25, O35, and O50.

No BC-only or loss-weight ablation is part of this stage.

## Resource configuration

Training resource use is controlled only by ordinary YAML or Hydra parameters,
including batch size, worker count, prefetching, precision, and trainer values.
These are operational settings rather than benchmark variables, and no runtime
check imposes a GPU-memory minimum.

The checked-in values are starting points selected by one-step smokes on the
current GPU. Before production they may be adjusted in configuration, after
which the forward/backward smoke is repeated. The resolved configuration and
selected batch size are recorded with every trained artifact.

## Frame cache and horizon views

The immutable frame-latent artifact is a representation cache: one released
LeWM latent per dataset frame plus dataset, checkpoint, split, and action
normalization lineage. A training horizon is not part of the encoded frame
identity.

Implementation must separate:

1. immutable frame-cache lineage; and
2. a configurable training view containing frameskip, maximum block horizon,
   eligible anchors, goal sampling, and action masks.

Changing the view from 25 to 50 steps must not rewrite or re-encode the
2,336,736 cached frame latents. Cache reuse is accepted only if the latent file
hash and source lineage remain unchanged.

For the 50-step view, the existing episode split contains:

| Split | Eligible episodes | Eligible anchors |
|---|---:|---:|
| Train | 16,721 | 1,261,619 |
| Validation | 1,863 | 140,968 |

## Paired evaluation manifest

One base manifest is generated from validation episodes long enough to support
the maximum offset of 50.

It contains:

- exactly 50 unique validation episode IDs;
- one deterministic start step per episode;
- allowed goal offsets `[25, 35, 50]`;
- maximum goal offset 50;
- dataset and source-checkpoint identity;
- split identity and validation-ID digest;
- evaluation seed and manifest version.

For a manifest row starting at `t`, O25, O35, and O50 use the same physical
initial state and goals `s_(t+25)`, `s_(t+35)`, and `s_(t+50)`
respectively. The three methods use the identical rows within each offset.

Production evaluation must not tune or replace manifest rows after seeing any
controller result. Small runtime checks use a separate smoke manifest.

## Closed-loop controller settings

### Released LeWM + CEM

- released autoregressive LeWM checkpoint, unchanged across offsets;
- horizons 5, 7, and 10 blocks for O25, O35, and O50;
- 300 candidates, 30 refinement iterations, top 30, variance scale 1.0;
- execute one action block and replan (`receding_horizon=1`);
- warm-start the unexecuted plan tail;
- terminated rows never enter the planner or consume random samples.

This is a matched-feedback CEM extension, not the completed H25 run's
full-five-block commitment. The old 82% CEM result is therefore not reused.

### GC-IDM-H50

GC-IDM encodes the fixed goal once, predicts one raw action, executes it, and
replans from the next observation.

### LARC-H50

LARC encodes the fixed goal once, predicts ten action blocks, executes only the
first block, and replans from the next observation.

## Metrics and timing

Every offset and method reports:

- successes as both a count and percentage;
- per-task success booleans in manifest order;
- Wilson 95% confidence interval;
- paired success differences between methods;
- environment steps executed;
- planner calls per task;
- synchronized planning time per call;
- total closed-loop wall time;
- amortized wall time per task (`total / 50`).

One task changes the success rate by two percentage points. Differences of one
or two tasks must not be described as statistically established improvements.

End-to-end timing starts after checkpoint/data loading and evaluation-pool
construction. It includes task-state initialization, first-use controller
kernels, goal encoding, controller inference or CEM optimization, physics,
rendering, and closed-loop bookkeeping. CUDA is synchronized at the total and
per-replanning timing boundaries.

Planning time covers observation preprocessing and the controller call. Because
learned controllers plan active rows as one batch, per-row planning time is an
amortized throughput measure, not batch-size-one latency. The report discloses
controller batch behavior and planner-call counts; no speedup is inferred from
unsynchronized wall time.

Fast-LeWM additionally reports prefix-wise open-loop latent MSE at every
5-step endpoint through 50, using the fixed validation subset.

## Production-training gate

Formal training must not start until all of the following are recorded as
passing in `docs/horizon_stress_progress.md`:

1. review of the complete implementation diff;
2. clean worktree provenance and `git diff --check`;
3. full project tests with reference-package imports explicitly blocked;
4. Ruff lint and format checks;
5. locked-environment and Hydra config composition checks;
6. unchanged frame-latent hash and verified reuse without encoder execution;
7. H50 dataset checks for episode boundaries, indices, shapes, masks, and
   uniform deterministic horizon coverage;
8. strict tensor-only construction and loading of every artifact;
9. forward/backward smoke tests for Fast-LeWM-H50, GC-IDM-H50, and LARC-H50;
10. manifest invariants and paired O25/O35/O50 row checks;
11. short closed-loop smoke runs for all methods, including horizon-10 CEM;
12. GPU-memory and batch-size review with any adjustment made only through
    configuration.

Passing narrow unit tests is not enough to approve production training.

## Artifact layout

Large artifacts remain outside Git:

```text
$ACT_LEWM_RUN_ROOT/pusht/horizon_h50/
  world_model/
  gc_idm/
  larc/
  open_loop/
  eval/
```

Versioned records are:

- this frozen protocol;
- `docs/horizon_stress_progress.md`, updated at every gate and run stage;
- `docs/horizon_stress_outcome.md`, written after evaluation;
- `results/RESULTS_pusht_horizon.md` and generated figures.

Raw checkpoints, logs, manifests, and JSON results are not committed.
