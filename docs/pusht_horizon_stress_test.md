# Push-T temporal-horizon stress-test protocol

## Status

This document freezes the protocol before implementation or production
training. It extends, rather than replaces, the completed 25-step experiment
documented in `docs/pusht_protocol.md`.

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

The proportional budgets preserve the released LeWM ratio of two allowed
environment steps per expert goal-offset step. A fixed-budget stress test is
out of scope for this stage.

## One H50 model family

The primary experiment trains one H50 artifact for each learned component and
evaluates the same artifacts at all three offsets. Separate H25 and H35 models
would confound goal distance with model capacity, training data, and checkpoint
selection.

The completed H25 artifacts and results remain available as historical
references. They are not mixed into the new paired result table, because the
new manifest and CEM replanning schedule differ.

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

Timing excludes checkpoint loading, data loading, environment construction,
and first-use CUDA initialization. It includes observation/goal encoding,
controller inference or CEM optimization, and closed-loop execution. CUDA
measurements require warm-up and synchronization at timing boundaries.

The report distinguishes:

- vectorized 50-task throughput; and
- batch-size-one controller latency.

It must disclose the controller batch size and number of replans. A speedup
claim cannot be inferred from unsynchronized total runtime alone.

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
