# Two-Room temporal-horizon stress-test protocol

## Status

This document freezes the Two-Room production protocol before cache construction
or training. It extends the completed Push-T temporal-horizon experiment without
importing reference-project runtime code.

## Research question

Can the same H50 Fast-LeWM, GC-IDM, and LARC design retain goal-conditioned
navigation success as the future expert state moves from 25 to 35 and 50 raw
steps away, while avoiding released-LeWM CEM search cost?

This is a temporal-offset goal-conditioned benchmark. It is not a fixed-target
full-episode navigation benchmark.

## Fixed experiment matrix

| Evaluation | Goal offset | CEM horizon | Evaluation budget |
|---|---:|---:|---:|
| O25 | 25 raw steps | 5 blocks | 50 raw steps |
| O35 | 35 raw steps | 7 blocks | 70 raw steps |
| O50 | 50 raw steps | 10 blocks | 100 raw steps |

Common settings:

- frameskip: five raw actions per model block;
- raw action dimension: 2; model action-block dimension: 10;
- training and episode-split seed: `3072`;
- evaluation seed: `42`; evaluation tasks: `50`;
- one H50 checkpoint each for Fast-LeWM, GC-IDM, and LARC;
- the same learned checkpoints and paired start rows at all offsets;
- one training seed only; no multi-seed claim;
- all runtime source code is project-owned.

## Dataset audit

Input: external `tworoom.h5`, containing 920,809 frames in 10,000 episodes.

| Field | Value |
|---|---:|
| Episode length minimum / median / maximum | 31 / 101 / 101 |
| Finite-action rows | 910,809 |
| Terminal NaN action rows | 10,000 |
| Raw action shape | `[frames, 2]` |
| Proprio/state shape used for reset | `[frames, 2]` |
| RGB shape | `[frames, 224, 224, 3]` |

The seed-3072 whole-episode 90/10 split yields:

| Offset | Train episodes | Train anchors | Validation episodes | Validation anchors |
|---|---:|---:|---:|---:|
| 25 | 9,000 | 603,678 | 1,000 | 67,131 |
| 35 | 8,991 | 513,695 | 1,000 | 57,131 |
| 50 | 8,827 | 379,531 | 980 | 42,221 |

Action statistics use finite rows from training episodes only. No training or
validation window crosses an episode boundary. The HDF5 compression filter is
provided by the locked `hdf5plugin` dependency.

## Released checkpoint publication

The official Two-Room `weights.pt` is a pure 303-entry tensor state dict. It
strictly loads into the project-owned ReleasedLeWM architecture with no missing
or unexpected keys. Its SHA-256 is
`566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd`.

A portable artifact is published from that tensor state dict using
`weights_only=True`, the shared explicit ReleasedLeWM YAML, strict construction,
and source-hash metadata. The legacy Python-object checkpoint is not loaded.

## Project-owned environment semantics

The production environment implements the audited fixed dataset geometry:

- 224 by 224 white canvas;
- red Gaussian agent, radius 7;
- fixed speed 5 and clipped two-dimensional actions;
- vertical center wall at x=112, thickness 10;
- one door centered at y=49 with half-extent 14;
- border at 14 pixels with agent-radius-aware clamping;
- central-wall collision with agent-radius and door-margin handling;
- success when Euclidean distance to the selected goal position is below 16.

Reset assigns the dataset `proprio` directly. For offset `h`, the goal pixels and
goal position are the same episode row at `t+h`. Rendering and collision rollouts
must match stored HDF5 frames/states before production.

## One H50 model family

Fast-LeWM predicts dense latent prefixes at raw steps 5 through 50. GC-IDM target
offsets cover every integer from 1 through 50. LARC target offsets cover every
five-step endpoint through 50, masks behavior loss after the selected goal, and
uses the matching Fast-LeWM prefix for rollout consistency.

The representation is frozen. Two-Room frames require a new latent cache because
the released Two-Room weights differ from Push-T; no Push-T latent or learned
weight is reused.

## Paired manifest

One version-2 manifest selects 50 unique validation episodes and one deterministic
start per episode, all able to support offset 50. The same rows are used for all
methods and offsets. Goals are `s_(t+25)`, `s_(t+35)`, and `s_(t+50)`.

Manifest rows are not replaced after observing results and are not filtered by
controller performance. Under the frozen seed, geometric start/goal distance is
already below the environment success radius for 4 O25, 1 O35, and 1 O50 rows;
these naturally sampled rows remain part of the benchmark and are disclosed.

## Closed-loop controllers

- CEM: 300 samples, 30 refinements, top 30, variance scale 1, warm start;
- CEM horizons: 5/7/10 blocks; execute one five-action block then replan;
- GC-IDM: predict and execute one raw action, then replan;
- LARC: predict ten blocks, execute one five-action block, then replan;
- terminated rows never enter a controller or consume CEM random samples.

## Metrics

Each method/offset reports count and percentage success, Wilson 95% interval,
paired discordant counts, raw steps, replans, synchronized planning time, and
full closed-loop wall time. Timing per task is amortized batched throughput, not
batch-size-one latency. Fast-LeWM reports all ten held-out prefix MSE values and
the persistence baseline.

## Production gate

Training may start only after all of the following pass:

1. generic latent/evaluation refactor preserves every Push-T test;
2. portable Two-Room tensor artifact strictly loads;
3. episode layout, NaN actions, split, anchors, shapes, and balanced goals pass;
4. project Two-Room reset pixels and collision rollouts match HDF5 goldens;
5. fresh-process runtime imports succeed with reference packages blocked;
6. Fast-LeWM, GC-IDM, and LARC forward/backward smokes pass;
7. the paired manifest passes unique validation row and boundary checks;
8. reduced-cost CEM/GC-IDM/LARC closed-loop smokes pass at horizon 10;
9. Ruff, formatting, full tests, dependency check, shell syntax, and every Hydra
   composition pass;
10. normal YAML batch/loader parameters are reviewed on the current GPU.

No GPU-memory minimum or hidden gate is introduced.

## Artifact and cleanup policy

Large artifacts stay under the data-disk cache/run roots. During training,
Lightning recovery state exists only until a successful fit writes the best tensor
artifact. After the final report passes, only the portable released artifact,
frame latents, three best learned weights, configs/metadata, manifest, per-task
JSON, and open-loop artifacts remain. Source object weights, periodic weights,
recovery state, smoke outputs, and redundant archives are deleted.
