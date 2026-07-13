# Push-T temporal-horizon stress-test outcome

## Outcome

Production training and all nine closed-loop evaluations completed successfully.
The experiment used one H50 checkpoint per learned component and evaluated the
same checkpoints at goal offsets 25, 35, and 50. The frozen production revision
is `01eccc3ccafba8c1eae140d504ad8abe523484f0`.

This is a goal-conditioned temporal-offset Push-T experiment: each task starts
from expert state `s_t` and targets `s_(t+offset)`. It is not the classic
fixed-target, full-episode Push-T benchmark.

## Protocol

- training/data split seed: `3072`;
- evaluation seed: `42`;
- 50 unique validation episode/start pairs;
- the same starts for every offset and method;
- offsets/budgets: `25/50`, `35/70`, and `50/100` raw steps;
- five raw actions per model action block;
- CEM and LARC replan after one block; GC-IDM replans every raw action;
- one training seed only; this experiment does not estimate seed variance.

The paired manifest SHA-256 is
`aa8c8845d82374807db8917f3876cc5db3722e3ef7b078841fd300abed5ae50d`.

## Training

| Component | Epochs | Batch size | Precision | Best validation loss |
|---|---:|---:|---|---:|
| Fast-LeWM-H50 | 10 | 15,000 | bf16 mixed | 0.216866 |
| GC-IDM-H50 | 100 | 700,000 | bf16 mixed | 0.381006 |
| LARC-H50 | 50 | 26,000 | bf16 mixed | 0.450797 |

Fast-LeWM reached about 22.5 GB allocated GPU memory on the current 24 GB device.
That utilization came from ordinary YAML batch/loader settings; no source-level
memory target, minimum, or runtime gate exists.

The existing 2,336,736 frame latents were reused without image re-encoding. Their
SHA-256 remained
`0d055b64c7ca486e107d3325a851bd28d5e7307e81d4e6b58763d0435b537781`.

Final tensor-weight SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| Fast-LeWM | `a991a52dd061841ee6f1dd5a29cda616f88c1c07f77c1e3df14411438231d685` |
| GC-IDM | `4762d615087d705d02bf097a9e5be2abe2997df4ac789e6fa434b60e77bbf92f` |
| LARC | `580aaebeed988b006699caa798d42737fc48aebaaff4b494a561935118ec1972` |

## Closed-loop results

| Offset | CEM | GC-IDM | LARC | LARC - CEM | LARC - GC-IDM |
|---:|---:|---:|---:|---:|---:|
| 25 | 9/50 (18%) | 22/50 (44%) | 40/50 (80%) | +62 pp | +36 pp |
| 35 | 7/50 (14%) | 14/50 (28%) | 41/50 (82%) | +68 pp | +54 pp |
| 50 | 4/50 (8%) | 6/50 (12%) | 25/50 (50%) | +42 pp | +38 pp |

LARC leads both baselines at every tested offset on the paired tasks. Its 82%
at offset 35 versus 80% at offset 25 is a one-task difference and is not evidence
that the longer offset is easier; the Wilson intervals overlap substantially.
Offset 50 is the clearest stress point: LARC falls to 50%, but CEM and GC-IDM
fall further to 8% and 12%.

Wilson intervals and per-task discordant counts are in the
[formal result report](../results/RESULTS_pusht_horizon.md).

## Runtime

| Offset | CEM s/task | GC-IDM s/task | LARC s/task | CEM/LARC throughput ratio |
|---:|---:|---:|---:|---:|
| 25 | 9.4019 | 0.1347 | 0.0964 | 97.5x |
| 35 | 17.6393 | 0.2139 | 0.1283 | 137.5x |
| 50 | 31.9673 | 0.3072 | 0.2250 | 142.1x |

These are amortized wall-clock seconds per task (`total batch wall time / 50`).
Learned controllers process active rows together, so the values measure batched
throughput rather than batch-size-one latency. CUDA synchronization is included.

## Fast-LeWM open-loop result

Prefix latent MSE rises smoothly with horizon:

| Raw steps | Fast-LeWM MSE | Persistence MSE |
|---:|---:|---:|
| 5 | 0.060238 | 0.165494 |
| 25 | 0.198838 | 1.372736 |
| 35 | 0.261813 | 1.677159 |
| 50 | 0.368971 | 1.843363 |

Fast-LeWM remains substantially below persistence at every one of the ten
measured prefixes. The increasing MSE still shows accumulated long-horizon
uncertainty, consistent with the harder O50 closed-loop result.

## Interpretation and limitations

- The main supported conclusion is paired performance on this fixed manifest:
  LARC is more successful and far cheaper than the evaluated CEM and GC-IDM.
- Only one training seed and one 50-task evaluation manifest were used. The
  result is a production run, not a multi-seed statistical estimate.
- The released representation and released CEM dynamics retain their upstream
  clip-level split provenance; the new learned policies never train on the 50
  held-out tasks, but those tasks cannot be claimed unseen by the released model.
- The old 82%/94%/92% O25 files used a different manifest and a five-block
  commitment schedule. They are not comparable and were removed after this run
  passed verification.
- The unusually low CEM result is not a loader or state-restoration failure:
  its cost, state reset, action scaling, goal indexing, and termination semantics
  were audited against the locked reference behavior. It belongs to this stricter
  paired H50-compatible, one-block-replanning protocol.

## Self-contained runtime and cleanup

Project runtime imports only project-owned dataset, model, environment, planner,
policy, and reporting code. The existing virtual environment supplies ordinary
dependencies such as Python, PyTorch, Gymnasium, and Pymunk; reference project
packages are not imported.

Only tensor-only portable/best weights remain. Periodic epoch exports, `last.pt`,
completed-run Lightning checkpoints, old H25 artifacts, the legacy Push-T object
checkpoint, and the extracted raw Push-T archive were deleted. The finalized H50
run directory is 77 MB.

## Next phase

The separate Two-Room production protocol has since completed; its outcome is in
`docs/tworoom_horizon_stress_outcome.md`. This document remains the frozen Push-T
record. Any later architecture or optimizer-budget ablation must use separate
checkpoints, revisions, and reports rather than changing this result in place.
