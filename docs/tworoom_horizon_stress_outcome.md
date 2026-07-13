# Two-Room temporal-horizon stress-test outcome

## Outcome

Production cache construction, training, open-loop validation, and all nine
paired closed-loop evaluations completed successfully. The frozen production
revision is `b33bef913b0d8f237c1c4306660e5755d406c76f`.

This is a temporal-offset goal-conditioned Two-Room experiment. Each task starts
from expert state `s_t` and targets `s_(t+offset)`; it is not a fixed-target
full-episode navigation benchmark.

## Protocol

- training/data split seed: `3072`;
- evaluation seed: `42`;
- 50 unique validation episode/start pairs;
- the same starts for every offset and method;
- offsets/budgets: `25/50`, `35/70`, and `50/100` raw steps;
- five raw actions per model action block;
- CEM and LARC replan after one block; GC-IDM replans every raw action;
- one training seed only; no seed-variance claim.

The paired manifest SHA-256 is
`32511ee2e85bb33275478e30aa44cf8d7ab2d7479f14a3dd0735ee55756e7cc4`.

## Training

| Component | Epochs | Batch size | Approx. optimizer updates | Precision |
|---|---:|---:|---:|---|
| Fast-LeWM-H50 | 10 | 15,000 | 250 | bf16 mixed |
| GC-IDM-H50 | 100 | 350,000 | 100 | bf16 mixed |
| LARC-H50 | 50 | 27,000 | 700 | bf16 mixed |

GC-IDM reached a best validation loss of `0.88276511`. LARC reached
`1.79862714`; the objectives differ, so these values are not directly
comparable. LARC used the explicit production override
`loader.batch_size=27000`, which is recorded in its metadata.

Observed compute-process memory was about 21,870 MiB for Fast-LeWM, 11,848 MiB
for GC-IDM, and 22,738 MiB for LARC on the 24,564 MiB GPU. These are operational
batch choices, not benchmark variables or source-level memory gates.

The 338 MB cache contains latents for the audited 920,809-frame dataset. The
frame-array SHA-256 is
`01abe90b7179c7946c95b4539433eb52b6a98fd76f73b858d9b480a1f9306406`.

Final tensor-weight SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| Portable released LeWM | `8388bdd66894e0ef8075d85d951cff7251f8e56e6f37c9cf7ab515f8236aa762` |
| Fast-LeWM | `999c236f4ed776be2405c1243f49867daa4ee89a259031df29ac1551e140387e` |
| GC-IDM | `67c2ca04860025ac3cee446d456cab7df882845e9317a1d4d5dd43eecef9c2d1` |
| LARC | `4fe1f3a508756c16f83fe8b1133f8c297dd058bbcbdc8bf1d50550da68200da3` |

## Closed-loop results

| Offset | CEM | GC-IDM | LARC | LARC - CEM | LARC - GC-IDM |
|---:|---:|---:|---:|---:|---:|
| 25 | 41/50 (82%) | 49/50 (98%) | 50/50 (100%) | +18 pp | +2 pp |
| 35 | 26/50 (52%) | 47/50 (94%) | 48/50 (96%) | +44 pp | +2 pp |
| 50 | 19/50 (38%) | 42/50 (84%) | 48/50 (96%) | +58 pp | +12 pp |

The LARC-versus-GC-IDM discordant counts are 1/0 at O25, 2/1 at O35, and 7/1
at O50. The first two offsets differ by only one net task. O50 is the meaningful
stress point in this run, but the single seed and 50-task manifest still do not
support a general architecture-level claim.

Wilson intervals and every paired comparison are in the
[formal result report](../results/RESULTS_tworoom_horizon.md).

## Runtime

| Offset | CEM s/task | GC-IDM s/task | LARC s/task | CEM/LARC ratio |
|---:|---:|---:|---:|---:|
| 25 | 5.4254 | 0.0489 | 0.0337 | 161x |
| 35 | 13.7262 | 0.0764 | 0.0573 | 240x |
| 50 | 27.8960 | 0.1154 | 0.0884 | 316x |

These are synchronized, amortized end-to-end seconds per task. Learned
controllers batch active rows, so the numbers measure batched throughput rather
than batch-size-one latency. CEM's cost grows sharply with horizon and with rows
that remain active for later replans.

## Fast-LeWM open-loop result

| Raw steps | Fast-LeWM MSE | Persistence MSE |
|---:|---:|---:|
| 5 | 0.772871 | 1.339171 |
| 25 | 0.916956 | 1.951431 |
| 35 | 0.921186 | 1.979735 |
| 50 | 0.944226 | 1.993175 |

Fast-LeWM remains below persistence at all ten measured prefixes. Its error rises
with horizon, while the learned closed-loop controllers retain high success by
replanning from fresh observations.

## Interpretation and limitations

- The supported conclusion is paired performance of the complete frozen recipes
  on this manifest: both learned controllers outperform CEM, and LARC retains
  96% success through O50.
- The experiment does not isolate architecture or loss contributions. The safe
  batches yield about 100 GC-IDM optimizer updates and 700 LARC updates.
- Only one training seed and one 50-task evaluation manifest were used.
- Two-Room has fixed geometry and deterministic dynamics; high success does not
  establish robustness to new layouts, observation noise, or dynamics shifts.
- The released representation has upstream pretraining provenance. The learned
  policies do not train on the held-out evaluation episodes, but the released
  model cannot be claimed unaware of all source data.
- Any later optimizer-budget or architecture ablation must use separate
  checkpoints, revisions, and reports. It is not part of this production result.

## Self-contained runtime and cleanup

Project runtime imports only project-owned data, model, environment, planner,
policy, and reporting code. Tests block imports from the reference projects and
their upstream module namespace.

Only the portable released artifact, frame latents, three best learned tensor
weights, configs/metadata, paired manifest, nine per-method JSON files, and
open-loop artifacts remain in the production roots. Completed Lightning recovery
state and temporary memory-smoke artifacts were removed. The final experiment
directory is 76 MB.

## Status

The frozen Two-Room production protocol is complete. No Transformer experiment
or other architecture ablation is included in this outcome.
