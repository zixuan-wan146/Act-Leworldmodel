# Push-T horizon-stress progress

## Current state

- Stage: production training, evaluation, report generation, and cleanup complete.
- Production revision: `01eccc3ccafba8c1eae140d504ad8abe523484f0`.
- Formal production training and all nine closed-loop runs: complete.
- Final report: `results/RESULTS_pusht_horizon.md`.
- Active protocol: `docs/pusht_horizon_stress_test.md`.

## Fixed decisions

- Train one H50 Fast-LeWM, one H50 GC-IDM, and one H50 LARC checkpoint.
- Evaluate the same learned checkpoints at goal offsets 25, 35, and 50.
- Use one paired 50-task base manifest shared across offsets and methods.
- Use proportional evaluation budgets 50, 70, and 100.
- CEM and LARC replan every five raw actions; GC-IDM replans every action.
- Use only project-owned runtime source code.
- Keep no duplicate H25/H50 configuration paths or compatibility fields.

## Dataset feasibility

The existing split was inspected without rewriting frame latents:

| Offset | Train episodes | Train anchors | Validation episodes | Validation anchors |
|---|---:|---:|---:|---:|
| 25 | 16,816 | 1,681,924 | 1,869 | 187,687 |
| 35 | 16,816 | 1,513,764 | 1,869 | 168,997 |
| 50 | 16,721 | 1,261,619 | 1,863 | 140,968 |

The H50 view has sufficient data and reuses all 2,336,736 existing per-frame
latents.

## Implementation log

### Protocol freeze

- Initial remote revision: `7e5394e7c76d72097ecbcdf725f1ecaa2f641454`.
- Protocol and stage log were committed before implementation.
- One H50 model family and one paired manifest were selected.

### Frame cache and configuration cleanup

- Frame-cache format version 2 stores representation lineage only.
- `frameskip`, `max_horizon`, and `max_goal_offset` exist only in training
  or evaluation views.
- Legacy cache-reading branches and compatibility aliases were removed.
- The four duplicate `*_h50.yaml` configs were deleted; default configs are
  the sole H50 configs.
- Unused CEM `history_len`, GC-IDM `commit_steps`, and empty result
  `seeds` fields were deleted.

### H50 target construction

- Fast-LeWM produces ten dense prefixes through 50 raw steps.
- Review found that the previous linear hash was not coprime with 10 or 50 and
  therefore exposed LARC to only two goal bins and GC-IDM to only a subset.
- Goal assignment was replaced by deterministic strict balancing. Every
  declared bin is present and counts differ by at most one.

### Evaluation and reporting

- Manifest version 2 records allowed offsets `[25, 35, 50]` and validates
  every row against the maximum offset.
- Closed-loop configuration enforces the proportional budget and the matching
  CEM horizon.
- The Push-T nonzero-angle state restoration and terminated-row planning paths
  have regression tests.
- Evaluation records per-task environment steps, planner calls, synchronized
  planning time, goal-encoding time, and end-to-end time.
- One joint summarizer validates all nine result files, shared artifacts,
  shared manifest, per-task success recomputation, timing arithmetic, and all
  ten Fast-LeWM prefixes.
- Training, open-loop evaluation, and closed-loop evaluation require the same
  full Git commit in artifact metadata.
- A fresh-process test blocks all reference-package imports while importing
  every project entry point. Reusing the reference virtual environment remains
  allowed; importing its source packages does not.
- Dataset-driven reset semantics were audited against the locked read-only
  source: block angle precedes position, the intentional one-step physics flush
  is retained, and the policy receives the dataset initial/goal pixels. The
  reference package is not imported by project code or tests.

## Verification

- Complete implementation diff and stale-reference review: passed.
- Ruff lint, Ruff format, `git diff --check`, shell syntax, and all 46 tests
  passed in the exact remote environment.
- Installed direct dependencies match the root `uv.lock`; `pip check` reports no
  broken requirements. Reference distributions and both external Lightning
  callback entry-point groups are absent.
- Every cache, training, open-loop, and O25/O35/O50 closed-loop Hydra
  composition passed through `scripts/preflight.sh`.
- The 2,336,736-frame cache migrated to version 2 without re-encoding. Its
  latent SHA-256 remained
  `0d055b64c7ca486e107d3325a851bd28d5e7307e81d4e6b58763d0435b537781`,
  and reuse was verified with encoder construction blocked.
- Real H50 indices, episode boundaries, balanced GC-IDM/LARC goal bins, masks,
  tensor shapes, and a vectorized open-loop loader batch were checked.
- Strict tensor-only construction/loading and forward/backward smokes passed
  for Fast-LeWM, GC-IDM, and LARC. The normal YAML batch/loader settings were
  reviewed on the current GPU; there is no memory threshold or runtime gate.
- The paired 50-task manifest passed episode/start/goal row checks. Its SHA-256
  is `aa8c8845d82374807db8917f3876cc5db3722e3ef7b078841fd300abed5ae50d`.
- Reduced-cost closed-loop interface smokes passed for horizon-10 CEM, GC-IDM,
  and LARC, including terminated-row masking and planner-call accounting.

## Pre-training review gate

Status: **passed**.

All frozen-protocol evidence is complete. Production must run from the clean
committed review revision; the workflow records that full revision in every
training and evaluation artifact.

## Cleanup transaction

The cleanup transaction completed after strict loading, open-loop validation,
all nine closed-loop runs, and joint summary validation passed:

- removed the superseded H25 learned weights, results, Lightning state, and logs;
- removed periodic/last tensor exports and completed-run Lightning checkpoints;
- removed the obsolete Push-T object checkpoint and extracted raw archive;
- retained the active HDF5 dataset, portable released-LeWM artifact, immutable
  frame latents, three best learned weights, per-task JSON, and paired manifest.

The verified H50 run directory was reduced from about 1.4 GB to 77 MB. Training
callbacks now export only the best portable tensor weights; Lightning state is
used only for interrupted-run recovery and is not part of a finalized run.

## Training

Completed from clean revision
`01eccc3ccafba8c1eae140d504ad8abe523484f0` with seed `3072`:

| Component | Epochs | Batch size | Best validation loss |
|---|---:|---:|---:|
| Fast-LeWM-H50 | 10 | 15,000 | 0.216866 |
| GC-IDM-H50 | 100 | 700,000 | 0.381006 |
| LARC-H50 | 50 | 26,000 | 0.450797 |

Fast-LeWM used about 22.5 GB on the current 24 GB GPU. This was achieved only
through the checked-in batch/loader configuration; no memory threshold or gate
exists. The 2,336,736 cached frame latents were reused without re-encoding.

## Evaluation

Completed for CEM, GC-IDM, and LARC at O25/O35/O50. The shared manifest contains
50 unique held-out episode/start pairs and has SHA-256
`aa8c8845d82374807db8917f3876cc5db3722e3ef7b078841fd300abed5ae50d`.
The joint summarizer accepted all artifact, code revision, manifest, method, seed,
task-count, success-recomputation, timing, and open-loop consistency checks.

## Results

The final success rates are:

| Goal offset | CEM | GC-IDM | LARC |
|---:|---:|---:|---:|
| 25 | 18% | 44% | 80% |
| 35 | 14% | 28% | 82% |
| 50 | 8% | 12% | 50% |

See `results/RESULTS_pusht_horizon.md` and
`docs/horizon_stress_outcome.md` for confidence intervals, paired differences,
timing, open-loop prefixes, provenance, limitations, and interpretation.

## Subsequent work

The separate Two-Room production protocol has since completed. Its progress,
formal result, and interpretation are recorded in `docs/tworoom_progress.md`,
`results/RESULTS_tworoom_horizon.md`, and
`docs/tworoom_horizon_stress_outcome.md`.
