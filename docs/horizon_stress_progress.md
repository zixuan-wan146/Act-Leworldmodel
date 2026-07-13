# Push-T horizon-stress progress

## Current state

- Stage: pre-training gate passed; ready for production from the reviewed commit.
- Formal production training: not started.
- Production evaluation: not started.
- Active protocol: `docs/pusht_horizon_stress_test.md`.
- Production workflow requires a clean worktree and records its full commit.

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
- Ruff lint, Ruff format, `git diff --check`, shell syntax, and all 45 tests
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

The raw dataset, portable released-LeWM weights, and frame latents are active
inputs and must remain. Earlier H25-only learned weights, Lightning state,
evaluation JSON, and versioned result files are deleted only after the H50
training, open-loop evaluation, all nine closed-loop runs, and final report are
verified. No backup, compatibility reader, or duplicate configuration is kept
after that replacement point.

## Training

Not started.

## Evaluation

Not started.

## Results

Not available.

## Queued next phase

After the Push-T report is verified, Two-Room will receive complete
project-owned dataset processing, latent caching, model training, closed-loop
evaluation, and result reporting. The available external `tworoom.h5` is an
input artifact only; no deleted stub config is counted as implementation.
