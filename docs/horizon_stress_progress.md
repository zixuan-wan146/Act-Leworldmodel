# Push-T horizon-stress progress

## Current state

- Stage: protocol freeze.
- Formal production training: not started.
- Production evaluation: not started.
- Active protocol: `docs/pusht_horizon_stress_test.md`.

## Decisions

- Use one H50 Fast-LeWM, one H50 GC-IDM, and one H50 LARC checkpoint.
- Evaluate the same learned checkpoints at goal offsets 25, 35, and 50.
- Use one paired 50-task base manifest shared across offsets and methods.
- Use proportional evaluation budgets 50, 70, and 100.
- CEM and LARC replan every five raw actions; GC-IDM replans every action.
- Use only project-owned runtime code.
- Preserve completed H25 artifacts and reports as historical results.

## Dataset feasibility

The existing split was inspected without modifying artifacts:

| Offset | Train episodes | Train anchors | Validation episodes | Validation anchors |
|---|---:|---:|---:|---:|
| 25 | 16,816 | 1,681,924 | 1,869 | 187,687 |
| 35 | 16,816 | 1,513,764 | 1,869 | 168,997 |
| 50 | 16,721 | 1,261,619 | 1,863 | 140,968 |

The 50-step view has sufficient data. Existing artifacts remain limited to a
25-step maximum and cannot be used for O35/O50 without new training.

## Stage log

### Protocol freeze

- Remote worktree inspected at commit
  `7e5394e7c76d72097ecbcdf725f1ecaa2f641454`.
- Remote worktree was clean before documentation changes.
- Local review copy matched the same commit.
- Existing H25 documentation and results were retained unchanged.
- Production-training gate defined in the protocol.

## Pre-training review gate

Status: **not reviewed**.

No production training may start while this status remains open.

## Training

Not started.

## Evaluation

Not started.

## Results

Not available.
