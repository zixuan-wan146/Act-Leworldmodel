# Two-Room horizon-stress progress

## Current state

- Stage: **production complete**.
- Production revision: `b33bef913b0d8f237c1c4306660e5755d406c76f`.
- Training/data seed: `3072`; evaluation seed: `42`.
- Active frozen protocol: `docs/tworoom_horizon_stress_test.md`.
- Formal report: `results/RESULTS_tworoom_horizon.md`.
- Interpretation: `docs/tworoom_horizon_stress_outcome.md`.

## Completed production artifacts

- The audited HDF5 contains 920,809 frames in 10,000 episodes, with 379,531
  H50 training anchors and 42,221 validation anchors.
- The 338 MB frame-latent cache was built from the portable 303-tensor released
  state. The frame array SHA-256 is
  `01abe90b7179c7946c95b4539433eb52b6a98fd76f73b858d9b480a1f9306406`.
- One H50 Fast-LeWM, one GC-IDM, and one LARC best tensor checkpoint remain.
  Completed Lightning recovery checkpoints were removed automatically.
- All ten Fast-LeWM prefix metrics, the paired manifest, all nine closed-loop
  JSON files, and the final plots/report are present.
- The finalized experiment directory is 76 MB; large artifacts remain on the
  data disk and are not committed.

## Closed-loop result

| Goal offset | CEM | GC-IDM | LARC |
|---:|---:|---:|---:|
| 25 | 41/50 (82%) | 49/50 (98%) | 50/50 (100%) |
| 35 | 26/50 (52%) | 47/50 (94%) | 48/50 (96%) |
| 50 | 19/50 (38%) | 42/50 (84%) | 48/50 (96%) |

The O50 LARC-minus-GC-IDM paired difference is +12 percentage points, with
discordant counts 7 versus 1. O25 and O35 each differ by only one net success.
Wilson intervals, all paired counts, timing, and hashes are in the formal report.

## Validation and resource record

- Ruff, formatting, `git diff --check`, all 56 tests, and every Hydra composition
  passed before production training.
- The project-owned environment matched stored reset/transition frames for 64
  deterministic dataset transitions and fixed wall-collision goldens.
- Fast-LeWM MSE is below persistence at every measured prefix from 5 to 50 raw
  steps; at 50 steps the values are 0.944226 versus 1.993175.
- Production batches were 15,000 for Fast-LeWM, 350,000 for GC-IDM, and 27,000
  for LARC. The LARC override is recorded in `policy_metadata.json`.
- Observed compute-process memory was about 21,870 MiB for Fast-LeWM, 11,848 MiB
  for GC-IDM, and 22,738 MiB for LARC on the 24,564 MiB GPU.

## Interpretation boundary

- This is one training seed and one 50-task paired manifest, not a multi-seed
  estimate.
- GC-IDM received about 100 optimizer updates, while LARC received about 700
  because their safe batch sizes differ. The result compares the frozen training
  recipes; it does not isolate architecture or loss contributions.
- Two-Room uses fixed geometry and deterministic dynamics. Architecture and
  optimizer-budget ablations are outside this completed production protocol.

## Production gate

Status: **passed**. No production work remains under the frozen protocol.
