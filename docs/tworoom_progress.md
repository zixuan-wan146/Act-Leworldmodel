# Two-Room horizon-stress progress

## Current state

- Stage: protocol and asset audit complete; implementation not started.
- Production training: not started.
- Production evaluation: not started.
- Active protocol: `docs/tworoom_horizon_stress_test.md`.

## Verified inputs

- HDF5: 920,809 frames, 10,000 episodes, action/proprio width 2.
- H50 training/validation anchors: 379,531 / 42,221.
- Released tensor state: 303 tensors, strict project-model load passed.
- Released tensor SHA-256:
  `566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd`.
- Legacy object checkpoint is present only as an obsolete external input and is
  never loaded by project code.
- Current data disk free space: about 157 GB; current GPU: 24 GB RTX 4090 D.

## Required implementation

- generic trajectory latent/cache and evaluation readers;
- portable Two-Room released artifact publication;
- project-owned fixed Two-Room environment with HDF5 golden tests;
- shared task-configured training/open-loop/closed-loop entry points;
- Two-Room task config, production workflow, manifest, and result summary;
- full verification, production training/evaluation, cleanup, and local sync.

## Production gate

Status: **not passed**. No Two-Room cache, learned checkpoint, or evaluation
result exists yet.
