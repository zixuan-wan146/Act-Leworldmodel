# Two-Room horizon-stress progress

## Current state

- Stage: project implementation complete; production gate validation in progress.
- Production latent cache: not started.
- Production training: not started.
- Production evaluation: not started.
- Active protocol: `docs/tworoom_horizon_stress_test.md`.

## Verified inputs and implementation

- HDF5: 920,809 frames, 10,000 episodes, action/proprio width 2.
- H50 training/validation anchors: 379,531 / 42,221.
- Released tensor state: 303 tensors and 18,042,672 tensor values.
- Released tensor SHA-256:
  `566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd`.
- The portable artifact was published with `weights_only=True` and strictly
  reloaded into the project-owned ReleasedLeWM definition.
- Generic latent/cache, trajectory evaluation, training, open-loop, closed-loop,
  summary, preflight, and production workflow entry points now select a task
  through `configs/task/`.
- The project-owned Two-Room environment exactly matched the stored reset frame,
  next state, and next frame for 64 deterministic random dataset transitions,
  in addition to fixed wall-collision goldens.
- Runtime import tests block `stable_worldmodel`, `stable_pretraining`, `jepa`,
- Single-step forward/backward smokes passed for Fast-LeWM, GC-IDM, and LARC.
- Measured peak compute-process memory was about 21,870 MiB for Fast-LeWM,
  11,768 MiB for GC-IDM, and 21,852 MiB for LARC. GC-IDM uses a task-specific
  350,000 batch because the complete Two-Room H50 training view has only 379,531
  anchors; a 700,000 drop-last batch produces no training batch.
  and the upstream `module` package.

## Remaining production work

1. finish the full production gate, including reduced-cost controller smokes;
2. review normal YAML loader/batch parameters on the current 24 GB GPU;
3. build the Two-Room frame-latent cache;
4. train H50 Fast-LeWM, GC-IDM, and LARC once with seed 3072;
5. run open-loop validation and all nine paired closed-loop evaluations;
6. summarize results, clean redundant raw/recovery artifacts, and sync locally.

## Production gate

Status: **not yet passed**. No Two-Room learned checkpoint or evaluation result
exists yet, so no Two-Room success-rate claim is currently valid.
