# Push-T experiment protocol

## Fixed data split

- Training seed: `3072`.
- Split unit: complete episode, never an overlapping clip.
- Split fraction: 90% train / 10% validation.
- Action mean and standard deviation: computed from finite rows in training
  episodes only.
- Visual representation: released Push-T LeWM encoder and projector, frozen.
- Cached latent dtype: float16; model and loss accumulation use float32 where
  numerically relevant.

The cache metadata records the dataset path/size, source-checkpoint SHA-256,
episode IDs, offsets, lengths, action statistics, frameskip, latent width, and
all split parameters.

This episode split governs the newly trained Fast-LeWM dynamics, GC-IDM, and
LARC. The released LeWM checkpoint has separate upstream provenance: its
official training script uses seed `3072` with a 90/10 random split over
already constructed clips, and fits column normalizers before that split. Its
encoder/projector and CEM dynamics can therefore have seen clips from episodes
that this project assigns to validation. Results are held out from the newly
trained stages, not from the upstream representation or CEM checkpoint.

## Released LeWM checkpoint provenance

The pinned upstream training config defines the checkpoint used by CEM and by
the reused visual representation:

- ViT-Tiny encoder, patch size 14, 224-pixel input, 192-dimensional CLS token.
- Projection MLP `192 -> 2048 -> 192` with BatchNorm; SIGReg uses 17 knots,
  1024 projections, and weight `0.09`.
- One-step autoregressive predictor with history size 3, depth 6, 16 heads,
  head dimension 64, and MLP width 2048.
- Push-T frameskip 5, so the action encoder receives `5 * 2 = 10` values.
- AdamW, learning rate `5e-5`, weight decay `1e-3`, batch size 128, BF16,
  gradient clipping 1.0, and 100 epochs.
- Seed `3072` and a clip-level 90/10 random split. Action/proprio/state
  normalizers are fitted on the complete source dataset before that split.

## Dynamics

- Raw action dimension: 2.
- Frameskip: 5 raw environment steps.
- Model action dimension: 10.
- Prefix horizon: 5 model steps / 25 environment steps.
- Objective: dense MSE at all five prefix endpoints.
- Encoder/projector: reused and frozen from released LeWM.
- Dynamics optimizer seed: `3072`.

## Learned controllers

GC-IDM is trained on deterministic goal offsets from 1 through 25 and predicts
the current normalized two-dimensional expert action. It replans every raw
environment step.

LARC is trained on deterministic goal offsets `{5, 10, 15, 20, 25}`. It
predicts five normalized 10-dimensional action blocks. Behavior cloning masks
blocks beyond the sampled goal, and rollout consistency gathers the frozen
world-model endpoint matching the sampled remaining horizon. At inference it
executes five decoded raw actions before replanning.

## Closed-loop evaluation

- Evaluation seed: `42`.
- Evaluation episodes: 50 unique validation episodes.
- Start state: one deterministic valid start sampled per episode.
- Goal offset: 25 environment steps.
- Evaluation budget: 50 environment steps.
- State/goal initialization: Push-T `_set_state` and `_set_goal_state` from the
  offline episode.
- Comparison: CEM, GC-IDM, and LARC use exactly the same manifest.

CEM reproduces the released checkpoint's full-file action scaler. Fast-LeWM,
GC-IDM, and LARC instead use statistics from this project's training episodes
only. Each model therefore receives the normalization it was trained with;
silently sharing one scaler would be incorrect.

Raw JSON results include resolved configs, per-episode successes, runtime, and
the manifest path. Only the compact Markdown summary and figures are copied to
`results/`.
