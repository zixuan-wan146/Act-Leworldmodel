# Project Structure

```text
configs/                    World-model, policy, controller, and training configs.
controllers/baselines/      Evaluation-only search baselines such as CEM.
controllers/learned/        Closed-loop GC-IDM and LARC controller wrappers.
data/dataset.py             Offline visual-trajectory loading for world-model training.
data/latent_policy_dataset.py  Frozen-latent caches for policy training.
models/world_model/         Encoder composition and latent dynamics backbone.
models/policies/            Trainable GC-IDM and LARC-Chunk networks.
losses/world_model/         Dense prefix prediction and SIGReg.
losses/policies/            GC-IDM regression and LARC rollout consistency.
train/                      Separate world-model, GC-IDM, and LARC entry points.
eval/                       Closed-loop, open-loop, and protocol evaluation.
scripts/                    Reproducible phase-level experiment commands.
tests/                      Unit and integration tests.
results/                    Versioned result reports and figures.
third_party/                Pinned external source dependencies.
```

## Boundaries

- `data/` contains loading code, not datasets.
- `configs/` owns paths, model choices, hyperparameters, and run modes.
- `models/`, `losses/`, and `controllers/` contain reusable implementation.
- `train/` and `eval/` orchestrate workflows without duplicating core logic.
- `scripts/` remain thin and call Python implementations.
- `results/` stores reviewable reports and figures, not raw logs or checkpoints.

## Dynamics boundary

`FastLeWMBackbone` owns pixel encoding and delegates latent transitions to
`PrefixDynamics`. `PrefixDynamics` is a pure function of an anchor latent and
an action sequence. It returns one predicted latent per action prefix and has
no knowledge of goals, CEM, MPC, or policy state.

The training path is:

```text
trajectory segment
  -> shared visual encoder
  -> anchor/future latent split
  -> causal action-prefix encoder
  -> parallel latent predictor
  -> dense prefix MSE + SIGReg
```

## Controller capabilities

The dependency boundary is expressed through two protocols:

```text
LatentEncoder
  encode_observations(...)

LatentDynamics extends LatentEncoder
  predict_latents(...)
```

- GC-IDM training and inference require only `LatentEncoder` outputs.
- LARC inference requires only `LatentEncoder`; its training loss additionally
  requires `LatentDynamics` for differentiable action rollout.
- CEM requires `LatentDynamics` online and remains isolated as a baseline.

All closed-loop methods return `ActionCommand [B,K,A]`. GC-IDM uses `K=1`,
LARC uses its configured chunk size, and CEM adapts the existing solver output.
