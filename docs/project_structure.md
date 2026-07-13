# Project Structure

```text
configs/                    World-model, policy, training, and evaluation configs.
controllers/baselines/      Evaluation-only search baselines such as CEM.
controllers/learned/        Closed-loop GC-IDM and LARC controller wrappers.
data/latent.py              Episode split, frame cache, and latent training datasets.
data/action_transform.py    Raw-action normalization and action-block conversion.
models/world_model/         Encoder composition and latent dynamics backbone.
models/policies/            Trainable GC-IDM and LARC-Chunk networks.
losses/world_model/         Dense prefix latent prediction objective.
losses/policies/            GC-IDM regression and LARC rollout consistency.
train/                      Separate world-model, GC-IDM, and LARC entry points.
eval/                       Closed-loop, open-loop, and protocol evaluation.
scripts/                    Preflight and production experiment commands.
tests/                      Unit and integration tests.
utils/                      Dependency-free helpers shared across package layers.
results/                    Versioned result reports and figures.
third_party/                Read-only implementation references; never imported at runtime.
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

The Push-T training path is:

```text
released LeWM encoder/projector
  -> one cached latent per dataset frame
  -> episode-safe anchor/future latent clips
  -> causal action-prefix encoder
  -> parallel latent predictor
  -> dense prefix MSE
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

The reused visual representation was already trained with SIGReg. Fast-LeWM
training freezes that representation and optimizes only prefix dynamics.

GC-IDM emits one raw-action coordinate per step. LARC emits model action blocks
and decodes them to raw environment actions only at the controller boundary.
