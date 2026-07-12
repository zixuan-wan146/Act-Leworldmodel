# Act-LeWorldModel

Action-aware world-model training and evaluation built on LeWorldModel.

The project separates a planner-agnostic Fast-LeWM backbone from two learned
controllers over frozen latents:

- **GC-IDM** predicts one action from current/goal latents and the remaining
  horizon, then re-encodes the real observation at every environment step.
- **LARC-Chunk** predicts an action chunk. During training, the chunk is rolled
  through frozen latent dynamics and its terminal prediction is constrained
  toward the goal latent.

The reproduced CEM path is retained only as an evaluation baseline under
`controllers/baselines`; it is not a model or a dependency of either policy.

## Setup

```bash
git submodule update --init --recursive
uv venv --python=3.10
uv pip install -e .
```

Datasets are resolved through `stable-worldmodel`; set `STABLEWM_HOME` or
`LOCAL_DATASET_DIR` for local storage.

## Train the world model

```bash
train-world-model data=pusht
```

Available dataset configurations are `pusht`, `tworoom`, `reacher`, and
`cube`. Defaults follow the Fast-LeWM paper where the implementation is
specified: horizon 5, latent width 192, a 3-layer/6-head prefix Transformer,
a 6-block action-modulated predictor, ten epochs, and batch size 128 (32 for
Cube).

The Fast-LeWM authors have not released their source code yet. Tokenizer and
residual-block details not fixed by the paper are therefore explicit in
[`configs/world_model/fast_lewm.yaml`](configs/world_model/fast_lewm.yaml) for later
reconciliation.

## Train amortized controllers

Both learned policies consume a tensor cache produced from a frozen encoder.
This keeps image encoding out of the policy-training hot path.

GC-IDM caches require:

```text
current_latent:  [N, latent_dim]
goal_latent:     [N, latent_dim]
steps_remaining:[N]
action:          [N, action_dim]  # normalized/model action coordinates
```

LARC caches require:

```text
current_latent:  [N, latent_dim]
goal_latent:     [N, latent_dim]
steps_remaining:[N]
action_chunk:    [N, chunk_size, action_dim]  # normalized/model coordinates
```

Train with:

```bash
train-gc-idm latent_cache=/path/to/gc_idm_cache.pt

train-larc \
  latent_cache=/path/to/larc_cache.pt \
  world_model.config_path=/path/to/model_config.yaml \
  world_model.weights_path=/path/to/fast_lewm_backbone_epoch_10.pt
```

The LARC loss freezes all world-model parameters while retaining gradients
from the terminal latent through the predicted actions to the chunk policy.
Learned policies operate in the same normalized action coordinates as the
world model. Their controllers accept an `ActionTransform` and decode policy
outputs back to environment action units before execution.

## Backbone API

```python
latents = model.encode_observations(pixels)
future_latents = model.predict_latents(latents[:, 0], action_sequence)
```

`predict_latents` accepts arbitrary leading batch dimensions, so a future
planner can pass `[batch, candidates, horizon, action_dim]` without coupling
its implementation to the dynamics model.

The latest static architecture review is recorded in
[`docs/controller_refactor_review.md`](docs/controller_refactor_review.md).
Project documentation is maintained under [`docs/`](docs/). Datasets,
checkpoints, caches, logs, and generated outputs stay outside this repository.
