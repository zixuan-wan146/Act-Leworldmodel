# Controller Refactor Review

Review status: **Approved for the next implementation phase**

Scope: package boundaries, frozen-world-model capabilities, GC-IDM and
LARC-Chunk policy definitions, controller contracts, policy losses, latent
cache loaders, training entry points, and Hydra configuration.

No tests, training runs, or environment rollouts were executed, per project
instruction. This is a static code and architecture review.

## Findings validated

- CEM is isolated under `controllers/baselines` and is not represented as a
  trainable model.
- GC-IDM depends only on frozen latent encoding and predicts one action per
  closed-loop step.
- LARC predicts an action chunk; its consistency objective invokes frozen
  latent dynamics during training but not controller inference.
- `FrozenWorldModel` disables parameter gradients while intentionally leaving
  autograd enabled through actions in `predict_latents`.
- Controller outputs share one `[batch, chunk, action_dim]` contract with an
  explicit `replan_after` value.
- Cached goal observations/latents are non-persistent buffers and therefore
  follow controller device moves.
- Policy checkpoints contain only learned policy weights and resolved policy
  construction config, never a duplicated frozen world model.
- Learned policies operate in model-action coordinates; controllers explicitly
  decode them through an `ActionTransform` before environment execution.

## Findings fixed during review

1. Goal caches were plain attributes and could remain on the old device after
   moving a controller. They are now registered non-persistent buffers.
2. `commit_steps=0` silently selected the full LARC chunk because of truthy
   fallback logic. `None` is now the only default sentinel.
3. Normalized world-model actions could be sent directly to the environment.
   The controller boundary now includes identity and z-score action transforms.
4. A frozen world model could accidentally cut LARC action gradients if all
   calls used `no_grad`. Only visual encoding is detached; differentiable
   dynamics rollout remains enabled.
5. Frozen world-model weights could be duplicated in every policy checkpoint.
   Policy training uses a model-only portable checkpoint callback.

## Remaining implementation work

- Build the latent-cache extraction command from the reproduced PushT dataset
  and checkpoint pipeline. The training loaders already define its tensor
  schema.
- Connect the reproduced stable-worldmodel CEM callable to `CEMController` when
  baseline evaluation is resumed.
- Implement the shared closed-loop evaluator in the currently empty `eval/`
  files using `ActionCommand.replan_after`.
- Reconcile the configurable LARC network details with its original source;
  only the method behavior supplied in the project brief is currently fixed.

These remaining items do not require another directory redesign.
