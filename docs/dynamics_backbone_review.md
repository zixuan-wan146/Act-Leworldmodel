# Dynamics Backbone Review

Review status: **Superseded by the world-model/controller package split; core
findings remain valid**

Scope reviewed: the CEM-free Fast-LeWM dynamics path introduced on top of
commit `531472a`, including configuration, data preparation, pixel encoding,
action-prefix encoding, parallel latent prediction, dense-prefix training,
and checkpoint persistence.

No tests or training runs were executed, per project instruction. This is a
static implementation review.

## Architecture findings

- The model boundary is clean: `PrefixDynamics` accepts only an anchor latent
  and action sequence. It has no dependency on goals, controllers, CEM, MPC,
  or evaluation state.
- Causal masking prevents a short prefix token from observing later actions.
- Every predicted horizon is anchored at the observed latent and is produced
  in parallel; predicted intermediate latents are not recurrent inputs.
- Leading batch dimensions are preserved, leaving a stable interface for
  future candidate-based policies.
- Dense prefix MSE supervises every available horizon and SIGReg is applied to
  the shared encoder outputs in float32 for mixed-precision stability.

## Findings fixed during review

1. Action inputs were forced to float32, which could mismatch explicitly
   half-cast model weights outside autocast. They now follow the state-token
   device and dtype.
2. Column standard deviation used the unbiased estimator, which becomes NaN
   for a one-row finite subset. It now uses the population estimator and a
   minimum scale.
3. SIGReg buffers could inherit a reduced dtype if the objective were manually
   cast. Its numerical path is now explicitly float32.
4. Checkpoint interval zero would fail at epoch end. Construction now rejects
   non-positive intervals.
5. The configured output model name was unused. Both Lightning and portable
   weight filenames now use it.
6. The adapted LeWM regularizer needed its original copyright retained in the
   repository-level MIT license; the notice is now included.
7. The console entry point referenced configs outside the installed Python
   packages. `configs` is now packaged so Hydra can resolve the same tree from
   editable and wheel installs.
8. Saving only the model subtree could retain parent-relative OmegaConf
   interpolations and produce a non-reloadable config. The callback now stores
   a fully resolved, self-contained model configuration.

## Known risks

- Fast-LeWM's official source is not public. The action tokenizer, prefix
  Transformer MLP width, and exact residual-predictor block are reasonable
  reconstructions of the paper rather than source-identical implementations.
- Dataset tensor shapes and optimizer integration follow the pinned LeWM
  submodule contract but remain runtime-unverified.
- Empty policy, planning, and evaluation placeholders are outside this review
  and are not imported by the dynamics training path.

These risks do not require coupling CEM into the backbone. They should be
reconciled against the official Fast-LeWM repository when its code is released.
