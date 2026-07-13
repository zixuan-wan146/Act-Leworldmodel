# Portable released-LeWM artifact

The project never loads the released Python-object checkpoint at runtime. That
legacy file embeds Python class references and is neither self-contained nor a
safe serialization boundary.

Runtime reconstruction has two independent inputs:

1. `configs/released_lewm/pusht.yaml` declares the complete shared architecture
   using modules owned by this repository. The historical filename is retained
   because it is part of the completed Push-T artifact provenance.
2. `PUSHT_LEWM_WEIGHTS` or `TWOROOM_LEWM_WEIGHTS` points to a task-specific
   tensor-only artifact loaded with `torch.load(..., weights_only=True)`.

The artifact is a mapping with this schema:

```text
format_version: 1
model_kind: released_lewm
metadata:
  source_checkpoint_sha256: <64 lowercase hexadecimal characters>
state_dict:
  <parameter name>: <tensor>
```

`source_checkpoint_sha256` identifies the upstream bytes from which the tensor
artifact was migrated. The latent cache, Fast-LeWM checkpoints, learned policy
checkpoints, CEM model, and evaluation manifest all retain this value as their
shared representation lineage.

A released bare tensor state dict can be published reproducibly with
`scripts/publish_released_lewm.py`. The command reads the source with
`weights_only=True`, strictly loads it into the configured project model,
writes atomically, reloads the portable artifact, and records the source and
model-config hashes.

Migration from a legacy Python object remains an offline operation in an isolated
environment that owns and trusts its classes; project runtime never opens such
an object. Every published artifact must strictly load into the declared project
architecture before use.

The production code rejects object modules, unexpected artifact kinds,
unsupported format versions, missing lineage metadata, non-tensor state-dict
entries, missing parameters, and unexpected parameters.
