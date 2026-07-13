# Portable released-LeWM artifact

The project never loads the released Python-object checkpoint at runtime. That
legacy file embeds Python class references and is neither self-contained nor a
safe serialization boundary.

Runtime reconstruction has two independent inputs:

1. `configs/released_lewm/pusht.yaml` declares the complete architecture using
   modules owned by this repository.
2. `PUSHT_LEWM_WEIGHTS` points to a tensor-only artifact loaded with
   `torch.load(..., weights_only=True)`.

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

Migration from a legacy object is an offline artifact-publication operation. It
must run in an isolated environment that owns and trusts the legacy classes;
it is intentionally not part of this project's install or runtime. Published
artifacts must be checked against the declared source SHA-256 and must strictly
load into the project architecture before use.

The production code rejects object modules, unexpected artifact kinds,
unsupported format versions, missing lineage metadata, non-tensor state-dict
entries, missing parameters, and unexpected parameters.
