# Push-T results

Fixed evaluation seed: `42`. All methods use the same episode/start manifest.
The episodes are held out from newly trained dynamics/policies; the released LeWM/CEM checkpoint retains its upstream clip-level split provenance. See [the protocol](../docs/pusht_protocol.md).

| Method | Successes | Success rate | 95% Wilson CI | Evaluation time |
|---|---:|---:|---:|---:|
| CEM | 41/50 | 82.0% | [69.2%, 90.2%] | 72.3 s |
| GC-IDM | 47/50 | 94.0% | [83.8%, 97.9%] | 5.2 s |
| LARC | 46/50 | 92.0% | [81.2%, 96.8%] | 4.6 s |

![Push-T success rates](pusht_success_rates.png)

## Evaluation provenance

Code revision: 1af58196d5a62bdcb7fc39e421e469a70d0aa974.
Manifest SHA-256: 9288432743b283da8b7d3e933f61b888cf9335da1170641f5c38a2ecc1ec389c.

| Artifact | SHA-256 |
|---|---|
| Latent cache metadata | eb2dd6ec89eb1702271cd0fbb997b25377cc769edb682b9a2ef363e19168276a |
| Released LeWM config | 843ce3f0d2db9853dc111adcbdadeacfe0fda1a1af2ecdbd86cb2bef8a13cc64 |
| Released LeWM weights | 446262af36abba313e4436287dc904b68a49f4fda045fa1f974dd9959f532002 |
| Fast-LeWM config | 718f3d9e188b32949fabd791002b2af33120bcd49e5a3b077087fc206fd53e49 |
| Fast-LeWM weights | a3d5dc3eda45d07af079e1c8738f8d1d6729715af68f7beed9a66595a0d808a8 |
| GC-IDM config | d7bbe89d439cb14fb1d215fe744e0c29b167d0030dd520c882353664fce2cbdd |
| GC-IDM weights | 2af3b36526ce2a4af9a39889fda44b18251198eaa0ed6d6f6c6a0fa3e0ec6905 |
| LARC config | 07cb3fb3c540de9c5ab316fbce11f3b3dbf08bfa6caf99f7754fae0a7fa31446 |
| LARC weights | 7fdabb6f44d8498d44383eb7b828c32998e61727266bf3e9a318f102881cdd77 |

## Fast-LeWM open-loop validation

Held-out validation clips: `100000`.

| Environment steps | Fast-LeWM MSE | Persistence MSE |
|---:|---:|---:|
| 5 | 0.008709 | 0.178751 |
| 10 | 0.014419 | 0.518204 |
| 15 | 0.022708 | 0.862032 |
| 20 | 0.034548 | 1.153437 |
| 25 | 0.051198 | 1.379066 |

![Fast-LeWM open-loop curve](open_loop_curve.png)

Raw metrics, per-episode successes, resolved configs, and the fixed protocol manifest remain in the external run directory and are not committed.
