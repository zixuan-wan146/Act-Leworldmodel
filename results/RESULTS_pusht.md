# Push-T results

Fixed evaluation seed: `42`. All methods use the same episode/start manifest.
The episodes are held out from newly trained dynamics/policies; the released LeWM/CEM checkpoint retains its upstream clip-level split provenance. See [the protocol](../docs/pusht_protocol.md).

| Method | Successes | Success rate | 95% Wilson CI | Evaluation time |
|---|---:|---:|---:|---:|
| CEM | 41/50 | 82.0% | [69.2%, 90.2%] | 52.4 s |
| GC-IDM | 47/50 | 94.0% | [83.8%, 97.9%] | 6.4 s |
| LARC | 46/50 | 92.0% | [81.2%, 96.8%] | 5.9 s |

![Push-T success rates](pusht_success_rates.png)

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
