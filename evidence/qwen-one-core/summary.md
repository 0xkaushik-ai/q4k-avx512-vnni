# CPU inference experiment

Model SHA-256: `ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a`

Tuned baseline: 1 physical threads; flash attention on.

Speedup = baseline time / candidate time. Above 1 is faster. Intervals are exploratory.

| Prompt / decode tokens | Prefill speedup | Decode speedup | Total speedup (95% interval) |
|---|---:|---:|---:|
| 128 / 64 | 1.103× | 1.182× | 1.157× (1.041–1.312) |

Full-vocabulary logits correctness passed on 1 fixed token sequences.

Exploratory 20% primary-workload target met: **False**.

- Synthetic fixed tokens, no sampling or text tokenization in timings
- One shared laptop, no controlled clock/thermal state or second machine
- Bootstrap resamples blocks; correlated host noise may invalidate nominal coverage
- Only this model, quantization and CPU; no general speed superiority established
