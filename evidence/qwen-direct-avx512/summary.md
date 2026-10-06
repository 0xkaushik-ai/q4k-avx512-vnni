# CPU inference experiment

Model SHA-256: `ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a`

Tuned baseline: 8 physical threads; flash attention on.

Speedup = baseline time / candidate time. Above 1 is faster. Intervals are exploratory.

| Prompt / decode tokens | Prefill speedup | Decode speedup | Total speedup (95% interval) |
|---|---:|---:|---:|
| 128 / 64 | 0.936× | 0.975× | 0.951× (0.889–1.037) |
| 512 / 64 | 0.972× | 1.015× | 0.986× (0.916–1.111) |
| 1024 / 64 | 1.007× | 1.040× | 1.037× (1.012–1.778) |

Full-vocabulary logits correctness passed on three fixed token sequences.

Exploratory 20% primary-workload target met: **False**.

- Synthetic fixed tokens, no sampling or text tokenization in timings
- One shared laptop, no controlled clock/thermal state or second machine
- Bootstrap resamples blocks; correlated host noise may invalidate nominal coverage
- Only this model, quantization and CPU; no general speed superiority established
