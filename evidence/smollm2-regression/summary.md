# CPU inference experiment

Model SHA-256: `2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d`

Tuned baseline: 4 physical threads; flash attention on.

Speedup = baseline time / candidate time. Above 1 is faster. Intervals are exploratory.

| Prompt / decode tokens | Prefill speedup | Decode speedup | Total speedup (95% interval) |
|---|---:|---:|---:|
| 128 / 64 | 0.997× | 0.979× | 0.985× (0.974–0.996) |
| 512 / 64 | 0.986× | 1.042× | 1.006× (1.004–1.009) |
| 1024 / 64 | 1.099× | 1.043× | 1.083× (0.954–1.213) |

Full-vocabulary logits correctness passed on 3 fixed token sequences.

Exploratory 20% primary-workload target met: **False**.

- Synthetic fixed tokens, no sampling or text tokenization in timings
- One shared laptop, no controlled clock/thermal state or second machine
- Bootstrap resamples blocks; correlated host noise may invalidate nominal coverage
- Only this model, quantization and CPU; no general speed superiority established
