# Q4_K AVX-512 VNNI GEMV for llama.cpp

An opt-in rewrite of llama.cpp's repacked Q4_K matrix-vector kernel using
AVX-512 VNNI, with a measurement harness that checks two separate questions:

1. **Is the kernel faster and still exact?** Yes: 1.35–1.47× in isolation, bitwise identical.
2. **Does that make the model faster?** Not at the tuned 8-core baseline (0.951×).
   Under a 1-core budget it was 1.157×.

The point of this repository is the gap between those two answers, and the
evidence needed to tell a real speedup from a misleading one.

## Results

Measured on an Intel i7-11800H laptop (Tiger Lake, AVX-512 VNNI), llama.cpp
commit [`836d571`](https://github.com/ggml-org/llama.cpp/tree/836d57176dc699a726c55418e4f96b8ca628e1bf).
Ratio = baseline time / candidate time; above 1× is faster.

### Kernel (isolated, single thread, warm cache)

| Iteration | What changed | Median ratio |
| --- | --- | ---: |
| 1. 256-bit VNNI | `vpdpbusd` in place of `vpmaddubsw` pairs, one accumulator | 0.94–0.96× |
| 2. Four accumulators | Shorter dependency chain | 0.98–1.02× |
| 3. **Direct AVX-512** | Reads the existing 8-byte interleave directly; keeps two integer lanes per column until the block ends | **1.35–1.47×** |

Static disassembly: the baseline's 16 `vpmaddubsw`, 16 `vpblendd`, 35 `vpshufd`
and 3 `vpshufb` become 8 512-bit `vpdpbusd`; vector stack accesses drop from 7 to 0.
Correctness: **875 fixtures, 42,000 output floats, bitwise equal**, including
signed-byte extremes and nonbinary scales.
[Kernel notes](docs/kernel-notes.md) · [timings](evidence/kernel/)

### Whole model (Qwen3-0.6B Q4_K_M, model resident, loading excluded)

| Configuration | Prompt / decode | Decode | Total (95% bootstrap) |
| --- | --- | ---: | ---: |
| Tuned baseline: 8 threads, flash attention on | 128 / 64 | 0.975× | **0.951×** (0.889–1.037) |
| | 512 / 64 | 1.015× | 0.986× (0.916–1.111) |
| | 1024 / 64 | 1.040× | 1.037× (1.012–1.778) |
| One-core diagnostic | 128 / 64 | 1.182× | **1.157×** (1.041–1.312) |

All captured logits matched bitwise: 7,748,736 values at 8 threads, 2,582,912 at
one core, 2,582,912 on an odd prefill length (131) that exercises GEMV remainder
rows, and 2,506,752 on a SmolLM2-135M regression model. Two greedy text
completions matched byte-for-byte. [Full results](docs/results.md)

### Interpretation

The kernel does less arithmetic work per byte of weights. With eight cores, Q4_K
decode is limited mostly by memory bandwidth and the other operations in the
graph, so removing shuffle instructions does not shorten the critical path. With
one core, compute is a larger share of the time and the gain becomes visible.
This is a hypothesis consistent with the data, not yet confirmed with hardware
performance counters.

The pre-registered target (≥1.20× total speedup at the tuned baseline) was
**not met**, and the screening threshold (≥1.10×) was missed, so no
confirmation run was done. The kernel stays opt-in.

## Method

- **Pre-registered plan:** model, workloads, thresholds and stopping rules are in
  [`engine/experiment-plan.json`](engine/experiment-plan.json), written before the
  whole-model candidate measurements.
- **Tuned baseline:** threads (1/2/4/8 physical cores) and flash attention are
  swept on the baseline, then held fixed for both paths.
- **Same binary:** baseline and candidate are selected at runtime by
  `DEVICEBENCH_Q4K_VNNI=0|1`, so weights, build flags and the rest of the graph are identical.
- **Paired AB/BA blocks:** alternating order, one warm-up and three measured
  trials per process; bootstrap intervals resample blocks, not tokens.
- **Correctness first:** full-vocabulary logits at 17 positions on three fixed
  sequences, plus the bitwise kernel fixture test.
- **Host state recorded:** load, memory, swap, temperatures and governor before and
  after every launch; peak RSS and page faults for every trial.

Full protocol and limitations: [docs/protocol.md](docs/protocol.md).

## Repository layout

| Path | Contents |
| --- | --- |
| `engine/patches/q4k-vnni.patch` | The kernel (applied to pinned llama.cpp) |
| `engine/kernel_test.cpp` | Bitwise fixture test and isolated kernel benchmark |
| `engine/bench.cpp` | Resident-model benchmark and logit dump |
| `engine/CMakeLists.txt`, `kernel-tests.cmake` | CPU-only Release build and CTest |
| `engine/source-lock.json`, `experiment-plan.json` | Pinned source, model hashes and pre-registered plan |
| `harness/engine_lab.py` | Tuning, paired runs, bootstrap and logit comparison (standard library only) |
| `tests/` | Harness unit tests |
| `scripts/prepare-engine.py` | Downloads llama.cpp, checks its hash and applies the patch |
| `evidence/` | Recorded summaries, per-run execution records, kernel timings and earlier patch iterations |
| `docs/` | Protocol, results and kernel notes |

The C symbols use a `devicebench_` prefix because this work began inside a
larger project, DeviceBench. The file contents are unchanged from the measured
versions: the patch and `bench.cpp` SHA-256 values match those recorded in
`evidence/*/metadata.json`.

## Reproduce

Requires Linux x86-64 with AVX-512 VNNI, CMake 3.24+, a C++20 compiler, OpenMP,
`patch`, `taskset` and Python 3.11+.

```sh
python3 scripts/prepare-engine.py
mkdir -p .cache
curl -fL 'https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/50968a4468ef4233ed78cd7c3de230dd1d61a56b/Qwen3-0.6B-Q4_K_M.gguf' \
  -o .cache/qwen3-0.6b-q4_k_m.gguf
echo 'ac2d97712095a558e31573f62f466a3f9d93990898b0ec79d7c974c1780d524a  .cache/qwen3-0.6b-q4_k_m.gguf' | sha256sum -c -

cmake -S engine -B build/engine \
  -DLLAMA_SOURCE_DIR="$PWD/.cache/llama.cpp-836d57176dc699a726c55418e4f96b8ca628e1bf" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/engine -j 2
ctest --test-dir build/engine --output-on-failure        # bitwise kernel fixtures
PYTHONPATH=harness python3 -m unittest discover -s tests -v

taskset -c 2 build/engine/bin/engine-kernel-test --benchmark   # isolated kernel
python3 harness/engine_lab.py --model .cache/qwen3-0.6b-q4_k_m.gguf   # whole model
```

`--blocks 4 --threads 8` gives the shorter screening run; `--threads 1 --primary-only`
gives the one-core diagnostic. Results go to a new timestamped folder under `reports/engine/`.

Text demo: `DEVICEBENCH_Q4K_VNNI=1 build/engine/bin/engine-demo -m .cache/qwen3-0.6b-q4_k_m.gguf -ngl 0 -n 24 "The capital of France is"`

SmolLM2 regression model:
`https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf`,
SHA-256 `2e8040ceae7815abe0dcb3540b9995eaa1fa0d2ca9e797d0a635ae4433c68c2d`.

The native build uses `-march=native`; do not copy binaries to other CPUs.

## Limits

One shared laptop, two small models and synthetic fixed-token workloads. Clock
frequency, thermal state and background load were recorded but not controlled.
Only single-row GEMV is changed; prefill GEMM and other quantizations use upstream
code. Next steps are a quiet-host repeat, hardware counters to test the
memory-bandwidth explanation, larger models and a second CPU.

## License

MIT for this repository's code (see [`LICENSE`](LICENSE)). The patch modifies
llama.cpp, which is MIT-licensed by its authors; its notice is in
[`engine/LLAMA-LICENSE`](engine/LLAMA-LICENSE). Raw logit dumps and full runtime
logs are not committed because of their size.
