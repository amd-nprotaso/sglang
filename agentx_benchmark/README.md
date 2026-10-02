# Qwen3.8-Flash-Next MXFP4 — AgentX Benchmark Guide

**MI355X · SGLang · MTP Speculative Decoding · InferenceX AgentX Replay**

> This guide walks through reproducing the agentic coding benchmark for
> Qwen3.8-Flash-Next MXFP4 on a single MI355X node. It covers building
> the Docker image, preparing the Slurm environment, and running the
> full AIPerf AgentX trace replay.

---

## 1. What This Benchmark Does

The benchmark replays real-world agentic coding traces (from SemiAnalysis
Weka) against a Qwen3.8-Flash-Next MXFP4 inference server running on AMD
MI355X GPUs. It measures latency (interactivity, ITL, TTFT) and throughput
under sustained concurrent load with multi-turn, variable-context sessions
up to 256k tokens.

| Component            | Details                                                      |
| -------------------- | ------------------------------------------------------------ |
| **Model**            | `fanwu103/Qwen3.8-Flash-Next-MXFP4-PLEFP8` — Hybrid GDN+QSA, Quark MXFP4 |
| **Speculative Decoding** | NEXTN MTP · 3 steps · 4 draft tokens · golden acceptance length 2.32 (K=3) |
| **Workload**         | AIPerf AgentX trace replay — 256k-capped Weka coding traces with subagents |

### InferenceX Architecture Background

[InferenceX](https://github.com/SemiAnalysisAI/InferenceX) is SemiAnalysis's
end-to-end inference benchmarking framework. Understanding its layered
architecture helps when debugging or customizing runs. The full pipeline:

```
amd-master.yaml
    │
    ▼
infx.matrix.generate        ← expands config into per-concurrency-point JSON rows
    │
    ▼
GitHub Actions / manual      ← each row becomes a Slurm job
    │
    ▼
python -m infx.launch run    ← resolves runner, binds srt-slurm recipe variant
    │
    ▼
srtctl apply                 ← submits Slurm job with enroot container
    │
    ▼
SGLang server starts         ← model loading, CUDA graph capture, warmup
    │
    ▼
srt_agentic.sh               ← client-side script, sources benchmark_lib.sh
    │
    ▼
aiperf profile --scenario agentx   ← replays Weka coding traces
    │
    ▼
infx.results.agentic.process_agentic_result  ← writes aggregated JSON
```

#### Layer 1: Config & Matrix (`configs/amd-master.yaml`)

The master config defines every model×hardware×precision×framework combination.
Each entry specifies an image, model, runner label (e.g. `cluster:mi355x-amds`),
and a `scenarios` block. For agentic benchmarks, the scenario is
`agentic-coding` with a `search-space` listing TP, EP, concurrency points,
KV offloading, and an `srt-recipe` path.

`infx.matrix.generate` (in `infx/matrix/generate.py`) parses the YAML,
walks the search-space, and emits **one JSON matrix row per concurrency
point** — each row owns its own server deployment. Runner labels like
`cluster:mi355x-amds` are resolved to concrete node pools via
`configs/runners.yaml`.

> **Note:** Qwen3.8-Flash-Next MI355X is not yet in `amd-master.yaml` —
> `qwen3.5-fp4-mi355x-sglang-agentic-mtp` is the current AMD MI355X
> AgentX reference. Qwen3.8 AgentX recipes exist for NVIDIA (B300, H200)
> in `nvidia-master.yaml`. Our smoke test script bypasses the matrix
> system and runs the benchmark directly.

#### Layer 2: Launcher (`infx.launch`)

`python -m infx.launch run` reads the matrix env, resolves the cluster from
`runners.yaml`, and dispatches to a driver. For single-node SGLang AgentX,
it routes to `infx.launch.drivers.srt.run_single_node()`, which:

1. Binds the matrix point to exactly one recipe variant (e.g. `override_tp4_c4`)
   via `infx.srt_slurm.single_node.prepare()`
2. Runs `srtctl apply` to submit the Slurm job
3. Streams logs and collects artifacts after completion

#### Layer 3: srt-slurm Recipe (`agentic.yaml`)

Each recipe YAML (e.g. `srt-slurm-recipes/qwen3.5/sglang/mi355x-fp4-mtp/agentic.yaml`)
contains:

- **`base`**: SGLang engine args (attention backend, MTP config, mem-fraction,
  chunked-prefill-size, etc.) and `benchmark.type: custom` with
  `benchmark.command: bash /infmax-workspace/benchmarks/srt_agentic.sh`
- **`benchmark.env`**: `MODEL`, `CONC`, `KV_OFFLOADING`, `WEKA_LOADER_OVERRIDE`,
  `AIPERF_*` flags
- **Per-point overrides** (`override_tp4_c4`, `override_tp4_c8`, etc.):
  set `tensor-parallel-size`, `max-running-requests` (= 2×CONC), and
  `cuda-graph-max-bs-decode`

#### Layer 4: Benchmark Execution (`srt_agentic.sh` + `benchmark_lib.sh`)

Inside the container, `srt_agentic.sh` sources `benchmark_lib.sh` (the
shared library with ~3500 lines of AgentX logic) and:

1. **`install_agentic_deps()`** — creates a uv venv and installs AIPerf from
   `utils/aiperf` (editable install)
2. **`resolve_trace_source()`** — maps a loader name to `--public-dataset`
   and pre-downloads the HuggingFace trace dataset
3. **`build_replay_cmd()`** — constructs the full `aiperf profile` command
4. **`run_agentic_replay_and_write_outputs()`** — runs the replay, optional
   GPU power monitor, then calls `write_agentic_result_json()`

#### Layer 5: AIPerf Trace Replay

[AIPerf](https://github.com/SemiAnalysisAI/InferenceX/tree/main/utils/aiperf)
is the workload generator. The `agentx` scenario
(`aiperf/common/scenario/inferencex_agentx_mvp.py`) replays multi-turn
agentic coding sessions from the Weka corpus:

- **Dataset**: `semianalysis_cc_traces_weka_with_subagents_256k` — 393 real
  coding traces from HuggingFace (`semianalysisai/cc-traces-weka-062126-256k`),
  capped at 256k tokens, with full subagent SPAWN/JOIN topology
- **Replay**: each trace is a multi-turn conversation with variable context
  lengths. AIPerf maintains `CONC` concurrent lanes, each replaying its own
  session with cache-bust on first-turn prefix
- **Duration**: 3600s canonical, 900s minimum (smoke test)
- **Warmup**: 10 requests per lane with 1800s grace period
- **Failure threshold**: 10% max failed requests

#### Layer 6: Result Collection

`write_agentic_result_json()` in `benchmark_lib.sh` calls
`infx.results.agentic.process_agentic_result`, which reads
`profile_export.jsonl` (per-request records) and the aiperf aggregate JSON,
then writes the final `{RESULT_FILENAME}.json` containing:

- `request_metrics.latency` — `intvty`, `itl`, `ttft`, `tpot` (each with
  p50/p90/p99/mean)
- `request_metrics.throughput.per_gpu` — `total_tput_tps`, `output_tput_tps`
- `request_accounting` — `records_error_dropped`, success/total counts
- `config` — full topology, concurrency, duration metadata

#### Our Smoke Test Script (Bypassing InferenceX)

Since Qwen3.8-Flash-Next is not yet in `amd-master.yaml`, our smoke test
script (`1002_local_test_qwen3.8next_mxfp4_mi355x_sglang-261002.sh`)
**bypasses** the InferenceX launcher pipeline and directly:

1. Submits a Slurm `sbatch` job with `srun --container-image=<squash>`
2. Mounts the InferenceX repo at `/workspace/` inside the container
3. Runs the per-model benchmark script
   (`benchmarks/single_node/agentic/qwen3.8next_mxfp4_mi355x_sglang_mtp.sh`)
   which sources `benchmark_lib.sh` and follows the same AIPerf flow
4. Collects results to a local log directory

This is equivalent to what the full InferenceX pipeline does, but without
the matrix generation, GitHub Actions, or srt-slurm recipe binding.

---

## 2. Prerequisites

### Hardware & Software

| Component          | Requirement                                                  |
| ------------------ | ------------------------------------------------------------ |
| **GPU**            | AMD Instinct MI355X (gfx950), 1 node = 1 GPU for TP1        |
| **Cluster**        | Slurm with Pyxis/Enroot plugin                              |
| **Container Runtime** | Enroot (squash files)                                     |
| **Model Weights**  | Pre-downloaded to `/it-share/hf-hub-cache/Qwen3.8-Flash-Next-MXFP4-PLEFP8` |
| **InferenceX**     | Cloned with `aiperf` submodule initialized                   |

### Model Weights

The Qwen3.8-Flash-Next MXFP4 checkpoint (~123 GB) must be available on the
cluster's shared storage. On our cluster it is pre-downloaded at:

```
/it-share/hf-hub-cache/Qwen3.8-Flash-Next-MXFP4-PLEFP8
```

If you need to download it yourself:

```bash
# Option 1: HuggingFace CLI (requires HF_TOKEN with gated model access)
huggingface-cli download fanwu103/Qwen3.8-Flash-Next-MXFP4-PLEFP8 \
  --local-dir /your/shared/storage/Qwen3.8-Flash-Next-MXFP4-PLEFP8

# Option 2: Git LFS
git lfs install
git clone https://huggingface.co/fanwu103/Qwen3.8-Flash-Next-MXFP4-PLEFP8 \
  /your/shared/storage/Qwen3.8-Flash-Next-MXFP4-PLEFP8
```

Then update `MODEL_HOST_DIR` in the test script to point to your local path.

### Repository Setup

```bash
# Clone InferenceX (if not already present)
git clone git@github.com:SemiAnalysisAI/InferenceX.git
cd InferenceX
git submodule update --init --recursive   # initializes utils/aiperf
```

A pre-cloned copy is included in this directory at `./InferenceX/`.

---

## 3. Build the Docker Image

The build system uses the `qwen_3_8_improvement` branches of both SGLang and
AITER. All FlyDSL kernel flags are baked into the image.

### Build Steps

```bash
# Unzip the build kit (located at /it-share/changliu/scripts/qwen3.8-flash-next/integration/)
unzip local_build-261002.zip
cd local_build

# Run the full build (takes ~1-2 hours)
# This clones qwen_3_8_improvement branches, builds from ROCm base image
./build.sh

# Output: sglang:local-rocm724-mi35x (Docker image)
# Logs: build.log, finalize.log
```

<details>
<summary><strong>What build.sh does under the hood</strong></summary>

1. Clones AITER and SGLang from `amd-nprotaso/aiter` and `amd-nprotaso/sglang` (`qwen_3_8_improvement` branches)
2. Builds the main image with `rocm.local.Dockerfile` (ROCm 7.2.4, PyTorch 2.11, Triton 3.7)
3. Runs `finalize.Dockerfile`: TileLang deps, import checks, FlyDSL entry-point validation
4. Bakes in env vars: `SGLANG_AITER_GDN_CONV=1`, `GDN_DECODE=1`, `HC_MIX=1`, `QSA_PA_DECODE=1`

</details>

### Create & Distribute Squash File

```bash
# Create enroot squash file from Docker image
enroot import -o /var/lib/squash/sglang_local-rocm724-mi35x-261002.sqsh \
  dockerd://sglang:local-rocm724-mi35x

# Copy to all compute nodes (run from head node)
# /var/lib/squash/ is node-local storage, NOT shared — you must copy to every node
for node in mia1-p01-g09 mia1-p01-g11 mia1-p01-g14 \
             mia1-p01-g15 mia1-p01-g16 mia1-p01-g18 \
             mia1-p01-g19 mia1-p01-g31 mia1-p01-g37 mia1-p01-g50; do
  scp /var/lib/squash/sglang_local-rocm724-mi35x-261002.sqsh \
    "$node:/var/lib/squash/" &
done
wait
```

---

## 4. Benchmark Configuration

### Server Launch Parameters

| Parameter                        | Value     | Notes                             |
| -------------------------------- | --------- | --------------------------------- |
| `--quantization`                 | `quark`   | MXFP4 via Quark                   |
| `--attention-backend`            | `aiter`   | AMD AITER kernels                 |
| `--tp-size`                      | `1`       | Single-GPU tensor parallelism     |
| `--mem-fraction-static`          | `0.90`    | KV cache memory allocation        |
| `--chunked-prefill-size`         | `16384`   | Prefill chunk size                |
| `--page-size`                    | `64`      | KV cache page size                |
| `--speculative-algorithm`        | `NEXTN`   | MTP speculative decoding          |
| `--speculative-num-steps`        | `3`       | Draft depth                       |
| `--speculative-num-draft-tokens` | `4`       | Total draft tokens                |
| `--speculative-eagle-topk`       | `1`       | Top-k for draft selection         |
| `--max-running-requests`         | `16`      | 2× concurrency                    |

### Environment Variables

| Variable                                   | Value | Purpose                                              |
| ------------------------------------------ | ----- | ---------------------------------------------------- |
| `SGLANG_EXACT_CHUNK_FILL`                  | `0`   | **Required workaround** — crashes without it (see §8) |
| `SGLANG_AITER_GDN_CONV`                    | `1`   | FlyDSL GDN prefill causal conv1d kernel              |
| `SGLANG_AITER_GDN_DECODE`                  | `1`   | FlyDSL GDN packed decode recurrence                  |
| `SGLANG_AITER_HC_MIX`                      | `1`   | FlyDSL hyper-connection mix kernel                   |
| `SGLANG_AITER_QSA_PA_DECODE`               | `1`   | FlyDSL BF16 paged decode for QSA layers             |
| `AITER_GDR_FLYDSL`                         | `1`   | Enable FlyDSL backend in AITER                       |
| `SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION` | `1`   | Force `mem_fraction_static=0.90` (no auto-scaling)   |

> ⚠️ **Warning:** `SGLANG_EXACT_CHUNK_FILL=0` is currently required. With the
> default value (1), the server crashes with `HSA_STATUS_ERROR_EXCEPTION`
> during 16k-token prefill batches. This workaround costs approximately 10-15%
> interactivity. We are actively testing a fix in the Oct 2 build.

### Creating Your Own Test Script

The reference script
`1002_local_test_qwen3.8next_mxfp4_mi355x_sglang-261002.sh` is included in
this directory. To create your own script (e.g. after building a new Docker
image), copy it and update the following variables:

```bash
cp 1002_local_test_qwen3.8next_mxfp4_mi355x_sglang-261002.sh \
   my_test_qwen3.8next_mxfp4_mi355x.sh
```

Edit the new script and update these key variables at the top:

| Variable          | What to change                                               | Example                                      |
| ----------------- | ------------------------------------------------------------ | -------------------------------------------- |
| `IMAGE`           | Your newly built Docker image name                           | `sglang:my-custom-build`                     |
| `SQUASH_FILE`     | Path to the enroot squash file created from your image       | `/var/lib/squash/sglang_my-custom-build.sqsh` |
| `MODEL_HOST_DIR`  | Path to the model weights on the host                        | `/your/shared/storage/Qwen3.8-Flash-Next-MXFP4-PLEFP8` |
| `INFERENCEX_DIR`  | Path to InferenceX repo (default: `/it-share/changliu/InferenceX`) | `/your/path/to/InferenceX`             |
| `LOG_DIR`         | Where to write output logs                                   | `/your/logs/qwen3.8-flash-next`              |

The environment variables (FlyDSL flags, `SGLANG_EXACT_CHUNK_FILL`, etc.)
and server launch parameters should generally stay the same unless you are
testing a specific change.

### Recommended Smoke Test Configurations

| Run type         | `DURATION` | `CONC_LIST` | Approx. wall time | Purpose                                |
| ---------------- | ---------- | ----------- | ------------------ | -------------------------------------- |
| **Single smoke** | `3600`     | `8`         | ~1.5 hours         | Quick validation at our baseline concurrency |
| **Sweep run**    | `3600`     | `8 12 16`   | ~4.5 hours         | Full Pareto sweep matching CI matrix   |

Example usage:

```bash
# Single smoke test (concurrency 8, 1-hour profiling)
CONC_LIST="8" DURATION=3600 bash 1002_local_test_qwen3.8next_mxfp4_mi355x_sglang-261002.sh

# Full sweep (concurrency 8, 12, 16 — runs sequentially)
CONC_LIST="8 12 16" DURATION=3600 bash 1002_local_test_qwen3.8next_mxfp4_mi355x_sglang-261002.sh
```

> ℹ️ Each concurrency point runs as a separate Slurm job sequentially (the
> script waits for one to finish before submitting the next). Wall time
> includes ~15 min overhead per point for model loading, CUDA graph capture,
> and warmup, plus the profiling duration itself.

---

## 5. Running the Benchmark

The smoke test script handles: Slurm job submission via enroot, server
startup, AIPerf trace replay, and result collection.

### Quick Start

```bash
# Run the benchmark (submits Slurm job, waits for completion)
bash 1002_local_test_qwen3.8next_mxfp4_mi355x_sglang-261002.sh

# Override defaults:
CONC_LIST="8 12 16" DURATION=900 bash 1002_local_test_*.sh

# Pin to a specific node:
SLURM_NODELIST=mia1-p01-g18 bash 1002_local_test_*.sh
```

<details>
<summary><strong>What happens during a run</strong></summary>

1. Script submits a Slurm job with enroot container (squash image)
2. Inside the container, the InferenceX benchmark script starts SGLang server
3. Server loads model weights, captures CUDA graphs, performs warmup
4. AIPerf replays Weka coding traces at the target concurrency for the specified duration
5. Results are written to JSON and server is stopped
6. Script repeats for each concurrency level in `CONC_LIST`

</details>

### Key Script Variables

You can override any of these via environment variables before running:

| Variable           | Default                                              | Description                |
| ------------------ | ---------------------------------------------------- | -------------------------- |
| `CONC_LIST`        | `8`                                                  | Concurrency levels to sweep |
| `DURATION`         | `3600`                                               | Profiling duration (seconds) |
| `TP`               | `1`                                                  | Tensor parallelism degree  |
| `IMAGE`            | `sglang:local-rocm724-mi35x`                         | Docker image name          |
| `SQUASH_FILE`      | `/var/lib/squash/sglang_local-rocm724-mi35x-261002.sqsh` | Enroot squash path     |
| `INFERENCEX_DIR`   | `/it-share/changliu/InferenceX`                      | InferenceX repo path       |
| `LOG_DIR`          | `/it-share/changliu/logs/qwen3.8-flash-next`         | Output log directory       |
| `SLURM_NODELIST`   | *(unset — Slurm picks)*                              | Pin to a specific node     |

### AIPerf Replay Parameters

| Parameter         | Value                                                |
| ----------------- | ---------------------------------------------------- |
| Scenario          | `agentx` (agentic coding trace replay)               |
| Dataset           | `semianalysis_cc_traces_weka_with_subagents_256k`    |
| Duration          | 3600s (canonical) or 900s (smoke test)               |
| Concurrency       | 8 (default), sweep [8, 12, 16]                       |
| Warmup            | 10 requests per lane, 1800s grace period             |
| MTP Acceptance    | `SGLANG_SIMULATE_ACC_LEN=2.32` (golden AL)           |
| Error threshold   | 10% max failed requests                              |

---

## 6. Reading Results

### Output Structure

```
logs/qwen3.8-flash-next/sglang_mxfp4_mtp_agentx_tp1_<TIMESTAMP>/
├── slurm_tp1_conc8.out              # Full Slurm job output
└── results_conc8/
    ├── server.log                   # SGLang server log
    ├── sglang_mxfp4_mtp_tp1_conc8_agentx.json   # Aggregate result
    ├── aiperf_artifacts/            # Raw AIPerf output
    │   ├── profile_export.jsonl     # Per-request records
    │   └── profile_export_aiperf.json
    ├── gpu_metrics.csv              # GPU utilization data
    └── metrics_plots.png            # Latency/throughput plots
```

### Extract Key Metrics

```python
import json

f = "results_conc8/sglang_mxfp4_mtp_tp1_conc8_agentx.json"
d = json.load(open(f))
rm = d["request_metrics"]
lat = rm["latency"]
pg = rm["throughput"]["per_gpu"]

print(f"Interactivity p50: {lat['intvty']['p50']:.1f} tok/s")
print(f"Interactivity p90: {lat['intvty']['p90']:.1f} tok/s")
print(f"ITL p50:           {lat['itl']['p50']*1000:.1f} ms")
print(f"TTFT p50:          {lat['ttft']['p50']*1000:.0f} ms")
print(f"Throughput/GPU:    {pg['total_tput_tps']:.0f} tok/s")
print(f"Requests:          {d['num_requests_successful']}/{d['num_requests_total']}")
print(f"Errors:            {d['request_accounting']['records_error_dropped']}")

# Spec accept rate (from server log)
# grep "accept rate" results_conc8/server.log | tail -3
```

The test script also prints a summary table at the end automatically.

---

## 7. Reference Results

All runs: Qwen3.8-Flash-Next MXFP4, MI355X, TP=1, Conc=8, AgentX replay
with MTP (NEXTN, 3 steps).

| Metric               | Baseline (Sep 25)    | qwen_3_8_improvement (Oct 1) | andy_pr latest (Oct 2) |
| --------------------- | -------------------- | ---------------------------- | ---------------------- |
| **Branch**            | andy_pr (old)        | qwen_3_8_improvement         | andy_pr (latest)       |
| **EXACT_CHUNK_FILL**  | 1 (default)          | 0 (workaround)               | 0 (workaround)         |
| **RoPE fusion**       | Yes                  | Yes                          | Yes                    |
| **FlyDSL flags**      | GDN conv only        | All enabled                  | All enabled            |
| **Interactivity p50** | 117 tok/s            | 105 tok/s                    | 101 tok/s              |
| **ITL p50**           | 9 ms                 | 10 ms                        | 10 ms                  |
| **TTFT p50**          | 666 ms               | 564 ms                       | 557 ms                 |
| **Throughput/GPU**    | 34,190 tok/s         | 34,266 tok/s                 | 33,555 tok/s           |
| **Spec Accept Rate**  | 44%                  | 44%                          | 44%                    |
| **Errors**            | 0                    | 2                            | 0                      |

> ℹ️ The Sep 25 baseline achieved 117 tok/s interactivity with
> `EXACT_CHUNK_FILL=1` (default). The ~15% gap in later runs is attributed
> to the `EXACT_CHUNK_FILL=0` workaround.

---

## 8. Troubleshooting

### HSA_STATUS_ERROR_EXCEPTION during prefill

Server crashes with SIGABRT during a 16384-token prefill batch. This is
caused by `EXACT_CHUNK_FILL=1` (default on gfx950).

**Fix:**
```bash
export SGLANG_EXACT_CHUNK_FILL=0
```

### FlyDSL HC_MIX SIGSEGV (if not using Oct 2 build)

The FlyDSL compiler crashes in `_infer_int_tuple_type` during CUDA graph
capture when `HC_MIX` is enabled on older builds.

**Fix:** The Oct 2 build (`local_build-261002`) includes the fix in the
`qwen_3_8_improvement` AITER branch. For older images, patch
`flydsl/expr/primitive.py`: change `return value.value` to
`return _expand_int_tuple_leaves(value.value)` in `_infer_int_tuple_type`.

### Squash file not found on compute node

Error: `No such file or directory` for the squash file. The
`/var/lib/squash/` directory is **node-local storage**, not shared.

**Fix:** Copy the squash file to the target compute node(s) via `scp` before
submitting the job, or use `SLURM_NODELIST` to pin to a node that already
has it. See §3 for the distribution script.

### Warmup failure / ProfileAborted

AIPerf aborts with "A root AgentX warmup request failed". This usually means
the server crashed during the first inference requests.

**Fix:** Check `results_concN/server.log` for the root cause (typically
`HSA_STATUS_ERROR_EXCEPTION` or FlyDSL SIGSEGV). Apply the corresponding
fix above.

---

## 9. File Inventory

| File                                                             | Description                          |
| ---------------------------------------------------------------- | ------------------------------------ |
| `README.md`                                                      | This guide                           |
| `1002_local_test_qwen3.8next_mxfp4_mi355x_sglang-261002.sh`     | Main smoke test script               |
| `InferenceX/`                                                    | InferenceX repo clone (with aiperf submodule) |

---

*Last updated: October 2, 2026 · Chang Liu · AMD AI Group*
