#!/usr/bin/env bash
# =============================================================================
# Local smoke test: Qwen3.8-Flash-Next MXFP4 (Quark) on MI355X via SGLang
# single-node agentic with NEXTN MTP (speculative decoding) — using AIPerf
# AgentX trace replay.
#
# 261002 build variant:
#   Built from qwen_3_8_improvement branches of amd-nprotaso/aiter and sglang.
#   Includes: GDN conv/decode kernels, FlyDSL BF16 paged decode for QSA,
#   HC mix kernels, PA decode, FlyDSL HC_MIX fix, EXACT_CHUNK_FILL=1 (default).
#   FlyDSL flags baked in via ENV directives; set explicitly here for safety.
#
# Runs the exact same CI benchmark script inside an enroot container, matching
# the launch_mi355x-amds.sh runner flow:
#   benchmarks/single_node/agentic/qwen3.8next_mxfp4_mi355x_sglang_mtp.sh
#
# CI config reference (main → qwen3.8-flash-next-mxfp4-mi355x-sglang-agentic-mtp):
#   image: aigmkt/qwen3.8-flash-next:v0.5.19-rocm724-mi35x-20260910-pr36601 (upstream v0.5.20)
#   model: fanwu103/Qwen3.8-Flash-Next-MXFP4-PLEFP8
#   runner: cluster:mi355x-amds
#   precision: mxfp4 (quark)
#   framework: sglang
#   dram-utilization: 0.80
#   spec-decoding: mtp (NEXTN, num-steps 3, draft-tokens 4, topk 1)
#   TP=4, EP=1: conc-list [8, 12, 16]
#   kv-offloading: none
#   kv-cache-dtype: auto
#   SGLANG_SIMULATE_ACC_LEN=2.32, method=match-expected, token-mode=real-draft-token
#
# No runtime patches needed — PR #36601 fixes are baked into the Docker image.
#
# Usage:
#   bash 0914_local_test_qwen3.8next_mxfp4_mi355x_sglang-agentic-mtp.sh
#   CONC_LIST="8 12" bash 0914_local_test_qwen3.8next_mxfp4_mi355x_sglang-agentic-mtp.sh
#   TP=8 EP_SIZE=8 NUM_GPUS=8 CONC_LIST="8" bash 0914_local_test_qwen3.8next_mxfp4_mi355x_sglang-agentic-mtp.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFERENCEX_DIR="${INFERENCEX_DIR:-/it-share/changliu/InferenceX}"

# ---- Model & image (GDN conv + HC mix kernel patches baked in) ----
IMAGE="${IMAGE:-sglang:local-rocm724-mi35x}"
MODEL="${MODEL:-fanwu103/Qwen3.8-Flash-Next-MXFP4-PLEFP8}"
MODEL_PREFIX="${MODEL_PREFIX:-qwen3.8-flash-next}"
PRECISION="${PRECISION:-mxfp4}"
FRAMEWORK="${FRAMEWORK:-sglang}"

# ---- Benchmark parameters (match CI sweep config: TP4/EP1 arm) ----
TP="${TP:-1}"
EP_SIZE="${EP_SIZE:-1}"
CONC_LIST="${CONC_LIST:-8}"
NUM_GPUS="${NUM_GPUS:-1}"
PORT="${PORT:-8888}"

# ---- KV offloading (none for the baseline run) ----
KV_OFFLOADING="${KV_OFFLOADING:-none}"
KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-}"
DRAM_UTILIZATION="${DRAM_UTILIZATION:-0.80}"
TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-2399}"

# ---- MTP (NEXTN for Qwen3.8 Flash Next) ----
SPEC_DECODING="${SPEC_DECODING:-mtp}"

# ---- Agentic replay settings ----
DURATION="${DURATION:-3600}"

# ---- Model weights ----
# The CI benchmark script resolves MODEL_PATH itself: when MODEL_PATH is unset,
# it runs `hf download "$MODEL"` (cache hit) then sets MODEL_PATH="$MODEL" (the
# HuggingFace model ID).  We mount the node-local HF cache.
# Model pre-downloaded to /it-share/hf-hub-cache/Qwen3.8-Flash-Next-MXFP4-PLEFP8.
HF_HUB_CACHE_MOUNT="${HF_HUB_CACHE_MOUNT:-/var/lib/hf-hub-cache/}"
MODEL_HOST_DIR="${MODEL_HOST_DIR:-/it-share/hf-hub-cache/Qwen3.8-Flash-Next-MXFP4-PLEFP8}"
MODEL_CONTAINER_DIR="/model_weights/Qwen3.8-Flash-Next-MXFP4-PLEFP8"

# ---- Slurm settings ----
PARTITION="${PARTITION:-compute}"
TIME_LIMIT="${TIME_LIMIT:-03:00:00}"
SLURM_EXCLUDE_NODES="${SLURM_EXCLUDE_NODES:-}"

# ---- Output ----
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-/it-share/changliu/logs/qwen3.8-flash-next}"
mkdir -p "$LOG_DIR"
BENCHMARK_LOGS_DIR="${BENCHMARK_LOGS_DIR:-${LOG_DIR}/sglang_mxfp4_mtp_agentx_tp${TP}_${TIMESTAMP}}"
mkdir -p "$BENCHMARK_LOGS_DIR"

# ---- Benchmark script (from InferenceX branch) ----
BENCHMARK_SCRIPT="benchmarks/single_node/agentic/qwen3.8next_${PRECISION}_mi355x_${FRAMEWORK}_mtp.sh"

# ---- Enroot squash file ----
SQUASH_FILE="${SQUASH_FILE:-/var/lib/squash/sglang_local-rocm724-mi35x-261002.sqsh}"

echo "======================================================"
echo " Qwen3.8-Flash-Next MXFP4 SGLang MTP AgentX Smoketest — MI355X"
echo "======================================================"
echo "  INFERENCEX_DIR:    $INFERENCEX_DIR"
echo "  Image:             $IMAGE"
echo "  Squash:            $SQUASH_FILE"
echo "  Model:             $MODEL"
echo "  Precision:         $PRECISION (quark)"
echo "  TP:                $TP"
echo "  EP:                $EP_SIZE"
echo "  SPEC_DECODING:     $SPEC_DECODING"
echo "  CONC_LIST:         $CONC_LIST"
echo "  KV_OFFLOADING:     $KV_OFFLOADING"
echo "  TOTAL_CPU_DRAM_GB: $TOTAL_CPU_DRAM_GB"
echo "  Duration:          ${DURATION}s"
echo "  HF_HUB_CACHE_MOUNT: $HF_HUB_CACHE_MOUNT"
echo "  Benchmark script:  $BENCHMARK_SCRIPT (from InferenceX branch)"
echo "  Logs:              $BENCHMARK_LOGS_DIR"
echo "======================================================"

# ---- Verify prerequisites ----
if [[ ! -d "$INFERENCEX_DIR/benchmarks" ]]; then
    echo "ERROR: InferenceX repo not found at $INFERENCEX_DIR" >&2
    exit 1
fi
if [[ ! -f "$INFERENCEX_DIR/$BENCHMARK_SCRIPT" ]]; then
    echo "ERROR: Benchmark script not found: $INFERENCEX_DIR/$BENCHMARK_SCRIPT" >&2
    exit 1
fi
if [[ ! -f "$INFERENCEX_DIR/utils/aiperf/setup.py" ]] && [[ ! -f "$INFERENCEX_DIR/utils/aiperf/pyproject.toml" ]]; then
    echo "ERROR: aiperf submodule not initialized. Run:" >&2
    echo "  git -C $INFERENCEX_DIR submodule update --init utils/aiperf" >&2
    exit 1
fi

_slurm_ok=false
if command -v sinfo >/dev/null 2>&1 && timeout 8 sinfo -N --noheader >/dev/null 2>&1; then
    _slurm_ok=true
fi
if [[ "$_slurm_ok" != true ]]; then
    echo "ERROR: Slurm is not reachable from this host." >&2
    exit 2
fi

# ---- Check squash file availability ----
if [[ ! -f "$SQUASH_FILE" ]]; then
    echo "WARNING: Squash file not found: $SQUASH_FILE"
    echo "  The image may need to be pulled and squashed first:"
    echo "    enroot import docker://${IMAGE}"
    echo "  Proceeding anyway (srun will fail if the image is unavailable)..."
fi

# ---- Run benchmark for each concurrency ----
_rc=0
for CONC in $CONC_LIST; do
    echo ""
    echo "==================== CONC=$CONC  TP=$TP  EP=$EP_SIZE  MTP (NEXTN) ===================="

    JOB_NAME="qwen3.8next-mxfp4-mtp-agentx-tp${TP}-c${CONC}"
    RESULT_FILENAME="sglang_mxfp4_mtp_tp${TP}_conc${CONC}_agentx"
    RESULT_DIR="${BENCHMARK_LOGS_DIR}/results_conc${CONC}"
    mkdir -p "$RESULT_DIR"

    EXCLUDE_FLAG=""
    if [[ -n "${SLURM_NODELIST:-}" ]]; then
        EXCLUDE_FLAG="--nodelist=$SLURM_NODELIST"
    elif [[ -n "$SLURM_EXCLUDE_NODES" ]]; then
        EXCLUDE_FLAG="--exclude=$SLURM_EXCLUDE_NODES"
    fi

    SLURM_OUT="${BENCHMARK_LOGS_DIR}/slurm_tp${TP}_conc${CONC}.out"

    echo "Submitting Slurm job..."
    set +e
    JOB_ID=$(sbatch \
        --partition="$PARTITION" \
        --gres="gpu:$NUM_GPUS" \
        --exclusive \
        --cpus-per-task=128 \
        --time="$TIME_LIMIT" \
        --job-name="$JOB_NAME" \
        $EXCLUDE_FLAG \
        --output="$SLURM_OUT" \
        --error="$SLURM_OUT" \
        --export=ALL \
        --wrap="
srun \
    --container-image=$SQUASH_FILE \
    --container-mounts=${INFERENCEX_DIR}:/workspace/,${HF_HUB_CACHE_MOUNT}:/root/.cache/huggingface/hub,${MODEL_HOST_DIR}:${MODEL_CONTAINER_DIR},${RESULT_DIR}:/result_dir \
    --container-writable \
    --container-workdir=/workspace/ \
    --container-remap-root \
    --no-container-entrypoint \
    env \
        MODEL='${MODEL_CONTAINER_DIR}' \
        MODEL_PREFIX='${MODEL_PREFIX}' \
        TP='${TP}' \
        EP_SIZE='${EP_SIZE}' \
        CONC='${CONC}' \
        DURATION='${DURATION}' \
        PORT='${PORT}' \
        KV_OFFLOADING='${KV_OFFLOADING}' \
        KV_OFFLOAD_BACKEND='${KV_OFFLOAD_BACKEND}' \
        TOTAL_CPU_DRAM_GB='${TOTAL_CPU_DRAM_GB}' \
        SPEC_DECODING='${SPEC_DECODING}' \
        RESULT_DIR='/result_dir' \
        RESULT_FILENAME='${RESULT_FILENAME}' \
        AGENTIC_OUTPUT_DIR='/result_dir' \
        EVAL_ONLY='false' \
        MODEL_PATH='${MODEL_CONTAINER_DIR}' \
        HF_HUB_CACHE='/root/.cache/huggingface/hub/' \
        SGLANG_AITER_GDN_CONV='1' \
        SGLANG_AITER_GDN_DECODE='1' \
        SGLANG_AITER_HC_MIX='1' \
        SGLANG_AITER_QSA_PA_DECODE='1' \
        AITER_GDR_FLYDSL='1' \
        SGLANG_AITER_HONOR_EXPLICIT_MEM_FRACTION='1' \
        SCHEDULER_RECV_INTERVAL='${SCHEDULER_RECV_INTERVAL:-10}' \
        STREAM_INTERVAL='${STREAM_INTERVAL:-50}' \
        CUDA_GRAPH_MAX_BS='${CUDA_GRAPH_MAX_BS:-${CONC}}' \
        TOKENIZER_WORKER_NUM='${TOKENIZER_WORKER_NUM:-}' \
    bash -c 'umask 000; exec bash /workspace/${BENCHMARK_SCRIPT}'
" 2>&1 | grep -oP '\d+')
    _sbatch_rc=$?
    set -e

    if [[ $_sbatch_rc -ne 0 ]] || [[ -z "${JOB_ID:-}" ]]; then
        echo "ERROR: sbatch failed (rc=$_sbatch_rc) for CONC=$CONC" >&2
        _rc=1
        continue
    fi

    echo "  Slurm job: $JOB_ID"
    echo "  Output:    $SLURM_OUT"

    echo "  Waiting for job $JOB_ID to finish..."
    while squeue --job "$JOB_ID" --noheader 2>/dev/null | grep -q "$JOB_ID"; do
        sleep 15
    done

    if [[ -f "$SLURM_OUT" ]]; then
        echo ""
        echo "---- Last 40 lines of Slurm output (CONC=$CONC) ----"
        tail -40 "$SLURM_OUT"
        echo "----------------------------------------------------"
    fi
done

# ---- Post-processing: summarize agentic results ----
echo ""
echo "================ Qwen3.8-Flash-Next MXFP4 MTP AgentX Results ================"
python3 - "$BENCHMARK_LOGS_DIR" "$NUM_GPUS" <<'PY'
import sys, json, glob, os

bld, gpus = sys.argv[1], int(sys.argv[2])

files = sorted(glob.glob(os.path.join(bld, "results_conc*", "*.json"), recursive=True))
files += sorted(glob.glob(os.path.join(bld, "*.json")))
if not files:
    print("No result JSONs found under", bld)
    sys.exit(0)

rows = []
seen_conc = set()
for f in files:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if not isinstance(d, dict):
        continue

    rm = d.get("request_metrics", {})
    if rm:
        conc = d.get("concurrency") or d.get("conc") or 0
        if not conc:
            conc = d.get("config", {}).get("concurrency", 0)
        if not conc or conc in seen_conc:
            continue
        seen_conc.add(conc)

        tput = rm.get("throughput", {})
        per_gpu = tput.get("per_gpu", {})
        total_tput_per_gpu = per_gpu.get("total_tput_tps", 0)
        output_tput_per_gpu = per_gpu.get("output_tput_tps", 0)

        if not total_tput_per_gpu:
            total_total = tput.get("total", {}).get("tokens_per_second", 0)
            total_tput_per_gpu = total_total / gpus if total_total else 0
        if not output_tput_per_gpu:
            output_total = tput.get("output", {}).get("tokens_per_second", 0)
            output_tput_per_gpu = output_total / gpus if output_total else 0

        latency = rm.get("latency", {})
        tpot = latency.get("tpot", {}).get("mean", 0)
        ttft = latency.get("ttft", {}).get("mean", 0)
        intvty = latency.get("intvty", {}).get("mean", 0)
        if not intvty and tpot:
            intvty = 1000.0 / tpot

        rows.append((conc, output_tput_per_gpu, total_tput_per_gpu, tpot, ttft, intvty, f))
        continue

    try:
        conc = int(d.get("max_concurrency") or d.get("concurrency") or 0)
    except Exception:
        conc = 0
    if conc == 0 or conc in seen_conc:
        continue
    seen_conc.add(conc)
    out = d.get("output_throughput", 0.0) or 0.0
    tot = d.get("total_token_throughput", 0.0) or 0.0
    tpot = d.get("mean_tpot_ms", 0.0) or 0.0
    ttft = d.get("mean_ttft_ms", 0.0) or 0.0
    interact = 1000.0 / tpot if tpot else 0.0
    rows.append((conc, out / gpus, tot / gpus, tpot, ttft, interact, f))

if not rows:
    print("No valid benchmark results found.")
    sys.exit(0)

rows.sort(key=lambda r: r[0])

print(f"\n  Qwen3.8-Flash-Next MXFP4 MTP (NEXTN) | TP={gpus} EP=1 | AgentX replay | {gpus} GPUs")
hdr = f"{'conc':>5} | {'interact (tok/s)':>16} | {'total/GPU (tok/s)':>17} | {'out/GPU (tok/s)':>15} | {'TPOT (ms)':>9} | {'TTFT (ms)':>9}"
print("  " + hdr)
print("  " + "-" * len(hdr))
for conc, out_gpu, tot_gpu, tpot, ttft, interact, _ in rows:
    print(f"  {conc:>5} | {interact:>16.1f} | {tot_gpu:>17.1f} | {out_gpu:>15.1f} | {tpot:>9.2f} | {ttft:>9.1f}")

print(f"\n  interactivity = 1000 / mean_TPOT_ms (or from aiperf intvty field)")
print(f"  throughput/GPU = total_token_throughput / {gpus} GPUs")
PY

echo ""
echo "================ MTP acceptance (from server logs) ================"
grep -aiEh "accept.?length|accept.?rate|speculative|spec.*accept|draft.?accept" \
    "$BENCHMARK_LOGS_DIR"/results_conc*/server.log \
    "$BENCHMARK_LOGS_DIR"/slurm_*.out 2>/dev/null | tail -12 \
    || echo "(no acceptance lines found — grep server logs manually)"

echo ""
echo "======================================================"
echo "Done. Results in: $BENCHMARK_LOGS_DIR"
echo "======================================================"

exit $_rc
