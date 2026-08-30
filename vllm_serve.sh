#!/usr/bin/env bash
# ==============================================================================
# vLLM Serving Launch Script for MOSS-Transcribe-Diarize
# ==============================================================================

set -euo pipefail

MODEL_ID=${MODEL_ID:-"OpenMOSS-Team/MOSS-Transcribe-Diarize"}
PORT=${PORT:-18000}
HOST=${HOST:-"0.0.0.0"}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-"0.90"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-"131072"}
DTYPE=${DTYPE:-"bfloat16"}
TP_SIZE=${TP_SIZE:-1}

echo "======================================================================"
echo " 🚀 正在啟動 vLLM 服務: ${MODEL_ID}"
echo "    監聽位址: http://${HOST}:${PORT}"
echo "    顯存佔用率: ${GPU_MEM_UTIL}"
echo "    最大長度: ${MAX_MODEL_LEN} (128k)"
echo "    資料型態: ${DTYPE}"
echo "======================================================================"

# Check if vllm command exists
if ! command -v vllm &> /dev/null; then
    echo "⚠️ 未檢測到 vllm 指令。請確保已安裝支援 MOSS 的 vLLM 版本："
    echo ""
    echo "  [CUDA 12 環境安裝]:"
    echo "  uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/68b4a1d582818e67adc903bf1b8fc5a5447da2fa/cu129"
    echo ""
    echo "  [CUDA 13 環境安裝]:"
    echo "  uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/68b4a1d582818e67adc903bf1b8fc5a5447da2fa/cu130"
    echo ""
fi

# Run vLLM serve
exec vllm serve "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --trust-remote-code \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --dtype "${DTYPE}" \
    --tensor-parallel-size "${TP_SIZE}"
