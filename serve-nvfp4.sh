#!/bin/bash
# usage: serve-nvfp4.sh   (VLLM_PY = python of a vLLM venv, MODEL_DIR = NVFP4 checkpoint)
# Same flags as the BF16 baseline; only the weights differ.
exec ${VLLM_PY:-python} -m vllm.entrypoints.openai.api_server \
  --model ${MODEL_DIR:?set MODEL_DIR} --dtype auto --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.87 --max-model-len 262144 --attention-backend FLASHINFER \
  --enable-prefix-caching --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --max-num-seqs 480 --port ${LLM_PORT:-8000} --host ${LLM_HOST:-127.0.0.1} --served-model-name ${LLM_MODEL:-qwen3.8-27b} \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
