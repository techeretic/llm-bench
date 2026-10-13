#!/bin/bash
# usage: serve-gguf.sh <quant>   e.g. UD-Q8_K_XL
# GGUF_DIR = directory with Qwen3.8-27B-<quant>.gguf and MTP/mtp-Qwen3.8-27B-Q4_0.gguf
# LLAMA_SERVER = path to a llama.cpp llama-server binary
D=${GGUF_DIR:?set GGUF_DIR}
exec ${LLAMA_SERVER:-llama-server} -m $D/Qwen3.8-27B-$1.gguf \
  --spec-type draft-mtp -md $D/MTP/mtp-Qwen3.8-27B-Q4_0.gguf --spec-draft-n-max 2 -ngld 999 \
  -ngl 999 -c 262144 -np 1 -fa on --jinja --alias qwen3.8-27b \
  --host 127.0.0.1 --port 8000 --metrics
