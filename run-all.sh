#!/bin/bash
# usage: run-all.sh <label>
# Runs the concurrency sweep, all quality generations, then scoring, against the server on :8000.
# CORPUS: directory of .py files used as prompt filler (we used an installed vllm package dir).
L=$1; cd ~/evals; mkdir -p out/$L
SRC=${CORPUS:?set CORPUS to a directory of .py source files}
venv/bin/python -I harness/conc.py $SRC 1,4,16,64,128 > out/$L/conc.jsonl 2>out/$L/conc.err
CONC=64 venv/bin/python -I harness/gen.py $L humaneval,mbpp,ifeval,spec
CONC=48 venv/bin/python -I harness/gen.py $L lcb
cd /tmp && ~/evals/venv/bin/python -I ~/evals/harness/score.py $L humaneval,mbpp,ifeval,spec,lcb
echo ALLDONE
