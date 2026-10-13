# llm-bench

Scripts for benchmarking a self-hosted LLM served through an OpenAI-compatible endpoint, either
vLLM or llama.cpp's `llama-server`. They answer two questions about a model and serving
configuration: **how fast is it under real workloads** (long-context coding agents, several agents
at once), and **how good is it for coding, following instructions and writing technical specs**.

Written to compare a BF16 + MTP (multi-token prediction speculative decoding) deployment of
Qwen3.8-27B against Unsloth's NVFP4 and GGUF quants, but nothing is specific to that model apart
from the default model name and the serve scripts.

## Principles

- **Measure the workload, not a synthetic one.** Prompts are built from real Python source, because
  coding agents send large code contexts. Each prompt gets a random nonce prefix so prefix caching
  can't make TTFT look better than it is.
- **Measure from the client.** TTFT and decode rate are timed on the streaming client. Server
  counters are only used for what the client can't see, like the speculative-decoding acceptance
  rate from vLLM's `/metrics` or llama-server's `timings`.
- **Same server flags for every variant.** Only the weights change (for example, the same
  `--max-num-seqs`, fp8 KV cache and MTP setting on both vLLM variants). Otherwise you're comparing
  configs, not quants.
- **Grade by execution or by rules, never by eye.** Code runs against hidden tests (EvalPlus,
  LiveCodeBench). Instructions are checked by IFEval's rule checkers. Spec formatting is checked by
  the rules in `specs.py`.
- **Paired comparisons with enough items to mean something.** The same prompts go to every model,
  and differences are reported per item (pairs where only one model passed) with significance. A
  12-task quick check (`quality.py`) proved too easy to tell variants apart, which is why the bigger
  suites exist.
- **Subjective quality is judged blind.** Spec quality beyond formatting is judged pairwise, with the
  X/Y labels randomized by `blind.py`. The judge doesn't open `key.json` until the verdicts are
  written down.
- **Benchmark the baseline in place when you can.** If the baseline is already serving, run against
  it without downtime. Only candidate models need the server swapped.

## Suites

| Suite | What it measures | Size | Mode | Grading |
|---|---|---|---|---|
| `longctx_bench.py` | TTFT, prefill tok/s, single-stream decode tok/s, speculative-decoding acceptance at 32K/64K/125K prompts | 3 runs × 3 lengths + warmup | thinking off, 1500 tokens with `ignore_eos` | timing |
| `conc.py` | Total and per-stream decode tok/s and TTFT at 1/4/16/64/128 parallel streams, ~6.6K-token prompts | 1 burst per level | thinking off, 1024 tokens with `ignore_eos` | timing |
| HumanEval+ / MBPP+ (`gen.py`, `score.py`) | Basic function-level coding | 164 + 378 | thinking off, greedy | EvalPlus base and plus tests |
| LiveCodeBench v6 | Hard competitive programming (Jan–Apr 2025 problems, 43 easy / 52 medium / 80 hard) | 175 | thinking on, temp 0.6, 32K max tokens | LCB hidden tests |
| IFEval | Following verifiable instructions (length, format, keywords, ...) | 541 | thinking off, greedy | lm-eval IFEval checkers, strict and loose |
| Spec writing (`specs.py`) | Writing a technical spec with exact section order, required tables, diagrams, RFC 2119 wording and word limits | 6 prompts × 3 samples | thinking on, temp 0.6 | rule checks + blind pairwise judgement |
| `quality.py` | Quick check: 12 coding tasks + needle retrieval at 32K/125K | small | greedy | asserts / exact match |

Speed runs use `ignore_eos` so every run generates the same number of tokens.

## Layout

```
~/evals/
  venv/            uv venv: evalplus, lm-eval[ifeval], datasets, httpx, matplotlib
  lcb/             git clone of LiveCodeBench (its testing_util grades LCB)
  data/lcb/test6.jsonl         livecodebench/code_generation_lite (release v6 additions)
  data/ifeval/                 google/IFEval
  harness/         this repo's *.py
  out/<label>/     <task>.jsonl (raw generations + usage), <task>.meta.json, <task>.scores.json, conc.jsonl
```

Setup:

```bash
mkdir -p ~/evals && cd ~/evals
uv venv -p 3.12 venv
uv pip install -p venv/bin/python evalplus "lm-eval[ifeval]" datasets httpx numpy matplotlib
git clone --depth 1 https://github.com/LiveCodeBench/LiveCodeBench.git lcb
hf download livecodebench/code_generation_lite test6.jsonl --repo-type dataset --local-dir data/lcb
hf download google/IFEval --repo-type dataset --local-dir data/ifeval
mkdir -p harness && cp <this repo>/*.py harness/ && cp <this repo>/run-all.sh .
```

## Configuration

Every script, including the serve scripts, reads the endpoint and model name from the environment:

| Variable | Meaning | Default |
|---|---|---|
| `LLM_HOST` | Host of the OpenAI-compatible server | `127.0.0.1` |
| `LLM_PORT` | Its port | `8000` |
| `LLM_MODEL` | Served model name sent in requests (and set by the serve scripts) | `qwen3.8-27b` |

The bench and the server it measures should normally run on the same machine, so network latency
doesn't end up in the TTFT numbers.

## Running

```bash
export LLM_HOST=127.0.0.1 LLM_PORT=8000 LLM_MODEL=my-model

# full suite, results under out/<label>/
CORPUS=/path/to/some/python/package nohup ~/evals/run-all.sh baseline > ~/evals/run-baseline.log 2>&1 &

# long-context speed only
python -I longctx_bench.py /path/to/some/python/package 32000,64000,125000 3
BACKEND=llama python -I longctx_bench.py ...      # against llama-server

# candidate model: stop the baseline server, then
MODEL_DIR=/path/to/nvfp4-checkpoint VLLM_PY=/path/to/vllm-venv/bin/python ./serve-nvfp4.sh
GGUF_DIR=/path/to/gguf LLAMA_SERVER=/path/to/llama-server ./serve-gguf.sh UD-Q4_K_XL
# ... run-all.sh candidate ... then restore the baseline server

# blind spec judging
python blind.py baseline candidate ~/evals/blind   # read pair_*.md, write verdicts, only then open key.json
```

`gen.py` resumes: rerunning it skips (key, sample) pairs already in the output. `LIMIT=2` runs a
smoke test. `CONC` sets client concurrency (64 by default; `run-all.sh` uses 48 for LCB, because
thinking runs can each grow to 32K tokens and oversubscribe the KV cache).

## Gotchas

- **vLLM's first NVFP4 start JIT-compiles FlashInfer FP4 GEMMs** (cached in `~/.cache/flashinfer`).
  The compile uses a lot of RAM and can be OOM-killed if something else heavy, like a llama.cpp
  build, runs at the same time. Let it compile alone, with `MAX_JOBS` capped.
- **Don't use `pkill -f` / `pgrep -f` over ssh.** The pattern matches the ssh shell's own command
  line and kills it. Use bracket patterns (`grep "[a]pi_server"`).
- **LCB private tests are zlib + base64 pickles.** `gen.lcb_tests` unpickles with an unpickler that
  refuses every class (the payload is just a JSON string), so a tampered dataset can't run code.
- **Generated code is executed** by EvalPlus and LCB (with their timeouts and process isolation) on
  the machine running the scorer. Run the scorers from `/tmp`, not from inside the harness directory.
- **Shared-server noise.** If the baseline also serves real traffic, check
  `vllm:num_requests_running` before starting and note it in the results.
- **Building llama.cpp with CUDA** needs `nvcc` on `PATH` (e.g. `export PATH=/usr/local/cuda/bin:$PATH`),
  or cmake fails with "No CMAKE_CUDA_COMPILER".
- **Make serve scripts executable** (`chmod +x`) or launch them with `bash`; a permission error kills the launch instantly and is easy to miss if you're only polling for the port.
- **EvalPlus needs every problem** in a samples file. A smoke test with `LIMIT` can't be scored
  for humaneval/mbpp.

## Files

| File | Purpose |
|---|---|
| `longctx_bench.py` | Long-context single-stream speed (vLLM or `BACKEND=llama`) |
| `conc.py` | Concurrency sweep |
| `gen.py` | Async generation for humaneval, mbpp, ifeval, lcb, spec |
| `lcb_prompt.py` | LiveCodeBench generic prompt (copied from lcb_runner) |
| `specs.py` | Spec-writing prompts and their rule checks |
| `score.py` | Grading: EvalPlus, IFEval, LCB, spec checks → per-item `scores.json` |
| `blind.py` | Builds randomized X/Y spec pairs for blind judging |
| `compare.py` | Paired comparison of two labels: pass rates, Wilson 95% CIs, exact McNemar p, token usage, wall time, concurrency rows |
| `compare_charts.py` | matplotlib charts from `compare.py` output plus the unblinded spec verdicts |
| `quality.py` | 12-task + needle quick check |
| `longctx_charts.py` | matplotlib charts from `longctx_bench.py` outputs (edit the `VARIANTS` paths) |
| `run-all.sh` | Concurrency sweep → generation → scoring for one label |
| `serve-nvfp4.sh`, `serve-gguf.sh` | Launch candidate models with flags matching the baseline |
