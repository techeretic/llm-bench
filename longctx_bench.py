"""Long-context bench: client TTFT, prefill rate, decode rate, spec-decode acceptance."""
import glob, http.client, json, random, re, sys, time, uuid

import os
HOST, PORT, MODEL = "127.0.0.1", 8000, "qwen3.8-27b"
LLAMA = os.environ.get("BACKEND") == "llama"
GEN = 1500

def post(path, body):
    c = http.client.HTTPConnection(HOST, PORT, timeout=600)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    return json.loads(c.getresponse().read())

def ntok(text):
    if LLAMA:
        return len(post("/tokenize", {"content": text})["tokens"])
    return post("/tokenize", {"model": MODEL, "prompt": text})["count"]

def metrics():
    if LLAMA:
        return {}
    c = http.client.HTTPConnection(HOST, PORT, timeout=30)
    c.request("GET", "/metrics")
    out = {}
    for line in c.getresponse().read().decode().splitlines():
        m = re.match(r'^(vllm:spec_decode_num_(?:drafts|draft_tokens|accepted_tokens))_total\{[^}]*\} (\S+)', line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out

# Real code as filler (coding-agent workload); shuffle files per run so no two prompts share a prefix.
files = sorted(glob.glob(sys.argv[1] + "/**/*.py", recursive=True))
corpus = [open(f, errors="ignore").read() for f in files]

def build(target, seed):
    rnd = random.Random(seed)
    order = corpus[:]
    rnd.shuffle(order)
    parts, total = [], 0
    for src in order:
        parts.append(src)
        total += len(src) // 3  # rough; refined below
        if total > target * 1.3:
            break
    text = "\n\n".join(parts)
    lo, hi = 0, len(text)
    while hi - lo > 2000:  # binary search on chars to hit target tokens
        mid = (lo + hi) // 2
        if ntok(text[:mid]) < target:
            lo = mid
        else:
            hi = mid
    return text[:lo]

def run(target, seed):
    nonce = uuid.uuid4().hex
    body = build(target, seed) if target else ""
    prompt = (f"[session {nonce}]\n\n" + (f"```python\n{body}\n```\n\n" if body else "")
              + "Explain in detail how the code above works, module by module, "
                "then write a thorough set of pytest unit tests for it.")
    if not body:
        prompt = f"[session {nonce}]\nImplement an LRU cache in Python with thorough pytest tests and explain the design."
    req = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": GEN,
           "ignore_eos": True, "temperature": 0.7, "top_p": 0.8, "stream": True,
           "stream_options": {"include_usage": True},
           "chat_template_kwargs": {"enable_thinking": False}}
    m0 = metrics()
    c = http.client.HTTPConnection(HOST, PORT, timeout=900)
    t0 = time.perf_counter()
    c.request("POST", "/v1/chat/completions", json.dumps(req), {"Content-Type": "application/json"})
    r = c.getresponse()
    ttft = None; usage = None; chunks = 0; timings = None
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        d = json.loads(line[6:])
        if d.get("timings"):
            timings = d["timings"]
        if d.get("usage"):
            usage = d["usage"]
        if d.get("choices") and d["choices"][0]["delta"].get("content"):
            if ttft is None:
                ttft = time.perf_counter() - t0
            chunks += 1
    t1 = time.perf_counter()
    m1 = metrics()
    ptok, ctok = usage["prompt_tokens"], usage["completion_tokens"]
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
    if LLAMA and timings and timings.get("draft_n"):
        # llama.cpp reports drafted/accepted tokens; accept_len approximated as tokens per verify step
        d = {"vllm:spec_decode_num_draft_tokens": timings["draft_n"],
             "vllm:spec_decode_num_accepted_tokens": timings["draft_n_accepted"],
             "vllm:spec_decode_num_drafts": ctok - timings["draft_n_accepted"]}
    drafts = d.get("vllm:spec_decode_num_drafts", 0)
    acc = d.get("vllm:spec_decode_num_accepted_tokens", 0)
    dtok = d.get("vllm:spec_decode_num_draft_tokens", 0)
    res = {"target": target, "prompt_tokens": ptok, "completion_tokens": ctok,
           "ttft_s": round(ttft, 3), "prefill_tok_s": round(ptok / ttft),
           "decode_tok_s": round((ctok - 1) / (t1 - t0 - ttft), 1),
           "accept_len": round(1 + acc / drafts, 2) if drafts else None,
           "draft_accept_rate": round(acc / dtok, 3) if dtok else None}
    print(json.dumps(res), flush=True)
    return res

if __name__ == "__main__":
    targets = [int(x) for x in sys.argv[2].split(",")]
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    run(0, 0)  # warmup
    for t in targets:
        for i in range(reps):
            run(t, hash((t, i, time.time())))
