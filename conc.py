"""Concurrency sweep: C parallel streams, ~8K-token code prompt each (unique prefix), 1024 tokens generated.

usage: conc.py <srcdir> 1,4,16,64  -> JSON line per level
"""
import asyncio, glob, json, os, random, statistics, sys, time, uuid

import httpx

HOST = os.environ.get("LLM_HOST", "127.0.0.1")
PORT = int(os.environ.get("LLM_PORT", "8000"))
MODEL = os.environ.get("LLM_MODEL", "qwen3.8-27b")
BASE = f"http://{HOST}:{PORT}/v1"
GEN, PROMPT_CHARS = 1024, 26000

files = sorted(glob.glob(sys.argv[1] + "/**/*.py", recursive=True))
corpus = [open(f, errors="ignore").read() for f in files]

def prompt(seed):
    rnd = random.Random(seed)
    text, out = "", []
    while len(text) < PROMPT_CHARS:
        text += "\n\n" + rnd.choice(corpus)
    return (f"[session {uuid.uuid4().hex}]\n```python\n{text[:PROMPT_CHARS]}\n```\n\n"
            "Review the code above: list the bugs and risky patterns you see, then propose fixes with code.")

async def stream(client, seed):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt(seed)}], "max_tokens": GEN,
            "ignore_eos": True, "temperature": 0.7, "top_p": 0.8, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
    t0 = time.perf_counter(); ttft = None; usage = None
    async with client.stream("POST", f"{BASE}/chat/completions", json=body) as r:
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            if ttft is None and d.get("choices") and d["choices"][0]["delta"].get("content"):
                ttft = time.perf_counter() - t0
    t1 = time.perf_counter()
    c = usage["completion_tokens"]
    return {"ttft": ttft, "end": t1, "start": t0, "ptok": usage["prompt_tokens"], "ctok": c,
            "decode": (c - 1) / (t1 - t0 - ttft)}

async def level(client, C):
    t0 = time.perf_counter()
    rs = await asyncio.gather(*(stream(client, hash((C, i, time.time()))) for i in range(C)))
    wall = max(r["end"] for r in rs) - t0
    return {"concurrency": C, "prompt_tokens": round(statistics.median(r["ptok"] for r in rs)),
            "agg_out_tok_s": round(sum(r["ctok"] for r in rs) / wall, 1),
            "per_stream_decode_med": round(statistics.median(r["decode"] for r in rs), 1),
            "ttft_med_s": round(statistics.median(r["ttft"] for r in rs), 2),
            "ttft_max_s": round(max(r["ttft"] for r in rs), 2), "wall_s": round(wall, 1)}

async def main():
    async with httpx.AsyncClient(timeout=httpx.Timeout(3600.0)) as client:
        await stream(client, 0)  # warmup
        for C in [int(x) for x in sys.argv[2].split(",")]:
            print(json.dumps(await level(client, C)), flush=True)

asyncio.run(main())
