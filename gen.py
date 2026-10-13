"""Generate completions for all quality evals against an OpenAI-compatible vLLM endpoint.

usage: gen.py <label> <task>[,<task>...]   tasks: humaneval mbpp ifeval lcb spec
Writes ~/evals/out/<label>/<task>.jsonl (one row per sample, with usage + wall time).
"""
import asyncio, base64, json, os, pickle, sys, time, zlib

import httpx

BASE = os.environ.get("BASE", "http://127.0.0.1:8000/v1")
MODEL = "qwen3.8-27b"
CONC = int(os.environ.get("CONC", "64"))
ROOT = os.path.expanduser("~/evals")
sys.path.insert(0, os.path.join(ROOT, "harness"))

NOTHINK = {"temperature": 0.0, "max_tokens": 4096, "chat_template_kwargs": {"enable_thinking": False}}
THINK = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "chat_template_kwargs": {"enable_thinking": True}}

class _Safe(pickle.Unpickler):
    def find_class(self, module, name):  # LCB tests are pickled JSON strings; refuse any object construction
        raise pickle.UnpicklingError(f"blocked {module}.{name}")

def lcb_tests(row):
    raw = row["private_test_cases"]
    try:
        priv = json.loads(raw)
    except Exception:
        priv = json.loads(_Safe(__import__("io").BytesIO(zlib.decompress(base64.b64decode(raw)))).load())
    return json.loads(row["public_test_cases"]) + priv

def items(task):
    if task in ("humaneval", "mbpp"):
        from evalplus.data import get_human_eval_plus, get_mbpp_plus
        probs = get_human_eval_plus() if task == "humaneval" else get_mbpp_plus()
        for tid, p in probs.items():
            msg = ("Please provide a self-contained Python script that solves the following problem in a "
                   f"markdown code block:\n```\n{p['prompt'].strip()}\n```\n")
            yield tid, [{"role": "user", "content": msg}], NOTHINK, 0
    elif task == "ifeval":
        for l in open(f"{ROOT}/data/ifeval/ifeval_input_data.jsonl"):
            d = json.loads(l)
            yield d["key"], [{"role": "user", "content": d["prompt"]}], NOTHINK, 0
    elif task == "lcb":
        from lcb_prompt import lcb_messages
        for l in open(f"{ROOT}/data/lcb/test6.jsonl"):
            d = json.loads(l)
            yield d["question_id"], lcb_messages(d), dict(THINK, max_tokens=32768, seed=1), 0
    elif task == "spec":
        from specs import SPECS
        for s in SPECS:
            for k in range(3):
                yield s["id"], [{"role": "user", "content": s["prompt"]}], dict(THINK, max_tokens=24576, seed=100 + k), k

async def one(client, sem, task, key, msgs, params, k, fout):
    async with sem:
        body = {"model": MODEL, "messages": msgs, **params}
        for attempt in range(3):
            try:
                t0 = time.perf_counter()
                r = await client.post(f"{BASE}/chat/completions", json=body)
                r.raise_for_status()
                d = r.json()
                break
            except Exception as e:
                if attempt == 2:
                    print("FAILED", task, key, e, flush=True)
                    return
                await asyncio.sleep(5)
        wall = time.perf_counter() - t0
        m = d["choices"][0]["message"]
        row = {"key": key, "sample": k, "content": m.get("content") or "",
               "reasoning": m.get("reasoning_content") or m.get("reasoning") or "",
               "finish": d["choices"][0]["finish_reason"], "usage": d["usage"], "wall_s": round(wall, 2)}
        fout.write(json.dumps(row) + "\n")
        fout.flush()

async def main(label, tasks):
    os.makedirs(f"{ROOT}/out/{label}", exist_ok=True)
    async with httpx.AsyncClient(timeout=httpx.Timeout(7200.0)) as client:
        for task in tasks:
            sem = asyncio.Semaphore(CONC)
            path = f"{ROOT}/out/{label}/{task}.jsonl"
            done = set()
            if os.path.exists(path):
                done = {(json.loads(l)["key"], json.loads(l)["sample"]) for l in open(path)}
            todo = [x for x in items(task) if (x[0], x[3]) not in done][: int(os.environ.get("LIMIT", 10**9))]
            t0 = time.time()
            with open(path, "a") as fout:
                await asyncio.gather(*(one(client, sem, task, *x, fout) for x in todo))
            el = time.time() - t0
            rows = [json.loads(l) for l in open(path)]
            ctoks = sum(r["usage"]["completion_tokens"] for r in rows)
            meta = {"task": task, "label": label, "n": len(rows), "new": len(todo), "elapsed_s": round(el, 1),
                    "completion_tokens": ctoks, "concurrency": CONC}
            open(f"{ROOT}/out/{label}/{task}.meta.json", "w").write(json.dumps(meta))
            print(json.dumps(meta), flush=True)

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2].split(",")))
