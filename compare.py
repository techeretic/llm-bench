"""Paired comparison of two labels. usage: compare.py <A> <B> -> JSON summary on stdout"""
import json, math, os, statistics, sys

ROOT = os.path.expanduser("~/evals")
A, B = sys.argv[1:3]
ld = lambda l, t: json.load(open(f"{ROOT}/out/{l}/{t}.scores.json"))
meta = lambda l, t: json.load(open(f"{ROOT}/out/{l}/{t}.meta.json"))

def mcnemar(a, b):
    """Exact two-sided McNemar p on paired booleans."""
    only_a = sum(x and not y for x, y in zip(a, b)); only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    if n == 0:
        return only_a, only_b, 1.0
    k = min(only_a, only_b)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return only_a, only_b, round(p, 4)

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]

def paired(task, field, filt=None):
    sa, sb = ld(A, task), ld(B, task)
    keys = sorted(k for k in sa if k in sb and (filt is None or filt(sa[k])))
    a = [bool(sa[k][field]) for k in keys]; b = [bool(sb[k][field]) for k in keys]
    oa, ob, p = mcnemar(a, b)
    return {"n": len(keys), A: round(100 * sum(a) / len(keys), 1), B: round(100 * sum(b) / len(keys), 1),
            f"{A}_ci": wilson(sum(a), len(keys)), f"{B}_ci": wilson(sum(b), len(keys)),
            f"only_{A}": oa, f"only_{B}": ob, "p": p}

out = {"quality": {}, "speed": {}, "usage": {}}
q = out["quality"]
for t in ("humaneval", "mbpp"):
    q[f"{t}_base"] = paired(t, "base"); q[f"{t}_plus"] = paired(t, "plus")
q["lcb_all"] = paired("lcb", "pass")
for d in ("easy", "medium", "hard"):
    q[f"lcb_{d}"] = paired("lcb", "pass", lambda v, d=d: v["difficulty"] == d)
q["ifeval_prompt_strict"] = paired("ifeval", "prompt_strict")
q["ifeval_prompt_loose"] = paired("ifeval", "prompt_loose")
for l in (A, B):
    s = ld(l, "ifeval")
    inst = [x for v in s.values() for x in v["inst_strict"]]
    q.setdefault("ifeval_inst_strict", {})[l] = round(100 * sum(inst) / len(inst), 1)
    sp = ld(l, "spec")
    q.setdefault("spec_checks", {})[l] = f'{sum(v["passed"] for v in sp.values())}/{sum(v["total"] for v in sp.values())}'
    q.setdefault("spec_words_median", {})[l] = statistics.median(v["words"] for v in sp.values())
    fails = {}
    for k, v in sp.items():
        for c, ok in v["checks"].items():
            if not ok:
                fails.setdefault(c, []).append(k)
    q.setdefault("spec_failed_checks", {})[l] = fails
    lc = ld(l, "lcb")
    q.setdefault("lcb_length_capped", {})[l] = sum(v["finish"] == "length" for v in lc.values())

# token usage / wall time per task
for l in (A, B):
    for t in ("humaneval", "mbpp", "ifeval", "spec", "lcb"):
        rows = [json.loads(x) for x in open(f"{ROOT}/out/{l}/{t}.jsonl")]
        m = meta(l, t)
        out["usage"].setdefault(t, {})[l] = {
            "elapsed_s": m["elapsed_s"], "completion_tokens": m["completion_tokens"],
            "agg_tok_s": round(m["completion_tokens"] / m["elapsed_s"], 1),
            "median_completion_tokens": statistics.median(r["usage"]["completion_tokens"] for r in rows)}
    out["speed"][l] = [json.loads(x) for x in open(f"{ROOT}/out/{l}/conc.jsonl")]
print(json.dumps(out, indent=1))
