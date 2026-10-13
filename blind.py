"""Build blinded pairwise spec comparisons. usage: blind.py <labelA> <labelB> <outdir>
Writes pair_<n>.md (prompt + Spec X + Spec Y) and key.json (which label is X). Judge without reading key.json."""
import json, os, random, sys

ROOT = os.path.expanduser("~/evals")
a, b, outdir = sys.argv[1:4]
os.makedirs(outdir, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "harness"))
from specs import SPECS
prompts = {s["id"]: s["prompt"] for s in SPECS}
load = lambda l: {(r["key"], r["sample"]): r["content"] for r in map(json.loads, open(f"{ROOT}/out/{l}/spec.jsonl"))}
A, B = load(a), load(b)
rnd = random.Random(os.urandom(8))
key, n = {}, 0
for k in sorted(A):
    if k not in B:
        continue
    n += 1
    x_is_a = rnd.random() < 0.5
    X, Y = (A[k], B[k]) if x_is_a else (B[k], A[k])
    key[f"pair_{n:02d}"] = {"spec": k[0], "sample": k[1], "X": a if x_is_a else b, "Y": b if x_is_a else a}
    with open(f"{outdir}/pair_{n:02d}.md", "w") as f:
        f.write(f"# pair_{n:02d}\n\n## PROMPT\n\n{prompts[k[0]]}\n\n=====================  SPEC X  =====================\n\n{X}\n\n"
                f"=====================  SPEC Y  =====================\n\n{Y}\n")
json.dump(key, open(f"{outdir}/key.json", "w"), indent=1)
print(n, "pairs")
