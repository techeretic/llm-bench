"""Score generated outputs. usage: score.py <label> <task>[,...]  -> out/<label>/<task>.scores.json (per-item)"""
import json, os, re, subprocess, sys

ROOT = os.path.expanduser("~/evals")
sys.path.insert(0, os.path.join(ROOT, "harness"))
sys.path.insert(0, os.path.join(ROOT, "lcb"))

def rows(label, task):
    return [json.loads(l) for l in open(f"{ROOT}/out/{label}/{task}.jsonl")]

def last_py_block(text):
    blocks = re.findall(r"```(?:python|py|Python)?[^\n]*\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else ""

def evalplus_score(label, task):
    from evalplus.sanitize import sanitize
    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    probs = get_human_eval_plus() if task == "humaneval" else get_mbpp_plus()
    d = f"{ROOT}/out/{label}/{task}_evalplus"
    os.makedirs(d, exist_ok=True)
    samples = f"{d}/samples.jsonl"
    with open(samples, "w") as f:
        for r in rows(label, task):
            code = sanitize(r["content"], probs[r["key"]]["entry_point"])
            f.write(json.dumps({"task_id": r["key"], "solution": code}) + "\n")
    subprocess.run([f"{ROOT}/venv/bin/evalplus.evaluate", "--dataset", task, "--samples", samples,
                    "--parallel", "14", "--i_just_wanna_run"], check=True, cwd=d,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    res = json.load(open(samples.replace(".jsonl", "_eval_results.json")))
    out = {}
    for tid, lst in res["eval"].items():
        e = lst[0]
        out[tid] = {"base": e["base_status"] == "pass", "plus": e["base_status"] == "pass" and e["plus_status"] == "pass"}
    return out

def ifeval_score(label):
    from lm_eval.tasks.ifeval.utils import process_results
    docs = {json.loads(l)["key"]: json.loads(l) for l in open(f"{ROOT}/data/ifeval/ifeval_input_data.jsonl")}
    out = {}
    for r in rows(label, "ifeval"):
        res = process_results(docs[r["key"]], [r["content"]])
        out[str(r["key"])] = {"prompt_strict": res["prompt_level_strict_acc"], "prompt_loose": res["prompt_level_loose_acc"],
                              "inst_strict": res["inst_level_strict_acc"], "inst_loose": res["inst_level_loose_acc"]}
    return out

def lcb_score(label):
    from gen import lcb_tests
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
    probs = {}
    for l in open(f"{ROOT}/data/lcb/test6.jsonl"):
        d = json.loads(l)
        probs[d["question_id"]] = d
    rs = rows(label, "lcb")
    samples, gens = [], []
    for r in rs:
        d = probs[r["key"]]
        tests = lcb_tests(d)
        fn = json.loads(d["metadata"]).get("func_name") if d["metadata"] else None
        samples.append({"input_output": json.dumps({"inputs": [t["input"] for t in tests],
                                                    "outputs": [t["output"] for t in tests], "fn_name": fn})})
        gens.append([last_py_block(r["content"])])
    metrics, results, _ = codegen_metrics(samples, gens, k_list=[1], num_process_evaluate=14, timeout=6)
    out = {}
    for i, r in enumerate(rs):
        res = results[i][0]
        out[r["key"]] = {"pass": bool(all(x is True or x == 1 for x in res)), "difficulty": probs[r["key"]]["difficulty"],
                         "finish": r["finish"], "has_code": bool(gens[i][0].strip())}
    return out

def spec_score(label):
    from specs import SPECS, check, words
    by = {s["id"]: s for s in SPECS}
    out = {}
    for r in rows(label, "spec"):
        c = check(by[r["key"]], r["content"])
        out[f'{r["key"]}#{r["sample"]}'] = {"checks": c, "passed": sum(c.values()), "total": len(c),
                                            "words": words(r["content"]), "finish": r["finish"]}
    return out

if __name__ == "__main__":
    label = sys.argv[1]
    for task in sys.argv[2].split(","):
        s = (evalplus_score(label, task) if task in ("humaneval", "mbpp") else ifeval_score(label) if task == "ifeval"
             else lcb_score(label) if task == "lcb" else spec_score(label))
        json.dump(s, open(f"{ROOT}/out/{label}/{task}.scores.json", "w"), indent=1)
        print(task, len(s), "scored", flush=True)
