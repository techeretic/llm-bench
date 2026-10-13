"""Small quality probe: coding tasks graded by executing asserts + long-context needle retrieval."""
import glob, http.client, json, os, random, re, subprocess, sys, tempfile, uuid

HOST = os.environ.get("LLM_HOST", "127.0.0.1")
PORT = int(os.environ.get("LLM_PORT", "8000"))
MODEL = os.environ.get("LLM_MODEL", "qwen3.8-27b")

def chat(prompt, max_tokens=2048):
    c = http.client.HTTPConnection(HOST, PORT, timeout=900)
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.0, "seed": 0, "chat_template_kwargs": {"enable_thinking": False}}
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    return json.loads(c.getresponse().read())["choices"][0]["message"]["content"] or ""

TASKS = [
 ("roman_to_int(s) converting a Roman numeral string to an int",
  "assert roman_to_int('MCMXCIV')==1994; assert roman_to_int('LVIII')==58; assert roman_to_int('IV')==4"),
 ("merge_intervals(iv) merging overlapping [start,end] intervals, returning a sorted list of lists",
  "assert merge_intervals([[1,3],[2,6],[8,10],[15,18]])==[[1,6],[8,10],[15,18]]; assert merge_intervals([[1,4],[4,5]])==[[1,5]]; assert merge_intervals([])==[]"),
 ("lis_length(a) returning the length of the longest strictly increasing subsequence in O(n log n)",
  "assert lis_length([10,9,2,5,3,7,101,18])==4; assert lis_length([0,1,0,3,2,3])==4; assert lis_length([7,7,7])==1; assert lis_length([])==0"),
 ("eval_rpn(tokens) evaluating a Reverse Polish Notation list of string tokens with + - * / where / truncates toward zero",
  "assert eval_rpn(['2','1','+','3','*'])==9; assert eval_rpn(['4','13','5','/','+'])==6; assert eval_rpn(['10','6','9','3','+','-11','*','/','*','17','+','5','+'])==22"),
 ("min_window(s, t) returning the minimum window substring of s containing all chars of t (with multiplicity), or '' if none",
  "assert min_window('ADOBECODEBANC','ABC')=='BANC'; assert min_window('a','aa')==''; assert min_window('aa','aa')=='aa'"),
 ("topo_sort(n, edges) returning a topological order (list) of nodes 0..n-1 for directed edges [u,v] meaning u before v, or None if there is a cycle",
  "o=topo_sort(4,[[0,1],[1,2],[0,3],[3,2]]); assert o is not None and all(o.index(u)<o.index(v) for u,v in [[0,1],[1,2],[0,3],[3,2]]) and sorted(o)==[0,1,2,3]; assert topo_sort(2,[[0,1],[1,0]]) is None"),
 ("edit_distance(a, b) returning the Levenshtein distance",
  "assert edit_distance('horse','ros')==3; assert edit_distance('intention','execution')==5; assert edit_distance('','abc')==3"),
 ("parse_duration(s) parsing strings like '1h30m', '45s', '2h5s', '3m' into total seconds as int",
  "assert parse_duration('1h30m')==5400; assert parse_duration('45s')==45; assert parse_duration('2h5s')==7205; assert parse_duration('3m')==180"),
 ("class LRUCache(capacity) with get(key) returning -1 if missing and put(key, value), evicting least recently used",
  "c=LRUCache(2); c.put(1,1); c.put(2,2); assert c.get(1)==1; c.put(3,3); assert c.get(2)==-1; c.put(4,4); assert c.get(1)==-1; assert c.get(3)==3 and c.get(4)==4"),
 ("count_islands(grid) counting 4-connected islands of '1' in a list of lists of '0'/'1' strings",
  "assert count_islands([['1','1','0','0','0'],['1','1','0','0','0'],['0','0','1','0','0'],['0','0','0','1','1']])==3; assert count_islands([])==0"),
 ("next_permutation(a) returning the next lexicographic permutation of list a as a new list (wrapping to sorted ascending at the end)",
  "assert next_permutation([1,2,3])==[1,3,2]; assert next_permutation([3,2,1])==[1,2,3]; assert next_permutation([1,1,5])==[1,5,1]; assert next_permutation([1,3,2])==[2,1,3]"),
 ("int_to_words(n) converting a non-negative int below one billion to English words like 'One Hundred Twenty Three' ('Zero' for 0)",
  "assert int_to_words(123)=='One Hundred Twenty Three'; assert int_to_words(12345)=='Twelve Thousand Three Hundred Forty Five'; assert int_to_words(1000010)=='One Million Ten'; assert int_to_words(0)=='Zero'"),
]

def code_tasks():
    passed = 0
    for spec, tests in TASKS:
        out = chat(f"Write a Python function/class {spec}. Reply with only one ```python code block, no tests, no explanation.")
        m = re.search(r"```(?:python)?\n(.*?)```", out, re.S)
        code = m.group(1) if m else out
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(code + "\n\n" + tests.replace("; ", "\n") + "\n")
        ok = subprocess.run([sys.executable, "-I", f.name], capture_output=True, timeout=30).returncode == 0
        passed += ok
    return passed

def needles(target, n_trials=3):
    files = sorted(glob.glob(sys.argv[1] + "/**/*.py", recursive=True))
    found = 0
    for t in range(n_trials):
        rnd = random.Random(1000 + t)
        srcs = [open(f, errors="ignore").read() for f in rnd.sample(files, min(len(files), 1500))]
        text = "\n\n".join(srcs)[: int(target * 3.2)]
        keys = {f"DEPLOY_TOKEN_{i}": uuid.UUID(int=rnd.getrandbits(128)).hex[:12] for i in range(3)}
        lines = text.split("\n")
        for i, (k, v) in enumerate(keys.items()):
            pos = int(len(lines) * (0.1 + 0.4 * i))
            lines.insert(pos, f'{k} = "{v}"  # do not rotate')
        prompt = ("```python\n" + "\n".join(lines) + "\n```\n\nWhat are the exact values of DEPLOY_TOKEN_0, "
                  "DEPLOY_TOKEN_1 and DEPLOY_TOKEN_2 in the code above? Answer as three lines NAME=value.")
        out = chat(prompt, 200)
        found += sum(v in out for v in keys.values())
    return found, n_trials * 3

if __name__ == "__main__":
    res = {"code_pass": code_tasks(), "code_total": len(TASKS)}
    for tgt in (32000, 125000):
        f, n = needles(tgt)
        res[f"needle_{tgt//1000}k"] = f"{f}/{n}"
    print(json.dumps(res))
