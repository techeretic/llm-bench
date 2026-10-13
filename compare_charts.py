"""Charts for a two-label quality+speed comparison. usage: compare_charts.py <compare.json> <judge.json> <outdir>
Labels are hard-coded below as bf16 / nvfp4; edit A, B and NAMES for other runs."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

cmp_, judge, OUT = json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), sys.argv[3]
A, B = "bf16", "nvfp4"
NAMES = {A: "BF16 (current)", B: "NVFP4"}
COL = {A: "#2a78d6", B: "#eb6834"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 13,
                     "axes.titlelocation": "left"})

def style(ax, grid="y"):
    ax.set_facecolor(SURFACE); ax.set_axisbelow(True)
    ax.grid(axis=grid, color=GRID, lw=0.8)
    for sp in ("top", "right", "left" if grid == "y" else "bottom"):
        ax.spines[sp].set_visible(False)

# 1. quality bars with 95% CI
q = cmp_["quality"]
items = [("HumanEval+\nbase", "humaneval_base"), ("HumanEval+\n+extra tests", "humaneval_plus"),
         ("MBPP+\nbase", "mbpp_base"), ("MBPP+\n+extra tests", "mbpp_plus"),
         ("LiveCodeBench\nv6 (all)", "lcb_all"), ("IFEval\nstrict", "ifeval_prompt_strict"),
         ("IFEval\nloose", "ifeval_prompt_loose")]
fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=96, facecolor=SURFACE); style(ax)
x = np.arange(len(items)); w = 0.38
for i, m in enumerate((A, B)):
    vals = [q[k][m] for _, k in items]
    lo = [q[k][m] - q[k][f"{m}_ci"][0] for _, k in items]; hi = [q[k][f"{m}_ci"][1] - q[k][m] for _, k in items]
    bars = ax.bar(x + (i - 0.5) * w, vals, w, color=COL[m], label=NAMES[m], edgecolor=SURFACE, linewidth=2,
                  yerr=[lo, hi], error_kw={"ecolor": "#8a8984", "elinewidth": 1, "capsize": 3})
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, 3, f"{v:.1f}", ha="center", va="bottom", fontsize=8.5,
                color="white", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([n for n, _ in items], fontsize=9)
ax.set_ylim(0, 105); ax.set_ylabel("% passed")
ax.set_title("Coding and instruction-following accuracy (whiskers = 95% CI)", pad=12)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1))
fig.tight_layout(); fig.savefig(f"{OUT}/quality.png", facecolor=SURFACE); plt.close(fig)

# 2. LCB by difficulty
fig, ax = plt.subplots(figsize=(7, 3.8), dpi=96, facecolor=SURFACE); style(ax)
diffs = ["easy", "medium", "hard"]; x = np.arange(3)
for i, m in enumerate((A, B)):
    vals = [q[f"lcb_{d}"][m] for d in diffs]
    bars = ax.bar(x + (i - 0.5) * w, vals, w, color=COL[m], label=NAMES[m], edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9, color=INK2)
ax.set_xticks(x); ax.set_xticklabels([f"{d} (n={q[f'lcb_{d}']['n']})" for d in diffs])
ax.set_ylim(0, 112); ax.set_ylabel("% solved")
ax.set_title("LiveCodeBench v6 by difficulty (thinking on)", pad=12)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/lcb.png", facecolor=SURFACE); plt.close(fig)

# 3. concurrency: aggregate throughput, per-stream decode, TTFT
sp = cmp_["speed"]
fig, axs = plt.subplots(1, 3, figsize=(12, 3.9), dpi=96, facecolor=SURFACE)
for ax, key, title, fmt in ((axs[0], "agg_out_tok_s", "Total output tok/s", "{:.0f}"),
                            (axs[1], "per_stream_decode_med", "Per-stream decode tok/s", "{:.0f}"),
                            (axs[2], "ttft_med_s", "Median TTFT (s)", "{:.1f}")):
    style(ax)
    ends = {m: sp[m][-1][key] for m in (A, B)}
    close = abs(ends[A] - ends[B]) < 0.08 * max(ends.values())
    for m, mk in ((A, "o"), (B, "s")):
        cs = [r["concurrency"] for r in sp[m]]; ys = [r[key] for r in sp[m]]
        ax.plot(cs, ys, color=COL[m], lw=2, marker=mk, ms=7, markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=NAMES[m])
        ax.annotate(fmt.format(ys[-1]), (cs[-1], ys[-1]),
                    xytext=(6, (7 if ys[-1] >= ends[A if m == B else B] else -7) if close else 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    ax.set_xscale("log", base=2); ax.set_xticks([1, 4, 16, 64, 128]); ax.set_xticklabels(["1", "4", "16", "64", "128"])
    ax.set_xlim(0.7, 220); ax.set_ylim(bottom=0); ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel("parallel streams")
axs[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("Concurrency sweep: ~6.5K-token code prompts, 1,024 tokens generated each", x=0.01, ha="left",
             fontweight="bold", fontsize=13, color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/concurrency.png", facecolor=SURFACE); plt.close(fig)

# 4. eval wall-clock per suite
u = cmp_["usage"]
suites = [("HumanEval+", "humaneval"), ("MBPP+", "mbpp"), ("IFEval", "ifeval"), ("Spec writing", "spec"),
          ("LiveCodeBench", "lcb")]
fig, ax = plt.subplots(figsize=(8, 3.6), dpi=96, facecolor=SURFACE); style(ax, grid="x")
y = np.arange(len(suites))[::-1]; h = 0.38
for i, m in enumerate((A, B)):
    vals = [u[k][m]["elapsed_s"] / 60 for _, k in suites]
    bars = ax.barh(y + (0.5 - i) * h, vals, h, color=COL[m], label=NAMES[m], edgecolor=SURFACE, linewidth=2)
    for b, v, (_, k) in zip(bars, vals, suites):
        ax.text(v + 0.6, b.get_y() + b.get_height() / 2, f"{v:.1f} min  ({u[k][m]['agg_tok_s']:.0f} tok/s)",
                va="center", fontsize=8.5, color=INK2)
ax.set_yticks(y); ax.set_yticklabels([n for n, _ in suites])
ax.set_xlim(0, max(u["lcb"][m]["elapsed_s"] for m in (A, B)) / 60 * 1.35)
ax.set_xlabel("wall-clock minutes (same prompts, same client concurrency)")
ax.set_title("Time to run each benchmark", pad=12)
ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.62))
fig.tight_layout(); fig.savefig(f"{OUT}/walltime.png", facecolor=SURFACE); plt.close(fig)

# 5. spec judging
t = judge["tally"]
fig, ax = plt.subplots(figsize=(8, 1.9), dpi=96, facecolor=SURFACE); style(ax, grid="x")
left = 0
for lab, n, c in ((f"BF16 better: {t['bf16']}", t["bf16"], COL[A]), (f"Tie: {t['tie']}", t["tie"], "#c9c8c3"),
                  (f"NVFP4 better: {t['nvfp4']}", t["nvfp4"], COL[B])):
    ax.barh(0, n, left=left, color=c, edgecolor=SURFACE, linewidth=2, height=0.6)
    ax.text(left + n / 2, 0, lab, ha="center", va="center", fontsize=10,
            color="white" if c != "#c9c8c3" else INK, fontweight="bold")
    left += n
ax.set_xlim(0, left); ax.set_yticks([]); ax.set_xticks([])
ax.set_title("Blind pairwise spec judgement (18 pairs)", pad=10)
fig.tight_layout(); fig.savefig(f"{OUT}/spec_judge.png", facecolor=SURFACE); plt.close(fig)
print("ok")
