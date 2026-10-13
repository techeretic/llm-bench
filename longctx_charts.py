"""Render long-context benchmark charts from longctx_bench.py JSONL outputs (edit VARIANTS paths)."""
import json, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
# Fixed categorical order from the reference palette (light mode).
VARIANTS = [
    ("BF16 (current)", "/tmp/longctx_bench.out", "#2a78d6"),
    ("NVFP4 (vLLM)", "/tmp/bench-nvfp4.out", "#eb6834"),
    ("UD-Q8_K_XL", "/tmp/bench-UD-Q8_K_XL.out", "#1baf7a"),
    ("UD-Q6_K_XL", "/tmp/bench-UD-Q6_K_XL.out", "#eda100"),
    ("UD-Q4_K_XL", "/tmp/bench-UD-Q4_K_XL.out", "#e87ba4"),
]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MARKERS = ["o", "s", "^", "D", "v"]

def load(path):
    rows = [json.loads(l) for l in open(path) if l.startswith("{")]
    by = {}
    for r in rows:
        by.setdefault(r["target"], []).append(r)
    return by

data = {name: load(p) for name, p, _ in VARIANTS}
summary = {}
for name, by in data.items():
    summary[name] = {t: {k: statistics.median(r[k] for r in rs) for k in
                         ("ttft_s", "prefill_tok_s", "decode_tok_s", "accept_len", "prompt_tokens")}
                     for t, rs in by.items()}
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 13,
                     "axes.titlelocation": "left"})
CTX = [32000, 64000, 125000]

def line_chart(key, title, ylabel, fname, baseline=None, fmt="{:.0f}"):
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=96, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ends = []
    for i, (name, _, color) in enumerate(VARIANTS):
        s = summary[name]
        ys = [s[t][key] for t in CTX if t in s]
        xs = [t / 1000 for t in CTX if t in s]
        ax.plot(xs, ys, color=color, lw=2, marker=MARKERS[i], ms=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=name, zorder=3)
        ends.append([ys[-1], xs[-1]])
    # dodge end labels so near-equal values don't overprint
    ends.sort()
    gap = ax.get_ylim()[1] * 0.05
    pos = [e[0] for e in ends]
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + gap)
    for (v, x), p in zip(ends, pos):
        ax.annotate(fmt.format(v), (x, v), xytext=(x + 2.5, p), textcoords="data",
                    va="center", fontsize=10, color=INK2)
    if baseline:
        bx, by_, blabel = baseline
        ax.plot(bx, by_, color="#8a8984", lw=1.5, ls="--", marker="o", ms=6, label=blabel, zorder=2)
    ax.set_title(title, pad=12)
    ax.set_xlabel("Prompt length (K tokens)")
    ax.set_ylabel(ylabel)
    ax.set_xticks([32, 64, 125])
    ax.set_xlim(25, 140)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", facecolor=SURFACE)
    plt.close(fig)

line_chart("decode_tok_s", "Decode speed vs prompt length (single stream, MTP n=2)", "tokens / s",
           "decode.png", baseline=([32, 125], [28, 27], "BF16, no MTP (2026-10-11)"))
line_chart("ttft_s", "Time to first token vs prompt length", "seconds", "ttft.png",
           baseline=([32, 125], [6.1, 33], "BF16, no MTP (2026-10-11)"), fmt="{:.1f}s")
line_chart("prefill_tok_s", "Prefill throughput vs prompt length", "prompt tokens / s", "prefill.png",
           baseline=([32, 125], [5334, 3791], "BF16, no MTP (2026-10-11)"), fmt="{:,.0f}")

# Short-prompt decode bars (warmup run = target 0)
fig, ax = plt.subplots(figsize=(8, 3.4), dpi=96, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
names = [n for n, _, _ in VARIANTS]
vals = [summary[n][0]["decode_tok_s"] for n in names]
cols = [c for _, _, c in VARIANTS]
ax.set_axisbelow(True)
bars = ax.barh(names[::-1], vals[::-1], color=cols[::-1], height=0.6, edgecolor=SURFACE, linewidth=2)
for b, v in zip(bars, vals[::-1]):
    ax.text(v + 2, b.get_y() + b.get_height() / 2, f"{v:.0f} tok/s", va="center", fontsize=10, color=INK2)
ax.axvline(28.5, color="#8a8984", ls="--", lw=1.5)
ax.text(29.5, 4.45, "BF16 without MTP: 28.5", fontsize=9, color=INK2)
ax.set_title("Decode speed, short prompt (1,500 tokens generated)", pad=12)
ax.set_xlim(0, max(vals) * 1.18)
ax.set_ylim(-0.5, 4.8)
ax.grid(axis="x", color=GRID, lw=0.8)
for sp in ("top", "right", "bottom"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(f"{OUT}/decode_short.png", facecolor=SURFACE)
plt.close(fig)
print(json.dumps(summary)[:3000])
