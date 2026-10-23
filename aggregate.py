"""Aggregate analyze_skip.py logs across seeds and generators -> mean +/- std table + plot.

Usage:
  python round_trip/aggregate.py --logs "round_trip/runs/gpu_logs/score_*_s*.log"
Log names must look like score_<generator>_s<seed>.log.
"""
import argparse, glob, os, re
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--logs", required=True, help="glob of analyze_skip output logs")
ap.add_argument("--plot", default="round_trip/runs/skip_vs_nfe.png")
a = ap.parse_args()

row_re = re.compile(r"^(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\[")
data = defaultdict(lambda: defaultdict(list))  # gen -> schedule -> [(nfe, ppl, ent)]
for f in sorted(glob.glob(a.logs)):
    m = re.search(r"score_(.+)_s(\d+)\.log$", os.path.basename(f))
    if not m:
        continue
    gen = m.group(1)
    for line in open(f, encoding="utf-8", errors="replace"):
        r = row_re.match(line.strip())
        if r:
            name, nfe, ppl, ent = r.group(1), int(r.group(2)), float(r.group(3)), float(r.group(4))
            data[gen][name].append((nfe, ppl, ent))

if not data:
    raise SystemExit("no rows parsed")

gens = sorted(data)
scheds = sorted({s for g in gens for s in data[g]}, key=lambda s: (-data[gens[0]].get(s, [(0,)])[0][0], s))
print(f"{'schedule':16s} {'NFE':>4} | " + " | ".join(f"{g:>22s}" for g in gens))
print("-" * (24 + 25 * len(gens)))
summary = defaultdict(dict)
for s in scheds:
    nfe = next(data[g][s][0][0] for g in gens if s in data[g])
    cells = []
    for g in gens:
        v = data[g].get(s)
        if not v:
            cells.append(f"{'-':>22s}")
            continue
        ppl = np.array([x[1] for x in v]); ent = np.array([x[2] for x in v])
        sd = ppl.std(ddof=1) if len(ppl) > 1 else 0.0
        summary[g][s] = (nfe, ppl.mean(), sd, ent.mean(), len(ppl))
        cells.append(f"{ppl.mean():8.2f} ±{sd:5.2f} e{ent.mean():.2f} n{len(ppl)}")
    print(f"{s:16s} {nfe:4d} | " + " | ".join(cells))

# relative cost vs each generator's own full schedule, and vs the released full schedule
ref_full = summary.get("released", {}).get("full")
print("\nrelative genPPL vs released/full (1.00 = baseline quality):")
for g in gens:
    parts = []
    for s in scheds:
        if s in summary[g] and ref_full:
            parts.append(f"{s}={summary[g][s][1] / ref_full[1]:.2f}")
    print(f"  {g:10s} " + "  ".join(parts))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 4.3), dpi=150)
    colors = {"released": "#444444", "drop00": "#9ecae1", "drop03": "#fc9272", "c00": "#1f77b4", "c03": "#d62728"}
    for g in gens:
        pts = sorted(((v[0], v[1], v[2]) for v in summary[g].values()))
        # best (lowest PPL) schedule at each NFE = Pareto frontier
        best = {}
        for nfe, m, sd in pts:
            if nfe not in best or m < best[nfe][0]:
                best[nfe] = (m, sd)
        xs = sorted(best)
        ax.errorbar(xs, [best[x][0] for x in xs], yerr=[best[x][1] for x in xs], fmt="o-",
                    color=colors.get(g, None), capsize=3, label=g)
    ax.set_xlabel("generator forward passes (NFE)")
    ax.set_ylabel("gen-PPL (lower = better), best schedule per NFE")
    ax.set_yscale("log")
    ax.grid(alpha=.3)
    ax.legend()
    ax.set_title("NCM TinyStories: level skipping, released vs fine-tuned")
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.plot), exist_ok=True)
    fig.savefig(a.plot)
    print(f"\nplot -> {a.plot}")
except ImportError:
    pass
