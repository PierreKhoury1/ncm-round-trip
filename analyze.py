"""Score early-exit logs: gen-PPL per level, stop-policy curves, plot.

Usage:
  python round_trip/analyze.py --in round_trip/runs/ts_n256.pkl --ref gpt2        # fast
  python round_trip/analyze.py --in round_trip/runs/ts_n256.pkl --ref gpt2-large  # paper protocol
"""
import argparse, math, os, pickle, sys
from collections import Counter
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from evaluation.gen_ppl import generation_perplexity  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--ref", default="gpt2")
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--plot", default=None, help="png path; default next to input")
a = ap.parse_args()

D = pickle.load(open(a.inp, "rb"))
levels, R = D["levels"], D["records"]
K = len(levels)
N = len(R)
print(f"{N} samples, levels={levels}, ref={a.ref}")

tok = AutoTokenizer.from_pretrained(a.ref)
lm = AutoModelForCausalLM.from_pretrained(a.ref).to(a.device).eval()


def genppl(texts):
    with torch.no_grad():
        return generation_perplexity(lm, tok, [""] * len(texts), texts, device=a.device,
                                     score_only_generated=True, batch_size=a.bs).mean_ppl


def entropy(ids):
    c, m = Counter(ids), len(ids)
    return -sum(v / m * math.log(v / m) for v in c.values()) if m else 0.0


# ---------- 1. fixed-depth curve: stop every sample after level k ---------------------
print("\n== fixed depth (all samples stop at level k) ==")
print(f"{'k':>3} {'NFE':>4} {'genPPL':>9} {'entropy':>8} {'changed':>8} {'vs_final':>9} {'dec_conf':>9} {'gen_conf':>9}")
fixed = []
for k in range(K):
    texts = [r["levels"][k]["text"] for r in R]
    ppl = genppl(texts)
    ent = float(np.mean([entropy(r["levels"][k]["ids"]) for r in R]))
    ch = float(np.mean([r["levels"][k]["changed"] for r in R]))
    cf = float(np.mean([r["levels"][k]["changed_final"] for r in R]))
    dc = float(np.mean([r["levels"][k]["dec_conf"] for r in R]))
    gc = [r["levels"][k]["gen_conf"] for r in R if r["levels"][k]["gen_conf"] is not None]
    gc = float(np.mean(gc)) if gc else float("nan")
    fixed.append((k, k, ppl, ent))
    print(f"{k:3d} {k:4d} {ppl:9.2f} {ent:8.3f} {ch:8.3f} {cf:9.3f} {dc:9.3f} {gc:9.3f}")

# ---------- 2. per-sample stop policies on the logged signals -------------------------
def run_policy(stop_fn):
    """stop_fn(level_dict, k) -> bool. Returns (mean_nfe, genppl, frac_stopped_early)."""
    texts, nfes = [], []
    for r in R:
        chosen = r["levels"][-1]
        for k in range(1, K):
            if stop_fn(r["levels"][k], k):
                chosen = r["levels"][k]
                break
        texts.append(chosen["text"])
        nfes.append(chosen["nfe"])
    return float(np.mean(nfes)), genppl(texts), float(np.mean([n < K - 1 for n in nfes]))


policies = []
print("\n== policy: stop when decoder confidence >= tau ==")
for tau in (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99):
    nfe, ppl, fe = run_policy(lambda L, k, t=tau: L["dec_conf"] >= t)
    policies.append(("dec_conf", tau, nfe, ppl))
    print(f"tau={tau:.2f}  mean NFE={nfe:5.2f}  genPPL={ppl:8.2f}  stopped early={fe:.2f}")

print("\n== policy: stop when fraction of tokens changed vs previous level <= tau ==")
for tau in (0.02, 0.05, 0.08, 0.10, 0.15, 0.20):
    nfe, ppl, fe = run_policy(lambda L, k, t=tau: k >= 2 and L["changed"] <= t)
    policies.append(("changed", tau, nfe, ppl))
    print(f"tau={tau:.2f}  mean NFE={nfe:5.2f}  genPPL={ppl:8.2f}  stopped early={fe:.2f}")

print("\n== policy: stop when generator confidence >= tau ==")
for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
    nfe, ppl, fe = run_policy(lambda L, k, t=tau: L["gen_conf"] is not None and L["gen_conf"] >= t)
    policies.append(("gen_conf", tau, nfe, ppl))
    print(f"tau={tau:.2f}  mean NFE={nfe:5.2f}  genPPL={ppl:8.2f}  stopped early={fe:.2f}")

# ---------- 3. plot ------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
    ax.plot([f[1] for f in fixed[1:]], [f[2] for f in fixed[1:]], "o-", color="#444", label="fixed depth (every sample stops at k)")
    for name, c in (("dec_conf", "#1f77b4"), ("changed", "#d62728"), ("gen_conf", "#2ca02c")):
        pts = [(p[2], p[3]) for p in policies if p[0] == name]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "s--", color=c, label=f"adaptive: {name} threshold")
    ax.set_xlabel("mean generator forward passes (NFE)")
    ax.set_ylabel(f"gen-PPL under {a.ref}  (lower = better)")
    ax.set_yscale("log")
    ax.set_title(f"NCM TinyStories early exit, N={N}")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    out = a.plot or a.inp.replace(".pkl", f"_{a.ref}.png")
    fig.tight_layout(); fig.savefig(out)
    print(f"\nplot -> {out}")
except ImportError:
    print("matplotlib missing, no plot")
