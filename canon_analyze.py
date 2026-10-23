"""Does a draft's non-canonicality predict its quality, and where does the round trip's gain land?

Inputs: a refine_eval pickle with per-sample "canon" (share of level 0-5 codes the re-encode changes) and a
score_ppl.py JSON with per-sample perplexities of the final texts and, for refined variants, of the drafts.
Reports Spearman rank correlations (canon vs log draft-PPL, canon vs log-PPL gain of the round trip) and a
quintile table (drafts binned by canonicality: mean draft / refined PPL, share of samples the round trip improves).

Usage: python round_trip/canon_analyze.py --pkl round_trip/runs/p9_canon_ts_s1.pkl --scores round_trip/runs/p9_g2l_canon_ts_s1.json \
           --out round_trip/runs/p9_canon_ts_s1.json
"""
import argparse, json, pickle
import numpy as np
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--pkl", required=True)
ap.add_argument("--scores", required=True)
ap.add_argument("--bins", type=int, default=5)
ap.add_argument("--out", required=True)
a = ap.parse_args()

D = pickle.load(open(a.pkl, "rb"))["results"]
S = json.load(open(a.scores))["variants"]
out = {}
for name, r in D.items():
    if not r.get("canon") or name not in S:
        continue
    c = np.asarray(r["canon"], dtype=float)
    fin = np.log(np.asarray(S[name]["ppl_per_sample"], dtype=float))
    row = {"nfe": r["nfe"], "n": int(len(c)), "canon_mean": float(c.mean()), "canon_std": float(c.std())}
    has_draft = "draft_ppl_per_sample" in S[name]
    base = np.log(np.asarray(S[name]["draft_ppl_per_sample"], dtype=float)) if has_draft else fin
    rho, p = spearmanr(c, base)
    row["rho_canon_vs_logppl"], row["p_canon_vs_logppl"] = float(rho), float(p)
    if has_draft:
        gain = base - fin  # > 0: the round trip lowered this sample's perplexity
        rho_g, p_g = spearmanr(c, gain)
        row.update(rho_canon_vs_gain=float(rho_g), p_canon_vs_gain=float(p_g),
                   share_improved=float((gain > 0).mean()), mean_gain_logppl=float(gain.mean()))
    edges = np.quantile(c, np.linspace(0, 1, a.bins + 1))
    bins = np.clip(np.searchsorted(edges, c, side="right") - 1, 0, a.bins - 1)
    row["quintiles"] = []
    for b in range(a.bins):
        m = bins == b
        q = {"canon_lo": float(edges[b]), "canon_hi": float(edges[b + 1]), "n": int(m.sum()),
             "ppl": float(np.exp(fin[m].mean()))}
        if has_draft:
            q["draft_ppl"] = float(np.exp(base[m].mean()))
            q["share_improved"] = float((gain[m] > 0).mean())
        row["quintiles"].append(q)
    out[name] = row
    tag = f"  rho(canon, gain) {row['rho_canon_vs_gain']:+.3f}  improved {row['share_improved']:.0%}" if has_draft else ""
    print(f"{name:16s} n={row['n']}  canon {row['canon_mean']:.3f}  rho(canon, log ppl) {rho:+.3f} (p={p:.1e}){tag}")
    for q in row["quintiles"]:
        d = f"  draft {q['draft_ppl']:7.2f}  improved {q['share_improved']:.0%}" if has_draft else ""
        print(f"    canon {q['canon_lo']:.2f}-{q['canon_hi']:.2f}  n={q['n']:3d}  ppl {q['ppl']:7.2f}{d}")
json.dump(out, open(a.out, "w"), indent=1)
print(f"wrote {a.out}")
