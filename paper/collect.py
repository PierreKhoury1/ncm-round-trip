"""Collect every number the paper reports from the run artifacts (no retyping).

gen-PPL and unigram entropy per seed come from the GPU score logs (analyze_skip.py / score_flm.py output);
corpus diversity (distinct-3, cross-sample 4-gram overlap, adjacent doubled words) is recomputed from the saved
samples with the definitions used during the study. Writes results.json next to this file.

Usage: python collect.py [RUNS_DIR]   (default: ../../ncm-adaptive/adaptive/runs)
"""
import json, os, pickle, re, sys
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "ncm-adaptive", "adaptive", "runs")
LOGS = os.path.join(RUNS, "gpu_logs")
CLS = 101  # LM1B: BERT [CLS] doubles as EOS; treated as a sentence break, not a word


def ngrams(seq, n):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def diversity(seqs):
    g3 = [g for s in seqs for g in ngrams(s, 3)]
    per = [set(ngrams(s, 4)) for s in seqs]
    cnt = Counter(g for p in per for g in p)
    xs = float(np.mean([np.mean([cnt[g] > 1 for g in p]) if p else 0 for p in per]))
    dup = 100 * float(np.mean([np.mean([a == b for a, b in zip(s, s[1:])]) if len(s) > 1 else 0 for s in seqs]))
    return {"dist3": len(set(g3)) / max(len(g3), 1), "xsamp4": xs, "dup": dup}


def parse_log(path):
    """Rows 'name NFE genPPL entropy ...' -> {name: (nfe, ppl, ent)}."""
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"^(\S+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)", line.strip())
        if m and m.group(1) != "schedule":
            out[m.group(1)] = (int(m.group(2)), float(m.group(3)), float(m.group(4)))
    return out


def add_set(store, name, pkls, logs, kind):
    S = store.setdefault(name, {})
    for pk, lg in zip(pkls, logs):
        if not (os.path.exists(pk) and os.path.exists(lg)):
            continue
        R = pickle.load(open(pk, "rb"))["results"]
        rows = parse_log(lg)
        for v, r in R.items():
            if v not in rows:
                continue
            nfe, ppl, ent = rows[v]
            seqs = [t.split() for t in r["texts"]] if kind == "text" else [[t for t in x if t != CLS] for x in r["raw"]]
            d = S.setdefault(v, {"nfe": nfe, "ppl": [], "ent": [], "dist3": [], "xsamp4": [], "dup": [], "changed": []})
            d["ppl"].append(ppl)
            d["ent"].append(ent)
            for k, val in diversity(seqs).items():
                d[k].append(val)
            if r.get("changed"):
                d["changed"].append(r["changed"])


def frontier(S, temps, ref_key="dist3"):
    """Per seed: gen-PPL of each variant / gen-PPL of the temperature curve interpolated at the same diversity."""
    out = {}
    n = min(len(S[t]["ppl"]) for t in temps if t in S)
    for v, d in S.items():
        rat = []
        for i in range(min(n, len(d["ppl"]))):
            xs = np.array([S[t][ref_key][i] for t in temps if t in S])
            ys = np.array([S[t]["ppl"][i] for t in temps if t in S])
            o = np.argsort(xs)
            x = d[ref_key][i]
            if xs[o][0] <= x <= xs[o][-1]:
                rat.append(d["ppl"][i] / float(np.interp(x, xs[o], ys[o])))
        d[f"vs_temp_{ref_key}"] = float(np.mean(rat) - 1) if rat else None
    return out


def summarize(store):
    for S in store.values():
        base = S.get("full")
        for v, d in S.items():
            for k in ("ppl", "ent", "dist3", "xsamp4", "dup"):
                if d[k]:
                    d[f"{k}_mean"] = float(np.mean(d[k]))
                    d[f"{k}_sd"] = float(np.std(d[k], ddof=1)) if len(d[k]) > 1 else 0.0
            if base and d["ppl"]:
                d["vs_full"] = d["ppl_mean"] / base["ppl_mean"] - 1
                d["seed_wins"] = int(sum(a < b for a, b in zip(d["ppl"], base["ppl"])))
            if d["changed"]:
                d["changed_mean"] = np.mean(np.array(d["changed"]), axis=0).tolist()


store = {}
S3 = (1, 2, 3)
P = lambda f: os.path.join(RUNS, f)
L = lambda *f: os.path.join(LOGS, *f)
TS_T = ["full", "t0.95@1", "t0.9@1", "t0.85@1", "t0.8@1", "t0.7@1"]
add_set(store, "ts_single", [P(f"gpu_var_s{s}.pkl") for s in S3], [L("refine", f"score_var_s{s}.log") for s in S3], "text")
add_set(store, "ts_more", [P(f"ts_more_s{s}.pkl") for s in S3], [L("phase1", f"score_ts_more_s{s}.log") for s in S3], "text")
for m in ("small", "large"):
    add_set(store, f"lm1b_{m}", [P(f"lm1b_{m}_s{s}.pkl") for s in S3], [L("lm1b2", f"score_lm1b_{m}_s{s}.log") for s in S3], "raw")
add_set(store, "lm1b_large_more", [P(f"lm1b_more_s{s}.pkl") for s in S3], [L("phase1", f"score_lm1b_more_s{s}.log") for s in S3], "raw")
for g in ("dist", "dist_soup", "ctrl", "ctrl_soup"):
    add_set(store, f"p2_{g}", [P(f"p2_{g}_s{s}.pkl") for s in S3], [L("phase2", f"score_p2_{g}_s{s}.log") for s in S3], "text")
for g in ("dist2k", "dist2k_soup", "ctrl2k", "ctrl2k_soup", "samplers"):
    add_set(store, f"p3_{g}", [P(f"p3_{g}_s{s}.pkl") for s in S3], [L("phase3", f"score_p3_{g}_s{s}.log") for s in S3], "text")
add_set(store, "p3_lm1b", [P(f"p3_lm1b_s{s}.pkl") for s in S3], [L("phase3", f"score_p3_lm1b_s{s}.log") for s in S3], "raw")
S456 = (4, 5, 6)
add_set(store, "p4_ts", [P(f"p4_ts_s{s}.pkl") for s in S456], [L("phase4", f"score_p4_ts_s{s}.log") for s in S456], "text")
add_set(store, "p4_grid", [P(f"p4_grid_s{s}.pkl") for s in S3], [L("phase4", f"score_p4_grid_s{s}.log") for s in S3], "text")
add_set(store, "p4_lm1bm", [P(f"p4_lm1bm_s{s}.pkl") for s in S3], [L("phase4", f"score_p4_lm1bm_s{s}.log") for s in S3], "raw")
add_set(store, "p4_lm1bl", [P(f"p4_lm1bl_s{s}.pkl") for s in S456], [L("phase4", f"score_p4_lm1bl_s{s}.log") for s in S456], "raw")
add_set(store, "p5_ts", [P(f"p5_ts_s{s}.pkl") for s in S456], [L("phase5", f"score_p5_ts_s{s}.log") for s in S456], "text")
add_set(store, "p5_guide", [P(f"p5_guide_s{s}.pkl") for s in S3], [L("phase5", f"score_p5_guide_s{s}.log") for s in S3], "text")
add_set(store, "p6_ts", [P(f"p6_ts_s{s}.pkl") for s in S456], [L("phase6", f"score_p6_ts_s{s}.log") for s in S456], "text")
add_set(store, "p6_lm1bl", [P(f"p6_lm1bl_s{s}.pkl") for s in S456], [L("phase6", f"score_p6_lm1bl_s{s}.log") for s in S456], "raw")

# multi-round TinyStories variants are compared against the single-round run's temperature curve (same seeds)
for v in TS_T:
    if v in store["ts_single"] and v != "full":
        store["ts_more"].setdefault(v, store["ts_single"][v])
summarize(store)
# one temperature curve for every TinyStories set: tau = 0.7..1.0 (single-round run) plus 1.1, 1.2 (sampler run,
# same seeds), so variants more diverse than the released sampler are compared too instead of being dropped
HOT = [t for t in ("t1.1@1", "t1.2@1") if t in store.get("p3_samplers", {})]
for name in ("ts_single", "ts_more", "p3_samplers"):
    if name in store and store[name]:
        for v in TS_T[1:]:
            store[name].setdefault(v, store["ts_single"][v])
        for v in HOT:
            store[name].setdefault(v, store["p3_samplers"][v])
        frontier(store[name], TS_T + HOT, "dist3")
        frontier(store[name], TS_T + HOT, "xsamp4")
if store.get("p4_grid"):  # tuning grid on seeds 1-3: same temperature curve as the other seed-1-3 TinyStories sets
    for v in TS_T[1:] + HOT:
        store["p4_grid"].setdefault(v, store["p3_samplers"][v] if v in store["p3_samplers"] else store["ts_single"][v])
    store["p4_grid"].setdefault("full", store["ts_single"]["full"])
    summarize({"g": store["p4_grid"]})
    frontier(store["p4_grid"], TS_T + HOT, "dist3")
    frontier(store["p4_grid"], TS_T + HOT, "xsamp4")
if store.get("p4_ts"):  # held-out seeds 4-6 carry their own full temperature curve
    frontier(store["p4_ts"], TS_T + ["t1.1@1", "t1.2@1"], "dist3")
    frontier(store["p4_ts"], TS_T + ["t1.1@1", "t1.2@1"], "xsamp4")
if store.get("p5_ts") and store.get("p4_ts"):  # grid winners on seeds 4-6: same seeds' temperature curve from run 14
    for v in TS_T[1:] + ["t1.1@1", "t1.2@1"]:
        store["p5_ts"].setdefault(v, store["p4_ts"][v])
    summarize({"h": store["p5_ts"]})
    frontier(store["p5_ts"], TS_T + ["t1.1@1", "t1.2@1"], "dist3")
    frontier(store["p5_ts"], TS_T + ["t1.1@1", "t1.2@1"], "xsamp4")
if store.get("p6_ts"):  # run 16 carries its own temperature curve (tau 0.6..1.3) on seeds 4-6
    P6T = ["full", "t0.6@1", "t0.7@1", "t0.8@1", "t0.85@1", "t0.9@1", "t0.95@1", "t1.1@1", "t1.2@1", "t1.3@1"]
    frontier(store["p6_ts"], P6T, "dist3")
    frontier(store["p6_ts"], P6T, "xsamp4")
if store.get("p6_lm1bl"):
    frontier(store["p6_lm1bl"], ["full", "t1.1@1", "t1.2@1"] + [v for v in ("t0.8@1", "t0.9@1") if v in store["p6_lm1bl"]], "ent")
if store.get("p5_guide"):  # guidance on seeds 1-3: seed-1-3 temperature curve
    for v in TS_T[1:] + HOT:
        store["p5_guide"].setdefault(v, store["p3_samplers"][v] if v in store["p3_samplers"] else store["ts_single"][v])
    summarize({"g": store["p5_guide"]})
    frontier(store["p5_guide"], TS_T + HOT, "dist3")
    frontier(store["p5_guide"], TS_T + HOT, "xsamp4")
for name in ("lm1b_small", "lm1b_large", "lm1b_large_more", "p3_lm1b", "p4_lm1bm", "p4_lm1bl"):
    if name in store and store[name]:
        if name in ("lm1b_large_more", "p3_lm1b"):
            for v in ("t0.9@1", "t0.85@1", "t0.8@1"):
                store[name].setdefault(v, store["lm1b_large"][v])
        frontier(store[name], ["full", "t0.9@1", "t0.85@1", "t0.8@1"], "ent")
for f, key in (("judge_ts.json", "judge_ts"), ("mauve_ts.json", "mauve_ts"), ("p5_sudoku_search_hard.json", "sudoku_search"),
               ("p8_var_screen.json", "var_screen"), ("p8_var_sanity.json", "var_sanity")):
    if os.path.exists(P(f)):
        store[key] = json.load(open(P(f)))
_sud = L("phase4", "score_p4_sudoku.log")
if os.path.exists(_sud):  # rows: variant NFE solved% solved/n cell% valid%
    store["sudoku_hard"] = {}
    for line in open(_sud, encoding="utf-8", errors="replace"):
        p = line.split()
        if len(p) == 6 and p[1].isdigit() and "/" in p[3]:
            store["sudoku_hard"][p[0]] = {"nfe": int(p[1]), "solved": float(p[2]), "solved_n": p[3],
                                          "cell": float(p[4]), "valid": float(p[5])}
    _sp = P("p4_sudoku_hard.pkl")
    if os.path.exists(_sp):  # paired exact McNemar test against the released greedy sampler
        from math import comb
        _D = pickle.load(open(_sp, "rb"))
        _ok = {v: [h == r for h, r in zip(x["texts"], _D["refs"])] for v, x in _D["results"].items()}
        for v, okv in _ok.items():
            b = sum(1 for f_, g_ in zip(_ok["full"], okv) if g_ and not f_)
            c = sum(1 for f_, g_ in zip(_ok["full"], okv) if f_ and not g_)
            n_ = b + c
            p_ = min(1.0, 2 * sum(comb(n_, j) for j in range(min(b, c) + 1)) / 2 ** n_) if n_ else 1.0
            store["sudoku_hard"][v].update({"new_solved": b, "new_broken": c, "mcnemar_p": p_})

# human-text reference (TinyStories validation openings, 3 x 256, word level) and its gen-PPL from the GPU log
real = pickle.load(open(P("real_ts.pkl"), "rb"))["results"]["real"]["texts"]
rd = [diversity([t.split() for t in real[256 * j:256 * (j + 1)]]) for j in range(3)]
rp = parse_log(L("lm1b", "score_real_ts.log"))
store["real_ts"] = {"real": {k: float(np.mean([x[k] for x in rd])) for k in rd[0]}}
store["real_ts"]["real"]["ppl_mean"] = next(iter(rp.values()))[1] if rp else None


def cond(path, cols):
    rows = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8", errors="replace"):
            p = line.split()
            if len(p) == len(cols) + 1 and re.match(r"^\d+$", p[1] or ""):
                rows[p[0]] = dict(zip(cols, [float(x) for x in p[1:]]))
    return rows


store["xsum"] = cond(L("phase1", "score_xsum.log"), ["nfe", "r1", "r2", "rl", "ppl", "words"])
store["wmt14"] = cond(L("phase1", "score_wmt14.log"), ["nfe", "bleu", "bleu_lc", "chrf", "len_ratio"])
json.dump(store, open(os.path.join(HERE, "results.json"), "w"), indent=1)

for name, S in store.items():
    print(f"\n== {name}")
    for v, d in S.items():
        if not isinstance(d, dict):
            print(f"  {v:16s} {d}")
        elif "ppl_mean" in d and d.get("ppl_sd") is not None:
            print(f"  {v:16s} NFE {d.get('nfe', ''):>3} ppl {d['ppl_mean']:7.2f} ± {d.get('ppl_sd', 0):5.2f}"
                  f"  ent {d.get('ent_mean', float('nan')):.3f}  d3 {d.get('dist3_mean', float('nan')):.3f}"
                  f"  xs4 {d.get('xsamp4_mean', float('nan')):.3f}  dup {d.get('dup_mean', float('nan')):.2f}"
                  f"  vsfull {100 * d.get('vs_full', 0):+5.1f}%  wins {d.get('seed_wins', '-')}"
                  f"  vsT@d3 {('%+.1f%%' % (100 * d['vs_temp_dist3'])) if d.get('vs_temp_dist3') is not None else '-':>7}"
                  f"  vsT@ent {('%+.1f%%' % (100 * d['vs_temp_ent'])) if d.get('vs_temp_ent') is not None else '-':>7}")
        else:
            print(f"  {v:16s} {d}")
