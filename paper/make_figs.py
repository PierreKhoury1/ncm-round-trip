"""Figures (figs/*.pdf) and LaTeX table bodies (tables/*.tex) from results.json. Missing runs are skipped."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False, "pdf.fonttype": 42})
C_T, C_R, C_M, C_D, C_A = "#8a8a8a", "#2b6cb0", "#1a365d", "#dd6b20", "#c53030"


def label(v):
    """Variant id -> paper notation."""
    if v == "full":
        return r"released ($\tau{=}1$)"
    if v.startswith("t") and v.endswith("@1"):
        return rf"$\tau{{=}}{v[1:-2]}$"
    parts, extra = [], []
    for p in v.split("+"):
        if p[0] in "rnq":
            k, n = (p[1:].split("x") + ["1"])[:2]
            name = {"r": "RT", "n": "Resample", "q": "LatentRQ"}[p[0]]
            parts.append(rf"{name}$_{{{k}}}$" + (rf"$\times{n}$" if n != "1" else ""))
        elif p[0] == "d":
            extra.append(rf"draft $\tau{{=}}{p[1:]}$")
        elif p[0] == "t":
            T, k = p[1:].split("@")
            extra.append(rf"$\tau_{{\geq {k}}}{{=}}{T}$")
    base, ext = "+".join(parts), ", ".join(extra)
    return base + (", " + ext if base and ext else ext)


def get(S, v):
    return R.get(S, {}).get(v)


def pct(x):
    return "--" if x is None else f"{100 * x:+.1f}\\%"


# ---------------- Figure: TinyStories frontier (gen-PPL vs corpus distinct-3) ----------------
def ts_frontier():
    fig, ax = plt.subplots(figsize=(5.2, 3.3))
    S = R["ts_single"]
    T = ["t0.7@1", "t0.8@1", "t0.85@1", "t0.9@1", "t0.95@1", "full"]
    pts = [(S[t]["dist3_mean"], S[t]["ppl_mean"]) for t in T]
    if get("p3_samplers", "t1.1@1"):
        for t in ("t1.1@1", "t1.2@1"):
            d = get("p3_samplers", t)
            pts.append((d["dist3_mean"], d["ppl_mean"]))
    pts.sort()
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", color=C_T, ms=3.5, lw=1.2, label=r"temperature sweep ($\tau$)")
    for t in T[:-1] + [t for t in ("t1.1@1", "t1.2@1") if get("p3_samplers", t)]:
        d = S.get(t) or get("p3_samplers", t)
        ax.annotate(t[1:-2], (d["dist3_mean"], d["ppl_mean"]), textcoords="offset points", xytext=(-4, -9), fontsize=6.5, color=C_T)
    single = ["r9", "r8", "r7", "r6", "r5", "r4", "r3"]
    ax.plot([S[v]["dist3_mean"] for v in single], [S[v]["ppl_mean"] for v in single], "s", color=C_R, ms=4,
            label="one round RT$_k$, $k=9\\ldots3$")
    M = R["ts_more"]
    multi = ["r6x2", "r4+r8", "r5x2", "r3+r6+r9"]
    ax.plot([M[v]["dist3_mean"] for v in multi], [M[v]["ppl_mean"] for v in multi], "D", color=C_M, ms=4,
            label="two or three rounds")
    for v, xy in (("r6", (5, -2)), ("r5x2", (-34, 5)), ("r3+r6+r9", (5, -7))):
        d = S.get(v) or M[v]
        ax.annotate(label(v), (d["dist3_mean"], d["ppl_mean"]), textcoords="offset points", xytext=xy, fontsize=6.5,
                    color=C_R if v in single else C_M)
    P3 = R.get("p3_samplers", {})
    hot = [v for v in P3 if v.startswith("d")]
    if hot:
        ax.plot([P3[v]["dist3_mean"] for v in hot], [P3[v]["ppl_mean"] for v in hot], "^", color=C_D, ms=4.5,
                label=r"hot draft ($\tau{=}1.1, 1.2$) + RT")
        for v, txt, xy in (("d1.1+r6", r"draft $\tau{=}1.1$ + RT$_6$", (-10, -13)),
                           ("d1.2+t0.8@6+r6", r"draft $\tau{=}1.2$ + RT$_6$", (5, 3))):
            if v in P3:
                ax.annotate(txt, (P3[v]["dist3_mean"], P3[v]["ppl_mean"]), textcoords="offset points", xytext=xy,
                            fontsize=6.5, color=C_D)
    rel = S["full"]
    ax.plot([rel["dist3_mean"]], [rel["ppl_mean"]], "*", color="k", ms=8, zorder=5)
    ax.annotate("released", (rel["dist3_mean"], rel["ppl_mean"]), textcoords="offset points", xytext=(-14, 7), fontsize=7)
    ax.set_xlabel("corpus distinct-3 (more diverse $\\rightarrow$)")
    ax.set_ylabel("Gen. PPL, GPT-2 Large ($\\downarrow$)")
    ax.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figs", "ts_frontier.pdf")); fig.savefig(os.path.join(HERE, "figs", "ts_frontier.png"), dpi=130)
    plt.close(fig)


# ---------------- Figure: LM1B frontier (gen-PPL vs unigram entropy) ----------------
def lm1b_frontier():
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for ax, m in zip(axs, ("small", "large")):
        S = R[f"lm1b_{m}"]
        T = ["t0.8@1", "t0.85@1", "t0.9@1", "full"]
        ax.plot([S[t]["ent_mean"] for t in T], [S[t]["ppl_mean"] for t in T], "-o", color=C_T, ms=3.5, lw=1.2,
                label=r"temperature $\tau{=}0.8\ldots1$")
        rs = [v for v in ("r9", "r7", "r6", "r5") if v in S]
        ax.plot([S[v]["ent_mean"] for v in rs], [S[v]["ppl_mean"] for v in rs], "s", color=C_R, ms=4, label="one round RT$_k$")
        off = {"r9": (3, 2), "r7": (-17, -3), "r6": (4, 1), "r5": (-17, -1)} if m == "small" else \
              {"r9": (3, 2), "r7": (3, 2), "r6": (-16, -3), "r5": (4, 1)}
        for v in rs:
            ax.annotate(label(v), (S[v]["ent_mean"], S[v]["ppl_mean"]), textcoords="offset points", xytext=off[v], fontsize=6.5, color=C_R)
        if m == "large":
            M = R["lm1b_large_more"]
            mm = [v for v in ("r6x2", "r5x2", "r4+r8") if v in M]
            ax.plot([M[v]["ent_mean"] for v in mm], [M[v]["ppl_mean"] for v in mm], "D", color=C_M, ms=4, label="two rounds")
            for v in mm:
                ax.annotate(label(v), (M[v]["ent_mean"], M[v]["ppl_mean"]), textcoords="offset points",
                            xytext={"r6x2": (-30, -3), "r5x2": (4, -3), "r4+r8": (-12, 6)}[v], fontsize=6.5, color=C_M)
            PL = R.get("p3_lm1b", {})
            hot = [v for v in ("d1.1+r6x2", "d1.2+r6x2") if v in PL]
            if hot:
                ax.plot([PL[v]["ent_mean"] for v in hot], [PL[v]["ppl_mean"] for v in hot], "^", color=C_D, ms=4.5,
                        label="hot draft + RT$_6\\times2$")
            ax.set_ylim(top=max(ax.get_ylim()[1], 82), bottom=min(ax.get_ylim()[0], 54))
        ax.set_title(f"LM1B {m}", fontsize=9)
        ax.set_xlabel("unigram entropy (more diverse $\\rightarrow$)")
    axs[0].set_ylabel("Gen. PPL, GPT-2 Large ($\\downarrow$)")
    axs[1].legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figs", "lm1b_frontier.pdf")); fig.savefig(os.path.join(HERE, "figs", "lm1b_frontier.png"), dpi=130)
    plt.close(fig)


# ---------------- Figure: quality vs generator passes ----------------
SKIP = [(6, 79.72), (7, 30.54), (8, 21.68), (9, 18.46), (10, 17.05), (11, 16.15)]  # RESULTS.md s4, released, GPT-2 Large


def nfe_plot():
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.plot([n for n, _ in SKIP], [p for _, p in SKIP], "-o", color=C_T, ms=3.5, lw=1.2, label="skip levels (fewer passes)")
    S, M = R["ts_single"], R["ts_more"]
    pts = sorted([(S[v]["nfe"], S[v]["ppl_mean"], v) for v in ("r9", "r8", "r7", "r6", "r5")] +
                 [(M[v]["nfe"], M[v]["ppl_mean"], v) for v in ("r5x2", "r3+r6+r9")])
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-s", color=C_R, ms=4, lw=1.2, label="round-trip refinement (more passes)")
    for n, p, v in pts:
        if v in ("r6", "r5x2", "r3+r6+r9", "r9"):
            ax.annotate(label(v), (n, p), textcoords="offset points", xytext=(3, 4), fontsize=6.5, color=C_R)
    ax.axhline(R["real_ts"]["real"]["ppl_mean"], color="k", lw=0.8, ls=":")
    ax.text(21.5, R["real_ts"]["real"]["ppl_mean"] + 0.6, "human text", fontsize=7)
    ax.set_yscale("log")
    ax.set_yticks([8, 10, 12, 16, 20, 30, 50, 80])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("generator passes (NFE)")
    ax.set_ylabel("Gen. PPL, GPT-2 Large ($\\downarrow$)")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figs", "ts_nfe.pdf")); fig.savefig(os.path.join(HERE, "figs", "ts_nfe.png"), dpi=130)
    plt.close(fig)


# ---------------- Figure: share of codes changed by the round trip, per level ----------------
def changed_plot():
    P3 = R.get("p3_samplers", {})
    if not (P3.get("r6", {}).get("changed_mean") and P3.get("q6", {}).get("changed_mean")):
        return
    fig, ax = plt.subplots(figsize=(5.2, 2.5))
    lv = list(range(12))
    for v, c, name in (("r6", C_R, "text round trip (decode, re-encode)"), ("q6", C_A, "latent re-quantization")):
        ax.plot(lv, [100 * x for x in P3[v]["changed_mean"]], "-o", color=c, ms=3.5, label=name)
    ax.set_xlabel("level")
    ax.set_ylabel("codes changed (\\%)")
    ax.set_xticks(lv)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "figs", "changed.pdf")); fig.savefig(os.path.join(HERE, "figs", "changed.png"), dpi=130)
    plt.close(fig)


# ---------------- Tables ----------------
def row_ts(d, v, base=None):
    sd = f"{d['ppl_mean']:.2f}{{\\scriptsize$\\pm${d['ppl_sd']:.2f}}}"
    vs = pct(d.get("vs_full")) if v != "full" else "--"
    fr = pct(d.get("vs_temp_dist3")) if (v != "full" and not (v.startswith("t") and v.endswith("@1"))) else "--"
    return (f"{label(v)} & {d['nfe']} & {sd} & {vs} & {d['ent_mean']:.3f} & {d['dist3_mean']:.3f} & "
            f"{d['xsamp4_mean']:.3f} & {d['dup_mean']:.2f} & {fr} \\\\")


def table_ts():
    S, M = R["ts_single"], R["ts_more"]
    L = [row_ts(S["full"], "full")]
    L.append("\\midrule")
    L += [row_ts(S[v], v) for v in ("t0.95@1", "t0.9@1", "t0.85@1", "t0.8@1", "t0.7@1")]
    L.append("\\midrule")
    L += [row_ts(S[v], v) for v in ("r9", "r8", "r7", "r6", "r5", "r4", "r3")]
    L.append("\\midrule")
    L += [row_ts(M[v], v) for v in ("r6x2", "r4+r8", "r5x2", "r3+r6+r9")]
    L.append("\\midrule")
    L += [row_ts(M[v], v) for v in ("t0.8@6+r6", "t0.8@6+r5", "t0.9@6+r5x2")]
    L.append("\\midrule")
    real = R["real_ts"]["real"]
    L.append(f"human text (validation) & -- & {real['ppl_mean']:.2f} & -- & -- & {real['dist3']:.3f} & {real['xsamp4']:.3f} & "
             f"$<$0.01 & -- \\\\")
    open(os.path.join(HERE, "tables", "ts.tex"), "w").write("\n".join(L) + "\n")


def table_lm1b():
    L = []
    for m in ("small", "large"):
        S = R[f"lm1b_{m}"]
        rows = [("full", S["full"])] + [(v, S[v]) for v in ("t0.9@1", "t0.85@1", "t0.8@1")] + \
               [(v, S[v]) for v in ("r9", "r7", "r6", "r5")]
        if m == "large":
            M = R["lm1b_large_more"]
            rows += [(v, M[v]) for v in ("r6x2", "r5x2", "r4+r8", "t0.9@6+r6")]
        L.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{LM1B {m}}}}} \\\\")
        for v, d in rows:
            sd = f"{d['ppl_mean']:.2f}{{\\scriptsize$\\pm${d['ppl_sd']:.2f}}}"
            L.append(f"{label(v)} & {d['nfe']} & {sd} & {pct(d.get('vs_full')) if v != 'full' else '--'} & "
                     f"{d['ent_mean']:.3f} & {d['dist3_mean']:.3f} & {d['xsamp4_mean']:.3f} & {(str(d.get('seed_wins', 0)) + '/3') if v != 'full' else '--'} \\\\")
        L.append("\\midrule")
        if m == "small" and R.get("p4_lm1bm"):
            S = R["p4_lm1bm"]
            L.append("\\multicolumn{8}{l}{\\emph{LM1B medium}} \\\\")
            for v in ("full", "t0.9@1", "t0.85@1", "t0.8@1", "r7", "r6", "r6x2"):
                if v in S:
                    d = S[v]
                    sd = f"{d['ppl_mean']:.2f}{{\\scriptsize$\\pm${d['ppl_sd']:.2f}}}"
                    L.append(f"{label(v)} & {d['nfe']} & {sd} & {pct(d.get('vs_full')) if v != 'full' else '--'} & "
                             f"{d['ent_mean']:.3f} & {d['dist3_mean']:.3f} & {d['xsamp4_mean']:.3f} & "
                             f"{(str(d.get('seed_wins', 0)) + '/3') if v != 'full' else '--'} \\\\")
            L.append("\\midrule")
    if L and L[-1] == "\\midrule":
        L.pop()
    open(os.path.join(HERE, "tables", "lm1b.tex"), "w").write("\n".join(L) + "\n")


def table_cond():
    X, W = R["xsum"], R["wmt14"]
    L = []
    for v in ("full", "r7", "r6", "r5", "r4", "r3", "r5x2"):
        if v in X:
            d = X[v]
            L.append(f"XSum & {label(v)} & {int(d['nfe'])} & {d['r1']:.2f} & {d['r2']:.2f} & {d['rl']:.2f} & {d['ppl']:.1f} \\\\")
    L.append("\\midrule")
    for v in ("full", "r8", "r7", "r6", "r5", "r4", "r6x2"):
        if v in W:
            d = W[v]
            L.append(f"WMT14 & {label(v)} & {int(d['nfe'])} & \\multicolumn{{3}}{{c}}{{BLEU {d['bleu']:.2f}, chrF {d['chrf']:.2f}}} & "
                     f"len.\\ {d['len_ratio']:.2f} \\\\")
    open(os.path.join(HERE, "tables", "cond.tex"), "w").write("\n".join(L) + "\n")


def table_distill():
    L = []
    rel = R["ts_more"]["full"]
    t08 = R["ts_single"]["t0.8@1"]
    for name, key in (("released (no fine-tune)", None), ("control fine-tune", "p2_ctrl"), ("control, soup", "p2_ctrl_soup"),
                      ("distilled", "p2_dist"), ("distilled, soup", "p2_dist_soup")):
        d = rel if key is None else R[key]["full"]
        bs = "--" if key is None else "128"
        L.append(f"{name} & {bs} & {d['ppl_mean']:.2f} & {d['dist3_mean']:.3f} & {d['xsamp4_mean']:.3f} \\\\")
    for key, name in (("p3_ctrl2k", "control fine-tune"), ("p3_dist2k", "distilled")):
        if R.get(key, {}).get("full"):
            d = R[key]["full"]
            L.append(f"{name} & 2048 & {d['ppl_mean']:.2f} & {d['dist3_mean']:.3f} & {d['xsamp4_mean']:.3f} \\\\")
    L.append("\\midrule")
    L.append(f"released, $\\tau{{=}}0.8$ & -- & {t08['ppl_mean']:.2f} & {t08['dist3_mean']:.3f} & {t08['xsamp4_mean']:.3f} \\\\")
    open(os.path.join(HERE, "tables", "distill.tex"), "w").write("\n".join(L) + "\n")


def table_ablation():
    P = R.get("p3_samplers", {})
    if not P:
        open(os.path.join(HERE, "tables", "ablation.tex"), "w").write("\\multicolumn{8}{c}{\\emph{pending}} \\\\\n")
        return
    order = ["full", "n6", "q6", "r6", "n5x2", "q5x2", "r5x2", "t1.1@1", "t1.2@1", "d1.1+r6", "d1.2+r6",
             "d1.2+t0.8@6+r6", "d1.1+r5x2", "d1.2+r5x2", "d1.2+t0.8@6+r5x2", "d1.2+r3+r6+r9"]
    L = []
    for v in order:
        if v not in P:
            continue
        d = P[v]
        if v in ("n6", "t1.1@1", "d1.1+r6"):
            L.append("\\midrule")
        sd = f"{d['ppl_mean']:.2f}{{\\scriptsize$\\pm${d['ppl_sd']:.2f}}}"
        fr = pct(d.get("vs_temp_dist3")) if not (v == "full" or (v.startswith("t") and v.endswith("@1"))) else "--"
        L.append(f"{label(v)} & {d['nfe']} & {sd} & {pct(d.get('vs_full')) if v != 'full' else '--'} & {d['ent_mean']:.3f} & "
                 f"{d['dist3_mean']:.3f} & {d['xsamp4_mean']:.3f} & {fr} \\\\")
    PL = R.get("p3_lm1b", {})
    if PL:
        L.append("\\midrule")
        L.append("\\multicolumn{8}{l}{\\emph{LM1B large (FLM protocol)}} \\\\")
        for v in ("full", "n6", "q6", "r6", "r6x2", "d1.1+r6x2", "d1.2+r6x2"):
            if v in PL:
                d = PL[v]
                sd = f"{d['ppl_mean']:.2f}{{\\scriptsize$\\pm${d['ppl_sd']:.2f}}}"
                L.append(f"{label(v)} & {d['nfe']} & {sd} & {pct(d.get('vs_full')) if v != 'full' else '--'} & "
                         f"{d['ent_mean']:.3f} & {d['dist3_mean']:.3f} & {d['xsamp4_mean']:.3f} & "
                         f"{pct(d.get('vs_temp_ent')) if v != 'full' else '--'} \\\\")
    open(os.path.join(HERE, "tables", "ablation.tex"), "w").write("\n".join(L) + "\n")


def table_heldout():
    S = R.get("p4_ts", {})
    order = ["full", "t0.8@1", "r6", "r5x2", "r3+r6+r9", "t0.8@6+r6", "d1.1+r6", "d1.2+t0.8@6+r6", "d1.2+t0.8@6+r5x2", "n6", "q6"]
    L = []
    for v in order:
        if v not in S:
            continue
        d = S[v]
        sd = f"{d['ppl_mean']:.2f}{{\\scriptsize$\\pm${d['ppl_sd']:.2f}}}"
        fr = pct(d.get("vs_temp_dist3")) if not (v == "full" or (v.startswith("t") and v.endswith("@1"))) else "--"
        L.append(f"{label(v)} & {d['nfe']} & {sd} & {pct(d.get('vs_full')) if v != 'full' else '--'} & "
                 f"{d['dist3_mean']:.3f} & {d['xsamp4_mean']:.3f} & {fr} \\\\")
        if v in ("t0.8@1", "r3+r6+r9", "d1.2+t0.8@6+r5x2"):
            L.append("\\midrule")
    open(os.path.join(HERE, "tables", "heldout.tex"), "w").write("\n".join(L) + "\n")


def table_judge():
    J = R.get("judge_ts", {}).get("pairs", {})
    names = {"full:r6": ("released", "RT$_6$", "main comparison"),
             "full:r3+r6+r9": ("released", "RT$_3$+RT$_6$+RT$_9$", ""),
             "full:d1.1+r6": ("released", "RT$_6$, draft $\\tau{=}1.1$", "more diverse than released"),
             "t0.8@1:r5x2": ("$\\tau{=}0.8$", "RT$_5\\times2$", "equal distinct-3"),
             "t0.8@1:r6": ("$\\tau{=}0.8$", "RT$_6$", "equal Gen.\\ PPL"),
             "q6:r6": ("LatentRQ$_6$", "RT$_6$", "text vs.\\ latent round trip"),
             "full:t0.8@1": ("released", "$\\tau{=}0.8$", "temperature alone")}
    L = []
    for k, (x, y, note) in names.items():
        if k in J:
            d = J[k]
            L.append(f"{x} & {y} & {d['p_prefer_Y']:.3f} & [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}] & "
                     f"{100 * d['both_orders_Y']:.1f} / {100 * d['both_orders_X']:.1f} & {note} \\\\")
    open(os.path.join(HERE, "tables", "judge.tex"), "w").write("\n".join(L) + "\n")


def table_mauve():
    M = R.get("mauve_ts", {})
    src = {**R["ts_single"], **R["ts_more"], **R.get("p3_samplers", {})}
    L = []
    for v in ("full", "t0.9@1", "t0.8@1", "n6", "q6", "r6", "t0.8@6+r6", "r5x2", "r3+r6+r9", "d1.1+r6", "d1.2+t0.8@6+r6"):
        if v in M and v in src:
            L.append(f"{label(v)} & {M[v]['mauve']:.3f}{{\\scriptsize$\\pm${M[v]['sd']:.3f}}} & {src[v]['ppl_mean']:.2f} & "
                     f"{src[v]['dist3_mean']:.3f} \\\\")
    open(os.path.join(HERE, "tables", "mauve.tex"), "w").write("\n".join(L) + "\n")


def table_sudoku():
    S = R.get("sudoku_hard", {})
    L = []
    for v in ("full", "r9", "r8", "r6", "r4", "r6x2", "r6x4", "r4x3", "r3+r6+r9"):
        if v in S:
            d = S[v]
            extra = (f"{d['new_solved']} / {d['new_broken']} & {d['mcnemar_p']:.1e}" if v != "full" and "new_solved" in d
                     else "-- & --")
            name = "released (greedy)" if v == "full" else label(v)
            L.append(f"{name} & {d['nfe']} & {d['solved']:.1f} & {d['cell']:.2f} & {extra} \\\\")
    open(os.path.join(HERE, "tables", "sudoku.tex"), "w").write("\n".join(L) + "\n")


def table_twins():
    S = R.get("p6_ts", {})
    rows = [("full", None), ("r6", None), ("\\midrule", None),
            ("d1.3+t0.8@6+r5x2", "best recipe"), ("t1.3@1+t1@5+t0.8@6", "its twin"),
            ("d1.2+t0.8@6+r5x2", "recipe"), ("t1.2@1+t1@5+t0.8@6", "its twin"),
            ("t1.2@1+t0.8@6+r6", "recipe"), ("t1.2@1+t0.8@6", "its twin"),
            ("\\midrule", None), ("t0.6@1", None), ("t0.7@1", None), ("t0.8@1", None),
            ("\\midrule", None), ("h", "human text"), ("h+r6", None), ("h+r5x2", None), ("h+r3+r6+r9", None)]
    L = []
    for v, note in rows:
        if v == "\\midrule":
            L.append(v)
            continue
        if v not in S:
            continue
        d = S[v]
        name = {"h": "human text"}.get(v, label(v).replace("h+", "human + "))
        if v.startswith("h+"):
            name = "human + " + label(v[2:])
        if note and v != "h":
            name += f" ({note})"
        plain_t = v.startswith("t") and v.endswith("@1") and "+" not in v
        fr = "--" if (v == "full" or v.startswith("h") or plain_t) else pct(d.get("vs_temp_dist3"))
        L.append(f"{name} & {d['nfe']} & {d['ppl_mean']:.2f} & {d['dist3_mean']:.3f} & {d['xsamp4_mean']:.3f} & {fr} \\\\")
    open(os.path.join(HERE, "tables", "twins.tex"), "w").write("\n".join(L) + "\n")


def table_var():
    V = R.get("var_screen", {})
    names = {"base": "VAR sampler (cfg 1.5, top-$k$ 900, top-$p$ 0.96)", "n7": "Resample$_{7}$ (no re-encode)",
             "q7": "LatentRQ$_{7}$", "r8": "RT$_{8}$", "r7": "RT$_{7}$", "r6": "RT$_{6}$", "r5": "RT$_{5}$",
             "r7x2": "RT$_{7}\\times2$"}
    L = []
    for v in ("base", "n7", "q7", "r8", "r7", "r6", "r5", "r7x2"):
        if v in V and "fid" in V[v]:
            ch = V[v].get("changed_per_scale")
            chs = f"{100 * min(ch[:-1]):.0f}--{100 * max(ch[:-1]):.0f}\\%" if ch else "--"
            L.append(f"{names[v]} & {V[v]['fid']:.2f} & {chs} \\\\")
            if v in ("base", "q7"):
                L.append("\\midrule")
    open(os.path.join(HERE, "tables", "var.tex"), "w").write("\n".join(L) + "\n")


ts_frontier()
lm1b_frontier()
table_twins()
table_var()
nfe_plot()
changed_plot()
table_ts()
table_lm1b()
table_cond()
table_distill()
table_ablation()
table_heldout()
table_judge()
table_mauve()
table_sudoku()
print("figs:", sorted(os.listdir(os.path.join(HERE, "figs"))), "tables:", sorted(os.listdir(os.path.join(HERE, "tables"))))
