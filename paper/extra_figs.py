"""Four extra figures (figs/*.pdf) drawn from the numbers already in tables/*.tex (3 seeds x 256, GPT-2 Large).
Same style as make_figs.py. Run after make_figs.py; independent of results.json."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False, "pdf.fonttype": 42})
C_T, C_R, C_M, C_D, C_A = "#8a8a8a", "#2b6cb0", "#1a365d", "#dd6b20", "#c53030"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, name + ".pdf")); fig.savefig(os.path.join(FIGS, name + ".png"), dpi=130)
    plt.close(fig)


def ablation_ladder():
    """Where the gain comes from: same passes, three ways to spend them (tables/ablation.tex)."""
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6), gridspec_kw={"width_ratios": [1.15, 1]})
    data = [  # (title, rows [(label, delta vs released, matched-diversity delta)])
        ("TinyStories, 6 extra passes", [("Resample$_6$\nno re-encode", 1.7, 2.1), ("LatentRQ$_6$\nre-quantize\nthe latent", -11.5, -6.8),
                                         ("RT$_6$\nre-encode\nthe text", -17.3, -12.4)]),
        ("LM1B large, 6 extra passes", [("Resample$_6$", 0.6, 2.4), ("LatentRQ$_6$", -17.9, -7.7), ("RT$_6$", -22.0, -20.9)]),
    ]
    for ax, (title, rows) in zip(axs, data):
        x = range(len(rows))
        ax.bar([i - 0.19 for i in x], [r[1] for r in rows], width=0.36, color=C_R, label="vs. released sampler")
        ax.bar([i + 0.19 for i in x], [r[2] for r in rows], width=0.36, color=C_D, label="vs. temperature at matched diversity")
        for i, r in enumerate(rows):
            for dx, v in ((-0.19, r[1]), (0.19, r[2])):
                ax.text(i + dx, v - 0.9 if v < 0 else v + 0.6, f"{v:+.1f}%", ha="center", va="top" if v < 0 else "bottom", fontsize=6.8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(list(x)); ax.set_xticklabels([r[0] for r in rows], fontsize=7.5)
        ax.set_ylim(-26, 6); ax.set_title(title, fontsize=9)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("change in Gen. PPL (%)")
    axs[0].legend(loc="lower left", fontsize=7, bbox_to_anchor=(0.0, 0.02))
    save(fig, "ablation_ladder")


def fifths():
    """Draft -> refined Gen. PPL per fifth of non-canonicality (tables/canon.tex)."""
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.6))
    sets = [
        ("TinyStories (per story)", ["0.47–0.72", "0.72–0.75", "0.75–0.78", "0.78–0.81", "0.81–0.90"],
         [13.62, 15.05, 15.95, 17.79, 20.14], [11.87, 12.61, 13.33, 14.09, 15.09]),
        ("LM1B large (per sequence)", ["0.66–0.77", "0.77–0.88", "0.88–0.91", "0.91–0.93", "0.93–0.97"],
         [76.40, 78.12, 79.79, 81.52, 84.67], [61.99, 61.11, 62.34, 61.71, 65.34]),
    ]
    for ax, (title, bins, d, r) in zip(axs, sets):
        x = list(range(5))
        for i in x:
            ax.annotate("", xy=(i, r[i]), xytext=(i, d[i]), arrowprops=dict(arrowstyle="-|>", color=C_R, lw=1.2, mutation_scale=9))
        ax.plot(x, d, "o", color=C_T, ms=5, label="draft (released sampler)")
        ax.plot(x, r, "s", color=C_R, ms=5, label="after RT$_6$")
        for i in x:
            ax.text(i + 0.12, d[i], f"{d[i]:.1f}", fontsize=6.5, va="center", color=C_T)
            ax.text(i + 0.12, r[i], f"{r[i]:.1f}", fontsize=6.5, va="center", color=C_R)
            g = 100 * (r[i] - d[i]) / d[i]
            ax.text(i, (d[i] + r[i]) / 2, f"{g:+.0f}%", fontsize=6.5, ha="right", va="center", color=C_M)
        ax.set_xticks(x); ax.set_xticklabels(bins, fontsize=7)
        ax.set_xlabel("non-canonicality of the draft (fifths)")
        ax.set_title(title, fontsize=9); ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.5, 4.9)
    axs[0].set_ylabel("Gen. PPL (geometric mean)")
    axs[0].legend(loc="upper left", fontsize=7)
    save(fig, "fifths")


def sizes():
    """Gain of RT_6 and RT_6 x2 against model size on LM1B (tables/lm1b.tex)."""
    fig, ax = plt.subplots(figsize=(3.9, 2.7))
    x = [0, 1, 2]
    r6 = [-14.8, -20.6, -22.0]
    r6x2 = [None, -28.8, -29.3]
    ax.plot(x, r6, "-s", color=C_R, ms=5, lw=1.3, label="RT$_6$ (one round)")
    ax.plot(x[1:], r6x2[1:], "-D", color=C_M, ms=5, lw=1.3, label="RT$_6\\times2$ (two rounds)")
    for i, v in zip(x, r6):
        ax.text(i, v + 1.0, f"{v:.1f}%", ha="center", fontsize=7, color=C_R)
    for i, v in zip(x[1:], r6x2[1:]):
        ax.text(i, v - 1.2, f"{v:.1f}%", ha="center", va="top", fontsize=7, color=C_M)
    ax.set_xticks(x); ax.set_xticklabels(["small", "medium", "large"])
    ax.set_xlim(-0.4, 2.4); ax.set_ylim(-34, -8)
    ax.set_ylabel("Gen. PPL vs. released (%)")
    ax.set_xlabel("LM1B checkpoint")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=7)
    save(fig, "sizes")


def timing():
    """Measured wall-clock split into generator passes and tokenizer work (tables/timing.tex), one RTX 4090."""
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [1.5, 1]})
    sets = [
        ("TinyStories, 256 stories, batch 128", ["released", "RT$_6$", "RT$_5\\times2$", "RT$_3$+RT$_6$+RT$_9$"],
         [1.21, 2.37, 3.57, 4.74], [0.20, 0.29, 0.49, 0.69], [11, 17, 25, 29]),
        ("LM1B large, 128 seq., batch 32", ["released", "RT$_6$"], [4.08, 7.83], [0.27, 0.43], [11, 17]),
    ]
    for ax, (title, names, gen, tok, nfe) in zip(axs, sets):
        x = list(range(len(names)))
        ax.bar(x, gen, color=C_R, width=0.6, label="generator passes")
        ax.bar(x, tok, bottom=gen, color=C_D, width=0.6, label="tokenizer: re-encode, decode, pick")
        for i in x:
            tot = gen[i] + tok[i]
            ax.text(i, tot + 0.08, f"{tot:.2f} s\n{tot / (gen[0] + tok[0]):.2f}$\\times$, {nfe[i]} passes", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7.5)
        ax.set_ylim(0, max(g + t for g, t in zip(gen, tok)) * 1.45)
        ax.set_title(title, fontsize=9); ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("seconds")
    axs[0].legend(loc="upper left", fontsize=7)
    save(fig, "timing")


if __name__ == "__main__":
    ablation_ladder(); fifths(); sizes(); timing()
    print("wrote", ", ".join(f"figs/{n}.pdf" for n in ("ablation_ladder", "fifths", "sizes", "timing")))
