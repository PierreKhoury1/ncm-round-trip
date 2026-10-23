"""Score conditional refine_eval.py pickles with the authors' metrics (copied from scripts/eval_merge.py):
XSum -> ROUGE-1/2/L (rouge_score, stemmer) + GPT-2 gen-PPL of the summaries; WMT14 -> sacreBLEU, lowercased
BLEU, chrF, length ratio. One row per variant.

Usage: python round_trip/score_cond.py --in round_trip/runs/xsum.pkl
"""
import argparse, os, pickle, sys
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()

D = pickle.load(open(a.inp, "rb"))
CFG = OmegaConf.load(D["config"])
refs = D["refs"]
metric = str(CFG.eval.metric)
print(f"config={D['config']}  metric={metric}  docs={len(refs)}\n")

if metric == "rouge":
    from rouge_score import rouge_scorer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from evaluation.gen_ppl import generation_perplexity
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    g2 = AutoTokenizer.from_pretrained("gpt2")
    lm = AutoModelForCausalLM.from_pretrained("gpt2").to(a.device).eval()
    print(f"{'variant':14s} {'NFE':>4} {'R-1':>7} {'R-2':>7} {'R-L':>7} {'genPPL':>8} {'words':>7}")
    for name, r in D["results"].items():
        hyps = r["texts"]
        R = {k: [] for k in ("rouge1", "rouge2", "rougeL")}
        for h, ref in zip(hyps, refs):
            v = sc.score(ref, h)
            for k in R:
                R[k].append(v[k].fmeasure)
        with torch.no_grad():
            gp = generation_perplexity(lm, g2, [""] * len(hyps), hyps, device=a.device,
                                       score_only_generated=True, batch_size=32).mean_ppl
        print(f"{name:14s} {r['nfe']:4d} {100 * np.mean(R['rouge1']):7.2f} {100 * np.mean(R['rouge2']):7.2f} "
              f"{100 * np.mean(R['rougeL']):7.2f} {gp:8.2f} {np.mean([len(h.split()) for h in hyps]):7.1f}")
elif metric == "bleu":
    import sacrebleu
    print(f"{'variant':14s} {'NFE':>4} {'BLEU':>7} {'BLEU-lc':>8} {'chrF':>7} {'len ratio':>10}")
    for name, r in D["results"].items():
        hyps = r["texts"]
        b = sacrebleu.corpus_bleu(hyps, [refs])
        bl = sacrebleu.corpus_bleu(hyps, [refs], lowercase=True)
        c = sacrebleu.corpus_chrf(hyps, [refs])
        lr = sum(len(h.split()) for h in hyps) / max(sum(len(x.split()) for x in refs), 1)
        print(f"{name:14s} {r['nfe']:4d} {b.score:7.2f} {bl.score:8.2f} {c.score:7.2f} {lr:10.3f}")
elif metric == "exact":  # Sudoku: copied from scripts/eval_merge.py exact()
    def cells(t):
        return t.replace("|", "")

    def valid(t):
        g = cells(t)
        if len(g) != 81 or not g.isdigit() or "0" in g:
            return False
        rows_ = [g[i * 9:(i + 1) * 9] for i in range(9)]
        cols_ = ["".join(r[j] for r in rows_) for j in range(9)]
        boxes = ["".join(rows_[r][c] for r in range(br, br + 3) for c in range(bc, bc + 3)) for br in (0, 3, 6) for bc in (0, 3, 6)]
        return all(sorted(x) == list("123456789") for x in rows_ + cols_ + boxes)

    n = len(refs)
    print(f"{'variant':14s} {'NFE':>4} {'solved %':>9} {'solved':>10} {'cell %':>7} {'valid %':>8}")
    for name, r in D["results"].items():
        hyps = r["texts"]
        exact_n = sum(h == ref for h, ref in zip(hyps, refs))
        cell = np.mean([np.mean([x == y for x, y in zip(cells(h), cells(ref))]) for h, ref in zip(hyps, refs)])
        valid_n = sum(valid(h) for h in hyps)
        print(f"{name:14s} {r['nfe']:4d} {100 * exact_n / n:9.2f} {exact_n:5d}/{n:<4d} {100 * cell:7.2f} {100 * valid_n / n:8.2f}")
else:
    raise SystemExit(f"unsupported metric {metric}")
