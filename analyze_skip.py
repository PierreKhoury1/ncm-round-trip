"""Score level-skip schedules: gen-PPL + entropy per schedule, sorted by NFE."""
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
a = ap.parse_args()

D = pickle.load(open(a.inp, "rb"))
levels, res = D["levels"], D["results"]
tok = AutoTokenizer.from_pretrained(a.ref)
lm = AutoModelForCausalLM.from_pretrained(a.ref).to(a.device).eval()


def genppl(texts):
    with torch.no_grad():
        return generation_perplexity(lm, tok, [""] * len(texts), texts, device=a.device,
                                     score_only_generated=True, batch_size=a.bs).mean_ppl


def entropy(text):
    ids = tok.encode(text)
    c, m = Counter(ids), len(ids)
    return -sum(v / m * math.log(v / m) for v in c.values()) if m else 0.0


rows = []
for name, r in res.items():
    ppl = genppl(r["texts"])
    ent = float(np.mean([entropy(t) for t in r["texts"]]))
    words = float(np.mean([len(t.split()) for t in r["texts"]]))
    rows.append((r["nfe"], name, r["skip"], ppl, ent, words))
rows.sort(key=lambda x: (-x[0], x[3]))
print(f"levels={levels}  ref={a.ref}  N={len(next(iter(res.values()))['texts'])}\n")
print(f"{'schedule':16s} {'NFE':>4} {'genPPL':>9} {'entropy':>8} {'words':>6}  skipped levels")
for nfe, name, skip, ppl, ent, w in rows:
    print(f"{name:16s} {nfe:5.1f} {ppl:9.2f} {ent:8.3f} {w:6.1f}  {skip}")
