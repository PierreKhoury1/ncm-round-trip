"""Score refine_eval.py pickles under the LM1B "FLM" protocol, copied from scripts/eval_merge.py genppl_flm:
decode the raw generator-tokenizer ids with special tokens kept, rescore the first 128 GPT-2 tokens,
corpus-level perplexity; entropy over the raw ids. One row per variant, one file per seed.

Usage: python round_trip/score_flm.py --in round_trip/runs/lm1b_small_s1.pkl [--ref gpt2-large]
"""
import argparse, os, pickle, sys
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--ref", default="gpt2-large")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()

D = pickle.load(open(a.inp, "rb"))
CFG = OmegaConf.load(D["config"])
gen_tok = AutoTokenizer.from_pretrained(str(CFG.tokenizer.pre_tokenizer.model_id))
rtok = AutoTokenizer.from_pretrained(a.ref)
if rtok.pad_token is None:
    rtok.pad_token = rtok.eos_token
rlm = AutoModelForCausalLM.from_pretrained(a.ref).to(a.device).eval()


def corpus_ppl(texts):
    enc = rtok(texts, return_tensors="pt", return_token_type_ids=False, truncation=True, padding=True, max_length=128)
    ids, attn = enc["input_ids"].to(a.device), enc["attention_mask"].to(a.device)
    nll_sum = tok_sum = 0.0
    with torch.no_grad():
        for i in range(0, ids.shape[0], 16):
            s_, m_ = ids[i:i + 16], attn[i:i + 16]
            logits = rlm(s_, attention_mask=m_)[0].transpose(-1, -2)
            nll = F.cross_entropy(logits[..., :-1], s_[..., 1:], reduction="none")
            first_eos = (s_ == rtok.eos_token_id).cumsum(-1) == 1
            valid = (first_eos[..., 1:] + (s_ != rtok.eos_token_id)[..., 1:]).to(nll.dtype)
            nll_sum += float((nll * valid).sum())
            tok_sum += float(valid.sum())
    return float(np.exp(nll_sum / tok_sum))


def entropy(ids):
    t = torch.tensor(ids)
    return float(torch.special.entr(torch.unique(t, return_counts=True)[1].float() / t.numel()).sum())


rows = []
for name, r in D["results"].items():
    texts = gen_tok.batch_decode(r["raw"])
    rows.append((r["nfe"], name, corpus_ppl(texts), float(np.mean([entropy(x) for x in r["raw"]]))))
rows.sort(key=lambda x: (-x[0], x[2]))
print(f"config={D['config']}  ref={a.ref}  N={len(next(iter(D['results'].values()))['raw'])}  (FLM protocol)\n")
print(f"{'schedule':16s} {'NFE':>4} {'genPPL':>9} {'entropy':>8} {'words':>6}  skipped levels")
for nfe, name, ppl, ent in rows:
    print(f"{name:16s} {nfe:5.1f} {ppl:9.2f} {ent:8.3f} {0.0:6.1f}  []")
