"""Perplexity of refine_eval.py pickles under any HF causal LM, with per-sample values.

Two protocols, picked from the pickle's config unless --protocol is given:
  text  TinyStories: the authors' generation_perplexity on "texts" (token-weighted mean, as in analyze_skip.py)
  flm   LM1B: decode the raw generator-tokenizer ids, rescore the first 128 reference-tokenizer tokens, corpus
        perplexity (as in score_flm.py, which copies scripts/eval_merge.py)
Big reference models load in bf16 (--dtype bf16); logits are cast to float before the log-softmax. "draft_texts"
(the decoded draft before any round) are scored too when present, for the canonicality analysis (canon_analyze.py).

Usage: python round_trip/score_ppl.py --in round_trip/runs/p9_canon_ts_s1.pkl --ref Qwen/Qwen2.5-7B --dtype bf16 \
           --out round_trip/runs/p9_qwen_canon_ts_s1.json
"""
import argparse, json, os, pickle, sys
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from evaluation.gen_ppl import generation_perplexity  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True)
ap.add_argument("--ref", default="gpt2-large")
ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--protocol", default="auto", choices=["auto", "text", "flm"])
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--out", required=True)
a = ap.parse_args()

D = pickle.load(open(a.inp, "rb"))
proto = a.protocol if a.protocol != "auto" else ("flm" if "lm1b" in str(D["config"]).lower() else "text")
DT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[a.dtype]
rtok = AutoTokenizer.from_pretrained(a.ref)
if rtok.pad_token is None:
    rtok.pad_token = rtok.eos_token
rtok.padding_side = "right"  # the FLM mask below assumes right padding (GPT-2's default)
rlm = AutoModelForCausalLM.from_pretrained(a.ref, torch_dtype=DT).to(a.device).eval()


class FloatLogits(torch.nn.Module):
    """generation_perplexity takes the model's logits as they come; give it float ones from a bf16 model."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, **kw):
        out = self.m(**kw)
        out.logits = out.logits.float()
        return out


wrapped = FloatLogits(rlm)
gen_tok = AutoTokenizer.from_pretrained(str(OmegaConf.load(D["config"]).tokenizer.pre_tokenizer.model_id)) if proto == "flm" else None


@torch.no_grad()
def score_text(texts):
    r = generation_perplexity(wrapped, rtok, [""] * len(texts), texts, device=a.device,
                              score_only_generated=True, batch_size=a.bs)
    return float(r.mean_ppl), [float(x) for x in r.ppl.tolist()]


@torch.no_grad()
def score_flm(raw):
    texts = gen_tok.batch_decode(raw)
    enc = rtok(texts, return_tensors="pt", return_token_type_ids=False, truncation=True, padding=True, max_length=128)
    ids, attn = enc["input_ids"].to(a.device), enc["attention_mask"].to(a.device)
    nll_sum = tok_sum = 0.0
    per = []
    for i in range(0, ids.shape[0], a.bs):
        s_, m_ = ids[i:i + a.bs], attn[i:i + a.bs]
        logits = rlm(input_ids=s_, attention_mask=m_).logits.float().transpose(-1, -2)
        nll = F.cross_entropy(logits[..., :-1], s_[..., 1:], reduction="none")
        first_eos = (s_ == rtok.eos_token_id).cumsum(-1) == 1
        valid = (first_eos[..., 1:] | (s_ != rtok.eos_token_id)[..., 1:]).to(nll.dtype)
        n_ = (nll * valid).sum(1)
        c_ = valid.sum(1)
        per += torch.exp(n_ / c_.clamp(min=1)).tolist()
        nll_sum += float(n_.sum())
        tok_sum += float(c_.sum())
    return float(np.exp(nll_sum / tok_sum)), per


out = {"ref": a.ref, "dtype": a.dtype, "protocol": proto, "config": D["config"], "variants": {}}
for name, r in D["results"].items():
    row = {"nfe": r["nfe"]}
    row["ppl"], row["ppl_per_sample"] = score_flm(r["raw"]) if proto == "flm" else score_text(r["texts"])
    if r.get("draft_raw") if proto == "flm" else r.get("draft_texts"):
        if proto == "flm":
            row["draft_ppl"], row["draft_ppl_per_sample"] = score_flm(r["draft_raw"])
        else:
            row["draft_ppl"], row["draft_ppl_per_sample"] = score_text(r["draft_texts"])
    if r.get("canon") is not None:
        row["canon_mean"] = float(np.mean(r["canon"]))
    out["variants"][name] = row
    extra = f"  draft {row['draft_ppl']:9.2f}" if "draft_ppl" in row else ""
    print(f"{name:20s} NFE={r['nfe']:3d}  ppl {row['ppl']:9.2f}{extra}", flush=True)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
print(f"wrote {a.out}  (ref={a.ref}, protocol={proto})")
