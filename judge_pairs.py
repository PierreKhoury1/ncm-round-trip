"""Blind pairwise preferences from an open-weights instruction-tuned judge, between two samplers' outputs for the same
seeds (paired drafts: sample i of every variant starts from the same level-0 draw).

Each pair is shown in both orders; the judge's probability of answering "A" vs "B" is read from its next-token
distribution (no free-form generation), and the two orders are averaged to cancel position bias. Reports, for each
pair X:Y, the mean probability that Y is preferred, the share of items where both orders prefer Y / prefer X, a 95%
bootstrap interval of the mean, and the judge's position bias.

Usage: python round_trip/judge_pairs.py --pkl "round_trip/runs/p3_samplers_s{s}.pkl,round_trip/runs/ts_more_s{s}.pkl" \
           --seeds 1,2,3 --pairs full:r6,t0.8@1:r6 --task stories --out round_trip/runs/judge_ts.json
For each seed, a variant is taken from the first listed pickle that contains it.
"""
import argparse, json, os, pickle
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--pkl", required=True, help="comma-separated pickle path templates containing {s} for the seed")
ap.add_argument("--seeds", default="1,2,3")
ap.add_argument("--pairs", required=True, help="comma-separated X:Y pairs of variant names")
ap.add_argument("--judge", default="Qwen/Qwen2.5-7B-Instruct")
ap.add_argument("--task", default="stories", choices=["stories", "news"])
ap.add_argument("--field", default="texts", help="'texts' (decoded) or 'raw' (token ids, decoded with --raw-tokenizer)")
ap.add_argument("--raw-tokenizer", default="bert-base-uncased")
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--limit", type=int, default=0, help="use only the first N items per variant (smoke tests)")
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", required=True)
a = ap.parse_args()

rt = AutoTokenizer.from_pretrained(a.raw_tokenizer) if a.field == "raw" else None


def items(r):
    if rt is None:
        return [" ".join(t.split()) for t in r["texts"]]
    return [" ".join(rt.decode([t for t in x if t not in (0, 101, 102)]).split()) for x in r["raw"]]


V = {}  # variant -> texts over seeds, in seed order
for s in a.seeds.split(","):
    loaded = [pickle.load(open(t.format(s=s), "rb"))["results"] for t in a.pkl.split(",") if os.path.exists(t.format(s=s))]
    for v in {v for R in loaded for v in R}:
        R = next(R for R in loaded if v in R)
        V.setdefault(v, []).extend(items(R[v]))

tok = AutoTokenizer.from_pretrained(a.judge)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.judge, torch_dtype=torch.bfloat16 if a.device == "cuda" else torch.float32)
model = model.to(a.device).eval()
if a.limit:
    V = {v: t[:a.limit] for v, t in V.items()}
IDA = tok.encode("A", add_special_tokens=False)[0]
IDB = tok.encode("B", add_special_tokens=False)[0]
KIND = {"stories": ("short children's stories", "story"), "news": ("short news-style passages", "passage")}[a.task]


def prompt(x, y):
    msg = [{"role": "user", "content":
            f"Below are two {KIND[0]}, A and B. Judge only the writing: which {KIND[1]} is more fluent, grammatical "
            f"and coherent? Ignore length and topic.\n\nA:\n{x}\n\nB:\n{y}\n\nAnswer with a single letter, A or B."}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def p_second(firsts, seconds):
    """P(judge prefers the text shown second), from the A/B next-token logits."""
    out = []
    for s in range(0, len(firsts), a.bs):
        enc = tok([prompt(x, y) for x, y in zip(firsts[s:s + a.bs], seconds[s:s + a.bs])], return_tensors="pt",
                  padding=True).to(a.device)
        logits = model(**enc).logits[:, -1, :].float()
        out += torch.stack([logits[:, IDA], logits[:, IDB]], -1).softmax(-1)[:, 1].tolist()
    return np.array(out)


res, rng = {}, np.random.default_rng(0)
for pair in a.pairs.split(","):
    X, Y = pair.split(":")
    n = min(len(V[X]), len(V[Y]))
    xs, ys = V[X][:n], V[Y][:n]
    y_second = p_second(xs, ys)        # shown X then Y: P(Y)
    y_first = 1 - p_second(ys, xs)     # shown Y then X: P(Y) = 1 - P(X, shown second)
    py = 0.5 * (y_second + y_first)
    boot = [float(np.mean(rng.choice(py, n))) for _ in range(2000)]
    res[pair] = {"n": n, "p_prefer_Y": float(py.mean()),
                 "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                 "both_orders_Y": float(np.mean((y_second > 0.5) & (y_first > 0.5))),
                 "both_orders_X": float(np.mean((y_second < 0.5) & (y_first < 0.5))),
                 "p_first_shown_wins": float(np.mean(((1 - y_second) + y_first) / 2))}
    r = res[pair]
    print(f"{X:>18s} vs {Y:<18s} n={n}  P(prefer {Y})={r['p_prefer_Y']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]  "
          f"both orders: {Y} {100 * r['both_orders_Y']:.1f}% / {X} {100 * r['both_orders_X']:.1f}%  "
          f"first-shown bias {r['p_first_shown_wins']:.3f}", flush=True)
json.dump({"judge": a.judge, "pairs": res}, open(a.out, "w"), indent=1)
