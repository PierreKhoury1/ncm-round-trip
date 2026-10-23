"""Verifier-guided test-time search on Sudoku with a released NCM checkpoint.

Every puzzle starts from the configured greedy draft. While a board fails the exact checker (rows, columns and boxes
are permutations of 1..9 AND every given clue is kept, which for unique-solution puzzles means it is THE solution),
more generator passes are spent, in one of two ways that use the checker identically:
  resample  a fresh full sample at temperature T for the unsolved puzzles (K-1 passes per attempt)        [baseline]
  refine    round trip on the current board: re-encode, keep levels < k, regenerate levels k..K-1 at temperature T
            (K-k passes per attempt)
Reports the solve rate (exact match to the reference solution) after each attempt together with the mean number of
generator passes spent per puzzle, so the two methods can be compared at equal compute.

Usage: python round_trip/sudoku_search.py --config config/eval/sudoku-hard.yaml --checkpoint-dir artifacts/sudoku/hard \
           --data-dir data/sudoku/hard --n 1000 --attempts 10 --methods resample@0.5,resample@1.0,refine6@0.5,refine4@0.5 \
           --out round_trip/runs/p5_sudoku_search.json
"""
import argparse, json, os, pickle, sys, time
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from omegaconf import OmegaConf  # noqa: E402
from models.next_concept import NextConceptTransformer  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--checkpoint-dir", required=True)
ap.add_argument("--data-dir", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--attempts", type=int, default=10)
ap.add_argument("--methods", default="resample@0.5,resample@1.0,refine6@0.5,refine4@0.5")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--smoke", action="store_true", help="tiny run that checks the plumbing and exits")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.chdir(ROOT)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = OmegaConf.load(a.config)
ck = a.checkpoint_dir
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.to(dev).eval()
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).to(dev)
sd = torch.load(os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
net.load_state_dict(sd["model"], strict=True)
net.eval()
vq = tk.vqvae.vq
levels = list(vq.levels)
K = len(levels)
P = int(CFG.model.prefix_len)
G = CFG.generation
temp0 = float(G.get("temperature", 0.0))
assert not G.get("frequency_penalty") and not G.get("adjacency_penalty"), "argmax token picking assumed"


def split_codes(idx):
    out, acc = [], 0
    for L in levels:
        out.append(idx[:, acc:acc + L])
        acc += L
    return out


_orig_sample = net._sample_level
_state = {"codes": None, "calls": 0, "upto": 0, "offset": 0, "temps": None}


def _sample_with_force(logits, temperature, top_k_, top_p_):
    level = _state["calls"] + 1 - _state["offset"]
    _state["calls"] += 1
    if level >= 1 and _state["codes"] is not None and level < _state["upto"]:
        return _state["codes"][level]
    t = _state["temps"][max(level, 0)] if _state["temps"] is not None else temperature
    return _orig_sample(logits, t, top_k_, top_p_)


net._sample_level = _sample_with_force


@torch.no_grad()
def generate(l0, prefix, temps, forced=None, upto=0):
    offset = 0
    if l0 is None:
        if net.query_level0:
            offset = levels[0] if getattr(net, "l0_ar", False) else 1
        elif net.predict_level0:
            offset = 1
    _state.update(codes=forced, calls=0, upto=upto, offset=offset, temps=temps)
    g = net.generate(l0, prefix, temperature=temps[-1], top_k=int(G.get("top_k", 0)), top_p=float(G.get("top_p", 1.0)),
                     cfg_scale=0.0, cfg_null_eos=tk.pre_tokenizer.eos_token_id)
    _state.update(codes=None, calls=0, upto=0, offset=0, temps=None)
    _, zl = vq.idx_to_zhats(g, return_zlevels=True)
    return tk.vqvae.decoder(zl[-1]).argmax(-1)


def dec(ids):
    return tk.pre_tokenizer.decode([int(t) for t in ids], skip_special_tokens=True)


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


dd = a.data_dir
items = pickle.load(open(os.path.join(dd, CFG.data[a.split]), "rb"))[:a.n]
refs = pickle.load(open(os.path.join(dd, CFG.data[f"{a.split}_refs"]), "rb"))[:a.n]
refs = [r if isinstance(r, str) else dec(r) for r in refs]
if a.smoke:
    items, refs, a.attempts = items[:32], refs[:32], 2
N = len(items)
prefix_all = torch.tensor([(list(x)[:P] + [0] * P)[:P] for x in items], device=dev)
clues = []
for x in items:  # the puzzle's given digits from its token ids (0 = blank, 1-9 = digit; '|' and BOS dropped)
    c = [int(t) for t in x if 0 <= int(t) <= 9]
    clues.append("".join("_" if t == 0 else str(t) for t in c[:81]))
# diagnostic only (the verifier never sees the solutions): clue cells must agree with the reference solutions
agree = sum(len(c) == 81 and all(ch == "_" or ch == s for ch, s in zip(c, cells(r))) for c, r in zip(clues, refs))
print(f"clue cells per puzzle: {sorted({len([t for t in x if 0 <= int(t) <= 9]) for x in items})}; "
      f"clues consistent with the reference solution: {agree}/{len(items)}", flush=True)
BASE = K if (net.query_level0 or net.predict_level0) else K - 1  # passes of one full sample (incl. a level-0 query)


def verified(i, board):
    g = cells(board)
    return valid(board) and len(clues[i]) == 81 and all(c == "_" or c == b for c, b in zip(clues[i], g))


def batched(idxs, fn):
    out = {}
    for s in range(0, len(idxs), a.bs):
        b = idxs[s:s + a.bs]
        toks = fn(b)
        for i, t in zip(b, toks.tolist()):
            out[i] = t
    return out


# greedy drafts, shared by every method
torch.manual_seed(a.seed)
draft = batched(list(range(N)), lambda b: generate(None, prefix_all[b], [temp0] * K))
draft_txt = {i: dec(t) for i, t in draft.items()}
print(f"{N} puzzles; clue strings of length 81: {sum(len(c) == 81 for c in clues)}/{N}; "
      f"draft solved {sum(draft_txt[i] == refs[i] for i in range(N))}, verified {sum(verified(i, draft_txt[i]) for i in range(N))}",
      flush=True)

results = {}
t0 = time.time()
for m in a.methods.split(","):
    kind, T = m.split("@")
    T = float(T)
    torch.manual_seed(a.seed)
    boards = dict(draft)
    txt = dict(draft_txt)
    done = {i: verified(i, txt[i]) for i in range(N)}
    nfe = {i: BASE for i in range(N)}
    curve = [{"attempt": 0, "solved": sum(txt[i] == refs[i] for i in range(N)) / N,
              "verified": sum(done.values()) / N, "mean_nfe": sum(nfe.values()) / N}]
    for att in range(1, a.attempts + 1):
        todo = [i for i in range(N) if not done[i]]
        if not todo:
            curve.append(dict(curve[-1], attempt=att))
            continue
        if kind == "resample":
            new = batched(todo, lambda b: generate(None, prefix_all[b], [T] * K))
            cost = BASE
        elif kind.startswith("plan"):  # fresh sample: coarse plan (levels < k, incl. level 0) at T, fine levels greedy
            kp = int(kind[len("plan"):])
            ptemps = [T] * kp + [0.0] * (K - kp)
            new = batched(todo, lambda b: generate(None, prefix_all[b], ptemps))
            cost = BASE
        else:
            k = int(kind[len("refine"):])
            temps = [temp0] * k + [T] * (K - k)

            def fn(b):
                cur = torch.tensor([boards[i] for i in b], device=dev)
                re = split_codes(tk.vqvae.encode(cur))
                return generate(re[0], prefix_all[b], temps, forced=re, upto=k)
            new = batched(todo, fn)
            cost = K - k
        for i, t in new.items():
            nfe[i] += cost
            s = dec(t)
            if verified(i, s):
                done[i], boards[i], txt[i] = True, t, s
            elif kind.startswith("refine"):
                boards[i], txt[i] = t, s  # refinement continues from its latest board
        curve.append({"attempt": att, "solved": sum(txt[i] == refs[i] for i in range(N)) / N,
                      "verified": sum(done.values()) / N, "mean_nfe": sum(nfe.values()) / N})
    results[m] = curve
    c = curve[-1]
    print(f"{m:14s} after {a.attempts} attempts: solved {100 * c['solved']:.1f}%  verified {100 * c['verified']:.1f}%  "
          f"mean NFE {c['mean_nfe']:.1f}  ({time.time() - t0:.0f}s)  curve " +
          " ".join(f"{100 * x['solved']:.1f}@{x['mean_nfe']:.1f}" for x in curve), flush=True)
json.dump({"n": N, "levels": levels, "results": results}, open(a.out, "w"), indent=1)
print(f"wrote {a.out}")
