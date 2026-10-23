"""Re-encode refinement on any released NCM checkpoint, using the authors' own sampler and decoder.

Mirrors scripts/eval_shard.py's unconditional path exactly (net.generate, level-0 unigram prior, and
pick_tokens with the config's frequency / adjacency penalties), then adds refinement:

  full        the authors' sampler as released
  r<k>        decode the draft, re-encode the decoded tokens, keep the re-encoded levels < k and resample
              levels k..K-1 (K-k extra generator passes), decode again with the same pick_tokens
  t<T>@1      global temperature T instead of 1.0 (fairness control: the temperature frontier)
  d<T>+r<k>   draft at temperature T, then refine at the usual temperature
  n<k>        ablation: like r<k> but keep the draft's own codes for levels < k (no decode / re-encode)
  q<k>        ablation: like r<k> but re-quantize the draft's latent z_hat instead of decoding and re-encoding
              the text (the latent re-quantization of Scale-travel / DiverseVAR)

Resampling reuses net.generate itself: _sample_level is wrapped so that, for levels 1..k-1, it returns
the re-encoded codes instead of sampling. z_hat construction, level embeddings, heads and masks are
therefore exactly the authors' code path.

Writes {"levels", "config", "results": {variant: {"nfe", "texts", "raw"}}}: analyze_skip.py scores "texts"
(TinyStories protocol); score_flm.py scores "raw" (LM1B FLM protocol).
"""
import argparse, os, pickle, sys, time
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from omegaconf import OmegaConf  # noqa: E402
from models.next_concept import NextConceptTransformer  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--checkpoint-dir", required=True)
ap.add_argument("--generator", default="", help="override generator checkpoint (fine-tunes, weight soups)")
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--variants", default="full,r9,r7,r6,r5,t0.9@1,t0.85@1")
ap.add_argument("--split", default="", help="conditional test split (XSum, WMT14); --n then caps the number of documents")
ap.add_argument("--data-dir", default="")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--human", default="data/tinystories/valid_bpe_8192.pickle",
                help="validation stories used by the 'h' variant part (real text instead of a draft)")
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.chdir(ROOT)
dev = torch.device(a.device)

CFG = OmegaConf.load(a.config)
G = CFG.generation
ck = a.checkpoint_dir
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
    if CFG.checkpoints.get(name):
        CFG.tokenizer.pre_tokenizer[key] = os.path.join(ck, CFG.checkpoints[name])
tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.to(dev).eval()
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).to(dev)
sd = torch.load(a.generator or os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
net.load_state_dict(sd["model"], strict=True)
net.eval()
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
vq = tk.vqvae.vq
levels = list(vq.levels)
K = len(levels)

temp0 = float(G.get("temperature", 1.0))
top_k = int(G.get("top_k", 0))
top_p = float(G.get("top_p", 1.0))
cfg_scale = float(G.get("cfg_scale", 0.0))
freq = float(G.get("frequency_penalty", 0.0))
mu = float(G.get("frequency_deficit", 0.0))
adj = str(G.get("adjacency_penalty", "") or "")
ADJ = [float(v) for v in adj.split(",")] if adj else None
CUT_AT_EOS = bool(G.get("cut_at_eos", True))


def load_spec(spec):
    path, ref = spec.split(":", 1)
    obj = torch.load(os.path.join(ck, path), map_location=dev, weights_only=False)
    for k in ref.split("."):
        obj = obj[k]
    return obj.double()


ALLOW = load_spec(str(G.frequency_allowance)) if G.get("frequency_allowance") else None
L0_P = load_spec(str(G.level0_unigram)).float() if G.get("level0_unigram") else None


def pick_tokens(logits):
    """Copy of scripts/eval_shard.py pick_tokens (frequency / adjacency penalties from the config)."""
    if freq <= 0 and not ADJ:
        return logits.argmax(-1)
    B, T, V = logits.shape
    lg = logits.double()
    allow = (ALLOW[:V] * T)[None, :] if ALLOW is not None else None
    al = ADJ[0] if ADJ else 0.0
    aw = int(ADJ[1]) if ADJ and len(ADJ) > 1 else 1
    ad = ADJ[2] if ADJ and len(ADJ) > 2 else 1.0
    cnt = torch.zeros(B, V, dtype=torch.float64, device=lg.device)
    out = torch.empty(B, T, dtype=torch.long, device=lg.device)
    hist = []
    for t in range(T):
        step = lg[:, t, :].clone()
        if freq > 0:
            excess = (cnt - allow).clamp_min(0.0) if allow is not None else cnt
            step -= freq * excess
            if mu > 0 and allow is not None:
                step += mu * ((T - t) / T) * (allow - cnt).clamp_min(0.0)
        for j, h in enumerate(reversed(hist[-aw:])):
            step.scatter_add_(1, h[:, None], torch.full((B, 1), -al * ad ** j, dtype=torch.float64, device=lg.device))
        p = step.argmax(-1)
        out[:, t] = p
        hist.append(p)
        cnt.scatter_add_(1, p[:, None], torch.ones(B, 1, dtype=torch.float64, device=lg.device))
    return out


def dec(ids):
    ids = list(ids)
    if CUT_AT_EOS and EOS in ids:
        ids = ids[:ids.index(EOS)]
    return tk.pre_tokenizer.decode([t for t in ids if t not in (EOS, PAD)], skip_special_tokens=True)


def split_codes(idx):
    out, acc = [], 0
    for L in levels:
        out.append(idx[:, acc:acc + L])
        acc += L
    return out


_orig_sample = net._sample_level
_state = {"codes": None, "calls": 0, "upto": 0, "offset": 0, "temps": None, "guide": 0.0, "B": None}


def _sample_with_force(logits, temperature, top_k_, top_p_):
    """Called once per sampled level by net.generate. Level-0 calls (query/predict level 0 when no l0 is
    given) come first and are counted by "offset". With guidance the batch is [canonical; draft] (2B rows):
    sample from (1 + w) * canonical - w * draft logits and give both halves the same codes."""
    level = _state["calls"] + 1 - _state["offset"]
    _state["calls"] += 1
    if level >= 1 and _state["codes"] is not None and level < _state["upto"]:
        return _state["codes"][level]  # forced to the re-encoded codes
    t = _state["temps"][max(level, 0)] if _state["temps"] is not None else temperature
    if _state["guide"]:
        B, w = _state["B"], _state["guide"]
        pred = _orig_sample((1.0 + w) * logits[:B] - w * logits[B:], t, top_k_, top_p_)
        return torch.cat([pred, pred], dim=0)
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
    g = net.generate(l0, prefix, temperature=temps[-1], top_k=top_k, top_p=top_p,
                     cfg_scale=cfg_scale, cfg_null_eos=EOS)
    _state.update(codes=None, calls=0, upto=0, offset=0, temps=None)
    _, zl = vq.idx_to_zhats(g, return_zlevels=True)
    return pick_tokens(tk.vqvae.decoder(zl[-1])), g


@torch.no_grad()
def generate_guided(src, anti, prefix, temps, upto, w):
    """Round-trip guidance: regenerate levels >= upto on [canonical; draft] coarse stacks and extrapolate the
    fine-level logits away from the draft-conditioned (off-manifold) branch. Returns the canonical half."""
    B = prefix.shape[0]
    forced = [torch.cat([s, x], dim=0) for s, x in zip(src, anti)]
    _state.update(codes=forced, calls=0, upto=upto, offset=0, temps=temps, guide=w, B=B)
    g = net.generate(forced[0], torch.cat([prefix, prefix], dim=0), temperature=temps[-1], top_k=top_k, top_p=top_p,
                     cfg_scale=0.0, cfg_null_eos=EOS)
    _state.update(codes=None, calls=0, upto=0, offset=0, temps=None, guide=0.0, B=None)
    g = g[:B]
    _, zl = vq.idx_to_zhats(g, return_zlevels=True)
    return pick_tokens(tk.vqvae.decoder(zl[-1])), g


def parse(name):
    """full | t<T>@<k> (temperature T for levels >= k) | d<T> (draft at temperature T, rounds keep the others) |
    r<k> | r<k>x<n> (n rounds) | n<k> (ablation: resample levels >= k from the draft's own codes, no re-encode) |
    q<k> (ablation: re-quantize the draft's latent) | g<w> (round-trip guidance weight for every round) |
    h (start from real validation text instead of a draft) |
    parts joined by '+'. Rounds run in the order written, e.g. r4+r8 = redo from level 4, then redo from level 8."""
    temps, draft, rounds, guide, human = [temp0] * K, None, [], 0.0, False
    if name == "full":
        return temps, temps, rounds, guide, human
    for part in name.split("+"):
        if part == "h":
            human = True
        elif part.startswith("g"):
            guide = float(part[1:])
        elif part.startswith("t"):
            T, k = part[1:].split("@")
            for j in range(int(k), K):
                temps[j] = float(T)
        elif part.startswith("d"):
            draft = [float(part[1:])] * K
        elif part[0] in "rnq":
            k, n = (part[1:].split("x") + ["1"])[:2]
            rounds += [(int(k), part[0])] * int(n)
        else:
            raise ValueError(name)
    return temps, draft or temps, rounds, guide, human


# ---------------- inputs: unconditional (TinyStories, LM1B) or conditional test set (XSum, WMT14) ----------------
COND = bool(a.split)
if COND:
    from transformers import AutoTokenizer
    dd = a.data_dir or os.path.join("data", CFG.task)
    items = pickle.load(open(os.path.join(dd, CFG.data[a.split]), "rb"))[:a.n]
    refs = pickle.load(open(os.path.join(dd, CFG.data[f"{a.split}_refs"]), "rb"))[:a.n]
    src_tok = AutoTokenizer.from_pretrained(str(CFG.tokenizer.pre_tokenizer.model_id)) if isinstance(items[0], str) else None
    P = int(CFG.model.prefix_len)

    def cond_prefix(item):
        ids = src_tok(item)["input_ids"] if src_tok is not None else list(item)
        return (ids[:P] + [PAD] * P)[:P]

    refs = [r if isinstance(r, str) else dec(r) for r in refs]
    N = len(items)
else:
    N = a.n

HUMAN = None
if any("h" in v.split("+") for v in a.variants.split(",")):
    T_ctx = int(CFG.tokenizer.vqvae.context_length)  # same openings as real_ref.py: first rows with >= T tokens
    HUMAN = [list(r)[:T_ctx] for r in pickle.load(open(a.human, "rb")) if len(r) >= T_ctx][:3 * N]
    HSTART = ((a.seed - 4) % 3) * N  # seeds 4, 5, 6 take disjoint slices of the first 3N openings

results = {}
t0 = time.time()
for name in a.variants.split(","):
    temps, draft, rounds, guide, human = parse(name)
    torch.manual_seed(a.seed)
    texts, raw = [], []
    changed, nchg = torch.zeros(K), 0  # share of each level's codes that the first re-encode changes
    for s in range(0, N, a.bs):
        B = min(a.bs, N - s)
        if COND:
            l0 = None
            prefix = torch.tensor([cond_prefix(items[i]) for i in range(s, s + B)], device=dev)
        else:
            l0 = torch.multinomial(L0_P.cpu(), B * levels[0], replacement=True).view(B, levels[0]).to(dev) \
                if L0_P is not None else None
            prefix = torch.full((B, 1), EOS, device=dev)
        if human:  # real text in place of the draft; rounds then run unchanged
            toks = torch.tensor(HUMAN[HSTART + s:HSTART + s + B], device=dev)
            g = tk.vqvae.encode(toks)
        else:
            toks, g = generate(l0, prefix, draft)
        for i, (k, mode) in enumerate(rounds):
            if mode == "r":
                src = split_codes(tk.vqvae.encode(toks))
            elif mode == "q":
                src = split_codes(vq.z_to_idx(vq.idx_to_zhats(g, return_zlevels=True)[1][-1]))
            else:
                src = split_codes(g)
            if i == 0 and mode != "n":
                changed += torch.stack([(s != d).float().mean() for s, d in zip(src, split_codes(g))]).cpu() * B
                nchg += B
            if guide and mode != "n":
                toks, g = generate_guided(src, split_codes(g), prefix, temps, k, guide)
            else:
                toks, g = generate(src[0], prefix, temps, forced=src, upto=k)
        texts += [dec(t) for t in toks.tolist()]
        raw += toks.tolist()
    nfe = (0 if human else K - 1) + sum(K - k for k, _ in rounds)
    results[name] = {"skip": [], "nfe": nfe, "texts": texts, "raw": raw,
                     "changed": (changed / nchg).tolist() if nchg else None}
    chg = " changed/level " + " ".join(f"{v:.2f}" for v in results[name]["changed"]) if nchg else ""
    print(f"{name:12s} NFE={nfe:2d} ({time.time() - t0:.0f}s){chg} | {texts[0][:90]!r}", flush=True)

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
out = {"levels": levels, "config": a.config, "results": results}
if COND:
    out["refs"] = refs
pickle.dump(out, open(a.out, "wb"))
print(f"wrote {a.out}")
