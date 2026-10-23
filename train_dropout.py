"""Level-dropout fine-tuning of a released NCM generator.

Teacher-forced training exactly as the generator expects (forward over all levels at once,
block-diagonal attention), except that the input for target level j is built from a random
subset of the coarser levels < j (level 0 always kept). The model learns to predict level j
from a sparser z_hat, so at inference we can skip levels and still get full quality.

  --p_drop 0.0  reproduces the original training objective (sanity / baseline)
  --p_drop 0.3  each level 1..K-2 is dropped from the inputs independently with prob 0.3

Also usable with --eval_only to measure teacher-forced CE of a checkpoint under full vs
dropped inputs (cheap mismatch measurement, no training).
"""
import argparse, math, os, pickle, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.next_concept import NextConceptTransformer  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config/eval/tinystories.yaml")
ap.add_argument("--checkpoint-dir", default="artifacts/tinystories")
ap.add_argument("--init", default="", help="generator checkpoint to start from (default: released)")
ap.add_argument("--train", default="data/tinystories/train_bpe_8192.bin")
ap.add_argument("--valid", default="data/tinystories/valid_bpe_8192.pickle")
ap.add_argument("--dtype", default="uint16")
ap.add_argument("--p_drop", type=float, default=0.3)
ap.add_argument("--distill", default="", help="refine_eval.py pickle(s), comma-separated, whose refined stories are "
                                             "mixed into training")
ap.add_argument("--resume", action="store_true", help="continue from --out (and its .opt optimizer state) if present")
ap.add_argument("--bsc", action="store_true", help="self-correction targets: re-quantize each level's target from the "
                                                   "residual left by the CORRUPTED coarser inputs (Infinity-style)")
ap.add_argument("--corrupt_nn", type=int, default=0, help="draw replacement codes from the k nearest codes to the "
                                                          "residual instead of uniformly (needs --bsc)")
ap.add_argument("--smoke_bsc", action="store_true", help="check that --bsc with corruption off reproduces vae.encode, then exit")
ap.add_argument("--distill_variant", default="r5x2")
ap.add_argument("--distill_frac", type=float, default=0.5, help="share of each batch drawn from the refined stories")
ap.add_argument("--self_mix", type=float, default=0.0, help="fraction of corrupted input codes taken from the "
                "model's own sampled guesses (one no-grad teacher-forced pass) instead of uniform random codes")
ap.add_argument("--corrupt", default="", help="'max,min': replace input codes uniformly from the level's codebook at a "
                "rate decreasing linearly over input levels (paper 3.4 / F.3 conditioning corruption); targets stay clean")
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--accum", type=int, default=1, help="gradient accumulation micro-steps; micro-batch = bs // accum")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--warmup", type=int, default=100)
ap.add_argument("--wd", type=float, default=0.1)
ap.add_argument("--eval_every", type=int, default=200)
ap.add_argument("--n_valid", type=int, default=512)
ap.add_argument("--amp", action="store_true")
ap.add_argument("--eval_only", action="store_true")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--out", default="round_trip/ckpt/generator_dropout.pt")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.chdir(ROOT)
torch.manual_seed(a.seed)
np.random.seed(a.seed)
dev = torch.device(a.device)

CFG = OmegaConf.load(a.config)
ck = a.checkpoint_dir
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
    if CFG.checkpoints.get(name):
        CFG.tokenizer.pre_tokenizer[key] = os.path.join(ck, CFG.checkpoints[name])
tk = VQVAETokenizer.from_config(CFG.tokenizer)
vae = tk.vqvae.to(dev).eval()
for p_ in vae.parameters():
    p_.requires_grad_(False)
vq = vae.vq
levels = list(vq.levels)
K = len(levels)
T = vae.context_length
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id

net = NextConceptTransformer(vqvae_local=vae, config=CFG.model).to(dev)
sd = torch.load(a.init or os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
net.load_state_dict(sd["model"], strict=True)
P = net.prefix_len
assert P == 1 and bool(CFG.model.raw_token_prefix) and not net.query_level0 and not net.predict_level0, \
    "script written for the TinyStories/LM1B-style unconditional configs (prefix = one raw EOS token)"

# ---------------- data ----------------
train = np.memmap(a.train, dtype=a.dtype, mode="r")
n_train = len(train)
_off_path = a.train.replace(".bin", "_offsets.npy")
OFFSETS = np.load(_off_path) if os.path.exists(_off_path) else None
if OFFSETS is not None:
    OFFSETS = OFFSETS[OFFSETS < n_train - 2]
valid_rows = pickle.load(open(a.valid, "rb"))
vseqs = [list(r)[:T] for r in valid_rows if len(r) >= 32][: a.n_valid]
vseqs = [s + [PAD] * (T - len(s)) for s in vseqs]
VALID = torch.tensor(vseqs, dtype=torch.long)
print(f"train stream {n_train:,} tokens; valid {VALID.shape[0]} x {T}; levels={levels}", flush=True)


DISTILL = None
if a.distill:
    _rows = [r for f in a.distill.split(",") for r in pickle.load(open(f, "rb"))["results"][a.distill_variant]["raw"]]
    DISTILL = torch.tensor([list(r)[:T] + [PAD] * (T - len(r)) for r in _rows], dtype=torch.long)
    print(f"distill: {DISTILL.shape[0]} refined stories ({a.distill_variant}), frac {a.distill_frac}", flush=True)


def train_batch(bs):
    """Real story-aligned windows, optionally mixed with refined (distillation) stories."""
    if DISTILL is not None:
        nd = int(np.random.binomial(bs, a.distill_frac))
        parts = [DISTILL[torch.randint(0, DISTILL.shape[0], (nd,))]] if nd else []
        if bs - nd:
            parts.append(real_batch(bs - nd))
        return torch.cat(parts)
    return real_batch(bs)


def real_batch(bs):
    """Story-aligned windows: start at a story offset, cut at the next story's start, pad to T."""
    if OFFSETS is None:
        starts = np.random.randint(0, n_train - T - 1, size=bs)
        return torch.from_numpy(np.stack([train[s:s + T] for s in starts]).astype(np.int64))
    i = np.random.randint(0, len(OFFSETS) - 1, size=bs)
    rows = []
    for j in i:
        s, e = int(OFFSETS[j]), int(min(OFFSETS[j + 1], OFFSETS[j] + T))
        r = train[s:e].astype(np.int64)
        rows.append(np.concatenate([r, np.full(T - len(r), PAD, dtype=np.int64)]))
    return torch.from_numpy(np.stack(rows))


# ---------------- inputs with level dropout ----------------
@torch.no_grad()
def build_inputs(idx, p_drop):
    """idx: [B, sum(levels)] codes. Returns x_in [B, P + sum(levels[1:]), D] for net.forward and keep mask."""
    B = idx.shape[0]
    keep = torch.ones(B, K, dtype=torch.bool, device=idx.device)
    if p_drop > 0:
        keep[:, 1:K - 1] = torch.rand(B, K - 2, device=idx.device) >= p_drop  # never drop level 0 or the last
    z_hat = torch.zeros(B, levels[-1], vq.embed_dim, device=idx.device)
    parts, acc = [], 0
    for li in range(K - 1):
        emb = vq.embeddings[li](idx[:, acc:acc + levels[li]])
        zq = vq._apply_quant_resi(vq._up_to_full(emb, li, levels[-1]), li)
        z_hat = z_hat + zq * keep[:, li, None, None].to(zq.dtype)
        parts.append(F.interpolate(z_hat.transpose(-1, -2), size=(levels[li + 1],), mode="area").transpose(-1, -2))
        acc += levels[li]
    x_target = torch.cat(parts, dim=1)
    z_prefix = vae.encoder.wte(torch.full((B, P), EOS, device=idx.device))
    return torch.cat([z_prefix, x_target], dim=1), keep


CORR = None
if a.corrupt:
    _cmax, _cmin = (float(v) for v in a.corrupt.split(","))
    CORR = [_cmax + (_cmin - _cmax) * li / (K - 2) for li in range(K - 1)]
    print("corruption per input level: " + " ".join(f"{v:.2f}" for v in CORR), flush=True)


@torch.no_grad()
def corrupt_codes(idx, guesses=None):
    """guesses: [B, sum(levels[1:])] model-sampled codes for levels 1..K-1 (None = paper's uniform corruption)."""
    if CORR is None:
        return idx
    out, acc, goff = idx.clone(), 0, 0
    for li in range(K - 1):
        L, V = levels[li], vq.embeddings[li].weight.shape[0]
        m = torch.rand(idx.shape[0], L, device=idx.device) < CORR[li]
        rep = torch.randint(0, V, (idx.shape[0], L), device=idx.device)
        if guesses is not None and li >= 1:
            own = torch.rand(idx.shape[0], L, device=idx.device) < a.self_mix
            rep = torch.where(own, guesses[:, goff:goff + L], rep)
            goff += L
        out[:, acc:acc + L] = torch.where(m, rep, idx[:, acc:acc + L])
        acc += L
    return out


@torch.no_grad()
def bsc_codes(x, corrupt=True):
    """Inputs and self-correction targets in one residual-quantization pass (mirrors quant.z_to_idx): at level i the
    target is the code nearest to the residual left by the inputs actually given at levels < i, and the input is that
    target, replaced at rate CORR[i] (uniformly, or by one of the k nearest codes). With corruption off this is exactly
    vae.encode(x)."""
    z = vae._encode_pre_quant(x).float()
    B, Lz, C = z.shape
    z_rest = z.clone()
    inp, tgt = [], []
    for i, level in enumerate(levels):
        emb = vq.embeddings[i]
        flat = vq._down_for_quant(z_rest, i, level).reshape(-1, C) if i != K - 1 else z_rest.reshape(-1, C)
        dist = vq._distances(flat, emb.weight)
        t = torch.argmin(dist, dim=1).view(B, level)
        u = t
        if corrupt and CORR is not None and i < K - 1:
            m = torch.rand(B, level, device=x.device) < CORR[i]
            if a.corrupt_nn > 0:
                near = torch.topk(dist, a.corrupt_nn + 1, dim=1, largest=False).indices[:, 1:]
                rep = near.gather(1, torch.randint(0, a.corrupt_nn, (near.shape[0], 1), device=x.device)).view(B, level)
            else:
                rep = torch.randint(0, emb.weight.shape[0], (B, level), device=x.device)
            u = torch.where(m, rep, t)
        tgt.append(t)
        inp.append(u)
        q = vq._up_to_full(emb(u), i, Lz) if i != K - 1 else emb(u)
        z_rest -= vq._apply_quant_resi(q, i)
    return torch.cat(inp, dim=1), torch.cat(tgt, dim=1)


@torch.no_grad()
def own_guesses(idx):
    """One clean teacher-forced pass in eval mode; sample every level at temperature 1, as at generation."""
    was = net.training
    net.eval()
    x_in, _ = build_inputs(idx, 0.0)
    with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=a.amp):
        logits = net(x_in)
    net.train(was)
    V = vq.embeddings[1].weight.shape[0]
    probs = logits[..., :V].float().softmax(-1)
    return torch.multinomial(probs.reshape(-1, V), 1).view(idx.shape[0], -1)


def loss_fn(logits, targets):
    # logits [B, sum(levels[1:]), Vmax]; targets [B, sum(levels[1:])]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1))


def per_level_ce(logits, targets):
    out, off = [], 0
    for li in range(1, K):
        L = levels[li]
        out.append(F.cross_entropy(logits[:, off:off + L].reshape(-1, logits.shape[-1]).float(),
                                   targets[:, off:off + L].reshape(-1)).item())
        off += L
    return out


@torch.no_grad()
def evaluate(p_drop_eval, n=None):
    net.eval()
    tot, cnt, per = 0.0, 0, np.zeros(K - 1)
    X = VALID if n is None else VALID[:n]
    ebs = max(1, a.bs // a.accum)
    for s in range(0, X.shape[0], ebs):
        x = X[s:s + ebs].to(dev)
        idx = vae.encode(x)
        x_in, _ = build_inputs(idx, p_drop_eval)
        targets = idx[:, levels[0]:]
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=a.amp):
            logits = net(x_in)
        tot += loss_fn(logits, targets).item() * x.shape[0]
        per += np.array(per_level_ce(logits, targets)) * x.shape[0]
        cnt += x.shape[0]
    net.train()
    return tot / cnt, per / cnt


if a.smoke_bsc:
    x = VALID[:64].to(dev)
    ref = vae.encode(x)
    _, tgt = bsc_codes(x, corrupt=False)
    print(f"[smoke_bsc] corruption off: codes identical to vae.encode: {bool((ref == tgt).all())} "
          f"(mismatch share {(ref != tgt).float().mean().item():.2e})", flush=True)
    inp, tgt = bsc_codes(x)
    acc, rows = 0, []
    for li, L in enumerate(levels):
        rows.append(f"L{li}: input changed {(inp[:, acc:acc + L] != ref[:, acc:acc + L]).float().mean():.2f} "
                    f"target changed {(tgt[:, acc:acc + L] != ref[:, acc:acc + L]).float().mean():.2f}")
        acc += L
    print("[smoke_bsc] with corruption: " + " | ".join(rows), flush=True)
    sys.exit(0)

if a.eval_only:
    for pd in (0.0, 0.15, 0.3, 0.5):
        ce, per = evaluate(pd)
        print(f"p_drop={pd:.2f}  CE={ce:.4f}  per-level: " + " ".join(f"{v:.2f}" for v in per), flush=True)
    sys.exit(0)

# ---------------- train ----------------
decay, no_decay = [], []
for n_, p_ in net.named_parameters():
    if not p_.requires_grad:
        continue
    (decay if p_.dim() >= 2 else no_decay).append(p_)
opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=a.lr, betas=(0.9, 0.95))
sched = lambda step: min(1.0, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(step, a.steps) / a.steps))

start = 0
if a.resume and os.path.exists(a.out) and os.path.exists(a.out + ".opt"):
    _ck = torch.load(a.out, map_location="cpu", weights_only=False)
    net.load_state_dict(_ck["model"], strict=True)
    opt.load_state_dict(torch.load(a.out + ".opt", map_location="cpu", weights_only=False))
    start = int(_ck["step"])
    torch.manual_seed(a.seed + start)
    np.random.seed(a.seed + start)
    print(f"[resume] from step {start}", flush=True)
else:
    ce0, per0 = evaluate(0.0, n=256)
    ced, perd = evaluate(a.p_drop, n=256)
    print(f"[init] CE full={ce0:.4f}  CE drop{a.p_drop}={ced:.4f}", flush=True)
net.train()
t0 = time.time()
for step in range(start, a.steps):
    for g in opt.param_groups:
        g["lr"] = a.lr * sched(step)
    opt.zero_grad(set_to_none=True)
    mb = a.bs // a.accum
    loss_acc = torch.zeros((), device=dev)
    for _ in range(a.accum):
        x = train_batch(mb).to(dev)
        if a.bsc:
            inp_idx, tgt_idx = bsc_codes(x)
            x_in, keep = build_inputs(inp_idx, a.p_drop)
            targets = tgt_idx[:, levels[0]:]
        else:
            with torch.no_grad():
                idx = vae.encode(x)
            guesses = own_guesses(idx) if a.self_mix > 0 else None
            if guesses is not None and step % 100 == 0:
                wrong = (guesses != idx[:, levels[0]:]).float().mean().item()
                print(f"step {step:5d} own-guess error rate {wrong:.3f}", flush=True)
            x_in, keep = build_inputs(corrupt_codes(idx, guesses), a.p_drop)
            targets = idx[:, levels[0]:]
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=a.amp):
            logits = net(x_in)
        loss = loss_fn(logits, targets) / a.accum
        loss.backward()
        loss_acc += loss.detach()
        del logits
    loss = loss_acc
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    opt.step()
    if step % 20 == 0:
        print(f"step {step:5d} loss {loss.item():.4f} lr {opt.param_groups[0]['lr']:.2e} "
              f"{(time.time() - t0) / (step + 1 - start):.2f}s/it", flush=True)
    if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
        ce0, per0 = evaluate(0.0, n=256)
        ced, perd = evaluate(a.p_drop, n=256)
        print(f"[eval {step + 1}] CE full={ce0:.4f}  CE drop{a.p_drop}={ced:.4f}  per-level(drop): "
              + " ".join(f"{v:.2f}" for v in perd), flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        torch.save({"model": net.state_dict(), "inference": sd.get("inference", {}),
                    "train_args": vars(a), "step": step + 1}, a.out)
        if a.resume:
            torch.save(opt.state_dict(), a.out + ".opt")
print(f"saved {a.out}")
