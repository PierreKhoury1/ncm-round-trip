"""Joint tokenizer + generator fine-tuning of a released NCM model (paper App. I open problem).

Generator: same teacher-forced objective as train_dropout.py (with conditioning corruption).
Tokenizer: encoder + decoder trainable with their reconstruction CE + VQ commitment loss; codebooks
frozen (EMA off) so the generator's tied heads stay valid. Coupling (--couple lam): the encoder's
residual at each level is pulled toward the generator's expected code embedding (generator detached),
i.e. the tokenizer learns codes the generator finds easier to predict, while reconstruction keeps
them informative. lam=0 = joint but uncoupled.

Original docstring of the base script follows.
Level-dropout fine-tuning of a released NCM generator.

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
ap.add_argument("--couple", type=float, default=0.0, help="weight of generator->tokenizer coupling loss")
ap.add_argument("--tok_lr", type=float, default=2e-5)
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
for n_, p_ in vae.named_parameters():
    p_.requires_grad_(not n_.startswith("vq."))
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


def train_batch(bs):
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
def corrupt_codes(idx):
    if CORR is None:
        return idx
    out, acc = idx.clone(), 0
    for li in range(K - 1):
        L, V = levels[li], vq.embeddings[li].weight.shape[0]
        m = torch.rand(idx.shape[0], L, device=idx.device) < CORR[li]
        out[:, acc:acc + L] = torch.where(m, torch.randint(0, V, (idx.shape[0], L), device=idx.device), idx[:, acc:acc + L])
        acc += L
    return out


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


if a.eval_only:
    for pd in (0.0, 0.15, 0.3, 0.5):
        ce, per = evaluate(pd)
        print(f"p_drop={pd:.2f}  CE={ce:.4f}  per-level: " + " ".join(f"{v:.2f}" for v in per), flush=True)
    sys.exit(0)

# ---------------- joint train ----------------
def set_modes():
    net.train(); vae.eval()  # tokenizer trains with grads but no dropout: codes must match the clean eval-time codes


def tok_losses(x):
    """Reconstruction CE + commitment from a grad-carrying encode; returns codes and the encoder output."""
    z = vae._encode_pre_quant(x)
    z_q, vq_loss, *_ = vae.vq(z)
    logits = vae.decoder(z_q)
    rec = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), x.reshape(-1), ignore_index=PAD)
    with torch.no_grad():
        idx = vq.z_to_idx(z.detach())
    return z, idx, rec, vq_loss


def couple_loss(z, idx, gen_logits):
    """Pull each level's encoder residual toward the generator's expected code embedding."""
    z = z.float()
    z_hat = torch.zeros_like(z)
    loss, acc, off = 0.0, 0, 0
    for li in range(K):
        L = levels[li]
        E = vq.embeddings[li].weight
        if li >= 1:
            r = vq._down_for_quant(z - z_hat, li, L) if li != K - 1 else (z - z_hat)
            p = gen_logits[:, off:off + L, :E.shape[0]].float().softmax(-1).detach()
            loss = loss + F.mse_loss(r, p @ E.detach())
            off += L
        with torch.no_grad():
            emb = vq.embeddings[li](idx[:, acc:acc + L])
            zq = vq._up_to_full(emb, li, levels[-1]) if li != K - 1 else emb
            z_hat = z_hat + vq._apply_quant_resi(zq, li)
        acc += L
    return loss / (K - 1)


@torch.no_grad()
def eval_recon(n=256):
    vae.eval()
    X = VALID[:n].to(dev)
    tok = vae.decode(vae.encode(X)).argmax(-1)
    m = X != PAD
    vae.eval()
    return ((tok == X) & m).sum().item() / m.sum().item()


gen_params = [p_ for p_ in net.parameters() if p_.requires_grad]
tok_params = [p_ for p_ in vae.parameters() if p_.requires_grad]
opt = torch.optim.AdamW([{"params": gen_params, "lr": a.lr, "weight_decay": a.wd, "base": a.lr},
                         {"params": tok_params, "lr": a.tok_lr, "weight_decay": 0.0, "base": a.tok_lr}],
                        betas=(0.9, 0.95))
sched = lambda step: min(1.0, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(step, a.steps) / a.steps))

vae.eval()
ce0, _ = evaluate(0.0, n=256)
print(f"[init] CE full={ce0:.4f}  recon_acc={eval_recon():.4f}  tok params {sum(p.numel() for p in tok_params)/1e6:.1f}M", flush=True)
set_modes()
t0 = time.time()
for step in range(a.steps):
    for g in opt.param_groups:
        g["lr"] = g["base"] * sched(step)
    opt.zero_grad(set_to_none=True)
    mb = a.bs // a.accum
    logs = np.zeros(4)
    for _ in range(a.accum):
        x = train_batch(mb).to(dev)
        z, idx, rec, vql = tok_losses(x)  # fp32: bf16 encoding changes fine-level codes vs eval-time encode
        x_in, keep = build_inputs(corrupt_codes(idx), a.p_drop)
        targets = idx[:, levels[0]:]
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=a.amp):
            logits = net(x_in)
        lg = loss_fn(logits, targets)
        lc = couple_loss(z, idx, logits) if a.couple > 0 else torch.zeros((), device=dev)
        loss = (lg + rec + vql + a.couple * lc) / a.accum
        loss.backward()
        logs += np.array([lg.item(), rec.item(), float(vql), lc.item()]) / a.accum
        del logits
    torch.nn.utils.clip_grad_norm_(gen_params + tok_params, 1.0)
    opt.step()
    if step % 20 == 0:
        print(f"step {step:5d} gen {logs[0]:.4f} rec {logs[1]:.4f} vq {logs[2]:.4f} couple {logs[3]:.4f} "
              f"{(time.time() - t0) / (step + 1):.2f}s/it", flush=True)
    if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
        vae.eval()
        ce0, per0 = evaluate(0.0, n=256)
        print(f"[eval {step + 1}] CE full={ce0:.4f}  recon_acc={eval_recon():.4f}  per-level: "
              + " ".join(f"{v:.2f}" for v in per0), flush=True)
        set_modes()
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        torch.save({"model": net.state_dict(), "inference": sd.get("inference", {}),
                    "train_args": vars(a), "step": step + 1}, a.out)
        torch.save({"model": vae.state_dict()}, a.out.replace(".pt", "_tok.pt"))
print(f"saved {a.out}")
