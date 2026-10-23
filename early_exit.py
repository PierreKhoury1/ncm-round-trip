"""Adaptive-depth (early-exit) study for Next-Concept Modeling.

One full NCM generation yields codes for every level. Because level k is sampled
conditioned only on levels < k, stopping generation after level k produces exactly
the first-k-level prefix of the full code sequence. So one generation gives the
early-exit sample for every k at once: decode the partial residual sum z_hat_k
with the VQ-VAE decoder and measure quality.

Per sample and level we log:
  text_k            decoded text using levels 0..k
  changed_k         fraction of the 128 positions whose decoded token differs from level k-1
  changed_final_k   fraction differing from the final (all-level) decode
  dec_conf_k        mean max-softmax of the decoder at level k (free stop signal)
  gen_conf_k        mean max-softmax of the generator's logits when it sampled level k

Outputs a pickle consumed by analyze.py (gen-PPL, NFE/quality curves, policies).
"""
import argparse, os, pickle, sys, time
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
ap.add_argument("--n", type=int, default=64, help="samples per seed")
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--seeds", default="42")
ap.add_argument("--temp", type=float, default=None)
ap.add_argument("--top_k", type=int, default=None)
ap.add_argument("--top_p", type=float, default=None)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--out", default="round_trip/runs/tinystories_ee.pkl")
a = ap.parse_args()

os.chdir(ROOT)
CFG = OmegaConf.load(a.config)
G = CFG.generation
ck = a.checkpoint_dir
dev = torch.device(a.device)
pick = lambda cli, key, cast, default: cast(cli if cli is not None else G.get(key, default))
temp = pick(a.temp, "temperature", float, 1.0)
top_k = pick(a.top_k, "top_k", int, 0)
top_p = pick(a.top_p, "top_p", float, 1.0)

CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
    if CFG.checkpoints.get(name):
        CFG.tokenizer.pre_tokenizer[key] = os.path.join(ck, CFG.checkpoints[name])
tk = VQVAETokenizer.from_config(CFG.tokenizer)
tk.vqvae = tk.vqvae.to(dev).eval()
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).to(dev)
sd = torch.load(os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
net.load_state_dict(sd["model"], strict=True)
net.eval()
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
vq = tk.vqvae.vq
levels = list(vq.levels)
n_lev = len(levels)
print(f"levels={levels}  (NFE for full generation = {n_lev - 1})")

# level-0 unigram prior, as in scripts/eval_shard.py
spec = str(G.level0_unigram)
path, ref = spec.split(":", 1)
obj = torch.load(os.path.join(ck, path), map_location="cpu", weights_only=False)
for k in ref.split("."):
    obj = obj[k]
L0_P = obj.double().float()

# --- hook generator sampling to record its confidence per level ---------------------
GEN_CONF = []  # list over levels of tensor [B]
_orig_sample = net._sample_level_raw


def _sample_rec(logits, temperature, top_k_, top_p_):
    p = F.softmax(logits.float(), dim=-1).max(-1).values  # [B, L]
    GEN_CONF.append(p.mean(-1).cpu())
    return _orig_sample(logits, temperature, top_k_, top_p_)


net._sample_level_raw = _sample_rec


@torch.no_grad()
def partial_zhats(idx):
    """Return list of z_hat_k (cumulative residual sum through level k), each a fresh tensor."""
    out = []
    z_hat = torch.zeros(idx.shape[0], levels[-1], vq.embed_dim, device=idx.device)
    acc = 0
    for li in range(n_lev):
        emb = vq.embeddings[li](idx[:, acc:acc + levels[li]])
        zq = vq._up_to_full(emb, li, levels[-1]) if li != n_lev - 1 else emb
        zq = vq._apply_quant_resi(zq, li)
        z_hat = z_hat + zq
        out.append(z_hat)
        acc += levels[li]
    return out


def dec_text(ids):
    ids = list(ids)
    if EOS in ids:
        ids = ids[:ids.index(EOS)]
    return tk.pre_tokenizer.decode([t for t in ids if t not in (EOS, PAD)], skip_special_tokens=True)


records = []
t0 = time.time()
for seed in [int(s) for s in a.seeds.split(",")]:
    torch.manual_seed(seed)
    for s in range(0, a.n, a.bs):
        B = min(a.bs, a.n - s)
        GEN_CONF.clear()
        l0 = torch.multinomial(L0_P, B * levels[0], replacement=True).view(B, levels[0]).to(dev)
        prefix = torch.full((B, 1), EOS, device=dev)
        with torch.no_grad():
            codes = net.generate(l0, prefix, temperature=temp, top_k=top_k, top_p=top_p, cfg_scale=0.0, cfg_null_eos=EOS)
            zs = partial_zhats(codes)
            toks, confs = [], []
            for z in zs:
                logits = tk.vqvae.decoder(z)
                pr = F.softmax(logits.float(), -1)
                toks.append(logits.argmax(-1))                       # [B, 128]
                confs.append(pr.max(-1).values.mean(-1).cpu())       # [B]
        toks = torch.stack(toks)        # [K, B, 128]
        final = toks[-1]
        gen_conf = torch.stack(GEN_CONF) if GEN_CONF else None  # [K-1, B]
        for b in range(B):
            rec = {"seed": seed, "i": s + b, "codes": codes[b].cpu(), "levels": []}
            for k in range(n_lev):
                tk_k = toks[k, b]
                rec["levels"].append({
                    "k": k,
                    "nfe": k,  # level 0 is sampled from the unigram prior: no generator pass
                    "text": dec_text(tk_k.tolist()),
                    "ids": tk_k.tolist(),
                    "changed": float((tk_k != toks[k - 1, b]).float().mean()) if k > 0 else 1.0,
                    "changed_final": float((tk_k != final[b]).float().mean()),
                    "dec_conf": float(confs[k][b]),
                    "gen_conf": float(gen_conf[k - 1, b]) if (gen_conf is not None and k > 0) else None,
                })
            records.append(rec)
        print(f"seed {seed}: {s + B}/{a.n}  ({time.time() - t0:.0f}s)", flush=True)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
pickle.dump({"levels": levels, "config": a.config, "temp": temp, "top_k": top_k, "top_p": top_p,
             "records": records}, open(a.out, "wb"))
print(f"wrote {len(records)} samples x {n_lev} levels -> {a.out}")

# quick peek
r = records[0]
for L in r["levels"]:
    print(f"\n--- k={L['k']} nfe={L['nfe']} changed={L['changed']:.2f} vs_final={L['changed_final']:.2f} "
          f"dec_conf={L['dec_conf']:.2f} gen_conf={L['gen_conf']}\n{L['text'][:300]}")
