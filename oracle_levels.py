"""Where does the information live? Oracle reconstruction from ground-truth codes.

Encode real TinyStories validation rows with the frozen VQ-VAE, then decode from
subsets of levels and measure token accuracy vs the input. No generator involved.

  prefix k      levels 0..k            (what early exit would see with perfect codes)
  only j        level j alone
  drop j        all levels except j
  suffix k      levels k..K-1          (do fine levels carry the text by themselves?)
"""
import argparse, os, pickle, sys
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config/eval/tinystories.yaml")
ap.add_argument("--checkpoint-dir", default="artifacts/tinystories")
ap.add_argument("--data", default="data/tinystories/valid_bpe_8192.pickle")
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--out", default="round_trip/runs/ts_oracle.pkl")
a = ap.parse_args()
os.chdir(ROOT)
CFG = OmegaConf.load(a.config)
ck = a.checkpoint_dir
dev = torch.device(a.device)
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
    if CFG.checkpoints.get(name):
        CFG.tokenizer.pre_tokenizer[key] = os.path.join(ck, CFG.checkpoints[name])
tk = VQVAETokenizer.from_config(CFG.tokenizer)
vae = tk.vqvae.to(dev).eval()
vq = vae.vq
levels = list(vq.levels)
K = len(levels)
T = vae.context_length
PAD = tk.pre_tokenizer.pad_token_id

rows = pickle.load(open(a.data, "rb"))
print(f"{len(rows)} validation rows; type={type(rows).__name__}")
# rows may be a list of token lists or a flat stream; normalise to [n, T]
if isinstance(rows, (list, tuple)) and len(rows) and isinstance(rows[0], (list, tuple, np.ndarray)):
    seqs = [list(r)[:T] for r in rows if len(r) >= 32][: a.n]
else:
    flat = np.asarray(rows).reshape(-1)
    seqs = [flat[i:i + T].tolist() for i in range(0, min(len(flat), a.n * T), T)]
seqs = [s + [PAD] * (T - len(s)) for s in seqs]
X = torch.tensor(seqs, dtype=torch.long)
print(f"{X.shape[0]} sequences of {T} tokens")


@torch.no_grad()
def zhat_subset(idx, keep):
    z = torch.zeros(idx.shape[0], levels[-1], vq.embed_dim, device=idx.device)
    acc = 0
    for li in range(K):
        if li in keep:
            emb = vq.embeddings[li](idx[:, acc:acc + levels[li]])
            zq = vq._up_to_full(emb, li, levels[-1]) if li != K - 1 else emb
            z = z + vq._apply_quant_resi(zq, li)
        acc += levels[li]
    return z


subsets = {}
for k in range(K):
    subsets[f"prefix{k}"] = set(range(k + 1))
for j in range(K):
    subsets[f"only{j}"] = {j}
for j in range(K):
    subsets[f"drop{j}"] = set(range(K)) - {j}
for k in range(1, K):
    subsets[f"suffix{k}"] = set(range(k, K))

acc = {name: [] for name in subsets}
with torch.no_grad():
    for s in range(0, X.shape[0], a.bs):
        x = X[s:s + a.bs].to(dev)
        idx = vae.encode(x)
        mask = (x != PAD)
        for name, keep in subsets.items():
            pred = vae.decoder(zhat_subset(idx, keep)).argmax(-1)
            acc[name].append(((pred == x) & mask).sum().item() / mask.sum().item())

res = {n: float(np.mean(v)) for n, v in acc.items()}
pickle.dump({"levels": levels, "acc": res}, open(a.out, "wb"))
print(f"\nlevels={levels}\n")
print(f"{'k':>3} {'codes':>6} | {'prefix 0..k':>12} {'only k':>8} {'drop k':>8} {'suffix k..':>11}")
for k in range(K):
    print(f"{k:3d} {levels[k]:6d} | {100 * res[f'prefix{k}']:11.2f}% {100 * res[f'only{k}']:7.2f}% "
          f"{100 * res[f'drop{k}']:7.2f}% {100 * res.get(f'suffix{k}', float('nan')):10.2f}%")
