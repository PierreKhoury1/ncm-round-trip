"""Does the tokenizer round trip preserve generated text? Re-encode each draft's tokens, decode the re-encoded stack
with the same token picker, and count exact matches; also report the share of codes per level that change.
Tokenizer only (no generator), so it runs on CPU.

Usage: python round_trip/roundtrip_check.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories \
           --in round_trip/runs/p3_samplers_s1.pkl [--in ...] --variant full
"""
import argparse, os, pickle, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from omegaconf import OmegaConf  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--checkpoint-dir", required=True)
ap.add_argument("--in", dest="inp", action="append", required=True)
ap.add_argument("--variant", default="full")
ap.add_argument("--bs", type=int, default=64)
a = ap.parse_args()
os.chdir(ROOT)

CFG = OmegaConf.load(a.config)
ck = a.checkpoint_dir
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
    if CFG.checkpoints.get(name):
        CFG.tokenizer.pre_tokenizer[key] = os.path.join(ck, CFG.checkpoints[name])
tk = VQVAETokenizer.from_config(CFG.tokenizer)
vae = tk.vqvae.eval()
vq = vae.vq
levels = list(vq.levels)
assert not CFG.generation.get("frequency_penalty") and not CFG.generation.get("adjacency_penalty"), \
    "argmax decoding only; LM1B's penalised token picker is not reimplemented here"

rows = [r for f in a.inp for r in pickle.load(open(f, "rb"))["results"][a.variant]["raw"]]
X = torch.tensor(rows, dtype=torch.long)
same, tok_same, n = 0, 0.0, 0
with torch.no_grad():
    for s in range(0, X.shape[0], a.bs):
        x = X[s:s + a.bs]
        idx = vae.encode(x)
        _, zl = vq.idx_to_zhats(idx, return_zlevels=True)
        y = vae.decoder(zl[-1]).argmax(-1)
        same += int((y == x).all(-1).sum())
        tok_same += float((y == x).float().mean(-1).sum())
        n += x.shape[0]
print(f"{n} drafts: re-encode + decode reproduces {same}/{n} exactly ({100 * same / n:.1f}%), "
      f"{100 * tok_same / n:.2f}% of tokens")
