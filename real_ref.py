"""Reference point for gen-PPL: real validation story openings, cut to the model's 128 tokens and decoded
the same way as samples. Writes a pickle in the analyze_skip.py format ({"results": {"real": ...}})."""
import argparse, os, pickle, sys
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config/eval/tinystories.yaml")
ap.add_argument("--checkpoint-dir", default="artifacts/tinystories")
ap.add_argument("--valid", default="data/tinystories/valid_bpe_8192.pickle")
ap.add_argument("--n", type=int, default=768)
ap.add_argument("--out", default="round_trip/runs/real_ref.pkl")
a = ap.parse_args()
os.chdir(ROOT)

CFG = OmegaConf.load(a.config)
ck = a.checkpoint_dir
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
    if CFG.checkpoints.get(name):
        CFG.tokenizer.pre_tokenizer[key] = os.path.join(ck, CFG.checkpoints[name])
tk = VQVAETokenizer.from_config(CFG.tokenizer)
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
T = int(CFG.tokenizer.vqvae.context_length)

rows = [list(r) for r in pickle.load(open(a.valid, "rb")) if len(r) >= T][: a.n]
texts = []
for r in rows:
    ids = r[:T]
    if EOS in ids:
        ids = ids[:ids.index(EOS)]
    texts.append(tk.pre_tokenizer.decode([t for t in ids if t not in (EOS, PAD)], skip_special_tokens=True))
os.makedirs(os.path.dirname(a.out), exist_ok=True)
pickle.dump({"levels": [], "results": {"real": {"skip": [], "nfe": 0, "texts": texts}}}, open(a.out, "wb"))
print(f"wrote {a.out}: {len(texts)} real openings")
