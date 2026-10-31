"""Step 1 of the bigger-model plan: does the released LM1B tokenizer (BERT-uncased, 128 tokens, 12 levels) reconstruct
OpenWebText as well as it reconstructs LM1B? If yes, a bigger generator can be trained on OWT with this tokenizer and no
tokenizer training. Tokenizer only, CPU.

OWT rows are built to mimic the LM1B stream format seen in valid_bert_30522.bin: a [CLS]-separated token stream cut
into 128-token rows whose last token is [CLS]. Sentence boundaries in OWT are approximated by inserting [CLS] after
'.', '?' and '!' tokens.
"""
import argparse, json, os, sys, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from omegaconf import OmegaConf  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config/eval/lm1b-large.yaml")
ap.add_argument("--checkpoint-dir", default="artifacts/lm1b/large")
ap.add_argument("--owt", default="data/owt/valid_gpt2.bin")
ap.add_argument("--lm1b", default="data/lm1b/valid_bert_30522.bin")
ap.add_argument("--n", type=int, default=1024, help="rows per corpus")
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--out", default="round_trip/runs/p14_owt_tokenizer_transfer.json")
a = ap.parse_args()
os.chdir(ROOT)
torch.manual_seed(0)

CFG = OmegaConf.load(a.config)
ck = a.checkpoint_dir
CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
tk = VQVAETokenizer.from_config(CFG.tokenizer)
vae = tk.vqvae.eval()
vq = vae.vq
levels = list(vq.levels)
L = sum(levels)
CTX = int(CFG.tokenizer.vqvae.context_length)
CLS = tk.pre_tokenizer.eos_token_id
print(f"levels={levels} ctx={CTX} CLS={CLS}", flush=True)

# ---- LM1B rows: the released validation stream, as is
lm = np.fromfile(a.lm1b, dtype=np.uint16).reshape(-1, CTX)
rng = np.random.default_rng(0)
lm_rows = torch.tensor(lm[rng.choice(len(lm), a.n, replace=False)].astype(np.int64))

# ---- OWT rows: GPT-2 ids -> text -> BERT-uncased ids with [CLS] after sentence ends -> 128-token rows ending in [CLS]
gpt2 = AutoTokenizer.from_pretrained("gpt2")
bert = AutoTokenizer.from_pretrained(str(CFG.tokenizer.pre_tokenizer.model_id))
SENT_END = {bert.convert_tokens_to_ids(t) for t in (".", "?", "!")}
owt = np.fromfile(a.owt, dtype=np.uint16)
off = np.load(a.owt.replace(".bin", "_offsets.npy"))
docs = rng.permutation(len(off) - 1)
import re  # noqa: E402
SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")  # sentence ends followed by whitespace and a capital: not inside URLs
stream = [CLS]
for d in docs:
    text = gpt2.decode(owt[off[d]:off[d + 1]].tolist())
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        for sent in SPLIT.split(para):
            ids = bert(sent, add_special_tokens=False)["input_ids"]
            if ids:
                stream.extend(ids); stream.append(CLS)
    if len(stream) > (a.n + 2) * (CTX - 1):
        break
owt_rows = []
p = 0
while len(owt_rows) < a.n:
    row = stream[p:p + CTX - 1] + [CLS]
    p += CTX - 1
    owt_rows.append(row)
owt_rows = torch.tensor(owt_rows, dtype=torch.long)
print(f"owt rows {tuple(owt_rows.shape)} from {len(docs)} docs; lm1b rows {tuple(lm_rows.shape)}", flush=True)


@torch.no_grad()
def reconstruct(x):
    idx = vae.encode(x)
    _, zl = vq.idx_to_zhats(idx, return_zlevels=True)
    rec = vae.decoder(zl[-1]).argmax(-1)
    return idx, rec


@torch.no_grad()
def run(name, rows):
    t0 = time.time()
    tok_ok, row_ok, n_tok, n_row = 0, 0, 0, 0
    code_use = [set() for _ in levels]
    for i in range(0, len(rows), a.bs):
        x = rows[i:i + a.bs]
        idx, rec = reconstruct(x)
        eq = (rec == x)
        tok_ok += int(eq.sum()); n_tok += eq.numel()
        row_ok += int(eq.all(1).sum()); n_row += len(x)
        acc = 0
        for li, Lk in enumerate(levels):
            code_use[li].update(idx[:, acc:acc + Lk].flatten().tolist()); acc += Lk
    out = {"rows": n_row, "token_acc": tok_ok / n_tok, "row_exact": row_ok / n_row,
           "codes_used_per_level": [len(s) for s in code_use], "sec": time.time() - t0}
    print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}), flush=True)
    return out


res = {"lm1b": run("lm1b", lm_rows), "owt": run("owt", owt_rows)}
# a few OWT examples: original vs reconstruction, so the kind of error is visible
ex = []
idx, rec = reconstruct(owt_rows[:8])
for x, r in zip(owt_rows[:8], rec):
    if not bool((x == r).all()):
        ex.append({"orig": bert.decode(x.tolist()), "recon": bert.decode(r.tolist()),
                   "n_wrong": int((x != r).sum())})
res["owt_examples"] = ex[:4]
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
for e in ex[:2]:
    print("ORIG ", e["orig"][:300]); print("RECON", e["recon"][:300]); print("wrong tokens", e["n_wrong"])
print("wrote", a.out)
