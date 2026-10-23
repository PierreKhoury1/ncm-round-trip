"""Level-skip ablation for NCM: which hierarchy levels can be dropped at inference?

Re-implements NextConceptTransformer.generate's level loop (eager path, no CFG) with a
`skip` set. A skipped level costs no generator pass and contributes no residual: the
next level's input is the area-interpolated z_hat exactly as the model would see it,
just missing one residual term. The generator was trained on the fixed schedule, so
this measures robustness to a schedule it never saw.

Writes a pickle of {schedule_name: [texts]} plus per-schedule NFE for analyze_skip.py.
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
ap.add_argument("--generator", default="", help="override generator checkpoint path")
ap.add_argument("--tokenizer", default="", help="override tokenizer checkpoint path (joint fine-tunes)")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--schedules", default="auto",
                help="comma list of names: full, drop<j>, drop<j>-<k>, keep<a>-<b>-<c>; 'auto' = full + every single drop + a few multi-drops")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--out", default="round_trip/runs/ts_skip.pkl")
a = ap.parse_args()

os.chdir(ROOT)
CFG = OmegaConf.load(a.config)
G = CFG.generation
ck = a.checkpoint_dir
dev = torch.device(a.device)
temp, top_k, top_p = float(G.get("temperature", 1.0)), int(G.get("top_k", 0)), float(G.get("top_p", 1.0))

CFG.tokenizer.vqvae.checkpoint_path = os.path.join(ck, CFG.checkpoints.tokenizer)
for key, name in (("vocab_file_path", "bpe_vocab"), ("merges_file_path", "bpe_merges")):
    if CFG.checkpoints.get(name):
        CFG.tokenizer.pre_tokenizer[key] = os.path.join(ck, CFG.checkpoints[name])
tk = VQVAETokenizer.from_config(CFG.tokenizer)
if a.tokenizer:
    tk.vqvae.load_state_dict(torch.load(a.tokenizer, map_location="cpu", weights_only=False)["model"], strict=True)
tk.vqvae = tk.vqvae.to(dev).eval()
net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).to(dev)
sd = torch.load(a.generator or os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
net.load_state_dict(sd["model"], strict=True)
net.eval()
assert not net.query_level0 and not net.use_ctx_encoder and net.prefix_pad_id is not None
EOS, PAD = tk.pre_tokenizer.eos_token_id, tk.pre_tokenizer.pad_token_id
vq = tk.vqvae.vq
levels = list(vq.levels)
K = len(levels)
P = net.prefix_len

spec = str(G.level0_unigram)
path, ref = spec.split(":", 1)
obj = torch.load(os.path.join(ck, path), map_location="cpu", weights_only=False)
for k in ref.split("."):
    obj = obj[k]
L0_P = obj.double().float()


def level_offset(j):
    return sum(levels[:j]) - levels[0]  # offset into level_pos_embed (level 0 excluded)


@torch.no_grad()
def generate_skip(l0, prefix_tokens, skip):
    """Return dict level->codes for levels not skipped, and NFE used."""
    B = prefix_tokens.shape[0]
    z_hat = torch.zeros(B, levels[-1], vq.embed_dim, device=dev)
    z_prefix = tk.vqvae.encoder.wte(prefix_tokens)[:, :P, :]
    prefix_emb = net.transformer.wte(z_prefix)
    if not net.use_rope:
        prefix_emb = prefix_emb + net.wpe(net.wp_prefix)
    lpe = net.transformer.wle(net.lvl_1L) if CFG.model.use_level_encoding else \
        torch.zeros(1, net.L - levels[0], CFG.model.n_embd, device=dev)
    if not net.use_rope:
        lpe = lpe + net.wpe(net.wp_1L)
    pm = (prefix_tokens[:, :P] == net.prefix_pad_id)
    pbias = torch.where(pm[:, None, None, :], torch.tensor(float("-inf"), device=dev), torch.tensor(0.0, device=dev))

    codes = {0: l0}
    nfe = 0
    # add level 0 residual
    z_hat, _ = vq.get_next_autoregressive_input(0, z_hat, vq.embeddings[0](l0))
    for j in range(1, K):
        if j in skip:
            continue
        # input for level j = area-interp of current z_hat to levels[j]
        x = F.interpolate(z_hat.transpose(-1, -2), size=(levels[j],), mode="area").transpose(-1, -2)
        if net.use_post_upsample_conv:
            x = net._apply_post_upsample_conv_single(x, j - 1 + (1 - net._lvl_first))
        x = net.transformer.wte(x) + lpe[:, level_offset(j):level_offset(j) + levels[j], :]
        hi = j - 1 + (1 - net._lvl_first)
        logits = net._decode_level_logits(x, prefix_emb, j - 1, pbias, hidx=hi)
        idx = net._sample_level_raw(logits, temp, top_k, top_p)
        nfe += 1
        codes[j] = idx
        emb = vq.embeddings[j](idx)
        if j != K - 1:
            zq = vq._apply_quant_resi(vq._up_to_full(emb, j, levels[-1]), j)
        else:
            zq = vq._apply_quant_resi(emb, j)
        z_hat = z_hat + zq
    return codes, z_hat, nfe


def dec_text(ids):
    ids = list(ids)
    if EOS in ids:
        ids = ids[:ids.index(EOS)]
    return tk.pre_tokenizer.decode([t for t in ids if t not in (EOS, PAD)], skip_special_tokens=True)


def parse_sched(name):
    if name == "full":
        return set()
    if name.startswith("drop"):
        return set(int(x) for x in name[4:].split("-"))
    if name.startswith("keep"):
        keep = set(int(x) for x in name[4:].split("-"))
        return set(range(1, K)) - keep
    raise ValueError(name)


if a.schedules == "auto":
    names = ["full"] + [f"drop{j}" for j in range(1, K)] + \
            ["drop1-2", "drop2-4", "drop3-5-7", "drop1-3-5-7-9", "keep4-8-11", "keep2-5-8-11", "keep8-11", "keep11"]
else:
    names = a.schedules.split(",")

results = {}
t0 = time.time()
for name in names:
    skip = parse_sched(name)
    torch.manual_seed(a.seed)
    texts, nfes = [], []
    for s in range(0, a.n, a.bs):
        B = min(a.bs, a.n - s)
        l0 = torch.multinomial(L0_P, B * levels[0], replacement=True).view(B, levels[0]).to(dev)
        prefix = torch.full((B, 1), EOS, device=dev)
        codes, z_hat, nfe = generate_skip(l0, prefix, skip)
        toks = tk.vqvae.decoder(z_hat).argmax(-1)
        texts += [dec_text(t) for t in toks.tolist()]
        nfes.append(nfe)
    results[name] = {"skip": sorted(skip), "nfe": nfes[0], "texts": texts}
    print(f"{name:16s} skip={sorted(skip)} NFE={nfes[0]:2d}  ({time.time() - t0:.0f}s)  | {texts[0][:90]!r}", flush=True)

os.makedirs(os.path.dirname(a.out), exist_ok=True)
pickle.dump({"levels": levels, "results": results}, open(a.out, "wb"))
print(f"wrote {a.out}")
