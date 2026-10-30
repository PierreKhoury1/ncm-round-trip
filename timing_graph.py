"""Wall-clock of the released sampler and RT6 with and without CUDA graphs (TinyStories checkpoint).

Modes:
  eager            the authors' code as released (what the paper's timing table measured)
  graphs           per-level transformer pass compiled with torch.compile(mode="reduce-overhead"): CUDA-graph replay
  compile          torch.compile default mode (kernel fusion, no CUDA graphs), to separate fusion from graph replay
Sampling (sort / multinomial) and the tokenizer stay eager in every mode. Batch 128 is the paper's setting; batch 8 is
the latency regime where launch overhead matters most. Each (mode, batch, sampler) is timed over --reps runs after one
warm-up, CUDA-synchronised. Output: one JSON with ms per sample and per pass.
"""
import argparse, json, os, sys, time
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from omegaconf import OmegaConf  # noqa: E402
from models.next_concept import NextConceptTransformer  # noqa: E402
from tokenization.vqvae_tokenizer import VQVAETokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config/eval/tinystories.yaml")
ap.add_argument("--checkpoint-dir", default="artifacts/tinystories")
ap.add_argument("--modes", default="eager,graphs,compile")
ap.add_argument("--batches", default="128,8")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--k", type=int, default=6)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--out", default="round_trip/runs/timing_graph.json")
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
sd = torch.load(os.path.join(ck, CFG.checkpoints.generator), map_location="cpu", weights_only=False)
net.load_state_dict(sd["model"], strict=True)
net.eval()
EOS = tk.pre_tokenizer.eos_token_id
vq = tk.vqvae.vq
levels = list(vq.levels)
K = len(levels)
temp0, top_k, top_p = float(G.get("temperature", 1.0)), int(G.get("top_k", 0)), float(G.get("top_p", 1.0))
assert float(G.get("frequency_penalty", 0.0)) <= 0 and not G.get("adjacency_penalty"), "argmax token picker only"


def load_spec(spec):
    path, ref = spec.split(":", 1)
    obj = torch.load(os.path.join(ck, path), map_location=dev, weights_only=False)
    for kk in ref.split("."):
        obj = obj[kk]
    return obj.double()


L0_P = load_spec(str(G.level0_unigram)).float() if G.get("level0_unigram") else None


def split_codes(idx):
    out, acc = [], 0
    for L in levels:
        out.append(idx[:, acc:acc + L]); acc += L
    return out


# ---- the same forcing hook as refine_eval.py ----
_orig_sample = net._sample_level
_state = {"codes": None, "calls": 0, "upto": 0}


def _sample_with_force(logits, temperature, top_k_, top_p_):
    level = _state["calls"] + 1
    _state["calls"] += 1
    if _state["codes"] is not None and 1 <= level < _state["upto"]:
        return _state["codes"][level]
    return _orig_sample(logits, temperature, top_k_, top_p_)


net._sample_level = _sample_with_force
_orig_decode = net._decode_level_logits


def set_mode(mode):
    if mode == "eager":
        net._decode_level_logits = _orig_decode
    elif mode == "graphs":
        net._decode_level_logits = torch.compile(_orig_decode, mode="reduce-overhead", dynamic=False)
    elif mode == "compile":
        net._decode_level_logits = torch.compile(_orig_decode, dynamic=False)
    else:
        raise ValueError(mode)


def sync():
    if dev.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def gen(l0, prefix, forced=None, upto=0):
    _state.update(codes=forced, calls=0, upto=upto)
    g = net.generate(l0, prefix, temperature=temp0, top_k=top_k, top_p=top_p, cfg_scale=0.0, cfg_null_eos=EOS)
    _state.update(codes=None, calls=0, upto=0)
    _, zl = vq.idx_to_zhats(g, return_zlevels=True)
    return tk.vqvae.decoder(zl[-1]).argmax(-1), g


@torch.no_grad()
def run(B, rt):
    torch.manual_seed(1)
    l0 = torch.multinomial(L0_P.cpu(), B * levels[0], replacement=True).view(B, levels[0]).to(dev) if L0_P is not None else None
    prefix = torch.full((B, 1), EOS, device=dev)
    sync(); t = time.time()
    toks, g = gen(l0, prefix)
    if rt:
        src = split_codes(tk.vqvae.encode(toks))
        toks, g = gen(src[0], prefix, forced=src, upto=a.k)
    sync()
    return time.time() - t


results = []
for mode in a.modes.split(","):
    try:
        set_mode(mode)
        for B in (int(b) for b in a.batches.split(",")):
            for name, rt in (("released", False), (f"RT{a.k}", True)):
                run(B, rt)  # warm-up (compiles / captures on the first call)
                if mode != "eager":
                    run(B, rt)  # second warm-up: reduce-overhead records graphs on the 2nd-3rd call
                    run(B, rt)
                ts = [run(B, rt) for _ in range(a.reps)]
                passes = (K - 1) + ((K - a.k) if rt else 0)
                med = sorted(ts)[len(ts) // 2]
                row = {"mode": mode, "batch": B, "sampler": name, "passes": passes, "sec_median": med,
                       "ms_per_sample": 1000 * med / B, "ms_per_pass_per_sample": 1000 * med / B / passes,
                       "sec_all": ts}
                results.append(row)
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != "sec_all"}), flush=True)
    except Exception as e:  # a compile failure must not kill the eager rows
        print(json.dumps({"mode": mode, "error": repr(e)[:300]}), flush=True)
        results.append({"mode": mode, "error": repr(e)[:300]})
        torch._dynamo.reset()

os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump({"device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu", "torch": torch.__version__,
           "k": a.k, "reps": a.reps, "rows": results}, open(a.out, "w"), indent=1)
print("wrote", a.out)
