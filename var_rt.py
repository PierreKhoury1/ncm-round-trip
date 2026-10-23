"""Round-trip refinement for VAR (next-scale image prediction, Tian et al. 2024) and FID screening.

The sampler is VAR.autoregressive_infer_cfg copied verbatim, with one hook: at scales listed in `forced` the sampled
token map is replaced by the given codes (the transformer still runs, so the KV cache sees the forced prefix).

Variants (comma-separated):
  base        VAR's own sampler (cfg, top-k, top-p as given)
  f           force every scale to the draft's own codes (sanity: must reproduce the draft exactly)
  r<k>        decode the draft to an 8-bit image, re-encode it with VAR's VQVAE, keep scales < k, regenerate k..9
  q<k>        re-quantize the draft's latent f_hat instead of the image (latent round trip, like scale-travel)
  n<k>        keep the draft's own codes for scales < k and regenerate k..9 (no re-encode)
  r<k>x<n>    n rounds
All variants of a run share the same labels and the same draft (same generator seed).

Writes Inception pool features (pytorch-fid, 2048-d) per variant and prints FID against a reference set.
Usage (inside a checkout of github.com/FoundationVision/VAR, with checkpoints in --ckpt-dir):
  python var_rt.py --var-dir /root/VAR --ckpt-dir /root/ckpt --depth 16 --n-per-class 10 --variants base,f,r7,q7,n7 \
      --ref /root/ref/VIRTUAL_imagenet256_labeled.npz --out /root/runs/var_d16.npz
"""
import argparse, json, os, sys, time
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--var-dir", required=True)
ap.add_argument("--ckpt-dir", required=True)
ap.add_argument("--depth", type=int, default=16)
ap.add_argument("--n-per-class", type=int, default=10)
ap.add_argument("--classes", type=int, default=1000)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--cfg", type=float, default=1.5)
ap.add_argument("--top-k", type=int, default=900)
ap.add_argument("--top-p", type=float, default=0.96)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--variants", default="base,f,r7,q7,n7")
ap.add_argument("--ref", default="")
ap.add_argument("--save-png", type=int, default=0, help="save the first N images per variant as a grid for eyeballing")
ap.add_argument("--save-feats", action="store_true", help="also save the Inception features (large)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

sys.path.insert(0, a.var_dir)
os.chdir(a.var_dir)
from models import build_vae_var  # noqa: E402
from models.helpers import sample_with_top_k_top_p_  # noqa: E402

dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
PN = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
vae, var = build_vae_var(V=4096, Cvae=32, ch=160, share_quant_resi=4, device=dev, patch_nums=PN, num_classes=1000,
                         depth=a.depth, shared_aln=False)
vae.load_state_dict(torch.load(os.path.join(a.ckpt_dir, "vae_ch160v4096z32.pth"), map_location="cpu"), strict=True)
var.load_state_dict(torch.load(os.path.join(a.ckpt_dir, f"var_d{a.depth}.pth"), map_location="cpu"), strict=True)
vae.eval()
var.eval()
for p_ in list(vae.parameters()) + list(var.parameters()):
    p_.requires_grad_(False)
SN = len(PN)


@torch.no_grad()
def infer(labels, rng, forced=None, upto=0):
    """VAR.autoregressive_infer_cfg with a forcing hook. Returns (f_hat, list of per-scale codes)."""
    B = labels.shape[0]
    with torch.autocast("cuda", enabled=True, dtype=torch.float16, cache_enabled=True):
        sos = cond_BD = var.class_emb(torch.cat((labels, torch.full_like(labels, fill_value=var.num_classes)), dim=0))
        lvl_pos = var.lvl_embed(var.lvl_1L) + var.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, var.first_l, -1) + var.pos_start.expand(2 * B, var.first_l, -1) \
            + lvl_pos[:, :var.first_l]
        cur_L = 0
        f_hat = sos.new_zeros(B, var.Cvae, var.patch_nums[-1], var.patch_nums[-1])
        codes = []
        for b in var.blocks:
            b.attn.kv_caching(True)
        for si, pn in enumerate(var.patch_nums):
            ratio = si / var.num_stages_minus_1
            cur_L += pn * pn
            cond_BD_or_gss = var.shared_ada_lin(cond_BD)
            x = next_token_map
            for b in var.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            if forced is not None and si < upto:
                idx_Bl = forced[si]
            else:
                logits_BlV = var.get_logits(x, cond_BD)
                t = a.cfg * ratio
                logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=a.top_k, top_p=a.top_p, num_samples=1)[:, :, 0]
            codes.append(idx_Bl)
            h_BChw = var.vae_quant_proxy[0].embedding(idx_Bl)
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, var.Cvae, pn, pn)
            f_hat, next_token_map = var.vae_quant_proxy[0].get_next_autoregressive_input(si, SN, f_hat, h_BChw)
            if si != var.num_stages_minus_1:
                next_token_map = next_token_map.view(B, var.Cvae, -1).transpose(1, 2)
                next_token_map = var.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + var.patch_nums[si + 1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)
        for b in var.blocks:
            b.attn.kv_caching(False)
    return f_hat.float(), codes


@torch.no_grad()
def to_uint8(f_hat):
    img = vae.fhat_to_img(f_hat.float())  # [-1, 1]
    return ((img + 1) * 127.5).round().clamp(0, 255)  # what a saved 8-bit image holds


@torch.no_grad()
def reencode(img8):
    return vae.img_to_idxBl(img8.float() / 127.5 - 1)


def parse(name):
    if name in ("base", "f"):
        return []
    rounds = []
    for part in name.split("+"):
        k, n = (part[1:].split("x") + ["1"])[:2]
        rounds += [(part[0], int(k))] * int(n)
    return rounds


from pytorch_fid.inception import InceptionV3  # noqa: E402
inc = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()


@torch.no_grad()
def feats(img8):
    return inc(img8 / 255.0)[0].squeeze(-1).squeeze(-1).double().cpu().numpy()


def fid(f1, mu2, s2):
    from scipy import linalg
    mu1, s1 = f1.mean(0), np.cov(f1, rowvar=False)
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


REF = None
if a.ref and os.path.exists(a.ref):
    R = np.load(a.ref)
    if "arr_0" in R.files:  # reference images -> features with the same extractor (consistent FID)
        imgs = R["arr_0"]
        fr = []
        for s in range(0, len(imgs), 200):
            x = torch.from_numpy(imgs[s:s + 200]).to(dev).permute(0, 3, 1, 2).float()
            fr.append(feats(x))
        fr = np.concatenate(fr)
        REF = (fr.mean(0), np.cov(fr, rowvar=False))
        print(f"reference: {len(fr)} images from {os.path.basename(a.ref)}", flush=True)

labels_all = torch.arange(a.classes).repeat_interleave(a.n_per_class)
out, report = {}, {}
t0 = time.time()
for name in a.variants.split(","):
    rng = torch.Generator(device=dev)
    rng.manual_seed(a.seed)
    rounds = parse(name)
    F_, changed, same_draft, nimg = [], np.zeros(SN), 0, 0
    grid = []
    for s in range(0, len(labels_all), a.bs):
        lab = labels_all[s:s + a.bs].to(dev)
        f_hat, codes = infer(lab, rng)
        img8 = to_uint8(f_hat)
        if name == "f":  # sanity: forcing every scale to the draft's own codes must give the identical image
            f2, _ = infer(lab, rng, forced=codes, upto=SN)
            same_draft += int((to_uint8(f2) == img8).flatten(1).all(1).sum())
        for i, (kind, k) in enumerate(rounds):
            if kind == "r":
                src = reencode(img8)
            elif kind == "q":
                src = vae.quantize.f_to_idxBl_or_fhat(f_hat, to_fhat=False)
            else:
                src = codes
            if i == 0 and kind != "n":
                changed += np.array([(x != y).float().mean().item() for x, y in zip(src, codes)]) * lab.shape[0]
            f_hat, codes = infer(lab, rng, forced=src, upto=k)
            img8 = to_uint8(f_hat)
        F_.append(feats(img8))
        nimg += lab.shape[0]
        if a.save_png and len(grid) * a.bs < a.save_png:
            grid.append(img8[: max(1, a.save_png // 8)].byte().cpu())
    F_ = np.concatenate(F_)
    out[name] = F_
    rep = {"n": int(nimg)}
    if rounds and rounds[0][0] != "n":
        rep["changed_per_scale"] = (changed / nimg).round(3).tolist()
    if name == "f":
        rep["forced_reproduces_draft"] = f"{same_draft}/{nimg}"
    if REF is not None:
        rep["fid"] = round(fid(F_, *REF), 3)
    report[name] = rep
    print(f"{name:10s} {json.dumps(rep)}  ({time.time() - t0:.0f}s)", flush=True)
    if grid:
        import torchvision
        g = torch.cat(grid)[: a.save_png].float() / 255.0
        torchvision.utils.save_image(g, a.out.replace(".npz", f"_{name}.png"), nrow=8)

if a.save_feats:
    np.savez_compressed(a.out, **{k: v.astype(np.float32) for k, v in out.items()})
json.dump(report, open(a.out.replace(".npz", ".json"), "w"), indent=1)
print(f"wrote {a.out.replace('.npz', '.json')}")
