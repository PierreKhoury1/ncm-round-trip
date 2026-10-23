"""Weight interpolation (WiSE-FT): theta = (1 - alpha) * released + alpha * fine-tuned. Writes a generator checkpoint."""
import argparse
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--released", default="artifacts/tinystories/generator.pt")
ap.add_argument("--finetuned", required=True)
ap.add_argument("--alpha", type=float, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
r = torch.load(a.released, map_location="cpu", weights_only=False)
f = torch.load(a.finetuned, map_location="cpu", weights_only=False)
mix = {k: ((1 - a.alpha) * v.float() + a.alpha * f["model"][k].float()).to(v.dtype) if v.is_floating_point() else v
       for k, v in r["model"].items()}
torch.save({"model": mix, "inference": r.get("inference", {})}, a.out)
print(f"wrote {a.out} (alpha={a.alpha})")
