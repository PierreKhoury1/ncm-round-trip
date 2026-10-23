"""MAUVE (Pillutla et al., 2021) between each sampler's outputs and human text, with GPT-2 Large features.

Generated texts are pooled over the seed pickles (e.g. 3 x 256 = 768); the human reference is a pickle of the same
number of real validation openings (real_ref.py output, variant "real"). MAUVE's k-means quantization is repeated with
several seeds and averaged.

Usage: python round_trip/score_mauve.py --pkl "round_trip/runs/p3_samplers_s{s}.pkl,round_trip/runs/ts_more_s{s}.pkl" \
           --seeds 1,2,3 --human round_trip/runs/real_ts.pkl --variants full,r6,t0.8@1 --out round_trip/runs/mauve_ts.json
"""
import argparse, importlib, json, os, pickle
import numpy as np

_cm = importlib.import_module("mauve.compute_mauve")  # the package re-exports only the function under the same name
compute_mauve, get_features_from_input = _cm.compute_mauve, _cm.get_features_from_input

ap = argparse.ArgumentParser()
ap.add_argument("--pkl", required=True, help="comma-separated pickle path templates containing {s}")
ap.add_argument("--seeds", default="1,2,3")
ap.add_argument("--human", required=True)
ap.add_argument("--variants", required=True)
ap.add_argument("--kmeans-seeds", default="1,2,3")
ap.add_argument("--max-len", type=int, default=256)
ap.add_argument("--out", required=True)
a = ap.parse_args()

V = {}
for s in a.seeds.split(","):
    loaded = [pickle.load(open(t.format(s=s), "rb"))["results"] for t in a.pkl.split(",") if os.path.exists(t.format(s=s))]
    for v in a.variants.split(","):
        R = next((R for R in loaded if v in R), None)
        if R is not None:
            V.setdefault(v, []).extend(" ".join(t.split()) for t in R[v]["texts"])
human = [" ".join(t.split()) for t in pickle.load(open(a.human, "rb"))["results"]["real"]["texts"]]

# features once per text set, then MAUVE with several k-means seeds
p_feat = get_features_from_input(None, None, human, "gpt2-large", a.max_len, device_id=0, name="p", verbose=False,
                                       batch_size=16, use_float64=False)
res = {}
for v, texts in V.items():
    n = min(len(texts), len(human))
    q_feat = get_features_from_input(None, None, texts[:n], "gpt2-large", a.max_len, device_id=0, name="q",
                                           verbose=False, batch_size=16, use_float64=False)
    scores = [compute_mauve(p_features=p_feat[:n], q_features=q_feat, seed=int(k), verbose=False).mauve
              for k in a.kmeans_seeds.split(",")]
    res[v] = {"n": n, "mauve": float(np.mean(scores)), "sd": float(np.std(scores)), "runs": scores}
    print(f"{v:22s} n={n}  MAUVE {res[v]['mauve']:.3f} +- {res[v]['sd']:.3f}", flush=True)
json.dump(res, open(a.out, "w"), indent=1)
