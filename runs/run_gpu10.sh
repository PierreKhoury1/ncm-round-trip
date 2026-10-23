#!/bin/bash
# Phase 1: (a) does re-encode refinement help conditional tasks (XSum ROUGE, WMT14 BLEU) under the authors'
# protocol; (b) can more rounds / coarse-then-fine rounds / fine-level temperature push TinyStories and
# LM1B large further.
mkdir -p /root/logs round_trip/runs
rm -f /root/logs/ALL_DONE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q omegaconf hydra-core regex rouge_score sacrebleu "transformers>=4.51,<5" "huggingface_hub[cli]" vastai > /root/logs/pip.log 2>&1
hf download hazyresearch/ncm-checkpoints --include "tinystories/*" "lm1b/tokenizer.pt" "lm1b/large/*" "xsum/*" "wmt14/*" --local-dir artifacts > /root/logs/dl.log 2>&1
hf download hazyresearch/ncm-datasets --repo-type dataset --include "xsum/*" "wmt14/*" --local-dir data >> /root/logs/dl.log 2>&1
touch /root/logs/SETUP_DONE

# sanity: forcing every level must reproduce the draft on the conditional models too
python -I round_trip/refine_eval.py --config config/eval/xsum.yaml --checkpoint-dir artifacts/xsum --split test --n 32 --bs 32 --variants full,r9 --out round_trip/runs/sanity_xsum.pkl > /root/logs/sanity_xsum.log 2>&1
python -I round_trip/refine_eval.py --config config/eval/wmt14.yaml --checkpoint-dir artifacts/wmt14 --split test --n 32 --bs 32 --variants full,r11 --out round_trip/runs/sanity_wmt14.pkl > /root/logs/sanity_wmt14.log 2>&1
python -I - > /root/logs/sanity.log 2>&1 <<'PY'
import pickle
for f, k in (("round_trip/runs/sanity_xsum.pkl", "r9"), ("round_trip/runs/sanity_wmt14.pkl", "r11")):
    R = pickle.load(open(f, "rb"))["results"]
    print(f, "forced==draft:", sum(a == b for a, b in zip(R["full"]["texts"], R[k]["texts"])), "/", len(R["full"]["texts"]))
PY

python -I round_trip/refine_eval.py --config config/eval/xsum.yaml --checkpoint-dir artifacts/xsum --split test --n 1000 --bs 50 --variants full,r7,r6,r5,r4,r3,r5x2 --out round_trip/runs/xsum.pkl > /root/logs/xsum.log 2>&1
python -I round_trip/score_cond.py --in round_trip/runs/xsum.pkl > /root/logs/score_xsum.log 2>&1
python -I round_trip/refine_eval.py --config config/eval/wmt14.yaml --checkpoint-dir artifacts/wmt14 --split test --n 1000 --bs 50 --variants full,r8,r7,r6,r5,r4,r6x2 --out round_trip/runs/wmt14.pkl > /root/logs/wmt14.log 2>&1
python -I round_trip/score_cond.py --in round_trip/runs/wmt14.pkl > /root/logs/score_wmt14.log 2>&1

TSV=full,r6,r5,r6x2,r5x2,r4+r8,r3+r6+r9,t0.8@6+r6,t0.8@6+r5,t0.9@6+r5x2
for s in 1 2 3; do
  python -I round_trip/refine_eval.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --n 256 --bs 128 --seed $s --variants $TSV --out round_trip/runs/ts_more_s$s.pkl > /root/logs/ts_more_s$s.log 2>&1
  python -I round_trip/analyze_skip.py --in round_trip/runs/ts_more_s$s.pkl --ref gpt2-large --bs 64 > /root/logs/score_ts_more_s$s.log 2>&1
done
LMV=full,r6,r6x2,r5x2,r4+r8,t0.9@6+r6
for s in 1 2 3; do
  python -I round_trip/refine_eval.py --config config/eval/lm1b-large.yaml --checkpoint-dir artifacts/lm1b/large --n 256 --bs 32 --seed $s --variants $LMV --out round_trip/runs/lm1b_more_s$s.pkl > /root/logs/lm1b_more_s$s.log 2>&1
  python -I round_trip/score_flm.py --in round_trip/runs/lm1b_more_s$s.pkl --ref gpt2-large > /root/logs/score_lm1b_more_s$s.log 2>&1
done
touch /root/logs/ALL_DONE
