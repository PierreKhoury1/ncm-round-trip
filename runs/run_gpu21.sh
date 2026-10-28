#!/bin/bash
# Run 21: adaptive restart + early stop. Per sample, a round runs only when the re-encode still rewrites more than a
# threshold of the level 0-5 codes (a<k>@<t>); a<k>x<n>@<t> repeats with the same test (early stop). Same seeds and
# sample counts as the paper's headline tables, so r6 / r5x2 / r6x2 rows are the fixed-cost references.
mkdir -p /root/logs round_trip/runs
rm -f /root/logs/ALL_DONE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q omegaconf hydra-core regex scipy "transformers>=4.51,<5" "huggingface_hub[cli]" vastai > /root/logs/pip.log 2>&1
hf download hazyresearch/ncm-checkpoints --include "tinystories/*" "lm1b/tokenizer.pt" "lm1b/large/*" --local-dir artifacts > /root/logs/dl.log 2>&1
touch /root/logs/SETUP_DONE

TS_V=full,r6,a6@0.70,a6@0.75,a6@0.80,r5x2,a5x2@0.72,a5x3@0.72,a5x3@0.78,a6x3@0.72,a6x3@0.78
for s in 1 2 3; do
  python -I round_trip/refine_eval.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --n 256 --bs 128 --seed $s \
    --variants $TS_V --out round_trip/runs/p10_adapt_ts_s$s.pkl > /root/logs/p10_adapt_ts_s$s.log 2>&1
  python -I round_trip/analyze_skip.py --in round_trip/runs/p10_adapt_ts_s$s.pkl --ref gpt2-large --bs 64 > /root/logs/score_p10_adapt_ts_s$s.log 2>&1
  python -I round_trip/score_ppl.py --in round_trip/runs/p10_adapt_ts_s$s.pkl --ref gpt2-large --bs 32 --out round_trip/runs/p10_g2l_adapt_ts_s$s.json > /root/logs/g2l_p10_adapt_ts_s$s.log 2>&1
done
touch /root/logs/TS_DONE

LM_V=full,r6,a6@0.80,a6@0.85,r6x2,a6x2@0.85,a6x3@0.85
for s in 1 2 3; do
  python -I round_trip/refine_eval.py --config config/eval/lm1b-large.yaml --checkpoint-dir artifacts/lm1b/large --n 256 --bs 32 --seed $s \
    --variants $LM_V --out round_trip/runs/p10_adapt_lm1bl_s$s.pkl > /root/logs/p10_adapt_lm1bl_s$s.log 2>&1
  python -I round_trip/score_flm.py --in round_trip/runs/p10_adapt_lm1bl_s$s.pkl --ref gpt2-large > /root/logs/score_p10_adapt_lm1bl_s$s.log 2>&1
  python -I round_trip/score_ppl.py --in round_trip/runs/p10_adapt_lm1bl_s$s.pkl --ref gpt2-large --bs 16 --out round_trip/runs/p10_g2l_adapt_lm1bl_s$s.json > /root/logs/g2l_p10_adapt_lm1bl_s$s.log 2>&1
done
touch /root/logs/LM_DONE
touch /root/logs/ALL_DONE
