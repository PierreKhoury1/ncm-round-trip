#!/bin/bash
# Run 22: (S) score the six run-21 pickles (uploaded to round_trip/runs/), whose score logs were lost in transit;
#         (E) early stop with a second-round threshold: after one round the coarse levels are canonical by
#             construction, so the drift of the refined text is far below the draft's. Low thresholds gate rounds 2-3.
#             Every variant now records the per-round non-canonicality (canon_rounds) to pick thresholds from.
mkdir -p /root/logs round_trip/runs
rm -f /root/logs/ALL_DONE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q omegaconf hydra-core regex scipy "transformers>=4.51,<5" "huggingface_hub[cli]" vastai > /root/logs/pip.log 2>&1
hf download hazyresearch/ncm-checkpoints --include "tinystories/*" "lm1b/tokenizer.pt" "lm1b/large/*" --local-dir artifacts > /root/logs/dl.log 2>&1
touch /root/logs/SETUP_DONE

# ---- S: score run 21 ----
for s in 1 2 3; do
  python -I round_trip/analyze_skip.py --in round_trip/runs/p10_adapt_ts_s$s.pkl --ref gpt2-large --bs 64 > /root/logs/score_p10_adapt_ts_s$s.log 2>&1
  python -I round_trip/score_ppl.py --in round_trip/runs/p10_adapt_ts_s$s.pkl --ref gpt2-large --bs 32 --out round_trip/runs/p10_g2l_adapt_ts_s$s.json > /root/logs/g2l_p10_adapt_ts_s$s.log 2>&1
  python -I round_trip/score_flm.py --in round_trip/runs/p10_adapt_lm1bl_s$s.pkl --ref gpt2-large > /root/logs/score_p10_adapt_lm1bl_s$s.log 2>&1
  python -I round_trip/score_ppl.py --in round_trip/runs/p10_adapt_lm1bl_s$s.pkl --ref gpt2-large --bs 16 --out round_trip/runs/p10_g2l_adapt_lm1bl_s$s.json > /root/logs/g2l_p10_adapt_lm1bl_s$s.log 2>&1
done
touch /root/logs/S_DONE

# ---- E: early stop with low second-round thresholds ----
TS_V=full,r5x2,r5x3,a5x3@0.30,a5x3@0.40,a5x3@0.50,a5x3@0.60,a6x3@0.40,a6x3@0.50
for s in 1 2 3; do
  python -I round_trip/refine_eval.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --n 256 --bs 128 --seed $s \
    --variants $TS_V --out round_trip/runs/p11_stop_ts_s$s.pkl > /root/logs/p11_stop_ts_s$s.log 2>&1
  python -I round_trip/analyze_skip.py --in round_trip/runs/p11_stop_ts_s$s.pkl --ref gpt2-large --bs 64 > /root/logs/score_p11_stop_ts_s$s.log 2>&1
  python -I round_trip/score_ppl.py --in round_trip/runs/p11_stop_ts_s$s.pkl --ref gpt2-large --bs 32 --out round_trip/runs/p11_g2l_stop_ts_s$s.json > /root/logs/g2l_p11_stop_ts_s$s.log 2>&1
done
LM_V=full,r6x2,r6x3,a6x3@0.40,a6x3@0.50,a6x3@0.60
for s in 1 2 3; do
  python -I round_trip/refine_eval.py --config config/eval/lm1b-large.yaml --checkpoint-dir artifacts/lm1b/large --n 256 --bs 32 --seed $s \
    --variants $LM_V --out round_trip/runs/p11_stop_lm1bl_s$s.pkl > /root/logs/p11_stop_lm1bl_s$s.log 2>&1
  python -I round_trip/score_flm.py --in round_trip/runs/p11_stop_lm1bl_s$s.pkl --ref gpt2-large > /root/logs/score_p11_stop_lm1bl_s$s.log 2>&1
  python -I round_trip/score_ppl.py --in round_trip/runs/p11_stop_lm1bl_s$s.pkl --ref gpt2-large --bs 16 --out round_trip/runs/p11_g2l_stop_lm1bl_s$s.json > /root/logs/g2l_p11_stop_lm1bl_s$s.log 2>&1
done
touch /root/logs/E_DONE
touch /root/logs/ALL_DONE
