#!/bin/bash
# Run 19: the referee shield + the one cheap new idea.
#   T  wall-clock cost of the samplers (generator passes vs tokenizer encode vs decode + token picking), TinyStories
#      and LM1B large; the first TinyStories pass is a warm-up and is thrown away
#   C  per-sample canonicality of drafts (does the re-encode change rate predict quality? where does the gain land?)
#      and best-of-N drafts by canonicality (b4, b8, with and without a round), seeds 1-3, gpt2-large per-sample scores
#   B  second perplexity scorer (Qwen2.5-7B, bf16) on the headline pickles of the paper and on the new ones
mkdir -p /root/logs round_trip/runs
rm -f /root/logs/ALL_DONE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q omegaconf hydra-core regex scipy "transformers>=4.51,<5" "huggingface_hub[cli]" vastai > /root/logs/pip.log 2>&1
hf download hazyresearch/ncm-checkpoints --include "tinystories/*" "lm1b/tokenizer.pt" "lm1b/large/*" --local-dir artifacts > /root/logs/dl.log 2>&1
touch /root/logs/SETUP_DONE

# ---- T: timing ----
TS_T=full,r6,r5x2,r3+r6+r9,t0.8@1,d1.1+r6
python -I round_trip/refine_eval.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --n 256 --bs 128 --seed 1 \
  --variants $TS_T --out round_trip/runs/p9_warmup.pkl > /root/logs/p9_warmup.log 2>&1
python -I round_trip/refine_eval.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --n 256 --bs 128 --seed 1 \
  --variants $TS_T --out round_trip/runs/p9_time_ts.pkl > /root/logs/p9_time_ts.log 2>&1
python -I round_trip/refine_eval.py --config config/eval/lm1b-large.yaml --checkpoint-dir artifacts/lm1b/large --n 128 --bs 32 --seed 1 \
  --variants full,r6,r6x2,d1.2+r6x2 --out round_trip/runs/p9_time_lm1bl.pkl > /root/logs/p9_time_lm1bl.log 2>&1
touch /root/logs/T_DONE

# ---- C: canonicality + best-of-N ----
for s in 1 2 3; do
  python -I round_trip/refine_eval.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --n 256 --bs 128 --seed $s \
    --variants full,r6,r5x2,d1.1+r6,b4,b8,b4+r6,b8+r6,b8+r5x2 --out round_trip/runs/p9_canon_ts_s$s.pkl > /root/logs/p9_canon_ts_s$s.log 2>&1
  python -I round_trip/analyze_skip.py --in round_trip/runs/p9_canon_ts_s$s.pkl --ref gpt2-large --bs 64 > /root/logs/score_p9_canon_ts_s$s.log 2>&1
  python -I round_trip/score_ppl.py --in round_trip/runs/p9_canon_ts_s$s.pkl --ref gpt2-large --bs 32 --out round_trip/runs/p9_g2l_canon_ts_s$s.json > /root/logs/g2l_p9_canon_ts_s$s.log 2>&1
  python -I round_trip/canon_analyze.py --pkl round_trip/runs/p9_canon_ts_s$s.pkl --scores round_trip/runs/p9_g2l_canon_ts_s$s.json --out round_trip/runs/p9_canon_ts_s$s.json > /root/logs/canon_p9_ts_s$s.log 2>&1
done
touch /root/logs/C1_DONE
for s in 1 2 3; do
  python -I round_trip/refine_eval.py --config config/eval/lm1b-large.yaml --checkpoint-dir artifacts/lm1b/large --n 256 --bs 32 --seed $s \
    --variants full,r6,b4,b4+r6 --out round_trip/runs/p9_canon_lm1bl_s$s.pkl > /root/logs/p9_canon_lm1bl_s$s.log 2>&1
  python -I round_trip/score_flm.py --in round_trip/runs/p9_canon_lm1bl_s$s.pkl --ref gpt2-large > /root/logs/score_p9_canon_lm1bl_s$s.log 2>&1
  python -I round_trip/score_ppl.py --in round_trip/runs/p9_canon_lm1bl_s$s.pkl --ref gpt2-large --bs 16 --out round_trip/runs/p9_g2l_canon_lm1bl_s$s.json > /root/logs/g2l_p9_canon_lm1bl_s$s.log 2>&1
  python -I round_trip/canon_analyze.py --pkl round_trip/runs/p9_canon_lm1bl_s$s.pkl --scores round_trip/runs/p9_g2l_canon_lm1bl_s$s.json --out round_trip/runs/p9_canon_lm1bl_s$s.json > /root/logs/canon_p9_lm1bl_s$s.log 2>&1
done
touch /root/logs/C_DONE

# ---- B: second scorer on the paper's headline pickles (uploaded) + the new ones ----
for f in p3_samplers_s1 p3_samplers_s2 p3_samplers_s3 p4_ts_s4 p4_ts_s5 p4_ts_s6 p6_ts_s4 p6_ts_s5 p6_ts_s6 \
         lm1b_small_s1 lm1b_small_s2 lm1b_small_s3 p4_lm1bm_s1 p4_lm1bm_s2 p4_lm1bm_s3 \
         lm1b_large_s1 lm1b_large_s2 lm1b_large_s3 p4_lm1bl_s4 p4_lm1bl_s5 p4_lm1bl_s6 p6_lm1bl_s4 p6_lm1bl_s5 p6_lm1bl_s6 \
         p9_canon_ts_s1 p9_canon_ts_s2 p9_canon_ts_s3 p9_canon_lm1bl_s1 p9_canon_lm1bl_s2 p9_canon_lm1bl_s3; do
  [ -f round_trip/runs/$f.pkl ] || { echo "missing $f" >> /root/logs/qwen_missing.log; continue; }
  python -I round_trip/score_ppl.py --in round_trip/runs/$f.pkl --ref Qwen/Qwen2.5-7B --dtype bf16 --bs 16 --out round_trip/runs/p9_qwen_$f.json > /root/logs/qwen_$f.log 2>&1
done
touch /root/logs/B_DONE
touch /root/logs/ALL_DONE
