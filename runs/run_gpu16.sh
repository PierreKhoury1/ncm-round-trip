#!/bin/bash
# Run 16 (from the design workflow's plan): the fairness controls a reviewer would demand before publication, plus the
# one remaining search idea for Sudoku.
#   S  Sudoku: plan<k>@T retries (hot coarse plan, greedy fine levels) vs fresh resampling, seeds 1,4,5,6; greedy RT on
#      easy/medium for the all-difficulty confirmation
#   T  TinyStories seeds 4-6: exact 11-NFE per-level temperature "twins" of every RT recipe (no round trip), the full
#      temperature curve tau 0.6..1.3, and RT applied to HUMAN text ('h': projection or sharpening?)
#   L  LM1B large seeds 4-6: 11-NFE twins of the LM1B recipes
mkdir -p /root/logs round_trip/runs
rm -f /root/logs/ALL_DONE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q omegaconf hydra-core regex "transformers>=4.51,<5" "huggingface_hub[cli]" vastai > /root/logs/pip.log 2>&1
hf download hazyresearch/ncm-checkpoints --include "tinystories/*" "lm1b/tokenizer.pt" "lm1b/large/*" "sudoku/*" --local-dir artifacts > /root/logs/dl.log 2>&1
hf download hazyresearch/ncm-datasets --repo-type dataset --include "tinystories/valid_bpe_8192.pickle" "sudoku/*" --local-dir data >> /root/logs/dl.log 2>&1
touch /root/logs/SETUP_DONE

for s in 1 4 5 6; do
  python -I round_trip/sudoku_search.py --config config/eval/sudoku-hard.yaml --checkpoint-dir artifacts/sudoku/hard \
    --data-dir data/sudoku/hard --n 1000 --attempts 10 --seed $s \
    --methods resample@1.0,plan3@1.0,plan6@1.0,plan9@1.0,plan4@1.5,plan6@1.5 \
    --out round_trip/runs/p6_sudoku_plan_s$s.json > /root/logs/p6_sudoku_plan_s$s.log 2>&1
done
for d in easy medium; do
  python -I round_trip/refine_eval.py --config config/eval/sudoku-$d.yaml --checkpoint-dir artifacts/sudoku/$d --split test --data-dir data/sudoku/$d --n 1000 --bs 128 --variants full,r6,r3+r6+r9 --out round_trip/runs/p6_sudoku_$d.pkl > /root/logs/p6_sudoku_$d.log 2>&1
  python -I round_trip/score_cond.py --in round_trip/runs/p6_sudoku_$d.pkl > /root/logs/score_p6_sudoku_$d.log 2>&1
done
touch /root/logs/S_DONE

REPRO=full,r6,d1.3+t0.8@6+r5x2,d1.2+t0.8@6+r5x2
CURVE=t0.6@1,t0.7@1,t0.8@1,t0.85@1,t0.9@1,t0.95@1,t1.1@1,t1.2@1,t1.3@1
TWINS=t1.1@1+t1@6,t1.2@1+t0.8@6,t1.2@1+t1@5+t0.8@6,t1.3@1+t1@5+t0.8@6,t1.3@1+t0.8@6,t1.2@1+t0.8@5,t1.15@1+t0.85@6,t1.25@1+t0.8@6,t0.8@6
EXTRA=t0.6@6,t0.7@1+t1@6,t1.4@1+t0.8@6,t1.3@1+t0.7@6,t1.3@1+t0.8@4,t1.3@1+t0.8@8,d1.2+t0.8@6+n6,t1.3@1+t1@5+t0.8@6+r5x2,t1.2@1+t0.8@6+r6
HUMAN=h,h+r6,h+r5x2,h+r3+r6+r9,h+t0.8@6+r6
for s in 4 5 6; do
  python -I round_trip/refine_eval.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --n 256 --bs 128 --seed $s \
    --variants $REPRO,$CURVE,$TWINS,$EXTRA,$HUMAN --out round_trip/runs/p6_ts_s$s.pkl > /root/logs/p6_ts_s$s.log 2>&1
  python -I round_trip/analyze_skip.py --in round_trip/runs/p6_ts_s$s.pkl --ref gpt2-large --bs 64 > /root/logs/score_p6_ts_s$s.log 2>&1
done
touch /root/logs/T_DONE

for s in 4 5 6; do
  python -I round_trip/refine_eval.py --config config/eval/lm1b-large.yaml --checkpoint-dir artifacts/lm1b/large --n 256 --bs 32 --seed $s \
    --variants full,t1.1@1,t1.2@1,t0.8@6,t0.9@6,t1.2@1+t1@6,t1.1@1+t1@6,r6x2,d1.2+r6x2 --out round_trip/runs/p6_lm1bl_s$s.pkl > /root/logs/p6_lm1bl_s$s.log 2>&1
  python -I round_trip/score_flm.py --in round_trip/runs/p6_lm1bl_s$s.pkl --ref gpt2-large > /root/logs/score_p6_lm1bl_s$s.log 2>&1
done
touch /root/logs/ALL_DONE
