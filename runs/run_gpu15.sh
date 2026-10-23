#!/bin/bash
# Run 15: "think until correct" on Sudoku (verifier-guided search: fresh resampling vs round-trip refinement at equal
# NFE) and held-out confirmation (seeds 4-6) of the best TinyStories settings from the tuning grid.
mkdir -p /root/logs round_trip/runs
rm -f /root/logs/ALL_DONE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q omegaconf hydra-core regex "transformers>=4.51,<5" "huggingface_hub[cli]" vastai > /root/logs/pip.log 2>&1
hf download hazyresearch/ncm-checkpoints --include "tinystories/*" "sudoku/*" --local-dir artifacts > /root/logs/dl.log 2>&1
hf download hazyresearch/ncm-datasets --repo-type dataset --include "sudoku/*" --local-dir data >> /root/logs/dl.log 2>&1
touch /root/logs/SETUP_DONE

# S0: plumbing check on 32 puzzles (clue parsing, verifier, both methods)
python -I round_trip/sudoku_search.py --config config/eval/sudoku-hard.yaml --checkpoint-dir artifacts/sudoku/hard \
  --data-dir data/sudoku/hard --smoke --methods resample@1.0,refine6@0.5 --out round_trip/runs/p5_sudoku_smoke.json > /root/logs/p5_sudoku_smoke.log 2>&1
# S1: full search on Sudoku-hard, 10 attempts per unsolved puzzle
python -I round_trip/sudoku_search.py --config config/eval/sudoku-hard.yaml --checkpoint-dir artifacts/sudoku/hard \
  --data-dir data/sudoku/hard --n 1000 --attempts 10 \
  --methods resample@0.5,resample@1.0,refine6@0.5,refine6@1.0,refine4@0.5,refine3@0.5,refine8@0.5 \
  --out round_trip/runs/p5_sudoku_search_hard.json > /root/logs/p5_sudoku_search_hard.log 2>&1
touch /root/logs/S_DONE

# A2: held-out seeds 4-6 for the tuning-grid winners (temperature curve for these seeds already measured in run 14)
AV=d1.3+t0.8@6+r5x2,d1.15+t0.85@6+r6x3,r5x3,d1.25+t0.8@6+r6x2,d1.3+t0.8@6+r6,d1.2+t0.8@5+r5x2
for s in 4 5 6; do
  python -I round_trip/refine_eval.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --n 256 --bs 128 --seed $s --variants full,$AV --out round_trip/runs/p5_ts_s$s.pkl > /root/logs/p5_ts_s$s.log 2>&1
  python -I round_trip/analyze_skip.py --in round_trip/runs/p5_ts_s$s.pkl --ref gpt2-large --bs 64 > /root/logs/score_p5_ts_s$s.log 2>&1
done
touch /root/logs/ALL_DONE
