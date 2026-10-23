#!/bin/bash
# Evidence run on released checkpoints (no training): held-out seeds, LLM judge, MAUVE, LM1B scale trend, OWT.
#   A. TinyStories held-out seeds 4-6: key samplers + full temperature curve (tau 0.7..1.2)
#   B. Blind pairwise judge (Qwen2.5-7B-Instruct, both orders) on seeds 1-3 outputs (uploaded pickles)
#   C. MAUVE vs 768 human TinyStories openings (uploaded real_ts.pkl)
#   D. LM1B medium, seeds 1-3 (small / medium / large trend)
#   E. LM1B large held-out seeds 4-6
#   F. OpenWebText checkpoint (32 levels, 1024 tokens), seeds 1-3
mkdir -p /root/logs round_trip/runs
rm -f /root/logs/ALL_DONE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q omegaconf hydra-core regex "transformers>=4.51,<5" "huggingface_hub[cli]" vastai mauve-text > /root/logs/pip.log 2>&1
hf download hazyresearch/ncm-checkpoints --include "tinystories/*" "lm1b/tokenizer.pt" "lm1b/medium/*" "lm1b/large/*" "owt/*" --local-dir artifacts > /root/logs/dl.log 2>&1
hf download hazyresearch/ncm-checkpoints --include "sudoku/hard/*" --local-dir artifacts >> /root/logs/dl.log 2>&1
hf download hazyresearch/ncm-datasets --repo-type dataset --include "sudoku/*" --local-dir data >> /root/logs/dl.log 2>&1
(hf download Qwen/Qwen2.5-7B-Instruct > /root/logs/dl_judge.log 2>&1; touch /root/logs/JUDGE_DL_DONE) &
touch /root/logs/SETUP_DONE

# H. Sudoku-hard (reasoning): does refinement = thinking longer raise the solve rate? Greedy, as configured.
find data -name "*.pickle" > /root/logs/sudoku_files.log 2>&1
SDIR=$(dirname "$(find data -name puzzles.pickle | grep -i hard | head -1)")
python -I round_trip/refine_eval.py --config config/eval/sudoku-hard.yaml --checkpoint-dir artifacts/sudoku/hard --split test --data-dir "$SDIR" --n 1000 --bs 128 --variants full,r9,r8,r6,r4,r6x2,r6x4,r4x3,r3+r6+r9 --out round_trip/runs/p4_sudoku_hard.pkl > /root/logs/p4_sudoku.log 2>&1
python -I round_trip/score_cond.py --in round_trip/runs/p4_sudoku_hard.pkl > /root/logs/score_p4_sudoku.log 2>&1
touch /root/logs/H_DONE

TV=full,t0.7@1,t0.8@1,t0.85@1,t0.9@1,t0.95@1,t1.1@1,t1.2@1,r6,r5x2,r3+r6+r9,t0.8@6+r6,d1.1+r6,d1.2+t0.8@6+r6,d1.2+t0.8@6+r5x2,n6,q6
for s in 4 5 6; do
  python -I round_trip/refine_eval.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --n 256 --bs 128 --seed $s --variants $TV --out round_trip/runs/p4_ts_s$s.pkl > /root/logs/p4_ts_s$s.log 2>&1
  python -I round_trip/analyze_skip.py --in round_trip/runs/p4_ts_s$s.pkl --ref gpt2-large --bs 64 > /root/logs/score_p4_ts_s$s.log 2>&1
done
touch /root/logs/A_DONE

# G. improvement grid on seeds 1-3 (tuning seeds; winners are confirmed on 4-6 in the next run):
#    hotter drafts, cooler regenerated levels, more rounds
GV=d1.2+t0.8@5+r5x2,d1.3+t0.8@6+r5x2,d1.3+t0.8@6+r3+r6+r9,d1.2+t0.9@3+r3+r6+r9,d1.25+t0.8@6+r6x2,d1.2+t0.8@4+r4+r8,d1.15+t0.85@6+r6x3,d1.3+t0.8@6+r6,r6x3,r5x3,d1.2+t0.7@6+r6
for s in 1 2 3; do
  python -I round_trip/refine_eval.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --n 256 --bs 128 --seed $s --variants $GV --out round_trip/runs/p4_grid_s$s.pkl > /root/logs/p4_grid_s$s.log 2>&1
  python -I round_trip/analyze_skip.py --in round_trip/runs/p4_grid_s$s.pkl --ref gpt2-large --bs 64 > /root/logs/score_p4_grid_s$s.log 2>&1
done
touch /root/logs/G_DONE

while [ ! -f /root/logs/JUDGE_DL_DONE ]; do sleep 20; done
python -I round_trip/judge_pairs.py --pkl "round_trip/runs/p3_samplers_s{s}.pkl,round_trip/runs/ts_more_s{s}.pkl,round_trip/runs/gpu_var_s{s}.pkl" --seeds 1,2,3 \
  --pairs full:r6,t0.8@1:r6,full:d1.1+r6,full:r3+r6+r9,t0.8@1:r5x2,full:t0.8@1,q6:r6 --task stories \
  --out round_trip/runs/judge_ts.json > /root/logs/judge_ts.log 2>&1
touch /root/logs/B_DONE

python -I round_trip/score_mauve.py --pkl "round_trip/runs/p3_samplers_s{s}.pkl,round_trip/runs/ts_more_s{s}.pkl,round_trip/runs/gpu_var_s{s}.pkl" --seeds 1,2,3 \
  --human round_trip/runs/real_ts.pkl --variants full,t0.9@1,t0.8@1,r6,r5x2,r3+r6+r9,t0.8@6+r6,d1.1+r6,d1.2+t0.8@6+r6,n6,q6 \
  --out round_trip/runs/mauve_ts.json > /root/logs/mauve_ts.log 2>&1
touch /root/logs/C_DONE

for s in 1 2 3; do
  python -I round_trip/refine_eval.py --config config/eval/lm1b-medium.yaml --checkpoint-dir artifacts/lm1b/medium --n 256 --bs 32 --seed $s --variants full,t0.9@1,t0.85@1,t0.8@1,r7,r6,r6x2 --out round_trip/runs/p4_lm1bm_s$s.pkl > /root/logs/p4_lm1bm_s$s.log 2>&1
  python -I round_trip/score_flm.py --in round_trip/runs/p4_lm1bm_s$s.pkl --ref gpt2-large > /root/logs/score_p4_lm1bm_s$s.log 2>&1
done
touch /root/logs/D_DONE

for s in 4 5 6; do
  python -I round_trip/refine_eval.py --config config/eval/lm1b-large.yaml --checkpoint-dir artifacts/lm1b/large --n 256 --bs 32 --seed $s --variants full,t0.9@1,t0.85@1,t0.8@1,r6,r6x2,d1.2+r6x2,n6,q6 --out round_trip/runs/p4_lm1bl_s$s.pkl > /root/logs/p4_lm1bl_s$s.log 2>&1
  python -I round_trip/score_flm.py --in round_trip/runs/p4_lm1bl_s$s.pkl --ref gpt2-large > /root/logs/score_p4_lm1bl_s$s.log 2>&1
done
touch /root/logs/E_DONE

for s in 1 2 3; do
  python -I round_trip/refine_eval.py --config config/eval/owt.yaml --checkpoint-dir artifacts/owt --n 256 --bs 16 --seed $s --variants full,r24,r16 --out round_trip/runs/p4_owt_s$s.pkl > /root/logs/p4_owt_s$s.log 2>&1
  python -I round_trip/analyze_skip.py --in round_trip/runs/p4_owt_s$s.pkl --ref gpt2-large --bs 8 > /root/logs/score_p4_owt_s$s.log 2>&1
done
touch /root/logs/ALL_DONE
