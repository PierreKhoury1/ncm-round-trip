#!/bin/bash
# Phase 2: distil re-encode refinement into the generator so 11 passes give refined quality.
# dist = half real, half refined (r5x2) stories; ctrl = real only, same gentler recipe. Then WiSE-FT soups.
mkdir -p /root/logs round_trip/runs round_trip/ckpt
rm -f /root/logs/ALL_DONE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q omegaconf hydra-core regex "transformers>=4.51,<5" "huggingface_hub[cli]" vastai > /root/logs/pip.log 2>&1
hf download hazyresearch/ncm-checkpoints --include "tinystories/*" --local-dir artifacts > /root/logs/dl.log 2>&1
hf download hazyresearch/ncm-datasets --repo-type dataset --include "tinystories/train_bpe_8192.bin" "tinystories/train_bpe_8192_offsets.npy" "tinystories/valid_bpe_8192.pickle" --local-dir data >> /root/logs/dl.log 2>&1
touch /root/logs/SETUP_DONE

python -I round_trip/refine_eval.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --n 60000 --bs 256 --seed 7 --variants r5x2 --out round_trip/runs/distill_r5x2.pkl > /root/logs/distill_gen.log 2>&1
touch /root/logs/GEN_DONE

COMMON="--p_drop 0.0 --corrupt 0.75,0.25 --steps 1500 --bs 128 --accum 8 --lr 3e-5 --warmup 100 --amp --eval_every 500"
python -I round_trip/train_dropout.py $COMMON --distill round_trip/runs/distill_r5x2.pkl --distill_variant r5x2 --distill_frac 0.5 --out round_trip/ckpt/dist.pt > /root/logs/train_dist.log 2>&1 &
python -I round_trip/train_dropout.py $COMMON --out round_trip/ckpt/ctrl.pt > /root/logs/train_ctrl.log 2>&1 &
wait
touch /root/logs/TRAIN_DONE
for g in dist ctrl; do
  python -I round_trip/soup.py --finetuned round_trip/ckpt/$g.pt --alpha 0.5 --out round_trip/ckpt/${g}_soup.pt > /root/logs/soup_$g.log 2>&1
done
for g in dist dist_soup ctrl ctrl_soup; do
  for s in 1 2 3; do
    python -I round_trip/refine_eval.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --generator round_trip/ckpt/$g.pt --n 256 --bs 128 --seed $s --variants full,r6 --out round_trip/runs/p2_${g}_s$s.pkl > /root/logs/p2_${g}_s$s.log 2>&1
    python -I round_trip/analyze_skip.py --in round_trip/runs/p2_${g}_s$s.pkl --ref gpt2-large --bs 64 > /root/logs/score_p2_${g}_s$s.log 2>&1
  done
done
touch /root/logs/ALL_DONE
