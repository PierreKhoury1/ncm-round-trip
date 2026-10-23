#!/bin/bash
# Phase 3: Phase 2 at the paper's batch size. Phase 2 fine-tuned with batch 128 and drifted +3.9%; the paper's
# TinyStories generator was trained with batch 2048 (Table 6). Only the batch changes here, plus 300k refined
# stories so the 50% refined share is not 60k stories seen 25 times. Same as Phase 2 otherwise: lr 3e-5 cosine,
# AdamW (0.9, 0.95), wd 0.1, clip 1, 1500 steps, corruption 0.75 -> 0.25. GPU 0 trains dist, GPU 1 trains ctrl.
# Then WiSE-FT soups, evals, and sampler ablations on the released model (no re-encode, hot drafts).
# ACCUM (micro-batch = 2048 / ACCUM) is set at launch; checkpoints + optimizer state every 100 steps (--resume).
mkdir -p /root/logs round_trip/runs round_trip/ckpt
rm -f /root/logs/ALL_DONE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ACCUM=${ACCUM:-128}
pip install -q omegaconf hydra-core regex "transformers>=4.51,<5" "huggingface_hub[cli]" vastai > /root/logs/pip.log 2>&1
hf download hazyresearch/ncm-checkpoints --include "tinystories/*" "lm1b/tokenizer.pt" "lm1b/large/*" --local-dir artifacts > /root/logs/dl.log 2>&1
hf download hazyresearch/ncm-datasets --repo-type dataset --include "tinystories/train_bpe_8192.bin" "tinystories/train_bpe_8192_offsets.npy" "tinystories/valid_bpe_8192.pickle" --local-dir data >> /root/logs/dl.log 2>&1
touch /root/logs/SETUP_DONE

if [ ! -f /root/logs/GEN_DONE ]; then
  for g in 0 1; do
    CUDA_VISIBLE_DEVICES=$g python -I round_trip/refine_eval.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --n 150000 --bs 256 --seed $((7 + g)) --variants r5x2 --out round_trip/runs/distill_r5x2_$g.pkl > /root/logs/distill_gen_$g.log 2>&1 &
  done
  wait
  touch /root/logs/GEN_DONE
fi

COMMON="--p_drop 0.0 --corrupt 0.75,0.25 --steps 1500 --bs 2048 --accum $ACCUM --lr 3e-5 --warmup 100 --amp --eval_every 100 --resume"
CUDA_VISIBLE_DEVICES=0 python -I round_trip/train_dropout.py $COMMON --distill round_trip/runs/distill_r5x2_0.pkl,round_trip/runs/distill_r5x2_1.pkl --distill_variant r5x2 --distill_frac 0.5 --out round_trip/ckpt/dist2k.pt >> /root/logs/train_dist2k.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -I round_trip/train_dropout.py $COMMON --out round_trip/ckpt/ctrl2k.pt >> /root/logs/train_ctrl2k.log 2>&1 &
wait
touch /root/logs/TRAIN_DONE
for g in dist2k ctrl2k; do
  python -I round_trip/soup.py --finetuned round_trip/ckpt/$g.pt --alpha 0.5 --out round_trip/ckpt/${g}_soup.pt > /root/logs/soup_$g.log 2>&1
done

evals() {  # $1 = GPU, rest = generator names
  local gpu=$1; shift
  for g in "$@"; do
    for s in 1 2 3; do
      CUDA_VISIBLE_DEVICES=$gpu python -I round_trip/refine_eval.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --generator round_trip/ckpt/$g.pt --n 256 --bs 128 --seed $s --variants full,r6 --out round_trip/runs/p3_${g}_s$s.pkl > /root/logs/p3_${g}_s$s.log 2>&1
      CUDA_VISIBLE_DEVICES=$gpu python -I round_trip/analyze_skip.py --in round_trip/runs/p3_${g}_s$s.pkl --ref gpt2-large --bs 64 > /root/logs/score_p3_${g}_s$s.log 2>&1
    done
  done
}
# n = no re-encode, q = latent re-quantization (Scale-travel), r = text re-encode (ours); d = hot draft
SV=full,r6,n6,q6,r5x2,n5x2,q5x2,d1.1+r6,d1.2+r6,d1.1+r5x2,d1.2+r5x2,d1.2+r3+r6+r9,t1.1@1,t1.2@1
samplers() {
  for s in 1 2 3; do
    CUDA_VISIBLE_DEVICES=1 python -I round_trip/refine_eval.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --n 256 --bs 128 --seed $s --variants $SV --out round_trip/runs/p3_samplers_s$s.pkl > /root/logs/p3_samplers_s$s.log 2>&1
    CUDA_VISIBLE_DEVICES=1 python -I round_trip/analyze_skip.py --in round_trip/runs/p3_samplers_s$s.pkl --ref gpt2-large --bs 64 > /root/logs/score_p3_samplers_s$s.log 2>&1
  done
}
LMV=full,r6,n6,q6,d1.1+r6x2
lm1b() {
  for s in 1 2 3; do
    CUDA_VISIBLE_DEVICES=0 python -I round_trip/refine_eval.py --config config/eval/lm1b-large.yaml --checkpoint-dir artifacts/lm1b/large --n 256 --bs 32 --seed $s --variants $LMV --out round_trip/runs/p3_lm1b_s$s.pkl > /root/logs/p3_lm1b_s$s.log 2>&1
    CUDA_VISIBLE_DEVICES=0 python -I round_trip/score_flm.py --in round_trip/runs/p3_lm1b_s$s.pkl --ref gpt2-large > /root/logs/score_p3_lm1b_s$s.log 2>&1
  done
}
(evals 0 dist2k dist2k_soup; lm1b) &
(evals 1 ctrl2k ctrl2k_soup; samplers) &
wait
touch /root/logs/ALL_DONE
