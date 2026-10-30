#!/bin/bash
# Run 23: own-error ("noise-matched") conditioning corruption at the paper's batch size. Section 7 found it hurt at batch
# 64-128, but run 17 showed that batch size alone removes the fine-tuning drift, so the small-batch negative is confounded.
# Arms (200 steps, batch 2048, lr 1e-4 warmup 30 cosine->0, AdamW 0.9/0.95, wd 0.1, clip 1, bf16, corruption 0.75->0.25):
#   ctrl    paper recipe, uniform replacement codes (same as run 17 ctrl, re-run on this box)
#   mix50   half of the replacement codes are the model's own sampled guesses (--self_mix 0.5)
#   mix100  all replacement codes are the model's own sampled guesses (--self_mix 1.0)
# Then every model + released at 11 passes, RT6 and tau 0.9 / 1.1 on held-out seeds 4-6, GPT-2 Large.
mkdir -p /root/logs round_trip/runs round_trip/ckpt
rm -f /root/logs/ALL_DONE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q omegaconf hydra-core regex "transformers>=4.51,<5" "huggingface_hub[cli]" vastai > /root/logs/pip.log 2>&1
hf download hazyresearch/ncm-checkpoints --include "tinystories/*" --local-dir artifacts > /root/logs/dl.log 2>&1
hf download hazyresearch/ncm-datasets --repo-type dataset --include "tinystories/train_bpe_8192.bin" "tinystories/train_bpe_8192_offsets.npy" "tinystories/valid_bpe_8192.pickle" --local-dir data >> /root/logs/dl.log 2>&1
touch /root/logs/SETUP_DONE

COMMON="--p_drop 0.0 --corrupt 0.75,0.25 --steps 200 --bs 2048 --accum 64 --lr 1e-4 --warmup 30 --amp --eval_every 50 --resume"
# checkpoints live under runs/ so the watcher's glob brings them home
CUDA_VISIBLE_DEVICES=0 python -I round_trip/train_dropout.py $COMMON --out round_trip/runs/p12_ctrl.pt > /root/logs/train_p12_ctrl.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -I round_trip/train_dropout.py $COMMON --self_mix 0.5 --out round_trip/runs/p12_mix50.pt > /root/logs/train_p12_mix50.log 2>&1 &
wait
touch /root/logs/TRAIN1_DONE

evals() {  # $1 GPU, $2 name, $3 generator path ("" = released)
  for s in 4 5 6; do
    G=""; [ -n "$3" ] && G="--generator $3"
    CUDA_VISIBLE_DEVICES=$1 python -I round_trip/refine_eval.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories $G --n 256 --bs 128 --seed $s --variants full,r6,r5x2,t0.9@1,t0.8@1,t1.1@1 --out round_trip/runs/p12_$2_s$s.pkl > /root/logs/p12_$2_s$s.log 2>&1
    CUDA_VISIBLE_DEVICES=$1 python -I round_trip/analyze_skip.py --in round_trip/runs/p12_$2_s$s.pkl --ref gpt2-large --bs 64 > /root/logs/score_p12_$2_s$s.log 2>&1
  done
}
CUDA_VISIBLE_DEVICES=0 python -I round_trip/train_dropout.py $COMMON --self_mix 1.0 --out round_trip/runs/p12_mix100.pt > /root/logs/train_p12_mix100.log 2>&1 &
(evals 1 released ""; evals 1 ctrl round_trip/runs/p12_ctrl.pt; evals 1 mix50 round_trip/runs/p12_mix50.pt) &
wait
touch /root/logs/TRAIN2_DONE
evals 0 mix100 round_trip/runs/p12_mix100.pt
rm -f round_trip/runs/p12_*.pt.opt
touch /root/logs/ALL_DONE
