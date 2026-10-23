#!/bin/bash
# Run 17 (pilot): can training teach NCM's fine levels to correct coarse errors?  2 GPUs, batch 2048 (the paper's).
#   ctrl   the paper's recipe: uniform conditioning corruption 0.75 -> 0.25, CLEAN targets   (drift control)
#   bscu   same corruption, targets RE-QUANTIZED from the residual of the corrupted inputs  (self-correction)
#   bscnn  same, replacement codes drawn from the 16 nearest codes (near-miss errors)
# 200 steps each, lr 1e-4 (warmup 30, cosine to 0), AdamW (0.9, 0.95), wd 0.1, clip 1, bf16; checkpoints every 50.
# Then every model at 11 passes and with RT6 on held-out seeds 4-6, plus tau 0.9 / 1.1 for its own frontier.
mkdir -p /root/logs round_trip/runs round_trip/ckpt
rm -f /root/logs/ALL_DONE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q omegaconf hydra-core regex "transformers>=4.51,<5" "huggingface_hub[cli]" vastai > /root/logs/pip.log 2>&1
hf download hazyresearch/ncm-checkpoints --include "tinystories/*" --local-dir artifacts > /root/logs/dl.log 2>&1
hf download hazyresearch/ncm-datasets --repo-type dataset --include "tinystories/train_bpe_8192.bin" "tinystories/train_bpe_8192_offsets.npy" "tinystories/valid_bpe_8192.pickle" --local-dir data >> /root/logs/dl.log 2>&1
touch /root/logs/SETUP_DONE

# safety: the self-correction target code must reproduce vae.encode exactly when corruption is off
python -I round_trip/train_dropout.py --smoke_bsc --corrupt 0.75,0.25 --p_drop 0.0 > /root/logs/smoke_bsc.log 2>&1
if ! grep -q "identical to vae.encode: True" /root/logs/smoke_bsc.log; then
  touch /root/logs/SMOKE_FAIL /root/logs/ALL_DONE; exit 1
fi
touch /root/logs/SMOKE_OK

COMMON="--p_drop 0.0 --corrupt 0.75,0.25 --steps 200 --bs 2048 --accum 64 --lr 1e-4 --warmup 30 --amp --eval_every 50 --resume"
CUDA_VISIBLE_DEVICES=0 python -I round_trip/train_dropout.py $COMMON --out round_trip/ckpt/p7_ctrl.pt >> /root/logs/train_p7_ctrl.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -I round_trip/train_dropout.py $COMMON --bsc --out round_trip/ckpt/p7_bscu.pt >> /root/logs/train_p7_bscu.log 2>&1 &
wait
touch /root/logs/TRAIN1_DONE

evals() {  # $1 GPU, $2 name, $3 generator path ("" = released)
  for s in 4 5 6; do
    G=""; [ -n "$3" ] && G="--generator $3"
    CUDA_VISIBLE_DEVICES=$1 python -I round_trip/refine_eval.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories $G --n 256 --bs 128 --seed $s --variants full,r6,t0.9@1,t1.1@1 --out round_trip/runs/p7_$2_s$s.pkl > /root/logs/p7_$2_s$s.log 2>&1
    CUDA_VISIBLE_DEVICES=$1 python -I round_trip/analyze_skip.py --in round_trip/runs/p7_$2_s$s.pkl --ref gpt2-large --bs 64 > /root/logs/score_p7_$2_s$s.log 2>&1
  done
}
CUDA_VISIBLE_DEVICES=0 python -I round_trip/train_dropout.py $COMMON --bsc --corrupt_nn 16 --out round_trip/ckpt/p7_bscnn.pt >> /root/logs/train_p7_bscnn.log 2>&1 &
(evals 1 released ""; evals 1 ctrl round_trip/ckpt/p7_ctrl.pt; evals 1 bscu round_trip/ckpt/p7_bscu.pt) &
wait
evals 0 bscnn round_trip/ckpt/p7_bscnn.pt
touch /root/logs/ALL_DONE
