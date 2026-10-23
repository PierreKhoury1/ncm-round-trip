#!/bin/bash
# LM1B small + large: re-encode refinement vs the authors' sampler, FLM scoring protocol (raw ids, specials kept).
mkdir -p /root/logs round_trip/runs
rm -f /root/logs/ALL_DONE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q omegaconf hydra-core regex "transformers>=4.51,<5" "huggingface_hub[cli]" vastai > /root/logs/pip.log 2>&1
hf download hazyresearch/ncm-checkpoints --include "lm1b/tokenizer.pt" "lm1b/small/*" "lm1b/large/*" --local-dir artifacts > /root/logs/dl.log 2>&1
touch /root/logs/SETUP_DONE
V=full,r9,r7,r6,r5,t0.9@1,t0.85@1,t0.8@1
for cfg in "small 64" "large 32"; do set -- $cfg
  for s in 1 2 3; do
    python -I round_trip/refine_eval.py --config config/eval/lm1b-$1.yaml --checkpoint-dir artifacts/lm1b/$1 --n 256 --bs $2 --seed $s --variants $V --out round_trip/runs/lm1b_$1_s$s.pkl > /root/logs/lm1b_$1_s$s.log 2>&1
    python -I round_trip/score_flm.py --in round_trip/runs/lm1b_$1_s$s.pkl --ref gpt2-large > /root/logs/score_lm1b_$1_s$s.log 2>&1
  done
done
touch /root/logs/ALL_DONE
