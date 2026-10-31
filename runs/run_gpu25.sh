#!/bin/bash
# Run 25: coarse-only round trip. c<k> decodes the draft truncated to levels < k (fine_drop-trained decoder), re-encodes that
# text, forces levels < k canonical and generates levels >= k once: in a real sampler no extra generator pass. Emulated on a
# full draft here. u<k> = the truncated decode itself (how good is the text the coarse levels alone carry).
# TinyStories + LM1B large, seeds 1-3, 256 samples, GPT-2 Large scoring. One A100 40 GB.
mkdir -p /root/logs round_trip/runs
rm -f /root/logs/ALL_DONE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q omegaconf hydra-core regex scipy "transformers>=4.51,<5" "huggingface_hub[cli]" vastai > /root/logs/pip.log 2>&1
hf download hazyresearch/ncm-checkpoints --include "tinystories/*" "lm1b/tokenizer.pt" "lm1b/large/*" --local-dir artifacts > /root/logs/dl.log 2>&1
touch /root/logs/SETUP_DONE
TS_V=full,r6,u4,u6,u8,c3,c4,c5,c6,c7,c8,c6+r6,c5+r5,c6x2
for s in 1 2 3; do
  python -I round_trip/refine_eval.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --n 256 --bs 128 --seed $s \
    --variants $TS_V --out round_trip/runs/p16_coarse_ts_s$s.pkl > /root/logs/p16_coarse_ts_s$s.log 2>&1
  python -I round_trip/analyze_skip.py --in round_trip/runs/p16_coarse_ts_s$s.pkl --ref gpt2-large --bs 64 > /root/logs/score_p16_coarse_ts_s$s.log 2>&1
done
touch /root/logs/TS_DONE
LM_V=full,r6,u6,c5,c6,c7,c6+r6,c6x2
for s in 1 2 3; do
  python -I round_trip/refine_eval.py --config config/eval/lm1b-large.yaml --checkpoint-dir artifacts/lm1b/large --n 256 --bs 32 --seed $s \
    --variants $LM_V --out round_trip/runs/p16_coarse_lm1bl_s$s.pkl > /root/logs/p16_coarse_lm1bl_s$s.log 2>&1
  python -I round_trip/score_flm.py --in round_trip/runs/p16_coarse_lm1bl_s$s.pkl --ref gpt2-large > /root/logs/score_p16_coarse_lm1bl_s$s.log 2>&1
done
touch /root/logs/LM_DONE
touch /root/logs/ALL_DONE
