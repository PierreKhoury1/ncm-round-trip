#!/bin/bash
# Sampler-only run on the released checkpoints (no training): the ablations the paper needs and hot drafts.
#   n<k>  keep the draft's own codes (no re-encode)        -> is the gain from re-encoding at all?
#   q<k>  re-quantize the draft's latent (Scale-travel)     -> does going through the text matter?
#   d<T>  hot draft, then refine; t0.8@6 cools the refined fine levels -> better matched-diversity frontier?
# TinyStories (GPT-2 Large, 3 seeds x 256) and LM1B large (FLM protocol, 3 seeds x 256).
mkdir -p /root/logs round_trip/runs
rm -f /root/logs/ALL_DONE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q omegaconf hydra-core regex "transformers>=4.51,<5" "huggingface_hub[cli]" vastai > /root/logs/pip.log 2>&1
hf download hazyresearch/ncm-checkpoints --include "tinystories/*" "lm1b/tokenizer.pt" "lm1b/large/*" --local-dir artifacts > /root/logs/dl.log 2>&1
touch /root/logs/SETUP_DONE

SV=full,r6,n6,q6,r5x2,n5x2,q5x2,d1.1+r6,d1.2+r6,d1.2+t0.8@6+r6,d1.1+r5x2,d1.2+r5x2,d1.2+t0.8@6+r5x2,d1.2+r3+r6+r9,t1.1@1,t1.2@1
for s in 1 2 3; do
  python -I round_trip/refine_eval.py --config config/eval/tinystories.yaml --checkpoint-dir artifacts/tinystories --n 256 --bs 128 --seed $s --variants $SV --out round_trip/runs/p3_samplers_s$s.pkl > /root/logs/p3_samplers_s$s.log 2>&1
  python -I round_trip/analyze_skip.py --in round_trip/runs/p3_samplers_s$s.pkl --ref gpt2-large --bs 64 > /root/logs/score_p3_samplers_s$s.log 2>&1
done
touch /root/logs/TS_DONE
LMV=full,r6,n6,q6,r6x2,d1.1+r6x2,d1.2+r6x2
for s in 1 2 3; do
  python -I round_trip/refine_eval.py --config config/eval/lm1b-large.yaml --checkpoint-dir artifacts/lm1b/large --n 256 --bs 32 --seed $s --variants $LMV --out round_trip/runs/p3_lm1b_s$s.pkl > /root/logs/p3_lm1b_s$s.log 2>&1
  python -I round_trip/score_flm.py --in round_trip/runs/p3_lm1b_s$s.pkl --ref gpt2-large > /root/logs/score_p3_lm1b_s$s.log 2>&1
done
touch /root/logs/ALL_DONE
