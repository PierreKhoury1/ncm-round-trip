#!/bin/bash
# Run 20: redo the LM1B-large canonicality stage of run 19. Its draft texts were empty (LM1B "texts" are cut at the
# leading [CLS]; the drafts are now also kept as raw ids and scored under the FLM protocol), then the Qwen scoring of
# those pickles. Same seeds and variants as run 19, so the samples are the same.
mkdir -p /root/logs round_trip/runs
rm -f /root/logs/ALL_DONE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q omegaconf hydra-core regex scipy "transformers>=4.51,<5" "huggingface_hub[cli]" vastai > /root/logs/pip.log 2>&1
hf download hazyresearch/ncm-checkpoints --include "lm1b/tokenizer.pt" "lm1b/large/*" --local-dir artifacts > /root/logs/dl.log 2>&1
touch /root/logs/SETUP_DONE

for s in 1 2 3; do
  python -I round_trip/refine_eval.py --config config/eval/lm1b-large.yaml --checkpoint-dir artifacts/lm1b/large --n 256 --bs 32 --seed $s \
    --variants full,r6,b4,b4+r6 --out round_trip/runs/p9_canon_lm1bl_s$s.pkl > /root/logs/p9_canon_lm1bl_s$s.log 2>&1
  python -I round_trip/score_flm.py --in round_trip/runs/p9_canon_lm1bl_s$s.pkl --ref gpt2-large > /root/logs/score_p9_canon_lm1bl_s$s.log 2>&1
  python -I round_trip/score_ppl.py --in round_trip/runs/p9_canon_lm1bl_s$s.pkl --ref gpt2-large --bs 16 --out round_trip/runs/p9_g2l_canon_lm1bl_s$s.json > /root/logs/g2l_p9_canon_lm1bl_s$s.log 2>&1
  python -I round_trip/canon_analyze.py --pkl round_trip/runs/p9_canon_lm1bl_s$s.pkl --scores round_trip/runs/p9_g2l_canon_lm1bl_s$s.json --out round_trip/runs/p9_canon_lm1bl_s$s.json > /root/logs/canon_p9_lm1bl_s$s.log 2>&1
done
touch /root/logs/C_DONE
for s in 1 2 3; do
  python -I round_trip/score_ppl.py --in round_trip/runs/p9_canon_lm1bl_s$s.pkl --ref Qwen/Qwen2.5-7B --dtype bf16 --bs 16 --out round_trip/runs/p9_qwen_p9_canon_lm1bl_s$s.json > /root/logs/qwen_p9_canon_lm1bl_s$s.log 2>&1
done
touch /root/logs/ALL_DONE
