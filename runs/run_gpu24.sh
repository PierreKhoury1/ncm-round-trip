#!/bin/bash
# Run 24: CUDA-graph timing. Released sampler and RT6 on the TinyStories checkpoint, RTX 4090 (the paper's timing GPU):
# eager (as released) vs per-level pass compiled with CUDA graphs (reduce-overhead) vs torch.compile default (fusion only).
# Batch 128 (paper) and batch 8 (latency regime). Median of 7 after warm-up.
mkdir -p /root/logs round_trip/runs
rm -f /root/logs/ALL_DONE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pip install -q omegaconf hydra-core regex "transformers>=4.51,<5" "huggingface_hub[cli]" vastai > /root/logs/pip.log 2>&1
hf download hazyresearch/ncm-checkpoints --include "tinystories/*" --local-dir artifacts > /root/logs/dl.log 2>&1
touch /root/logs/SETUP_DONE
nvidia-smi --query-gpu=name,driver_version --format=csv > /root/logs/gpu.txt 2>&1
python -I round_trip/timing_graph.py --modes eager,graphs,compile --batches 128,8 --reps 7 --out round_trip/runs/p13_timing_graph.json > /root/logs/p13_timing_graph.log 2>&1
touch /root/logs/ALL_DONE
