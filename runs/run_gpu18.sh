#!/bin/bash
# Run 18: does round-trip refinement work for IMAGE next-scale models? VAR-d16 (ImageNet 256, class-conditional).
#   S0 sanity (2 per class): 'f' must reproduce the draft exactly; per-scale code change of the image round trip
#   S1 screen (10 per class = 10k images, FID-10k vs the ADM reference batch): base, r5..r8, q7, n7, r7x2
#   S2 confirm (50 per class = 50k images, FID-50k): base and the two best r-variants from S1
mkdir -p /root/logs /root/ncm-round_trip/round_trip/runs /root/ckpt /root/ref
rm -f /root/logs/ALL_DONE
pip install -q pytorch-fid scipy "huggingface_hub[cli]" vastai > /root/logs/pip.log 2>&1
git clone -q https://github.com/FoundationVision/VAR.git /root/VAR > /root/logs/git.log 2>&1
hf download FoundationVision/var vae_ch160v4096z32.pth var_d16.pth --local-dir /root/ckpt > /root/logs/dl.log 2>&1
wget -q -O /root/ref/VIRTUAL_imagenet256_labeled.npz https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz >> /root/logs/dl.log 2>&1
cp /root/ncm-round_trip/round_trip/var_rt.py /root/VAR/var_rt.py
touch /root/logs/SETUP_DONE

cd /root/VAR
python var_rt.py --var-dir /root/VAR --ckpt-dir /root/ckpt --depth 16 --n-per-class 2 --bs 64 --variants base,f,r7,q7 \
  --save-png 32 --out /root/ncm-round_trip/round_trip/runs/p8_var_sanity.npz > /root/logs/p8_sanity.log 2>&1
touch /root/logs/S0_DONE
python var_rt.py --var-dir /root/VAR --ckpt-dir /root/ckpt --depth 16 --n-per-class 10 --bs 64 \
  --variants base,r8,r7,r6,r5,q7,n7,r7x2 --ref /root/ref/VIRTUAL_imagenet256_labeled.npz \
  --out /root/ncm-round_trip/round_trip/runs/p8_var_screen.npz > /root/logs/p8_screen.log 2>&1
touch /root/logs/S1_DONE
BEST=$(python - <<'EOF'
import json
R = json.load(open('/root/ncm-round_trip/round_trip/runs/p8_var_screen.json'))
r = sorted([(v['fid'], k) for k, v in R.items() if k.startswith('r') and 'fid' in v])[:2]
print(','.join(['base'] + [k for _, k in r]))
EOF
)
echo "confirm: $BEST" > /root/logs/p8_confirm_variants.log
python var_rt.py --var-dir /root/VAR --ckpt-dir /root/ckpt --depth 16 --n-per-class 50 --bs 64 --seed 1 \
  --variants $BEST --ref /root/ref/VIRTUAL_imagenet256_labeled.npz \
  --out /root/ncm-round_trip/round_trip/runs/p8_var_confirm.npz > /root/logs/p8_confirm.log 2>&1
touch /root/logs/ALL_DONE
