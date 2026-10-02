#!/bin/zsh
# Sequential 60k queue on MPS. Same protocol as the GPU runs: effective batch 8, AdamW lr 1e-4,
# frozen backbone for variants, no EMA, seed 0, latent cache. (fp32 instead of bf16.)
cd "$(dirname $0)/.."
PY=${PY:-python}
common=(--model SiT-S/2 --image-size 256 --latent-path data/imagenet50k_latents --vae ema
        --global-batch-size 8 --grad-accum-steps 1 --precision fp32 --no-ema
        --learning-rate 1e-4 --global-seed 0 --num-workers 2 --device mps
        --log-every 100 --sample-every 0 --ckpt-every 5000 --results-dir results)
$PY analysis/train_mps.py $common --variant linear --freeze-backbone \
    --pretrained pretrained_models/SiT-S-2-256.pt --max-steps 60000
$PY analysis/train_mps.py $common --variant linear_uvit --freeze-backbone \
    --pretrained pretrained_models/SiT-S-2-256.pt --max-steps 60000
$PY analysis/train_mps.py $common --variant uvit --freeze-backbone \
    --ckpt results/000-SiT-S-2-uvit-Linear-velocity-None/checkpoints/0040000.pt --max-steps 60000
$PY analysis/train_mps.py $common --variant baseline --no-freeze-backbone \
    --pretrained pretrained_models/SiT-S-2-256.pt --max-steps 10000 --ckpt-every 2500
