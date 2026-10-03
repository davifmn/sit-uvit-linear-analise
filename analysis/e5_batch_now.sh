#!/bin/zsh
# E5 on checkpoints that already exist (CPU). Waits for E3 to free the CPU first.
cd "$(dirname $0)"
PY=${PY:-python}
while pgrep -f e3_attention_capacity.py >/dev/null; do sleep 30; done
U=../results/000-SiT-S-2-uvit-Linear-velocity-None/checkpoints
L=../results/001-SiT-S-2-linear-Linear-velocity-None/checkpoints
$PY e5_grad_noise.py baseline - "baseline (pré-treinado, todos os 33M)"
$PY e5_grad_noise.py uvit - "uvit @0 (skips)"
$PY e5_grad_noise.py uvit $U/0040000.pt "uvit @40k (skips)"
$PY e5_grad_noise.py linear - "linear @0 (atenção LiT)"
$PY e5_grad_noise.py linear $L/0010000.pt "linear @10k (atenção LiT)"
