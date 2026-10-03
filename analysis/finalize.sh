#!/bin/zsh
# Runs after run_queue.sh: learning curves, interleaved latency, E5 at final checkpoints, figures.
cd "$(dirname $0)"
PY=${PY:-python}
while pgrep -f run_queue.sh >/dev/null; do sleep 60; done
while pgrep -f e5_batch_now.sh >/dev/null; do sleep 30; done
last() { ls ../results/*-SiT-S-2-$1-Linear-velocity-None/checkpoints/*.pt | sort -t/ -k5 | tail -1; }
echo "== E6"; $PY e6_learning_curves.py
echo "== E4b"; $PY e4b_latency_interleaved.py
echo "== E5 final"
$PY e5_grad_noise.py linear "$(last linear)" "linear @60k (atenção LiT)"
$PY e5_grad_noise.py linear_uvit "$(last linear_uvit)" "linear_uvit @60k"
$PY e5_grad_noise.py uvit "$(ls ../results/*-uvit-*/checkpoints/0060000.pt)" "uvit @60k (skips)"
$PY e5_grad_noise.py baseline "$(last baseline)" "baseline fine-tune @10k (33M)"
echo "== figures"; $PY make_figures.py
echo FINALIZE_DONE
