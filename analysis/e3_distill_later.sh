#!/bin/zsh
cd "$(dirname $0)"
while pgrep -f e5_batch_now.sh >/dev/null; do sleep 30; done
E3_DISTILL_ONLY=1 $PY e3_attention_capacity.py
$PY make_figures.py
echo E3_DONE
