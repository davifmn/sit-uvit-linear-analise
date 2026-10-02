"""E1: how far each variant is from the pretrained function at step 0, and where uvit is at 10k-40k.

Paired evaluation: identical (x1, y, t, x0) for every model, so loss differences are
not contaminated by noise/timestep sampling variance.
"""
import json

import numpy as np
import torch

from common import OUT, ROOT, build, eval_set, paired_bootstrap, per_sample_loss

x1, y, t, x0 = eval_set()
ref_loss, ref_pred = per_sample_loss(build("baseline"), x1, y, t, x0, return_pred=True)
uvit_dir = ROOT / "results" / "000-SiT-S-2-uvit-Linear-velocity-None" / "checkpoints"

entries = [(v, 0, None) for v in ("uvit", "linear", "linear_uvit", "full_uvit",
                                  "full_linear", "full_linear_uvit")]
entries += [("uvit", s, uvit_dir / f"{s:07d}.pt") for s in (10000, 20000, 30000, 40000)]

rows = [dict(variant="baseline(pretrained)", step=0, loss=ref_loss.mean().item(),
             delta=0.0, ci_low=0.0, ci_high=0.0, p=1.0, rel_output_dev=0.0)]
per_t = {"t": t.tolist(), "baseline(pretrained)@0": ref_loss.tolist()}
for variant, step, ckpt in entries:
    loss, pred = per_sample_loss(build(variant, ckpt), x1, y, t, x0, return_pred=True)
    d, lo, hi, p = paired_bootstrap((loss - ref_loss).numpy())
    dev = ((pred - ref_pred).flatten(1).norm(dim=1) / ref_pred.flatten(1).norm(dim=1)).mean().item()
    rows.append(dict(variant=variant, step=step, loss=loss.mean().item(), delta=d,
                     ci_low=lo, ci_high=hi, p=p, rel_output_dev=dev))
    per_t[f"{variant}@{step}"] = loss.tolist()
    print(f"{variant:>17} @{step:>5}: loss={loss.mean():.4f}  dL={d:+.5f} "
          f"[{lo:+.5f},{hi:+.5f}] p={p:.2g}  ||v-v_pre||/||v_pre||={dev:.4f}", flush=True)

json.dump(rows, open(OUT / "e1_rows.json", "w"), indent=1)
json.dump(per_t, open(OUT / "e1_per_sample.json", "w"))
