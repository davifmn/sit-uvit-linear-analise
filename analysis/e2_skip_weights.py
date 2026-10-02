"""E2: what the uvit skip projections actually learned, 0 -> 40k (weights only).

Each projection is P([h_deep, h_shallow]) = W_d h_deep + W_s h_shallow + b, initialized W_d = I, W_s = 0.
We track ||W_d - I||_F, ||W_s||_F, ||W_s||_2 (spectral), ||b||, and the step-to-step drift
||W_k - W_{k-10k}||_F, plus Adam state (|m|/sqrt(v) ~ signal-to-noise of the gradient).
"""
import json

import torch

from common import OUT, ROOT

ckdir = ROOT / "results" / "000-SiT-S-2-uvit-Linear-velocity-None" / "checkpoints"
D = 384
rows, prev = [], {}
states = {0: None}
for step in (0, 10000, 20000, 30000, 40000):
    ck = None if step == 0 else torch.load(ckdir / f"{step:07d}.pt", map_location="cpu", weights_only=False)
    for target in ("9", "10"):
        if ck is None:
            W = torch.cat([torch.eye(D), torch.zeros(D, D)], 1); b = torch.zeros(D)
        else:
            W = ck["model"][f"skip_projections.{target}.weight"].float(); b = ck["model"][f"skip_projections.{target}.bias"].float()
        Wd, Ws = W[:, :D], W[:, D:]
        row = dict(step=step, target=int(target), source={"9": 2, "10": 1}[target],
                   wd_minus_I_fro=(Wd - torch.eye(D)).norm().item(),
                   ws_fro=Ws.norm().item(), ws_spec=torch.linalg.matrix_norm(Ws, 2).item(),
                   wd_spec_dev=torch.linalg.matrix_norm(Wd - torch.eye(D), 2).item(),
                   bias=b.norm().item(),
                   drift=(W - prev[target]).norm().item() if target in prev else 0.0)
        # Effective rank (entropy of normalized singular values) of the learned shallow map.
        s = torch.linalg.svdvals(Ws)
        if s.sum() > 0:
            p = s / s.sum(); row["ws_eff_rank"] = torch.exp(-(p * (p + 1e-12).log()).sum()).item()
        else:
            row["ws_eff_rank"] = 0.0
        prev[target] = W
        rows.append(row)
    if ck is not None:
        opt = ck["opt"]
        snr = []
        for st in opt["state"].values():
            m, v = st["exp_avg"].float(), st["exp_avg_sq"].float()
            snr.append((m.abs() / (v.sqrt() + 1e-12)).flatten())
        snr = torch.cat(snr)
        rows.append(dict(step=step, target="adam", snr_median=snr.median().item(),
                         snr_mean=snr.mean().item(), lr=opt["param_groups"][0]["lr"]))
for r in rows:
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
json.dump(rows, open(OUT / "e2_skip_weights.json", "w"), indent=1)
