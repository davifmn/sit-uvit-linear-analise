"""E5: gradient noise scale B_noise = tr(Sigma) / ||G||^2 of the trainable parameters.

G = E[g] (true gradient), Sigma = per-sample gradient covariance. With batch B the expected update
direction has signal ||G||^2 vs noise tr(Sigma)/B; B_noise is the batch where they are equal
(McCandlish et al. 2018). Per-sample grads are computed exactly (loop, CPU) on real latents.
Usage: python e5_grad_noise.py VARIANT CKPT_OR_- LABEL
"""
import json, sys
import torch
from common import OUT, build, eval_set

variant, ckpt, label = sys.argv[1], sys.argv[2], sys.argv[3]
torch.set_num_threads(4)
model = build(variant, None if ckpt == "-" else ckpt).cpu().train()
model.y_embedder.dropout_prob = 0.0                    # deterministic labels for the estimate
if variant != "baseline":
    model.freeze_backbone()
params = [p for p in model.parameters() if p.requires_grad]
x1, y, t, x0 = eval_set(2000, seed=99)
n = 256
idx = torch.randperm(2000, generator=torch.Generator().manual_seed(0))[:n]
K = 8                                                   # blocks for jackknife CI, O(K*P) memory
P = sum(p.numel() for p in params)
S = torch.zeros(K, P, dtype=torch.float64)             # per-block sum of g_i
Q = torch.zeros(K, dtype=torch.float64)                # per-block sum of ||g_i||^2
for j, i in enumerate(idx):
    tt = t[i].view(1, 1, 1, 1)
    out = model(tt * x1[i:i + 1] + (1 - tt) * x0[i:i + 1], t[i:i + 1], y[i:i + 1])
    loss = ((out - (x1[i:i + 1] - x0[i:i + 1])) ** 2).mean()
    g = torch.cat([x.flatten() for x in torch.autograd.grad(loss, params)]).double()
    S[j % K] += g; Q[j % K] += g.pow(2).sum()


def b_noise(Ssum, Qsum, m):
    gbar = Ssum / m
    tr = ((Qsum - m * gbar.pow(2).sum()) / (m - 1)).item()
    g2 = gbar.pow(2).sum().item() - tr / m             # unbiased ||G||^2
    return tr, g2, (tr / g2 if g2 > 0 else float("inf"))


trace_sigma, g2, B = b_noise(S.sum(0), Q.sum(), n)
m_blk = n // K
jk = torch.tensor([b_noise(S.sum(0) - S[k], Q.sum() - Q[k], n - m_blk)[2] for k in range(K)])
se = ((K - 1) / K * (jk - jk.mean()).pow(2).sum()).sqrt().item()
res = dict(variant=variant, label=label, n=n, params=P, trace_sigma=trace_sigma, G_norm2=g2, B_noise=B,
           B_noise_ci=[max(B - 1.645 * se, 1e-3), B + 1.645 * se], snr_at_batch8=8 * g2 / trace_sigma)
print(json.dumps(res), flush=True)
p = OUT / "e5_grad_noise.json"
allres = json.load(open(p)) if p.exists() else []
allres = [r for r in allres if r["label"] != label] + [res]
json.dump(allres, open(p, "w"), indent=1)
