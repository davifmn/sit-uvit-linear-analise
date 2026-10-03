"""E3: can LiT linear attention represent what pretrained softmax attention does?

For each block of the pretrained SiT we capture the (modulated) attention input on real latents
at stratified t, and measure on the softmax maps A (per head, 256x256):
  * normalized entropy  H(A_i)/log N   (1 = uniform, 0 = one-hot)
  * rank-64 residual    ||(A - A_64) V||_F^2 / ||A V||_F^2, A_64 = best rank-64 approx (Eckart-Young).
    phi(Q)phi(K)^T has rank <= d_head = 64, so this lower-bounds the error of the global term.
  * local mass          fraction of attention inside the 5x5 neighbourhood (reachable by the dwc).
Then, for blocks 8-11, we distill a LinearAttention onto the softmax attention output and report the
held-out relative error (random init, as in training, vs warm-start from the pretrained q/k/v/proj).
"""
import json
import math
import os

import torch
import torch.nn.functional as F

from common import OUT, build, eval_set
from LiT_linearAttn import LinearAttention

torch.manual_seed(0)
torch.set_num_threads(4)
DEV = torch.device("cpu")
N_SAMPLES, D, HEADS, N = 160, 384, 6, 256
x1, y, t, x0 = eval_set(2000)
idx = torch.linspace(0, 1999, N_SAMPLES).long()
x1, y, t, x0 = x1[idx], y[idx], t[idx], x0[idx]

model = build("baseline").to(DEV)
captured = {i: [] for i in range(12)}
hooks = [blk.attn.register_forward_hook(lambda m, inp, out, i=i: captured[i].append(inp[0].detach()))
         for i, blk in enumerate(model.blocks)]
with torch.no_grad():
    for s in range(0, N_SAMPLES, 32):
        tt = t[s:s + 32].view(-1, 1, 1, 1)
        model(tt * x1[s:s + 32] + (1 - tt) * x0[s:s + 32], t[s:s + 32], y[s:s + 32])
for h in hooks:
    h.remove()
inputs = {i: torch.cat(v) for i, v in captured.items()}

coords = torch.stack(torch.meshgrid(torch.arange(16), torch.arange(16), indexing="ij"), -1).view(N, 2)
local_mask = ((coords[:, None] - coords[None]).abs().max(-1).values <= 2).float()   # 5x5 window


def softmax_parts(attn, x):
    B = x.shape[0]
    qkv = attn.qkv(x).reshape(B, N, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    A = (q @ k.transpose(-2, -1) * (D // HEADS) ** -0.5).softmax(-1)
    return A, v


stats = []
for i in ([] if os.environ.get("E3_DISTILL_ONLY") else range(12)):
    with torch.no_grad():
        A, v = softmax_parts(model.blocks[i].attn, inputs[i])
        ent = (-(A * (A + 1e-12).log()).sum(-1) / math.log(N)).mean().item()
        U, S, Vh = torch.linalg.svd(A)
        A64 = (U[..., :64] * S[..., None, :64]) @ Vh[..., :64, :]
        AV = A @ v
        resid = (((A - A64) @ v).pow(2).sum((-1, -2)) / AV.pow(2).sum((-1, -2))).mean().item()
        energy_out = (S[..., 64:].pow(2).sum(-1) / S.pow(2).sum(-1)).mean().item()
        local = (A * local_mask).sum(-1).mean().item()
        p = S / S.sum(-1, keepdim=True)
        eff_rank = torch.exp(-(p * (p + 1e-12).log()).sum(-1)).mean().item()
        # per-t-bin entropy (t small = noisy input)
        ent_t = (-(A * (A + 1e-12).log()).sum(-1) / math.log(N)).mean((1, 2))
    row = dict(block=i, entropy=ent, rank64_output_residual=resid, spectral_energy_beyond_64=energy_out,
               eff_rank=eff_rank, local5x5_mass=local,
               entropy_by_t=[ent_t[(t >= a) & (t < a + .25)].mean().item() for a in (0, .25, .5, .75)])
    stats.append(row)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
if stats:
    json.dump(stats, open(OUT / "e3_attention_stats.json", "w"), indent=1)


def warm_start(lin, attn):
    """Copy pretrained q/k/v/proj into the LiT module (the dwc stays zero so it starts as pure kernel attention)."""
    with torch.no_grad():
        W, b = attn.qkv.weight, attn.qkv.bias
        lin.q.weight.copy_(W[:D]); lin.q.bias.copy_(b[:D])
        lin.kv.weight.copy_(W[D:]); lin.kv.bias.copy_(b[D:])
        lin.proj.weight.copy_(attn.proj.weight); lin.proj.bias.copy_(attn.proj.bias)
        lin.dwc.weight.zero_(); lin.dwc.bias.zero_()


distill = []
train_idx, test_idx = torch.arange(0, N_SAMPLES, 5).ne(0), None
perm = torch.randperm(N_SAMPLES)
tr, te = perm[:128], perm[128:]
for i in (8, 9, 10, 11):
    X = inputs[i]
    with torch.no_grad():
        Y = model.blocks[i].attn(X)
    for init in ("random", "warm_start"):
        torch.manual_seed(0)
        lin = LinearAttention(D, num_heads=HEADS, qkv_bias=True)
        for mod in lin.modules():
            if isinstance(mod, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(mod.weight); torch.nn.init.zeros_(mod.bias)
        if init == "warm_start":
            warm_start(lin, model.blocks[i].attn)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-3)
        curve = []
        for it in range(1501):
            b = tr[torch.randint(0, len(tr), (16,))]
            loss = F.mse_loss(lin(X[b]), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 100 == 0:
                with torch.no_grad():
                    err = ((lin(X[te]) - Y[te]).pow(2).sum() / Y[te].pow(2).sum()).item()
                curve.append((it, err))
        distill.append(dict(block=i, init=init, curve=curve, final_rel_err=curve[-1][1]))
        json.dump(distill, open(OUT / "e3_distill.json", "w"), indent=1)   # incremental
        print(f"block {i} {init:>10}: rel.err start={curve[0][1]:.3f} final={curve[-1][1]:.4f}", flush=True)
json.dump(distill, open(OUT / "e3_distill.json", "w"), indent=1)
