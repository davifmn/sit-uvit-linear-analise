"""E4: is linear attention actually cheaper here? Exact FLOPs (torch flop counter) + measured latency.

Usage: python e4_speed_flops.py VARIANT   (one process per variant -> no allocator interference)
"""
import json, sys, time
import torch
from torch.utils.flop_counter import FlopCounterMode
from common import DEVICE, OUT, build, eval_set

variant = sys.argv[1]
x1, y, t, x0 = eval_set(64)
res = {}
# FLOPs on CPU (exact op-level count), batch 1.
m = build(variant).cpu()
with torch.no_grad(), FlopCounterMode(display=False) as fc:
    m(x1[:1], t[:1], y[:1])
res["gflops_per_image"] = fc.get_total_flops() / 1e9
attn = {}
for i, blk in enumerate(m.blocks):
    xin = torch.randn(1, 256, 384)
    with torch.no_grad(), FlopCounterMode(display=False) as fa:
        blk.attn(xin)
    attn[i] = fa.get_total_flops() / 1e6
res["attn_mflops_per_block"] = attn

def latency(model, dev, bs, reps):
    xb, tb, yb = x1[:bs].to(dev), t[:bs].to(dev), y[:bs].to(dev)
    sync = torch.mps.synchronize if dev.type == "mps" else (lambda: None)
    with torch.no_grad():
        for _ in range(3): model(xb, tb, yb)
        sync(); times = []
        for _ in range(reps):
            s = time.perf_counter(); model(xb, tb, yb); sync(); times.append(time.perf_counter() - s)
    times = torch.tensor(times)
    return dict(median_ms=times.median().item() * 1e3, p10_ms=times.quantile(.1).item() * 1e3,
                p90_ms=times.quantile(.9).item() * 1e3)

res["cpu_bs8"] = latency(m, torch.device("cpu"), 8, 15)
m = m.to(DEVICE)
res["mps_bs32"] = latency(m, DEVICE, 32, 30)
# Attention module alone, MPS, batch 32 (isolates the swapped component).
blk = m.blocks[11].attn
xa = torch.randn(32, 256, 384, device=DEVICE)
with torch.no_grad():
    for _ in range(5): blk(xa)
    torch.mps.synchronize(); ts = []
    for _ in range(50):
        s = time.perf_counter(); blk(xa); torch.mps.synchronize(); ts.append(time.perf_counter() - s)
res["attn_block11_mps_bs32_ms"] = float(torch.tensor(ts).median() * 1e3)
print(variant, json.dumps(res))
p = OUT / "e4_speed.json"
allres = json.load(open(p)) if p.exists() else {}
allres[variant] = res
json.dump(allres, open(p, "w"), indent=1)
