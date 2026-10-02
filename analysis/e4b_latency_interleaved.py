"""E4b: latency with variants interleaved round-robin in one process (removes drift between runs)."""
import json, time
import torch
from common import DEVICE, OUT, build, eval_set

variants = ["baseline", "uvit", "linear", "linear_uvit", "full_uvit", "full_linear", "full_linear_uvit"]
x1, y, t, x0 = eval_set(64)
xb, tb, yb = x1[:32].to(DEVICE), t[:32].to(DEVICE), y[:32].to(DEVICE)
models = {v: build(v) for v in variants}
times = {v: [] for v in variants}
with torch.no_grad():
    for v in variants:
        for _ in range(5): models[v](xb, tb, yb)
    torch.mps.synchronize()
    for r in range(40):
        for v in (variants if r % 2 == 0 else variants[::-1]):
            s = time.perf_counter(); models[v](xb, tb, yb); torch.mps.synchronize()
            times[v].append((time.perf_counter() - s) * 1e3)
p = OUT / "e4_speed.json"
res = json.load(open(p))
for v in variants:
    res[v]["interleaved_ms"] = times[v]
res["baseline"]["interleaved_desc"] = "F6c · Latência medida (MPS, batch 32, 40 rodadas intercaladas)"
json.dump(res, open(p, "w"), indent=1)
for v in variants:
    print(v, round(float(torch.tensor(times[v]).median()), 1))
