"""E6: paired held-out loss for every checkpoint (0 -> 60k), per-t breakdown, and asymptote fits.

dL(s) = L_variant(s) - L_pretrained on identical (x1, y, t, x0). For each run we fit
dL(s) = c + a * s^(-alpha) and bootstrap over evaluation samples: c is the gap that remains as
s -> infinity under this protocol, a CI excluding 0 means the variant does not catch up.
"""
import json
from glob import glob
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import curve_fit

from common import OUT, ROOT, build, eval_set, paired_bootstrap, per_sample_loss

x1, y, t, x0 = eval_set()
cache = OUT / "e6_losses.json"
store = json.load(open(cache)) if cache.exists() else {}
if "pretrained" not in store:
    store["pretrained"] = per_sample_loss(build("baseline"), x1, y, t, x0).tolist()
    store["t"] = t.tolist()


def runs_for(variant):
    return sorted(Path(p) for p in glob(str(ROOT / "results" / f"*-SiT-S-2-{variant}-Linear-velocity-None")))


for variant in ("uvit", "linear", "linear_uvit", "baseline"):
    if variant != "baseline" and f"{variant}@0" not in store:
        store[f"{variant}@0"] = per_sample_loss(build(variant), x1, y, t, x0).tolist()
    for run in runs_for(variant):
        for ck in sorted(run.glob("checkpoints/*.pt")):
            key = f"{variant}@{int(ck.stem)}"
            if key in store:
                continue
            store[key] = per_sample_loss(build(variant, ck), x1, y, t, x0).tolist()
            print("evaluated", key, flush=True)
            json.dump(store, open(cache, "w"))
json.dump(store, open(cache, "w"))

ref = np.array(store["pretrained"]); tt = np.array(store["t"])
bins = np.linspace(0, 1, 6)
summary = {}
for variant in ("uvit", "linear", "linear_uvit", "baseline"):
    steps = sorted(int(k.split("@")[1]) for k in store if k.startswith(variant + "@"))
    rows = []
    for s in steps:
        d = np.array(store[f"{variant}@{s}"]) - ref
        m, lo, hi, p = paired_bootstrap(d, n_boot=4000)
        by_t = [d[(tt >= a) & (tt < b)].mean() for a, b in zip(bins[:-1], bins[1:])]
        rows.append(dict(step=s, dL=m, ci=[lo, hi], p=p, rel=m / ref.mean(), dL_by_t=by_t))
    fit = None
    pts = [(r["step"], r["dL"]) for r in rows if r["step"] >= 5000]
    if variant in ("linear", "linear_uvit") and len(pts) >= 4:
        f = lambda s, c, a, al: c + a * np.power(s / 1e4, -al)
        S = np.array([p[0] for p in pts], float)
        D = np.stack([np.array(store[f"{variant}@{s}"]) - ref for s in S.astype(int)])  # (k, n)
        popt, _ = curve_fit(f, S, D.mean(1), p0=[D.mean(1)[-1], 0.05, 0.5], maxfev=20000)
        rng = np.random.default_rng(0); boots = []
        for _ in range(1000):
            j = rng.integers(0, D.shape[1], D.shape[1])
            try:
                boots.append(curve_fit(f, S, D[:, j].mean(1), p0=popt, maxfev=20000)[0])
            except RuntimeError:
                pass
        boots = np.array(boots)
        fit = dict(c=popt[0], a=popt[1], alpha=popt[2],
                   c_ci=np.percentile(boots[:, 0], [2.5, 97.5]).tolist(),
                   alpha_ci=np.percentile(boots[:, 2], [2.5, 97.5]).tolist(),
                   pred_100k=float(f(1e5, *popt)), pred_1M=float(f(1e6, *popt)))
    summary[variant] = dict(rows=rows, fit=fit)
    for r in rows:
        print(f"{variant:>12} @{r['step']:>6}: dL={r['dL']:+.5f} [{r['ci'][0]:+.5f},{r['ci'][1]:+.5f}] "
              f"({100 * r['rel']:+.2f}%)")
    if fit:
        print("   fit:", {k: np.round(v, 5).tolist() for k, v in fit.items()})
json.dump(summary, open(OUT / "e6_summary.json", "w"), indent=1, default=float)
