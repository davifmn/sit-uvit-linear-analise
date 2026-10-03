"""All figures for the variant diagnosis. Each figure is drawn only if its data exists."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import OUT

FIG = OUT / "figs"; FIG.mkdir(exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"baseline": "#2a78d6", "uvit": "#eb6834", "linear": "#1baf7a", "linear_uvit": "#eda100",
       "full_uvit": "#e87ba4", "full_linear": "#008300", "full_linear_uvit": "#4a3aa7"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .8, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "lines.linewidth": 2, "legend.frameon": False})


def load(name):
    p = OUT / name
    return json.load(open(p)) if p.exists() else None


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, dpi=160); plt.close(fig); print("wrote", name)


# F1: functional damage at step 0 ------------------------------------------------------------
e1 = load("e1_rows.json")
if e1:
    import torch
    from common import eval_set
    x1, y, t, x0 = eval_set()
    l_pre = [r for r in e1 if r["variant"].startswith("baseline")][0]["loss"]
    l_zero = ((x1 - x0) ** 2).mean().item()            # loss of predicting v = 0
    rows = [r for r in e1 if r["step"] == 0 and not r["variant"].startswith("baseline")]
    rows.sort(key=lambda r: r["delta"])
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, r in enumerate(rows):
        ax.barh(i, r["delta"], color=COL[r["variant"]], height=.6)
        ax.errorbar(r["delta"], i, xerr=[[r["delta"] - r["ci_low"]], [r["ci_high"] - r["delta"]]],
                    color=INK, capsize=3, lw=1)
        ax.text(r["delta"] + .06, i, f"+{r['delta']:.2f}  ({100 * r['delta'] / l_pre:.0f}%)" if r["delta"] > 1e-9
                else "0 (idêntico ao pré-treinado)", va="center", color=INK2, fontsize=9)
    ax.axvline(l_zero - l_pre, color=INK2, ls="--", lw=1.2)
    ax.text(l_zero - l_pre, 1.5, f" prever v ≡ 0\n (+{l_zero - l_pre:.2f})", color=INK2, fontsize=8.5)
    ax.set_yticks(range(len(rows)), [r["variant"] for r in rows])
    ax.set_xlabel("ΔL = L(variante) − L(pré-treinado), passo 0  [loss de velocidade, pareada]")
    ax.set_title("F1 · Dano funcional na inicialização (antes de qualquer treino)")
    ax.set_xlim(0, max(r["delta"] for r in rows) * 1.25)
    save(fig, "F1_init_damage.png")

# F2: uvit — skip weights random-walk and paired loss -----------------------------------------
e2 = load("e2_skip_weights.json")
if e2 and e1:
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axs[0]
    for tgt, src, c in ((9, 2, "#eb6834"), (10, 1, "#2a78d6")):
        rs = [r for r in e2 if r.get("target") == tgt]
        s = np.array([r["step"] for r in rs]); w = np.array([r["ws_fro"] for r in rs])
        k = (w[1:] @ np.sqrt(s[1:])) / (np.sqrt(s[1:]) @ np.sqrt(s[1:]))     # LS fit w = k*sqrt(s)
        r2 = 1 - ((w[1:] - k * np.sqrt(s[1:])) ** 2).sum() / ((w[1:] - w[1:].mean()) ** 2).sum()
        ss = np.linspace(0, 6e4, 200)
        ax.plot(ss / 1e3, k * np.sqrt(ss), color=c, lw=1, ls="--")
        ax.plot(s / 1e3, w, "o-", color=c, ms=6, label=f"skip {src}→{tgt}:  k√s, R²={r2:.3f}")
    ax.set_xlabel("passo (×1000)"); ax.set_ylabel("‖W_shallow‖_F")
    ax.set_title("F2a · Pesos dos skips crescem como √passos\n(assinatura de passeio aleatório)")
    ax.legend(fontsize=8.5, loc="upper left")
    ax = axs[1]
    adam = [r for r in e2 if r.get("target") == "adam"]
    ax.bar([r["step"] / 1e3 for r in adam], [r["snr_median"] for r in adam], width=5, color="#eb6834")
    pure = 0.6745 * np.sqrt(0.1 / 1.9)                  # median |m|/sqrt(v) if g is zero-mean iid noise
    ax.axhline(pure, color=INK, ls="--", lw=1.2)
    ax.text(5, pure + .02, f"ruído puro, teórico = {pure:.3f}", color=INK, fontsize=8.5)
    ax.axhline(1, color=INK2, ls=":", lw=1); ax.text(5, 1.02, "gradiente determinístico = 1", color=INK2, fontsize=8.5)
    ax.set_ylim(0, 1.15); ax.set_xlabel("passo (×1000)"); ax.set_ylabel("mediana |m| / √v  (Adam)")
    ax.set_title("F2b · SNR do gradiente (estado do Adam)")
    ax = axs[2]
    rs = sorted([r for r in e1 if r["variant"] == "uvit"], key=lambda r: r["step"])
    s = [r["step"] / 1e3 for r in rs]; d = np.array([r["delta"] for r in rs]) * 1e3
    lo = np.array([r["ci_low"] for r in rs]) * 1e3; hi = np.array([r["ci_high"] for r in rs]) * 1e3
    ax.fill_between(s, lo, hi, color="#eb6834", alpha=.18, lw=0)
    ax.plot(s, d, "o-", color="#eb6834", ms=6)
    ax.axhline(0, color=INK, lw=1); ax.text(0.5, 0.05, "pré-treinado", color=INK2, fontsize=8.5)
    ax.set_xlabel("passo (×1000)"); ax.set_ylabel("ΔL × 10³ (IC 95%)")
    ax.set_title("F2c · uvit vs pré-treinado: nenhum ganho\n(ΔL>0 ⇒ ligeiramente pior)")
    save(fig, "F2_uvit_random_walk.png")

# F3/F4: attention capacity ---------------------------------------------------------------------
e3 = load("e3_attention_stats.json")
if e3:
    b = [r["block"] for r in e3]
    hl = [COL["linear"] if i >= 8 else "#b7b6b0" for i in b]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    axs[0].bar(b, [r["eff_rank"] for r in e3], color=hl)
    axs[0].axhline(64, color=INK, ls="--", lw=1.2); axs[0].text(-.4, 68, "teto da LiT: posto ≤ d_head = 64", fontsize=8.5)
    axs[0].set_title("F3a · Posto efetivo da atenção softmax"); axs[0].set_ylabel("exp(entropia dos σ)")
    axs[1].bar(b, [100 * r["rank64_output_residual"] for r in e3], color=hl)
    axs[1].set_title("F3b · Saída inalcançável com posto 64\n‖(A−A₆₄)V‖² / ‖AV‖²  (Eckart-Young)")
    axs[1].set_ylabel("% da energia da saída")
    axs[2].bar(b, [100 * r["local5x5_mass"] for r in e3], color=hl)
    axs[2].set_title("F3c · Massa de atenção na vizinhança 5×5\n(o que a DWConv da LiT cobre)"); axs[2].set_ylabel("%")
    for ax in axs:
        ax.set_xticks(b); ax.set_xlabel("bloco (verde = trocado por linear em linear/linear_uvit)")
    save(fig, "F3_attention_capacity.png")

    M = np.array([r["entropy_by_t"] for r in e3])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    im = ax.imshow(M, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#0d366b", "#3987e5", "#cde2fb"]),
                   aspect="auto", vmin=0.2, vmax=1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] < .6 else INK)
    ax.set_xticks(range(4), ["t∈[0,.25)\nruído", "[.25,.5)", "[.5,.75)", "[.75,1)\ndados"])
    ax.set_yticks(range(12), [f"bloco {i}" + (" (linear)" if i >= 8 else "") for i in range(12)])
    ax.set_title("F4 · Entropia normalizada da atenção por bloco e t\n(menor = mais afiada, mais difícil p/ atenção linear)")
    fig.colorbar(im, ax=ax, fraction=.04)
    ax.grid(False)
    save(fig, "F4_entropy_by_t.png")

dis = load("e3_distill.json")
if dis:
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
    for ax, blk in zip(axs, (8, 9, 10, 11)):
        for r in [r for r in dis if r["block"] == blk]:
            it, err = zip(*r["curve"])
            c = "#1baf7a" if r["init"] == "random" else "#4a3aa7"
            ax.plot(it, err, color=c, label=f"{'aleatória (como no treino)' if r['init'] == 'random' else 'warm-start (pesos softmax)'}: {err[-1]:.3f}")
        ax.set_yscale("log"); ax.set_title(f"bloco {blk}"); ax.set_xlabel("iterações de distilação")
        ax.legend(fontsize=7.5, loc="upper right")
    axs[0].set_ylabel("erro relativo ‖Ŷ−Y‖²/‖Y‖² (held-out)")
    fig.suptitle("F5 · Teto de distilação: melhor LiT ajustada para imitar a atenção softmax de cada bloco",
                 fontweight="bold", fontsize=11)
    save(fig, "F5_distillation_ceiling.png")

# F6: compute -----------------------------------------------------------------------------------
e4 = load("e4_speed.json")
if e4:
    order = [v for v in COL if v in e4]
    sdpa = 4 * 256 * 256 * 384 / 1e9                     # FLOP counter misses fused SDPA
    nlin = {"baseline": 0, "uvit": 0, "full_uvit": 0, "linear": 4, "linear_uvit": 4, "full_linear": 12, "full_linear_uvit": 12}
    gfl = [e4[v]["gflops_per_image"] + (12 - nlin[v]) * sdpa for v in order]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    axs[0].bar(range(len(order)), gfl, color=[COL[v] for v in order])
    for i, g in enumerate(gfl):
        axs[0].text(i, g + .05, f"{100 * (g / gfl[0] - 1):+.1f}%", ha="center", fontsize=8, color=INK2)
    axs[0].set_ylim(0, max(gfl) * 1.12); axs[0].set_ylabel("GFLOPs / imagem / NFE")
    axs[0].set_title("F6a · Custo teórico\n(SDPA incluído)")
    attn_soft = 0.302 + sdpa; attn_lin = 0.332
    axs[1].bar([0, 1], [attn_soft * 1e3, attn_lin * 1e3], color=[COL["baseline"], COL["linear"]])
    axs[1].set_xticks([0, 1], ["softmax\n(4N²C + projeções)", "LiT linear\n(4NC·d + DWConv)"])
    axs[1].set_ylabel("MFLOPs / bloco de atenção"); axs[1].set_title("F6b · Atenção por bloco\n(N=256 tokens: −17%)")
    key = "interleaved_ms" if "interleaved_ms" in e4.get("baseline", {}) else None
    if key:
        lat = [np.median(e4[v][key]) for v in order]
        q = [np.percentile(e4[v][key], [25, 75]) for v in order]
        axs[2].bar(range(len(order)), lat, color=[COL[v] for v in order])
        axs[2].errorbar(range(len(order)), lat, yerr=np.array([[l - a, b - l] for l, (a, b) in zip(lat, q)]).T,
                        fmt="none", color=INK, capsize=3, lw=1)
        for i, l in enumerate(lat):
            axs[2].text(i, q[i][1] + 10, f"{100 * (l / lat[0] - 1):+.1f}%", ha="center", fontsize=8, color=INK2)
        axs[2].set_ylim(0, max(x[1] for x in q) * 1.15)
        axs[2].set_ylabel("ms por forward (mediana, IQR)")
        axs[2].set_title("F6c · Latência medida\n(MPS, batch 32, 40 rodadas intercaladas)")
    for ax in (axs[0], axs[2]) if key else (axs[0],):
        ax.set_xticks(range(len(order)), order, rotation=30, ha="right")
    save(fig, "F6_compute.png")

# F7/F8: learning curves 0 -> 60k ------------------------------------------------------------------
e6 = load("e6_summary.json")
if e6:
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[1.35, 1]))
    fits = []
    for v, d in e6.items():
        rows = d["rows"]
        if not rows:
            continue
        s = np.array([r["step"] for r in rows]); m = np.array([r["dL"] for r in rows])
        lo = np.array([r["ci"][0] for r in rows]); hi = np.array([r["ci"][1] for r in rows])
        for ax in axs:
            if ax is axs[1] and v not in ("uvit", "baseline"):
                continue
            scale = 1e3 if ax is axs[1] else 1
            ax.fill_between(s / 1e3, lo * scale, hi * scale, color=COL[v], alpha=.18, lw=0)
            ax.plot(s / 1e3, m * scale, "o-", color=COL[v], ms=4, label=v)
        if d.get("fit"):
            f = d["fit"]; ss = np.linspace(5e3, 1e5, 200)
            axs[0].plot(ss / 1e3, f["c"] + f["a"] * (ss / 1e4) ** (-f["alpha"]), color=COL[v], ls=":", lw=1.5)
            fits.append(f"{v}: c = {f['c']:+.4f} [{f['c_ci'][0]:+.4f}, {f['c_ci'][1]:+.4f}],  α = {f['alpha']:.2f}")
    axs[0].text(0.36, 0.55, "assíntota (IC 95%):\n" + "\n".join(fits), transform=axs[0].transAxes,
                fontsize=8.5, color=INK, va="top", bbox=dict(fc=SURF, ec=GRID, boxstyle="round,pad=.4"))
    axs[0].set_yscale("symlog", linthresh=0.01); axs[0].axhline(0, color=INK, lw=1)
    axs[0].axvline(60, color=INK2, lw=.8, ls="--")
    axs[0].set_title("F7a · ΔL vs passo (pontilhado: c + a·s^−α até 100k)")
    axs[0].set_xlabel("passo (×1000)"); axs[0].set_ylabel("ΔL vs pré-treinado (symlog)"); axs[0].legend()
    axs[1].axhline(0, color=INK, lw=1)
    axs[1].set_title("F7b · Zoom (×10³): uvit vs baseline fine-tunado")
    axs[1].set_xlabel("passo (×1000)"); axs[1].set_ylabel("ΔL × 10³ (IC 95%)"); axs[1].legend()
    save(fig, "F7_learning_curves_60k.png")

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    centers = ["[0,.2)\nruído", "[.2,.4)", "[.4,.6)", "[.6,.8)", "[.8,1]\ndados"]
    vs = [v for v in ("linear", "linear_uvit", "uvit", "baseline") if e6.get(v, {}).get("rows")]
    w = .8 / len(vs)
    for k, v in enumerate(vs):
        last = e6[v]["rows"][-1]
        ax.bar(np.arange(5) + (k - (len(vs) - 1) / 2) * w, last["dL_by_t"], width=w * .92, color=COL[v],
               label=f"{v} @{last['step'] // 1000}k")
    ax.axhline(0, color=INK, lw=1); ax.set_xticks(range(5), centers)
    ax.set_ylabel("ΔL vs pré-treinado"); ax.legend()
    ax.set_title("F8 · Onde está o déficit: ΔL por faixa de t (último checkpoint)")
    save(fig, "F8_gap_by_t.png")

# F9: gradient noise scale --------------------------------------------------------------------
e5 = load("e5_grad_noise.json")
if e5 and all("snr_at_batch8_ci" in r for r in e5):
    order = {"linear": 0, "linear_uvit": 1, "uvit": 2, "baseline": 3}
    e5 = sorted(e5, key=lambda r: (order[r["variant"]], r["label"]))[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(e5) + 1.6))
    for i, r in enumerate(e5):
        snr, (lo, hi) = r["snr_at_batch8"], r["snr_at_batch8_ci"]
        ax.barh(i, max(snr, 0), color=COL[r["variant"]], height=.6)
        ax.errorbar(max(snr, 0), i, xerr=[[max(snr, 0) - max(lo, 0)], [hi - max(snr, 0)]], color=INK, capsize=3, lw=1)
        b = r["B_noise"]; bci = r["B_noise_ci"]
        fmt = lambda v: "∞" if v == float("inf") else f"{v:.0f}" if v >= 10 else f"{v:.1f}"
        ax.text(max(hi, snr) * 1.25 + .002, i, f"B_noise ≈ {fmt(b)}  [{fmt(bci[0])}, {fmt(bci[1])}]",
                va="center", fontsize=8.5, color=INK2)
    ax.axvline(1, color=INK, ls="--", lw=1.2)
    ax.text(1.03, -0.9, "SNR = 1  ⇔  B_noise = 8 (batch usado)", fontsize=8.5)
    ax.set_yticks(range(len(e5)), [r["label"] for r in e5])
    ax.set_ylim(-1.2, len(e5) - .4)
    ax.set_xscale("symlog", linthresh=0.01, linscale=0.5)
    ax.set_xlim(0, max(r["snr_at_batch8_ci"][1] for r in e5) * 60)
    ax.set_xlabel("SNR do update com batch 8  =  8·‖G‖² / tr(Σ)   (IC 90%, jackknife; escala symlog)")
    ax.set_title("F9 · Sinal vs ruído do gradiente (< 1 ⇒ passo dominado por ruído)")
    save(fig, "F9_gradient_noise_scale.png")
