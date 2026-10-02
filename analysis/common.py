"""Shared helpers for the variant diagnostics (real latents, fixed noise, paired losses)."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import SiT_models  # noqa: E402
from checkpoint_utils import load_pretrained, read_checkpoint, model_weights  # noqa: E402
from transport import create_transport  # noqa: E402

PRETRAINED = ROOT / "pretrained_models" / "SiT-S-2-256.pt"
SUBSET = ROOT / "data" / "latent_subset" / "subset.npz"
OUT = ROOT / "analysis" / "out"
OUT.mkdir(parents=True, exist_ok=True)
LATENT_SCALE = 0.18215
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def build(variant, checkpoint=None, seed=0):
    """Variant built exactly as train.py does: pretrained load, then optional trained weights."""
    torch.manual_seed(seed)
    model = SiT_models["SiT-S/2"](input_size=32, num_classes=1000, learn_sigma=True, variant=variant)
    load_pretrained(model, model_weights(read_checkpoint(str(PRETRAINED))), log=lambda *_: None)
    if checkpoint is not None:
        model.load_state_dict(read_checkpoint(str(checkpoint))["model"], strict=True)
    return model.to(DEVICE).eval()


def eval_set(n=2000, seed=1234):
    """Fixed (x1, y, t, x0): common random numbers so variant differences are paired."""
    data = np.load(SUBSET)
    g = torch.Generator().manual_seed(seed)
    params = torch.from_numpy(data["posterior"][:n, 0])        # non-flipped view
    mean, logvar = params[:, 0], params[:, 1]
    std = torch.exp(0.5 * logvar.clamp(-30, 20))
    x1 = (mean + std * torch.randn(mean.shape, generator=g)) * LATENT_SCALE
    y = torch.from_numpy(data["labels"][:n]).long()
    # Stratified t so every t-bin is equally covered.
    t = ((torch.arange(n) + torch.rand(n, generator=g)) / n)[torch.randperm(n, generator=g)]
    t = t.clamp(1e-5, 1 - 1e-5)
    x0 = torch.randn(x1.shape, generator=g)
    return x1, y, t, x0


@torch.no_grad()
def per_sample_loss(model, x1, y, t, x0, batch=50, return_pred=False):
    """Velocity-matching loss of the Linear interpolant (t=1 is data), per sample."""
    losses, preds = [], []
    for i in range(0, len(x1), batch):
        b1, by, bt, b0 = (v[i:i + batch].to(DEVICE) for v in (x1, y, t, x0))
        tt = bt.view(-1, 1, 1, 1)
        xt = tt * b1 + (1 - tt) * b0
        ut = b1 - b0
        pred = model(xt, bt, by)
        losses.append(((pred - ut) ** 2).mean(dim=(1, 2, 3)).float().cpu())
        if return_pred:
            preds.append(pred.float().cpu())
    out = torch.cat(losses)
    return (out, torch.cat(preds)) if return_pred else out


def paired_bootstrap(diff, n_boot=10000, seed=0):
    """Mean of paired differences with 95% bootstrap CI and a paired t-test p-value."""
    from scipy import stats
    diff = np.asarray(diff, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), (n_boot, len(diff)))].mean(1)
    p = stats.ttest_1samp(diff, 0.0).pvalue
    return diff.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5), p
