"""The measurement suite.

Everything is on the log(1+x) scale after undoing the scaling used in training.
On standardised targets a mean predictor scores exactly 1.000 by construction,
which makes relative error degenerate there.
"""

import time
import numpy as np
import torch

from . import config as C


def relative_l2(pred, true):
    p, t = pred.reshape(len(pred), -1), true.reshape(len(true), -1)
    return float(np.mean(np.linalg.norm(p - t, axis=1) /
                         np.maximum(np.linalg.norm(t, axis=1), 1e-9)))


def per_sample_relative_l2(pred, true):
    p, t = pred.reshape(len(pred), -1), true.reshape(len(true), -1)
    return np.linalg.norm(p - t, axis=1) / np.maximum(np.linalg.norm(t, axis=1), 1e-9)


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def per_gene_r2(pred, true):
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.array([np.corrcoef(pred[:, j], true[:, j])[0, 1]
                      if true[:, j].std() > 1e-9 and pred[:, j].std() > 1e-9 else 0.0
                      for j in range(true.shape[1])])
    r = np.nan_to_num(r) ** 2
    return float(np.median(r)), float((r > 0.5).mean())


def band_error(pred, true, phi, bands=None):
    """Share of squared error by graph-frequency band.

    Note this covers only the retained modes, so it describes how error is
    distributed within the represented subspace and says nothing about the modes
    that were discarded.
    """
    k = phi.shape[1]
    bands = bands or {"low": (0, k // 3), "mid": (k // 3, 2 * k // 3), "high": (2 * k // 3, k)}
    e = phi.T @ (pred - true).T
    total = (e ** 2).sum() + 1e-12
    return {b: float((e[a:z] ** 2).sum() / total) for b, (a, z) in bands.items()}


def _ranks(a):
    return a.argsort(1).argsort(1).astype(np.float64)


def spearman(pred, true):
    rp, rt = _ranks(pred), _ranks(true)
    rp -= rp.mean(1, keepdims=True)
    rt -= rt.mean(1, keepdims=True)
    return (rp * rt).sum(1) / (np.sqrt((rp ** 2).sum(1) * (rt ** 2).sum(1)) + 1e-12)


def precision_at_k(pred, true, k):
    """Of the k genes that actually moved most, how many are in the model's own
    top k? This is the question a perturbation screen asks."""
    top_true = np.argsort(-np.abs(true), 1)[:, :k]
    top_pred = np.argsort(-np.abs(pred), 1)[:, :k]
    return np.array([len(set(a) & set(b)) / k for a, b in zip(top_true, top_pred)])


def full_suite(pred, true, pred_control=None, true_control=None, keep=None, phi=None, prefix=""):
    med, frac = per_gene_r2(pred, true)
    out = {f"{prefix}rel_l2": relative_l2(pred, true),
           f"{prefix}rmse": rmse(pred, true),
           f"{prefix}r2_med": med,
           f"{prefix}r2_frac": frac}
    if phi is not None:
        out.update({f"{prefix}band_{k}": v for k, v in band_error(pred, true, phi).items()})
    if pred_control is not None and keep is not None and keep.any():
        dp, dt = pred[keep] - pred_control, true[keep] - true_control
        out[f"{prefix}delta_rel_l2"] = relative_l2(dp, dt)
        out[f"{prefix}spearman"] = float(spearman(dp, dt).mean())
        out[f"{prefix}prec_at_k"] = float(precision_at_k(dp, dt, C.EVAL["top_k"]).mean())
    return out


def bootstrap_ci(per_sample, n=None, rng=None):
    """Three seeds give a paired t-test two degrees of freedom, which says almost
    nothing about gaps this small. The 600 test samples are the resolution
    actually available."""
    rng = rng or np.random.default_rng(0)
    n = n or C.EVAL["bootstrap"]
    mean = np.nanmean(per_sample, axis=0)
    idx = rng.integers(0, len(mean), size=(n, len(mean)))
    draws = np.nanmean(mean[idx], axis=1)
    return float(np.nanmean(mean)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def paired_bootstrap(a, b, n=None, rng=None):
    """Resamples the same samples for both models, separating sample noise from
    seed noise."""
    rng = rng or np.random.default_rng(0)
    n = n or C.EVAL["bootstrap"]
    d = np.nanmean(a, axis=0) - np.nanmean(b, axis=0)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    draws = np.nanmean(d[idx], axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(np.nanmean(d)), float(lo), float(hi), bool(lo > 0 or hi < 0)


@torch.no_grad()
def time_forward(model, geom, xs, n_calls, blocks=5):
    """Fastest of several timing blocks, per sample.

    The minimum is the right statistic: contention can only make a measurement
    slower, so the fastest block is the closest estimate of the model's real
    cost. A batch of one is tens of microseconds of real work, so too few
    repeats leaves the measurement dominated by scheduler jitter: identical
    architectures can differ sixfold. Five blocks of 200 brings the spread across
    seeds to a median of 3%.
    """
    model.eval()
    for _ in range(3):
        model(xs, geom)
    best = float("inf")
    for _ in range(blocks):
        t0 = time.perf_counter()
        for _ in range(n_calls):
            model(xs, geom)
        best = min(best, (time.perf_counter() - t0) / (n_calls * len(xs)))
    return best
