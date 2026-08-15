"""Training loop and run cache.

Every run is written to disk under a name that encodes the arm, the seed, the
task and the training fraction, and is reused if the settings it was trained
under still match. The fingerprint matters: changing an architecture otherwise
leaves stale files that later cells report as cached while being unloadable.
"""

import hashlib
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from . import metrics
from .operators import n_params


def fingerprint(arm, task):
    d = {k: C.MODEL[k] for k in C.MODEL}
    d.update({k: C.TRAIN[k] for k in ("lr", "weight_decay", "batch", "grad_clip")})
    d["task"] = task
    d["epochs"], d["patience"] = budget(arm, task)
    if arm.startswith("node_"):
        d.update(C.ODE)
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]


def budget(arm, task):
    if arm.startswith("node_"):
        return C.ODE["epochs"], C.ODE["patience"]
    epochs = C.TRAIN["traj_epochs"] if task == "B" else C.TRAIN["epochs"]
    return epochs, C.TRAIN["patience"]


def atomic_save(obj, path):
    """A torch.save interrupted mid-write leaves a truncated file that later
    looks cached but will not load."""
    tmp = path + ".tmp"
    torch.save(obj, C.ensure(tmp))
    try:
        os.replace(tmp, path)
    except OSError:
        torch.save(obj, path)
        if os.path.exists(tmp):
            os.remove(tmp)


@torch.no_grad()
def predict(model, xs, geom, batch=512, trajectory=False):
    model.eval()
    outs = []
    for i in range(0, len(xs), batch):
        chunk = xs[i:i + batch]
        outs.append(model(chunk, geom, return_traj=True)
                    if trajectory and hasattr(model, "steps") else model(chunk, geom))
    return torch.cat(outs)


def train(model, geom, X, y, train_idx, val_idx, arm, task="A", checkpoint=None, verbose=False):
    """Early stopping tracks validation relative error, the measure reported,
    rather than raw squared error."""
    max_epochs, patience = budget(arm, task)
    opt = torch.optim.AdamW(model.parameters(), lr=C.TRAIN["lr"],
                            weight_decay=C.TRAIN["weight_decay"])
    xt, yt = X[train_idx], y[train_idx]
    xv, yv = X[val_idx], y[val_idx]
    trajectory = task == "B"
    n = len(train_idx)
    best, best_state, bad, history, start = float("inf"), None, 0, [], 0

    if checkpoint and os.path.exists(checkpoint):
        try:
            ck = torch.load(checkpoint, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            best, bad, history, best_state = ck["best"], ck["bad"], ck["history"], ck["best_state"]
            start = ck["epoch"] + 1
        except Exception:
            pass

    gen = torch.Generator().manual_seed(hash(arm) % (2 ** 31))
    for _ in range(start):
        torch.randperm(n, generator=gen)

    for epoch in range(start, max_epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, C.TRAIN["batch"]):
            b = perm[i:i + C.TRAIN["batch"]]
            opt.zero_grad()
            out = (model(xt[b], geom, return_traj=True)
                   if trajectory and hasattr(model, "steps") else model(xt[b], geom))
            F.mse_loss(out, yt[b]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), C.TRAIN["grad_clip"])
            opt.step()

        pv = predict(model, xv, geom, trajectory=trajectory)
        score = metrics.relative_l2(pv.numpy(), yv.numpy())
        if not np.isfinite(score):
            raise RuntimeError(f"{arm} diverged at epoch {epoch}; fixed-step RK4 is stable "
                               f"only while h*|lambda| < 2.78, so lower ODE['h_max']")
        history.append(score)
        if score < best - 1e-6:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if checkpoint and (epoch + 1) % 10 == 0:
            atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "best": best,
                         "bad": bad, "history": history, "best_state": best_state,
                         "epoch": epoch}, checkpoint)
        if verbose and epoch % 10 == 0:
            print(f"    epoch {epoch:3d}  val {score:.4f}  best {best:.4f}")

    model.load_state_dict(best_state)
    return model, {"best_val": best, "epochs": len(history), "history": history}


def run_name(arm, seed, task="A", fraction=1.0):
    tag = f"{task}_{arm}_s{seed}"
    return tag if fraction >= 1.0 else f"{tag}_f{int(fraction * 100)}"


def is_current(path, arm, task):
    if not os.path.exists(path):
        return False
    try:
        return torch.load(path, weights_only=False).get("fingerprint") == fingerprint(arm, task)
    except Exception:
        return False


def subsample(idx, fraction, seed):
    if fraction >= 1.0:
        return idx
    rng = np.random.default_rng(1000 + seed)
    return np.sort(rng.choice(idx, size=int(round(fraction * len(idx))), replace=False))
