"""Shared plumbing for the training scripts: loading, geometry, run management."""

import json
import os
import numpy as np
import torch

from . import config as C
from . import build as B
from . import network, training


def load():
    data = np.load(C.path("dataset.npz"), allow_pickle=True)
    graphs = np.load(C.path("graphs.npz"), allow_pickle=True)
    norm = json.load(open(C.path("normalisation.json")))

    split, valid, is_control = data["split"].astype(str), data["valid"], data["is_control"]
    X = torch.tensor(data["X_norm"], dtype=torch.float32)
    y = torch.tensor(data["y_norm"], dtype=torch.float32)
    y_mu = torch.tensor(norm["y_log_mu"], dtype=torch.float32)
    y_sd = torch.tensor(norm["y_log_sd"], dtype=torch.float32)

    # Matched controls are references for the perturbation-response measure, not
    # evaluation samples: scoring on them would dilute every number with easy
    # control predictions.
    idx = {s: np.nonzero((split == s) & valid & ~is_control)[0] for s in C.ALL_SPLITS}
    control = {}
    for s in C.ALL_SPLITS:
        ci = data["control_index"][idx[s]]
        keep = ci >= 0
        control[s] = (keep, ci[keep])

    geom = B.Geometry({k.split("_", 1)[1]: graphs[k] for k in graphs.files
                       if k.startswith("signs_")},
                      len(norm["genes"]), data["t_eval"], norm)
    return data, X, y, y_mu, y_sd, idx, control, geom, norm


def denormalise(z, mu, sd, genes=None):
    return z * (sd if genes is None else sd[genes]) + (mu if genes is None else mu[genes])


def trajectory_targets(data, split, valid):
    from . import dataset as D
    mu, sd = D.trajectory_normalisation(data["y_traj"], split, valid)
    mu_t = torch.tensor(mu, dtype=torch.float32)
    sd_t = torch.tensor(sd, dtype=torch.float32)
    y_log = torch.tensor(np.log1p(np.clip(data["y_traj"], 0, None)), dtype=torch.float32)
    return (y_log - mu_t[None, :, None]) / sd_t[None, :, None], mu_t, sd_t


def do_run(arm, seed, geom, X, y, idx, control, task="A", fraction=1.0, verbose=False):
    """Train one arm unless a current run already exists. Fraction runs are kept
    in their own directory so nothing that scans the training directory mistakes
    them for full runs."""
    import time
    name = training.run_name(arm, seed, task, fraction)
    directory = C.path("runs" if fraction >= 1.0 else "runs_fraction")
    path = os.path.join(directory, name + ".pt")
    if training.is_current(path, arm, task):
        return False

    start = time.time()
    train_idx = training.subsample(idx["train"], fraction, seed)
    model = B.build(arm, seed, geom, task)
    model, info = training.train(model, geom.for_arm(arm), X, y, train_idx, idx["val"],
                                 arm, task, checkpoint=os.path.join(directory, name + ".part"),
                                 verbose=verbose)

    trajectory = task == "B"
    preds = {s: training.predict(model, X[idx[s]], geom.for_arm(arm),
                                 trajectory=trajectory).numpy() for s in C.ALL_SPLITS}
    control_preds = {}
    for s in C.ALL_SPLITS:
        keep, ci = control[s]
        control_preds[s] = (training.predict(model, X[ci], geom.for_arm(arm),
                                             trajectory=trajectory).numpy()
                            if keep.any() else np.zeros((0,) + preds[s].shape[1:], np.float32))

    training.atomic_save(
        {"state": model.state_dict(), "info": info, "preds": preds,
         "control_preds": control_preds, "arm": arm, "seed": seed, "task": task,
         "fraction": fraction, "n_params": B.ops.n_params(model),
         "nfe": getattr(model, "nfe", 1), "fingerprint": training.fingerprint(arm, task),
         "train_minutes": (time.time() - start) / 60}, path)
    part = os.path.join(directory, name + ".part")
    if os.path.exists(part):
        os.remove(part)
    print(f"  {name}: val {info['best_val']:.4f}, {info['epochs']} epochs, "
          f"{(time.time() - start) / 60:.1f} min")
    return True


def load_runs(directory=None, task=None):
    directory = directory or C.path("runs")
    out = []
    if not os.path.isdir(directory):
        return out
    for f in sorted(os.listdir(directory)):
        if not f.endswith(".pt"):
            continue
        try:
            r = torch.load(os.path.join(directory, f), weights_only=False)
        except Exception:
            print(f"  unreadable, skipping: {f}")
            continue
        if task is None or r.get("task", "A") == task:
            out.append(r)
    return out


def rebuild(run, geom):
    model = B.build(run["arm"], run["seed"], geom, run.get("task", "A"))
    model.load_state_dict(run["state"])
    model.eval()
    return model
