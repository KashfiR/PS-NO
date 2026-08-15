"""Sample design, simulation, assembly and the transfer domains.

Splits are decided when the data is generated rather than assigned afterwards,
and the damaged networks are given to the model only: the simulator always uses
the true one, so damage changes what a model is told and never what is true.
"""

import json
import os
import numpy as np

from . import config as C
from . import kinetics


def target_pools(n_genes, rng):
    order = rng.permutation(n_genes)
    n_tr, n_va, n_te = C.SPLITS["target_pools"]
    return dict(train=order[:n_tr],
                val=order[n_tr:n_tr + n_va],
                test=order[n_tr + n_va:n_tr + n_va + n_te])


def design(n_genes, pools):
    """One row per sample, drawn from an RNG stream keyed by (split, index) so
    that no draw is ever reused across splits."""
    s = C.SAMPLING
    spec = {"train": (C.SPLITS["train"], "train", False),
            "val": (C.SPLITS["val"], "val", False),
            "test_iid": (C.SPLITS["test_iid"], "train", False),
            "test_target": (C.SPLITS["test_target"], "test", False),
            "test_param": (C.SPLITS["test_param"], "train", True)}
    eval_splits = {"val", "test_iid", "test_target", "test_param"}

    rows = []
    for offset, (name, (n, pool_key, ood)) in enumerate(spec.items()):
        pool = pools[pool_key]
        k_range = s["k_prod_ood"] if ood else s["k_prod"]
        g_range = s["gamma_ood"] if ood else s["gamma"]
        n_control = int(round(s["control_frac"] * n))
        for i in range(n):
            r = np.random.default_rng([1000 + offset, i])
            row = dict(split=name, idx=i,
                       x0=r.uniform(*s["x0"], size=n_genes),
                       k=r.uniform(*k_range, size=n_genes),
                       gamma=r.uniform(*g_range, size=n_genes),
                       is_control=False)
            if i < n_control:
                row.update(target=-1, level=0.0)
            else:
                row.update(target=int(pool[r.integers(len(pool))]),
                           level=float(s["pert_levels"][r.integers(len(s["pert_levels"]))]))
            rows.append(row)

    # Matched controls exist so the perturbation-response measure has an exact
    # reference. Only evaluation splits need them.
    for row in list(rows):
        if row["target"] >= 0 and row["split"] in eval_splits:
            rows.append({**row, "target": -1, "level": 0.0, "is_control": True})
    return rows


def simulate_all(system, rows, t_eval, shard_dir, shard_size=None):
    """Sharded so an interrupted session costs at most one shard."""
    shard_size = shard_size or C.SPLITS["shard_size"]
    os.makedirs(shard_dir, exist_ok=True)
    n_genes = len(rows[0]["x0"])

    for s_i, start in enumerate(range(0, len(rows), shard_size)):
        path = os.path.join(shard_dir, f"shard_{s_i:04d}.npz")
        if os.path.exists(path):
            continue
        stop = min(start + shard_size, len(rows))
        steady, trajs, ok = [], [], []
        for row in rows[start:stop]:
            y, traj, conv = kinetics.simulate(system, row["x0"], row["k"], row["gamma"],
                                              row["target"], row["level"], t_eval)
            if y is None:
                y = np.full(n_genes, np.nan)
                traj = np.full((n_genes, len(t_eval)), np.nan)
            steady.append(y)
            trajs.append(traj)
            ok.append(conv)
        np.savez_compressed(path, start=start, stop=stop,
                            y_ss=np.array(steady, np.float32),
                            y_traj=np.array(trajs, np.float32),
                            converged=np.array(ok))
        yield s_i, stop, float(np.mean(ok))


def assemble(rows, shard_dir, t_eval):
    """Collect shards into arrays, build the input tensor, and freeze the
    normalisation on the training split alone."""
    n, n_genes = len(rows), len(rows[0]["x0"])
    y_ss = np.full((n, n_genes), np.nan, np.float32)
    y_traj = np.full((n, n_genes, len(t_eval)), np.nan, np.float32)
    converged = np.zeros(n, bool)
    for f in sorted(os.listdir(shard_dir)):
        if not f.endswith(".npz"):
            continue
        z = np.load(os.path.join(shard_dir, f))
        a, b = int(z["start"]), int(z["stop"])
        y_ss[a:b], y_traj[a:b], converged[a:b] = z["y_ss"], z["y_traj"], z["converged"]

    valid = converged & np.isfinite(y_ss).all(1)
    split = np.array([r["split"] for r in rows])
    target = np.array([r["target"] for r in rows])
    level = np.array([r["level"] for r in rows], np.float32)
    is_control = np.array([r["is_control"] for r in rows])
    idx = np.array([r["idx"] for r in rows])
    x0 = np.stack([r["x0"] for r in rows]).astype(np.float32)

    flag = np.zeros((n, n_genes), np.float32)
    dose = np.zeros((n, n_genes), np.float32)
    has = target >= 0
    flag[np.nonzero(has)[0], target[has]] = 1.0
    dose[np.nonzero(has)[0], target[has]] = level[has]
    X = np.stack([x0, np.stack([r["k"] for r in rows]), np.stack([r["gamma"] for r in rows]),
                  flag, dose], axis=-1).astype(np.float32)

    train = (split == "train") & valid
    y_log = np.log1p(np.clip(y_ss, 0, None))
    mu, sd = y_log[train].mean(0), y_log[train].std(0) + 1e-6
    x0_log = np.log1p(np.clip(x0, 0, None))
    x_mu, x_sd = x0_log[train].mean(0), x0_log[train].std(0) + 1e-6

    X_norm = X.copy()
    X_norm[..., 0] = (x0_log - x_mu) / x_sd
    for c in (1, 2):
        m, s = X[train, :, c].mean(), X[train, :, c].std() + 1e-6
        X_norm[..., c] = (X[..., c] - m) / s

    key = {(split[i], int(idx[i])): i for i in range(n) if is_control[i]}
    control_index = np.array([-1 if is_control[i] else key.get((split[i], int(idx[i])), -1)
                              for i in range(n)])

    norm = dict(y_log_mu=mu.tolist(), y_log_sd=sd.tolist(),
                x0_log_mu=x_mu.tolist(), x0_log_sd=x_sd.tolist(),
                fit_on="train split only")
    data = dict(X=X, X_norm=X_norm, y_ss=y_ss, y_norm=(y_log - mu) / sd, y_traj=y_traj,
                split=split, target=target, level=level, is_control=is_control,
                control_index=control_index, valid=valid, t_eval=t_eval)
    return data, norm


def trajectory_normalisation(y_traj, split, valid):
    """Refit over all time points, still on the training split only."""
    y_log = np.log1p(np.clip(y_traj[(split == "train") & valid], 0, None))
    return y_log.mean((0, 2)), y_log.std((0, 2)) + 1e-6


def connected_subset(signs, size, rng, n_candidates=300):
    """A connected induced subnetwork whose density matches the parent's.

    Breadth-first growth collects the busiest neighbourhood and returns a much
    denser subnetwork than the parent, which would confound a change of gene set
    with a change of density. Growing by uniformly random frontier expansion and
    keeping the density-closest of many candidates controls for that.
    """
    n = len(signs)
    nbrs = {i: set() for i in range(n)}
    for i, j in zip(*np.nonzero(signs)):
        nbrs[int(i)].add(int(j))
        nbrs[int(j)].add(int(i))
    parent_density = (signs != 0).sum() / (n * (n - 1))

    best, best_gap = None, np.inf
    for _ in range(n_candidates):
        cur = {int(rng.integers(n))}
        frontier = set(nbrs[next(iter(cur))])
        while len(cur) < size and frontier:
            pick = int(rng.choice(sorted(frontier)))
            cur.add(pick)
            frontier.discard(pick)
            frontier |= nbrs[pick] - cur
        if len(cur) != size:
            continue
        sub = np.array(sorted(cur))
        d = (signs[np.ix_(sub, sub)] != 0).sum() / (size * (size - 1))
        if abs(d - parent_density) < best_gap:
            best, best_gap = sub, abs(d - parent_density)
    if best is None:
        raise RuntimeError(f"no connected subset of size {size}")
    return best
