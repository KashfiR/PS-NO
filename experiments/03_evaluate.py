"""Every measurement, for every arm, in one pass.

Nothing is retrained: each quantity is read from the saved predictions or
measured by rebuilding a saved state dict and running it forward. Writes the
tables and figures the paper reports.

    python experiments/03_evaluate.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C, build as B, figures, metrics, network, operators as ops, runner

# Not every measurement applies to every arm, and that is a result rather than a
# gap: an arm can move to a different gene set only if nothing inside it is sized
# to the gene count. The dense baseline and the plain-field ODE have a fixed
# input width; both Fourier arms carry a fixed gene ordering.
TRANSFERABLE = C.GRAPH_AWARE


def unpack(run, split, idx, control, y, y_mu, y_sd, traj=None, traj_norm=None):
    keep, ci = control[split]
    if run.get("task") == "B":
        mu, sd = traj_norm
        p = (torch.tensor(run["preds"][split]) * sd[None, :, None] + mu[None, :, None]).numpy()
        pc = (torch.tensor(run["control_preds"][split]) * sd[None, :, None] + mu[None, :, None]).numpy()
        t = (traj[idx[split]] * sd[None, :, None] + mu[None, :, None]).numpy()
        tc = (traj[ci] * sd[None, :, None] + mu[None, :, None]).numpy()
        return p[..., -1], pc[..., -1], t[..., -1], tc[..., -1], keep, p, t
    p = runner.denormalise(torch.tensor(run["preds"][split]), y_mu, y_sd).numpy()
    pc = runner.denormalise(torch.tensor(run["control_preds"][split]), y_mu, y_sd).numpy()
    t = runner.denormalise(y[idx[split]], y_mu, y_sd).numpy()
    tc = runner.denormalise(y[ci], y_mu, y_sd).numpy()
    return p, pc, t, tc, keep, None, None


def main():
    data, X, y, y_mu, y_sd, idx, control, geom, norm = runner.load()
    split, valid = data["split"].astype(str), data["valid"]
    traj, mu_t, sd_t = runner.trajectory_targets(data, split, valid)
    geom.traj_norm = (mu_t, sd_t)
    phi = geom.eigs["real"][1][:, :C.MODEL["k_modes"]]
    rng = np.random.default_rng(0)

    runs = runner.load_runs()
    print(f"{len(runs)} runs found\n")

    # ---- accuracy, ranking, dose, per-sample curves for the bootstrap --------
    rows, dose_rows, curves = [], [], {}
    levels = data["level"][idx["test_iid"]]
    for run in runs:
        arm, task = run["arm"], run.get("task", "A")
        row = {"arm": arm, "seed": run["seed"], "task": task,
               "n_params": run["n_params"], "nfe": run["nfe"],
               "epochs": run["info"]["epochs"], "train_minutes": run["train_minutes"]}
        for s in C.EVAL_SPLITS:
            p, pc, t, tc, keep, ptraj, ttraj = unpack(run, s, idx, control, y, y_mu, y_sd,
                                                      traj, (mu_t, sd_t))
            row.update(metrics.full_suite(p, t, pc, tc, keep,
                                          phi if s == "test_iid" else None, prefix=f"{s}/"))
            if task == "B" and s == "test_iid":
                row["test_iid/traj_rel_l2"] = metrics.relative_l2(ptraj, ttraj)
                per_time = (np.linalg.norm(ptraj - ttraj, axis=1) /
                            np.maximum(np.linalg.norm(ttraj, axis=1), 1e-9)).mean(0)
                row.update({f"time_{i}": float(v) for i, v in enumerate(per_time)})
            if s == "test_iid":
                curves.setdefault((task, arm), []).append(metrics.per_sample_relative_l2(p, t))
                if keep.any():
                    per = metrics.per_sample_relative_l2(p[keep] - pc, t[keep] - tc)
                    lv = levels[keep]
                    for level in np.unique(lv):
                        mask = lv == level
                        if mask.sum() >= 5:
                            dose_rows.append({"task": task, "arm": arm, "seed": run["seed"],
                                              "level": float(level),
                                              "delta_rel_l2": float(per[mask].mean())})
        rows.append(row)
    results = pd.DataFrame(rows)
    results.to_csv(C.ensure(C.path("results_raw.csv")), index=False)

    task_a = results[results.task == "A"]
    summary = task_a.groupby("arm").agg(
        rel_l2=("test_iid/rel_l2", "mean"), rel_l2_sd=("test_iid/rel_l2", "std"),
        delta=("test_iid/delta_rel_l2", "mean"),
        spearman=("test_iid/spearman", "mean"), spearman_sd=("test_iid/spearman", "std"),
        prec_at_k=("test_iid/prec_at_k", "mean"),
        unseen_gene=("test_target/rel_l2", "mean"),
        unseen_rates=("test_param/rel_l2", "mean"),
        n_params=("n_params", "mean"), nfe=("nfe", "mean")).sort_values("rel_l2")
    print("=== steady-state task ===")
    print(summary.round(4).to_string())
    summary.to_csv(C.ensure(C.path("summary_task_a.csv")))

    # ---- bootstrap ----------------------------------------------------------
    per_sample = {arm: np.stack(v) for (task, arm), v in curves.items() if task == "A"}
    boot = pd.DataFrame([{"arm": a, **dict(zip(["mean", "ci_lo", "ci_hi"],
                                               metrics.bootstrap_ci(v, rng=rng)))}
                         for a, v in per_sample.items()]).sort_values("mean")
    reference = "psno_real"
    paired = pd.DataFrame([{"vs": a,
                            **dict(zip(["diff", "ci_lo", "ci_hi", "significant"],
                                       metrics.paired_bootstrap(per_sample[reference], v, rng=rng)))}
                           for a, v in per_sample.items() if a != reference])
    boot.to_csv(C.ensure(C.path("bootstrap.csv")), index=False)
    paired.to_csv(C.ensure(C.path("paired_bootstrap.csv")), index=False)
    print(f"\n=== paired bootstrap against {reference} (positive means the other is better) ===")
    print(paired.round(5).to_string(index=False))

    # ---- cost ---------------------------------------------------------------
    cost_rows = []
    for run in runs:
        model = runner.rebuild(run, geom)
        g = geom.for_arm(run["arm"])
        xs = X[idx["test_iid"]]
        single = metrics.time_forward(model, g, xs[:1], 200)
        batched = metrics.time_forward(model, g, xs[:C.EVAL["time_batch"]], 5)
        cost_rows.append({"arm": run["arm"], "seed": run["seed"], "task": run.get("task", "A"),
                          "n_params": run["n_params"], "nfe": run["nfe"],
                          "sec_single": single, "sec_batched": batched,
                          "speedup_single": C.EVAL["solver_sec"] / single,
                          "speedup_batched": C.EVAL["solver_sec"] / batched})
        del model
    cost = pd.DataFrame(cost_rows)
    cost.to_csv(C.ensure(C.path("cost.csv")), index=False)
    cost_a = cost[cost.task == "A"].groupby("arm")[
        ["n_params", "nfe", "sec_single", "sec_batched", "speedup_single"]].mean()
    print(f"\n=== cost (reference solver {C.EVAL['solver_sec'] * 1000:.1f} ms per sample, "
          f"unbatched) ===")
    print(cost_a.sort_values("sec_single").round(6).to_string())

    # ---- the graph-feature alignment control --------------------------------
    perm_rows = []
    for run in [r for r in runs if r.get("task") == "A" and r["arm"] in C.GRAPH_AWARE]:
        model = runner.rebuild(run, geom)
        g = geom.for_arm(run["arm"])
        t = runner.denormalise(y[idx["test_iid"]], y_mu, y_sd).numpy()
        base = metrics.relative_l2(
            runner.denormalise(runner.training.predict(model, X[idx["test_iid"]], g),
                               y_mu, y_sd).numpy(), t)
        for trial in range(3):
            pm = np.random.default_rng(100 + trial).permutation(geom.n_genes)
            out = runner.training.predict(model, X[idx["test_iid"]][:, pm, :], g)
            restored = torch.empty_like(out)
            restored[:, pm] = out
            shuffled = metrics.relative_l2(runner.denormalise(restored, y_mu, y_sd).numpy(), t)
            perm_rows.append({"arm": run["arm"], "seed": run["seed"], "trial": trial,
                              "aligned": base, "permuted": shuffled, "ratio": shuffled / base})
        del model
    permutation = pd.DataFrame(perm_rows)
    permutation.to_csv(C.ensure(C.path("permutation.csv")), index=False)
    print("\n=== shuffling genes across nodes (ratio >> 1 means the graph is used) ===")
    print(permutation.groupby("arm")[["aligned", "permuted", "ratio"]].mean()
          .sort_values("ratio", ascending=False).round(4).to_string())

    # ---- mode truncation ----------------------------------------------------
    sweep = []
    for run in [r for r in runs if r.get("task") == "A" and r["arm"] in C.SPECTRAL]:
        model = runner.rebuild(run, geom)
        name = "real" if run["arm"] == "wno_real" else run["arm"].split("_", 1)[1]
        t = runner.denormalise(y[idx["test_iid"]], y_mu, y_sd).numpy()
        for k in C.EVAL["k_sweep"]:
            if k > geom.eigs[name][1].shape[1]:
                continue
            g = (geom.wavelet(name, k) if run["arm"] == "wno_real" else geom.spectral(name, k))
            p = runner.denormalise(runner.training.predict(model, X[idx["test_iid"]], g),
                                   y_mu, y_sd).numpy()
            sweep.append({"arm": run["arm"], "seed": run["seed"], "k": k,
                          "rel_l2": metrics.relative_l2(p, t)})
        del model
    k_sweep = pd.DataFrame(sweep)
    k_sweep.to_csv(C.ensure(C.path("mode_truncation.csv")), index=False)
    print(f"\n=== mode truncation (trained at k={C.MODEL['k_modes']}, no retraining) ===")
    print(k_sweep.pivot_table(index="arm", columns="k", values="rel_l2").round(4).to_string())

    # ---- transfer to smaller gene sets --------------------------------------
    transfer_rows = []
    for size in C.EVAL["sub_sizes"]:
        path = C.path(f"transfer_{size}.npz")
        if not os.path.exists(path):
            continue
        T = np.load(path, allow_pickle=True)
        genes = T["gene_idx"]
        keep_mask = np.isfinite(T["y"]).all(1) & ~T["is_control"]
        x0 = np.log1p(np.clip(T["x0"][keep_mask], 0, None))
        x_mu = np.array(norm["x0_log_mu"])[genes]
        x_sd = np.array(norm["x0_log_sd"])[genes]
        flag = np.zeros((keep_mask.sum(), size), np.float32)
        dose = np.zeros_like(flag)
        tgt = T["target"][keep_mask]
        has = tgt >= 0
        flag[np.nonzero(has)[0], tgt[has]] = 1.0
        dose[np.nonzero(has)[0], tgt[has]] = T["level"][keep_mask][has]
        k_mu, k_sd = data["X"][:, :, 1].mean(), data["X"][:, :, 1].std() + 1e-6
        g_mu, g_sd = data["X"][:, :, 2].mean(), data["X"][:, :, 2].std() + 1e-6
        xs = torch.tensor(np.stack([(x0 - x_mu) / x_sd,
                                    (T["k"][keep_mask] - k_mu) / k_sd,
                                    (T["gamma"][keep_mask] - g_mu) / g_sd,
                                    flag, dose], -1), dtype=torch.float32)
        truth = np.log1p(np.clip(T["y"][keep_mask], 0, None))
        sub_eigs = network.laplacian_eigs(T["signs"])

        for run in [r for r in runs if r.get("task") == "A"]:
            arm = run["arm"]
            if arm not in TRANSFERABLE:
                transfer_rows.append({"arm": arm, "seed": run["seed"], "size": size,
                                      "rel_l2": np.nan})
                continue
            model = runner.rebuild(run, geom)
            if arm == "wno_real":
                profiles, _ = ops.sgwt_profiles(sub_eigs[0][:C.MODEL["k_modes"]],
                                                C.MODEL["wavelet_scales"])
                g = (torch.tensor(sub_eigs[1][:, :C.MODEL["k_modes"]], dtype=torch.float32),
                     torch.tensor(profiles))
            elif arm == "psno_er":
                er = network.erdos_renyi(T["signs"], np.random.default_rng(909 + size))
                g = geom.spectral(None, eigs=network.laplacian_eigs(er))
            elif arm.startswith("psno_"):
                g = geom.spectral(None, eigs=sub_eigs)
            elif arm in ("gcn", "node_graph"):
                g = geom.adjacency(signs=T["signs"])
            else:
                index, attr, deg = network.edge_list(T["signs"])
                g = (torch.tensor(index, dtype=torch.long),
                     torch.tensor(attr, dtype=torch.float32),
                     torch.tensor(deg, dtype=torch.float32))
            # the graph ODE carries a per-gene decay; slice it to the genes present
            saved = getattr(model, "decay", None)
            if saved is not None:
                model.decay = torch.nn.Parameter(saved.detach()[genes], requires_grad=False)
            try:
                p = runner.training.predict(model, xs, g).numpy()
                p = p * np.array(norm["y_log_sd"])[genes] + np.array(norm["y_log_mu"])[genes]
                transfer_rows.append({"arm": arm, "seed": run["seed"], "size": size,
                                      "rel_l2": metrics.relative_l2(p, truth)})
            except Exception:
                transfer_rows.append({"arm": arm, "seed": run["seed"], "size": size,
                                      "rel_l2": np.nan})
            if saved is not None:
                model.decay = saved
            del model
    transfer = pd.DataFrame(transfer_rows)
    transfer.to_csv(C.ensure(C.path("transfer.csv")), index=False)
    print("\n=== zero-shot transfer to smaller gene sets ===")
    print(transfer.pivot_table(index="arm", columns="size", values="rel_l2").round(4).to_string())
    blocked = sorted(transfer[transfer.rel_l2.isna()].arm.unique())
    if blocked:
        print("  cannot be evaluated on another gene set at all:", ", ".join(blocked))

    # ---- what the filters learned -------------------------------------------
    filt, scales = [], []
    for run in [r for r in runs if r.get("task") == "A"
                and r["arm"] in C.LEARNED_FILTER | {"wno_real"}]:
        model = runner.rebuild(run, geom)
        if run["arm"] == "wno_real":
            for bi, block in enumerate(model.blocks):
                for j, gain in enumerate(block.A.detach().flatten(1).norm(dim=1).numpy()):
                    scales.append({"arm": run["arm"], "seed": run["seed"], "block": bi,
                                   "scale": "low-pass" if j == 0 else f"band {j}",
                                   "gain": float(gain)})
        else:
            grid = torch.linspace(0, 2, 401)
            feats = ops.lambda_features(grid, model.n_freq)
            with torch.no_grad():
                for bi, block in enumerate(model.blocks):
                    response = block.filter_matrices(feats).flatten(1).norm(dim=1).numpy()
                    filt.append({"arm": run["arm"], "seed": run["seed"], "block": bi,
                                 "total_variation": float(np.abs(np.diff(response)).sum() /
                                                          max(response.mean(), 1e-9))})
        del model
    if filt:
        pd.DataFrame(filt).to_csv(C.ensure(C.path("filter_roughness.csv")), index=False)
        print("\n=== roughness of the learned filter (lower is smoother) ===")
        print(pd.DataFrame(filt).groupby("arm").total_variation.agg(["mean", "std"])
              .round(3).to_string())
    if scales:
        pd.DataFrame(scales).to_csv(C.ensure(C.path("wavelet_scales.csv")), index=False)
        print("\n=== weight given to each wavelet scale ===")
        print(pd.DataFrame(scales).groupby("scale").gain.agg(["mean", "std"]).round(3).to_string())

    # ---- trajectory task ----------------------------------------------------
    task_b = results[results.task == "B"]
    if len(task_b):
        print("\n=== trajectory task ===")
        print(task_b.groupby("arm")[["test_iid/traj_rel_l2", "test_iid/rel_l2"]]
              .agg(["mean", "std"]).round(4).to_string())

    # ---- figures ------------------------------------------------------------
    cost_plot = cost_a.join(summary["rel_l2"])
    dose = pd.DataFrame(dose_rows)
    dose = dose[dose.task == "A"] if len(dose) else dose
    dose.to_csv(C.ensure(C.path("dose_response.csv")), index=False)
    figures.results_grid(boot, cost_plot, permutation, k_sweep, transfer, dose,
                         C.EVAL["solver_sec"], C.path("figures/results_grid.png"))
    if len(task_b):
        time_cols = [c for c in task_b.columns if c.startswith("time_")]
        trajectories = {arm: sub[time_cols].mean().values
                        for arm, sub in task_b.groupby("arm")}
        figures.summary_grid(summary, cost_plot,
                             summary[["spearman", "spearman_sd"]].dropna(),
                             trajectories, data["t_eval"], C.EVAL["solver_sec"],
                             C.path("figures/summary_grid.png"))
    print("\nwrote results_raw.csv, summary_task_a.csv, bootstrap.csv, paired_bootstrap.csv, "
          "cost.csv, permutation.csv, mode_truncation.csv, transfer.csv, "
          "filter_roughness.csv, wavelet_scales.csv and figures/")


if __name__ == "__main__":
    main()
