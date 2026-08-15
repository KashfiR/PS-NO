"""Every plot in the paper."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C

NAVY, OCHRE, SAGE, BRICK, STONE = "#1B3A5C", "#A87225", "#46654F", "#7E3B36", "#7A7772"

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.grid": True, "grid.alpha": .25,
                     "axes.axisbelow": True, "axes.labelsize": 12, "axes.titlesize": 13,
                     "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
                     "legend.fontsize": 10, "figure.facecolor": "white"})


def data_validation(data, eigs, k_modes, path):
    valid, split = data["valid"], data["split"]
    fig, ax = plt.subplots(2, 2, figsize=(11.6, 8.4))

    ax[0, 0].hist(np.log1p(data["y_ss"][valid]).ravel(), bins=80, color=NAVY)
    ax[0, 0].set(xlabel="Steady-State Expression, $\\log(1+x)$", ylabel="Number of Genes",
                 title="Distribution of Simulated Steady States")

    first = np.nonzero(valid)[0][0]
    for i in range(8):
        ax[0, 1].plot(data["t_eval"], data["y_traj"][first, i, :], lw=1.8)
    ax[0, 1].set_xscale("symlog", linthresh=0.1)
    ax[0, 1].set(xlabel="Time (Arbitrary Units)", ylabel="Expression Level",
                 title="Example Trajectories for Eight Genes")

    for (name, label), colour, marker in zip(
            [("real", "True network"), ("partial", "Half-corrupted"), ("random", "Rewired")],
            [NAVY, OCHRE, SAGE], ["o", "s", "^"]):
        ax[1, 0].plot(eigs[name][0][:k_modes], marker=marker, ms=4.5, lw=1.7,
                      color=colour, label=label)
    ax[1, 0].set(xlabel="Eigenvalue Index", ylabel="Eigenvalue $\\lambda$",
                 title="Laplacian Spectra of the Three Networks")
    ax[1, 0].legend(frameon=False)

    train = (split == "train") & valid
    y = data["y_norm"][train]
    energy = np.abs(eigs["real"][1][:, :k_modes].T @ (y - y.mean(0)).T).mean(1)
    ax[1, 1].plot(energy, marker="o", ms=4.5, lw=1.7, color=NAVY)
    ax[1, 1].axvline(4.5, ls="--", lw=1.2, color="#888888")
    ax[1, 1].set(xlabel="Eigenvalue Index", ylabel="Mean Absolute Coefficient",
                 title="Where the Signal Sits in the Spectrum")

    plt.tight_layout()
    plt.savefig(C.ensure(path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def results_grid(bootstrap, cost, permutation, k_sweep, transfer, dose, solver_sec, path):
    fig, ax = plt.subplots(2, 3, figsize=(18, 9.5))

    b = bootstrap.set_index("arm")
    err = np.abs(np.stack([b.ci_lo, b.ci_hi]) - b["mean"].values)
    ax[0, 0].barh(range(len(b)), b["mean"], xerr=err, color=NAVY, capsize=3)
    ax[0, 0].set_yticks(range(len(b)))
    ax[0, 0].set_yticklabels(b.index, fontsize=9)
    ax[0, 0].invert_yaxis()
    ax[0, 0].set_xscale("log")
    ax[0, 0].set(xlabel="Relative $L_2$ Error", title="Accuracy, 95% Bootstrap")

    j = cost.dropna(subset=["sec_single", "rel_l2"])
    ax[0, 1].scatter(j.sec_single * 1000, j.rel_l2, s=70, color=OCHRE, zorder=3)
    for name, row in j.iterrows():
        ax[0, 1].annotate(name, (row.sec_single * 1000, row.rel_l2), fontsize=8,
                          xytext=(5, 4), textcoords="offset points")
    ax[0, 1].axvline(solver_sec * 1000, ls="--", c=BRICK, lw=1.5)
    ax[0, 1].set_xscale("log")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set(xlabel="Single-Query Time per Sample (ms)", ylabel="Relative $L_2$ Error",
                 title="Accuracy Against Cost")

    pm = permutation.groupby("arm")[["aligned", "permuted"]].mean()
    pe = permutation.groupby("arm")[["aligned", "permuted"]].std().fillna(0)
    x = np.arange(len(pm))
    ax[0, 2].bar(x - .2, pm.aligned, .4, yerr=pe.aligned, capsize=3, color=NAVY,
                 label="Genes on their own nodes")
    ax[0, 2].bar(x + .2, pm.permuted, .4, yerr=pe.permuted, capsize=3, color=BRICK,
                 label="Gene labels shuffled")
    ax[0, 2].set_xticks(x)
    ax[0, 2].set_xticklabels(pm.index, rotation=35, ha="right", fontsize=8)
    ax[0, 2].legend(frameon=False, fontsize=8)
    ax[0, 2].set(ylabel="Relative $L_2$ Error", title="Shuffling Genes Across Nodes")

    if not len(k_sweep):
        ax[1, 0].axis("off")
    for arm, sub in (k_sweep.groupby("arm") if len(k_sweep) else []):
        g = sub.groupby("k").rel_l2
        mu, sd = g.mean(), g.std().reindex(g.mean().index).fillna(0)
        line, = ax[1, 0].plot(mu.index, mu.values, marker="o", ms=5, label=arm)
        ax[1, 0].fill_between(mu.index, mu - sd, mu + sd, color=line.get_color(), alpha=.18, lw=0)
    ax[1, 0].axvline(C.MODEL["k_modes"], ls="--", c="#888888", lw=1)
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].set(xlabel="Modes Kept at Test Time", ylabel="Relative $L_2$ Error",
                 title="Sensitivity to the Number of Modes")

    tp = (transfer.pivot_table(index="arm", columns="size", values="rel_l2").dropna(how="all")
          if len(transfer) else None)
    if tp is None or tp.empty:
        ax[1, 1].axis("off")
        tp = None
    if tp is not None:
      ts = (transfer.pivot_table(index="arm", columns="size", values="rel_l2", aggfunc="std")
            .reindex(index=tp.index, columns=tp.columns).fillna(0))
      y = np.arange(len(tp))
      width = 0.8 / max(len(tp.columns), 1)
      for i, col in enumerate(tp.columns):
          ax[1, 1].barh(y + (i - (len(tp.columns) - 1) / 2) * width, tp[col], width,
                        xerr=ts[col], capsize=2, color=[NAVY, SAGE][i % 2], label=f"{col} genes")
      ax[1, 1].set_yticks(y)
      ax[1, 1].set_yticklabels(tp.index, fontsize=8)
      ax[1, 1].legend(fontsize=8)
      ax[1, 1].set(xlabel="Relative $L_2$ Error", title="Transfer to Smaller Gene Sets")

    if len(dose):
        dp = dose.pivot_table(index="level", columns="arm", values="delta_rel_l2")
        ds = (dose.pivot_table(index="level", columns="arm", values="delta_rel_l2", aggfunc="std")
              .reindex(index=dp.index, columns=dp.columns).fillna(0))
        for col in dp.columns:
            line, = ax[1, 2].plot(dp.index, dp[col], marker="o", ms=4, label=col)
            ax[1, 2].fill_between(dp.index, dp[col] - ds[col], dp[col] + ds[col],
                                  color=line.get_color(), alpha=.15, lw=0)
        ax[1, 2].legend(fontsize=7, ncol=2)
        ax[1, 2].set(xlabel="Perturbation Dose ($\\log_2$ Fold Change)",
                     ylabel="Relative $L_2$ Error on Response",
                     title="Error Against Perturbation Strength")
    else:
        ax[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(C.ensure(path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def summary_grid(summary, cost, ranking, trajectory, t_eval, solver_sec, path):
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9))

    m = summary["rel_l2"].dropna().sort_values()
    ax[0, 0].barh(range(len(m)), m.values, xerr=summary["rel_l2_sd"].reindex(m.index).fillna(0),
                  color=NAVY, capsize=3)
    ax[0, 0].set_yticks(range(len(m)))
    ax[0, 0].set_yticklabels(m.index, fontsize=9)
    ax[0, 0].invert_yaxis()
    ax[0, 0].set(xlabel="Relative $L_2$ Error", title="Every Model, Steady-State Task")

    j = cost.dropna(subset=["sec_single", "rel_l2"])
    ax[0, 1].scatter(j.sec_single * 1000, j.rel_l2, s=60, color=OCHRE)
    for name, row in j.iterrows():
        ax[0, 1].annotate(name, (row.sec_single * 1000, row.rel_l2), fontsize=8,
                          xytext=(4, 4), textcoords="offset points")
    ax[0, 1].axvline(solver_sec * 1000, ls="--", c=BRICK, lw=1.4)
    ax[0, 1].set_xscale("log")
    ax[0, 1].set_yscale("log")
    ax[0, 1].set(xlabel="Single-Query Time per Sample (ms)", ylabel="Relative $L_2$ Error",
                 title="Accuracy Against Cost")

    r = ranking["spearman"].sort_values()
    ax[1, 0].barh(range(len(r)), r.values, xerr=ranking["spearman_sd"].reindex(r.index).fillna(0),
                  color=SAGE, capsize=3)
    ax[1, 0].set_yticks(range(len(r)))
    ax[1, 0].set_yticklabels(r.index, fontsize=9)
    ax[1, 0].set(xlabel="Spearman Correlation", title="Ranking Which Genes Responded")

    for arm, curve in trajectory.items():
        ax[1, 1].plot(t_eval, curve, marker="o", ms=4, label=arm)
    ax[1, 1].set_xscale("symlog", linthresh=0.1)
    ax[1, 1].legend(fontsize=9)
    ax[1, 1].set(xlabel="Time", ylabel="Relative $L_2$ Error",
                 title="Trajectory Error Against Time")

    plt.tight_layout()
    plt.savefig(C.ensure(path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def data_efficiency(curves, advantage, path):
    fig, ax = plt.subplots(1, 2, figsize=(11.6, 4.3))
    markers = ["o", "s", "^", "D", "v", "P"]
    for (arm, series), marker in zip(curves.items(), markers):
        ax[0].plot(series.index, series.values, marker=marker, lw=1.9, ms=6.5, label=arm)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].legend(frameon=False, loc="upper right")
    ax[0].set(xlabel="Fraction of Training Data Used", ylabel="Relative $L_2$ Error",
              title="Accuracy Against Training-Set Size")

    ax[1].plot(advantage.index, advantage.values, marker="o", lw=2.2, ms=7, color=NAVY)
    ax[1].fill_between(advantage.index, 0, advantage.values, color=NAVY, alpha=.10)
    ax[1].set_xscale("log")
    ax[1].set(xlabel="Fraction of Training Data Used",
              ylabel="Advantage of the True Network (%)",
              title="Advantage of the True Network by Data Size")
    for x, y in advantage.items():
        ax[1].annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 9),
                       ha="center", fontsize=10, color=NAVY)

    plt.tight_layout()
    plt.savefig(C.ensure(path), dpi=200, bbox_inches="tight")
    plt.close(fig)
