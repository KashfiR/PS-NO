"""Build the regulatory network, simulate the dynamics, and write the dataset.

Runs in about 25 minutes on one CPU. Simulation is sharded, so an interrupted
session resumes where it stopped.

    python experiments/01_build_dataset.py
"""

import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C, dataset, figures, kinetics, network


def main():
    net, samp = C.NETWORK, C.SAMPLING
    rng = np.random.default_rng(net["seed"])

    links = network.normalize_links(network.fetch_collectri(), net["min_evidence"])
    genes, signs = network.select_subnetwork(links, net["n_genes"], net["max_density"], net["seed"])
    n = len(genes)
    print(f"{n} genes, {(signs != 0).sum()} links, density {(signs != 0).mean():.3f}, "
          f"{(signs > 0).sum()} activating, {(signs < 0).sum()} repressing")

    lam, null_mu, null_sd, z = network.modularity_z_score(signs, rng)
    print(f"algebraic connectivity {lam:.4f} vs degree-matched null {null_mu:.4f} "
          f"+- {null_sd:.4f}, z = {z:.2f}")
    if z > -2:
        print("  the network is not measurably more modular than degree alone predicts; "
              "the damage controls will have little to detect")

    # Damaged variants are defined relative to the measured randomisation floor,
    # because degree-preserving rewiring cannot randomise past a graph-specific
    # limit and fixed targets would collapse two arms into one.
    random_signs, floor = network.rewire(signs, 0.0, rng)
    partial_signs, partial_overlap = network.rewire(signs, 1.0 - 0.5 * (1.0 - floor), rng)
    graphs = {"real": signs, "partial": partial_signs, "random": random_signs,
              "er": network.erdos_renyi(signs, rng)}
    overlaps = {"real": 1.0, "partial": partial_overlap, "random": floor,
                "er": float((np.abs(graphs["er"]) * np.abs(signs)).sum() / np.abs(signs).sum())}
    for name, ov in overlaps.items():
        print(f"  {name:8s} edge overlap {ov:.3f}  lambda[1] "
              f"{network.laplacian_eigs(graphs[name], 2)[0][1]:.4f}")

    np.savez_compressed(C.ensure(C.path("graphs.npz")),
                        genes=np.array(genes),
                        **{f"signs_{k}": v for k, v in graphs.items()},
                        **{f"overlap_{k}": v for k, v in overlaps.items()},
                        modularity_z=z)

    system = kinetics.build_system(signs, np.random.default_rng(C.KINETICS["seed"]))
    np.savez_compressed(C.ensure(C.path("system.npz")), **system)

    pools = dataset.target_pools(n, np.random.default_rng(samp["seed"]))
    rows = dataset.design(n, pools)
    t_eval = kinetics.snapshot_times()
    print(f"\n{len(rows)} samples to simulate "
          f"({sum(r['is_control'] for r in rows)} matched controls)")

    for shard, stop, conv in dataset.simulate_all(system, rows, t_eval, C.path("shards")):
        print(f"  shard {shard + 1}: {stop}/{len(rows)} done, convergence {conv:.2f}")

    data, norm = dataset.assemble(rows, C.path("shards"), t_eval)
    np.savez_compressed(C.ensure(C.path("dataset.npz")), **data)
    json.dump({**norm, "pools": {k: v.tolist() for k, v in pools.items()},
               "genes": genes, "overlaps": overlaps},
              open(C.ensure(C.path("normalisation.json")), "w"), indent=2)
    print(f"\n{data['valid'].sum()}/{len(rows)} samples converged")

    # Smaller gene sets for the transfer test, density-matched to the parent so
    # that a change of gene set is not confounded with a change of density.
    held_out = set(pools["test"].tolist())
    for size in C.EVAL["sub_sizes"]:
        sub = dataset.connected_subset(signs, size, np.random.default_rng(samp["seed"] + size))
        sub_signs = signs[np.ix_(sub, sub)]
        local_pool = [i for i, g in enumerate(sub) if int(g) in held_out] or list(range(size))
        sub_system = {k: (v[np.ix_(sub, sub)] if v.ndim == 2 else v[sub])
                      if isinstance(v, np.ndarray) else v for k, v in system.items()}

        X, y, controls, parent = [], [], [], []
        n_control = int(round(samp["control_frac"] * C.EVAL["n_transfer"]))
        for i in range(C.EVAL["n_transfer"]):
            r = np.random.default_rng([7000 + size, i])
            x0 = r.uniform(*samp["x0"], size=size)
            k = r.uniform(*samp["k_prod"], size=size)
            gamma = r.uniform(*samp["gamma"], size=size)
            target, level = (-1, 0.0) if i < n_control else (
                int(local_pool[r.integers(len(local_pool))]),
                float(samp["pert_levels"][r.integers(len(samp["pert_levels"]))]))
            for tgt, lvl, is_ctrl in ([(target, level, False)] if target < 0
                                      else [(target, level, False), (-1, 0.0, True)]):
                out, _, _ = kinetics.simulate(sub_system, x0, k, gamma, tgt, lvl, t_eval)
                X.append((x0, k, gamma, tgt, lvl))
                y.append(out if out is not None else np.full(size, np.nan))
                controls.append(is_ctrl)
                parent.append(i)

        np.savez_compressed(C.ensure(C.path(f"transfer_{size}.npz")),
                            gene_idx=sub, signs=sub_signs,
                            x0=np.stack([a[0] for a in X]).astype(np.float32),
                            k=np.stack([a[1] for a in X]).astype(np.float32),
                            gamma=np.stack([a[2] for a in X]).astype(np.float32),
                            target=np.array([a[3] for a in X]),
                            level=np.array([a[4] for a in X], np.float32),
                            y=np.array(y, np.float32),
                            is_control=np.array(controls),
                            parent=np.array(parent))
        density = (sub_signs != 0).sum() / (size * (size - 1))
        print(f"  transfer domain {size} genes: {(sub_signs != 0).sum()} links, "
              f"density {density:.3f} (parent {(signs != 0).mean():.3f})")

    eigs = {k: network.laplacian_eigs(v) for k, v in graphs.items()}
    figures.data_validation(data, eigs, net["k_modes"], C.path("figures/data_validation.png"))
    print("\nwrote graphs.npz, system.npz, dataset.npz, normalisation.json, transfer_*.npz")


if __name__ == "__main__":
    main()
