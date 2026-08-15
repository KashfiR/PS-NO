"""Every tunable in one place.

Nothing here is searched over. The study fixes a single configuration and varies
only the operator kernel, because tuning each arm separately would reintroduce
the variable the comparison exists to control.
"""

import os

ROOT = os.environ.get("PSNO_ROOT", "results")

NETWORK = dict(n_genes=64, k_modes=32, max_density=0.06, min_evidence=1, seed=11)

KINETICS = dict(hill_n=2.0, k_edge=(1.0, 5.0), half_response=(0.5, 1.5),
                basal=(0.05, 0.30), basal_master=(0.80, 1.50), seed=22)

SAMPLING = dict(x0=(0.0, 2.0), k_prod=(0.50, 2.00), gamma=(0.40, 1.20),
                k_prod_ood=(2.00, 3.50), gamma_ood=(1.20, 2.00),
                pert_levels=[-4.0, -2.0, -1.0, 1.0, 2.0, 3.0], control_frac=0.10,
                t_max=40.0, t_points=16, rtol=1e-6, atol=1e-8, steady_tol=1e-3,
                seed=33)

SPLITS = dict(train=4000, val=500, test_iid=600, test_target=600, test_param=600,
              target_pools=(40, 8, 16), shard_size=250)

MODEL = dict(channels=32, blocks=3, head=128, k_modes=32, filter_basis=12,
             filter_hidden=32, lambda_freqs=4, wavelet_scales=4, fno_modes=16,
             gno_hidden=16)

# The vector field is sized directly rather than matched to the other arms.
# Parameter matching is the wrong control for a neural ODE: a feed-forward arm
# applies its weights once per prediction, an ODE once per solver stage. Matching
# forces 77 channels and 51 hours per run. Parameter count and evaluation count
# are both reported instead, and the asymmetry is stated in the paper.
ODE = dict(channels=32, head=32, blocks=2, h0=0.5, growth=2.0, h_max=2.0,
           epochs=80, patience=12, decay=True)

TRAIN = dict(lr=1e-3, weight_decay=1e-4, batch=64, epochs=200, patience=20,
             grad_clip=1.0, traj_epochs=120, seeds=[0, 1, 2], capacity_tol=0.10)

EVAL = dict(bootstrap=2000, top_k=10, k_sweep=[8, 16, 32, 48, 64],
            sub_sizes=[48, 32], fractions=[0.05, 0.10, 0.25, 0.50],
            time_batch=256, n_transfer=500,
            solver_sec=61.6e-3)   # measured: 8.6 min for 8,370 LSODA solves

ARMS_CORE = ["psno_real", "psno_partial", "psno_random", "gcn", "mlp"]
ARMS_EXTENDED = ["wno_real", "fno_fiedler", "fno_shuffled", "gno_real",
                 "psno_er", "node_mlp", "node_graph"]
ARMS_TRAJECTORY = ["psno_real", "gcn", "mlp", "node_graph"]
ARMS_ALL = ARMS_CORE + ARMS_EXTENDED

SPECTRAL = {"psno_real", "psno_partial", "psno_random", "psno_er", "wno_real"}
LEARNED_FILTER = {"psno_real", "psno_partial", "psno_random", "psno_er"}
GRAPH_AWARE = SPECTRAL | {"gcn", "gno_real", "node_graph"}

EVAL_SPLITS = ["test_iid", "test_target", "test_param"]
ALL_SPLITS = ["train", "val"] + EVAL_SPLITS


def path(name):
    os.makedirs(ROOT, exist_ok=True)
    return os.path.join(ROOT, name)


def ensure(p):
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    return p
