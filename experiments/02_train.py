"""Train every arm on both prediction targets.

Task A predicts where expression settles; Task B predicts the whole trajectory.
Runs are cached by name, so this can be stopped and resumed, and re-running only
trains what is missing.

    python experiments/02_train.py                 # everything
    python experiments/02_train.py --arms gcn mlp  # a subset
    python experiments/02_train.py --task B
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C, build as B, runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--task", choices=["A", "B", "both"], default="both")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--fractions", nargs="*", type=float, default=None,
                    help="reduced training-set sizes for the data-efficiency curve")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    data, X, y, y_mu, y_sd, idx, control, geom, norm = runner.load()
    seeds = args.seeds or C.TRAIN["seeds"]
    target = B.target_params()
    print(f"target parameter count {target:,}\n")

    if args.task in ("A", "both"):
        arms = args.arms or C.ARMS_ALL
        for arm in arms:
            n = B.ops.n_params(B.build(arm, 0, geom, "A"))
            note = "" if abs(n - target) / target <= C.TRAIN["capacity_tol"] else \
                "  (sized directly; see config)"
            print(f"  {arm:14s} {n:>9,} parameters{note}")
        print()
        for seed in seeds:
            for arm in arms:
                for fraction in (args.fractions or [1.0]):
                    runner.do_run(arm, seed, geom, X, y, idx, control, "A", fraction,
                                  args.verbose)

    if args.task in ("B", "both"):
        split, valid = data["split"].astype(str), data["valid"]
        y_traj, mu_t, sd_t = runner.trajectory_targets(data, split, valid)
        geom.traj_norm = (mu_t, sd_t)
        arms = args.arms or C.ARMS_TRAJECTORY
        print("\ntrajectory task:")
        for seed in seeds:
            for arm in arms:
                if arm not in C.ARMS_TRAJECTORY:
                    continue
                runner.do_run(arm, seed, geom, X, y_traj, idx, control, "B", 1.0, args.verbose)


if __name__ == "__main__":
    main()
