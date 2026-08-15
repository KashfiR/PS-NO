"""How accuracy depends on training-set size.

Structural priors are supposed to pay off when data is short rather than raise
the ceiling, so this trains a subset of arms on 5%, 10%, 25% and 50% of the
training split. About three hours; every run is cached.

    python experiments/04_data_efficiency.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C, figures, metrics, runner

ARMS = ["psno_real", "psno_partial", "psno_random", "psno_er", "gcn", "gno_real",
        "wno_real", "mlp"]


def main():
    data, X, y, y_mu, y_sd, idx, control, geom, norm = runner.load()

    for seed in C.TRAIN["seeds"]:
        for fraction in C.EVAL["fractions"]:
            for arm in ARMS:
                runner.do_run(arm, seed, geom, X, y, idx, control, "A", fraction)

    rows = []
    for directory, fraction_default in [(C.path("runs_fraction"), None), (C.path("runs"), 1.0)]:
        for run in runner.load_runs(directory, task="A"):
            if run["arm"] not in ARMS:
                continue
            p = runner.denormalise(torch.tensor(run["preds"]["test_iid"]), y_mu, y_sd).numpy()
            t = runner.denormalise(y[idx["test_iid"]], y_mu, y_sd).numpy()
            rows.append({"arm": run["arm"], "seed": run["seed"],
                         "fraction": run.get("fraction", fraction_default),
                         "rel_l2": metrics.relative_l2(p, t)})
    df = pd.DataFrame(rows)
    df.to_csv(C.ensure(C.path("data_efficiency.csv")), index=False)

    table = df.pivot_table(index="fraction", columns="arm", values="rel_l2")
    print("=== accuracy against training-set size ===")
    print(table.round(4).to_string())

    advantage = 100 * (table["psno_random"] - table["psno_real"]) / table["psno_random"]
    print("\nadvantage of the true network over a degree-matched rewiring:")
    for fraction, value in advantage.items():
        print(f"  {fraction:.2f}  {value:+.1f}%")

    figures.data_efficiency({a: table[a].dropna() for a in table.columns if a in ARMS},
                            advantage.dropna(), C.path("figures/data_efficiency.png"))
    print("\nwrote data_efficiency.csv and figures/data_efficiency.png")


if __name__ == "__main__":
    main()
