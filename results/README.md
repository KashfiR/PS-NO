# Results

Written here by the experiment scripts; the directory is gitignored because the
run checkpoints are large.

| File | Written by | Contents |
|---|---|---|
| `graphs.npz` | 01 | the network and its three damaged variants, with achieved edge overlaps |
| `system.npz` | 01 | the fixed kinetic parameters |
| `dataset.npz` | 01 | inputs, steady states, trajectories, splits, matched-control pairings |
| `normalisation.json` | 01 | scaling fitted on the training split alone |
| `transfer_*.npz` | 01 | smaller gene sets for the transfer test |
| `runs/`, `runs_fraction/` | 02, 04 | one checkpoint per arm, seed and task |
| `results_raw.csv` | 03 | every measurement, one row per run |
| `summary_task_a.csv` | 03 | the main table |
| `bootstrap.csv`, `paired_bootstrap.csv` | 03 | intervals over the 600 test samples |
| `cost.csv` | 03 | parameters, field evaluations, timing, speedup |
| `permutation.csv` | 03 | the graph-feature alignment control |
| `mode_truncation.csv` | 03 | evaluating at a different number of modes |
| `transfer.csv` | 03 | zero-shot on smaller gene sets |
| `filter_roughness.csv`, `wavelet_scales.csv` | 03 | what the filters learned |
| `data_efficiency.csv` | 04 | accuracy against training-set size |
| `figures/` | 01, 03, 04 | every plot in the paper |

Set `PSNO_ROOT` to write somewhere else.
