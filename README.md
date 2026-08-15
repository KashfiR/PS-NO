# Which basis should a neural operator use on a gene network?

A controlled comparison of spectral, wavelet, Fourier, message-passing and continuous-time
operators on gene-expression dynamics.

A neural operator is a lift, a stack of kernel layers and a projection. What distinguishes one
from another is the basis its kernel is written in. On a regular grid the answer is usually
Fourier; on a gene network nobody knows. This repository holds everything else fixed — the data,
the model size, the training budget, every other component — and varies only that choice, across
twelve models and five families.

## What it found

| | |
|---|---|
| Using a graph at all | worth about a factor of four |
| Shuffling which gene sits on which node | costs every graph model a factor of 4–6 |
| Best on the settled state | graph neural operator, 0.0133 relative error |
| Best at ranking which genes responded | graph neural operator, 0.89 rank correlation against 0.05–0.19 for the other eleven |
| Best on the whole trajectory | spectral operator — the ordering reverses |
| Correct wiring versus a degree-matched rewiring | 25.7% better at 5% of the data, 5.6% at full data |
| Speed against the ODE solver being replaced | 2–134× per query on the settled-state task |

Several of these are negative results for the spectral operator the project set out to study. It
places third on the settled-state task, and a fixed wavelet dictionary beats a freely learned
spectral filter. Establishing which bases fail, and why, is the point.

## Running it

```bash
pip install -r requirements.txt

python experiments/01_build_dataset.py       # ~25 min: network, simulation, splits
python experiments/02_train.py               # ~11 h: 12 arms x 3 seeds, both tasks
python experiments/03_evaluate.py            # ~15 min: every measurement and figure
python experiments/04_data_efficiency.py     # ~3 h, optional: reduced training sizes
python experiments/00_architecture_diagram.py
```

CPU only. Everything fits on a free Colab instance. Runs are cached by name and checkpointed
every ten epochs, so any script can be stopped and resumed, and re-running trains only what is
missing.

Subsets are easy:

```bash
python experiments/02_train.py --arms gcn gno_real --seeds 0
python experiments/02_train.py --task B
```

Outputs land in `results/` (override with `PSNO_ROOT`).

## The twelve arms

| Arm | Kernel | What it isolates |
|---|---|---|
| `psno_real` | Laplacian eigenbasis of the true network | the spectral operator itself |
| `psno_partial` | half-corrupted network, degrees preserved | how much topological accuracy is needed |
| `psno_random` | rewired to the randomisation floor, degrees preserved | wiring pattern versus degree sequence |
| `psno_er` | Erdős–Rényi, density matched, degrees not | whether the degree sequence carries the benefit |
| `wno_real` | fixed multiscale wavelet dictionary | learned filter versus fixed dictionary |
| `fno_fiedler` | 1-D Fourier, genes ordered by the Fiedler vector | is the Fourier basis usable here at all |
| `fno_shuffled` | 1-D Fourier, random gene order | was any FNO result just the ordering |
| `gno_real` | edge-conditioned kernel over neighbourhoods | a local kernel with no global transform |
| `gcn` | signed message passing | a plain graph network as reference |
| `mlp` | dense, no graph | is a graph needed at all |
| `node_mlp` | neural ODE, dense vector field | does continuous time help without structure |
| `node_graph` | neural ODE, message-passing vector field | continuous time plus structure |

Three reference predictors — the training mean, carrying the initial condition forward, and ridge
regression — are reported in the same table, so no architecture gets credit for beating nothing.

## Layout

```
src/
  config.py      every tunable; nothing here is searched over
  network.py     CollecTRI, community selection, damage controls, eigendecomposition
  kinetics.py    the Hill kinetic model and its numerical solution
  dataset.py     sample design, splits, assembly, transfer domains
  operators.py   the five families, written as one kernel layer
  build.py       arm name -> model and the geometry it needs
  training.py    training loop, run cache, checkpointing
  metrics.py     accuracy, ranking, spectral bands, bootstrap, timing
  runner.py      shared loading and run management
  figures.py     every plot
experiments/
  00_architecture_diagram.py
  01_build_dataset.py
  02_train.py
  03_evaluate.py
  04_data_efficiency.py
```

## Data

Regulatory links come from [CollecTRI](https://academic.oup.com/nar/article/51/20/10934/7318114)
through the OmniPath REST endpoint, fetched and cached on first run. Nothing needs downloading by
hand.

Dynamics are simulated rather than measured. No existing simulator fitted: SERGIO exposes only
settled states, BoolODE needs a hand-written Boolean rule file that does not exist for an
arbitrary CollecTRI subnetwork, and SBML models from BioModels carry their own fixed wiring. All
three use the same functional form, so it is implemented directly in `src/kinetics.py` —
SERGIO's additive production term with shifted-Hill repression and BoolODE's linear decay.

That is also the single largest caveat. Benchmarks have repeatedly shown that methods topping
simulated leaderboards can fall to near-random on real measurements.

## Notes on the design

**No hyperparameter search.** One configuration applied identically to every arm. The question is
whether the basis matters with everything else equal, and tuning each arm separately would
reintroduce exactly the variable the comparison exists to control. Absolute numbers are therefore
not the best achievable; only the comparisons are claimed.

**The spectral filter reads the eigenvalue, not the mode index.** An index-keyed filter fails
twice: mode number means nothing across networks, so it could never move to another gene set, and
eigenvectors are defined only up to sign and up to rotation within a repeated eigenvalue, so the
model would change if the solver returned an equally valid basis. Reading λ ∈ [0, 2] fixes both.

**Parameter matching is the wrong control for a neural ODE.** A feed-forward arm applies its
weights once per prediction; an ODE applies them once per solver stage. Matching parameters hands
the ODE about a hundred times the compute and takes 51 hours per run; matching compute would hand
it a hundredth of the parameters. Neither is uniquely right, so both the parameter count and the
field-evaluation count are reported for every arm.

**Timing needs care.** A batch of one is tens of microseconds of real work, so a naive
measurement is dominated by scheduler jitter, and identical architectures can differ sixfold.
`metrics.time_forward` takes the fastest of five blocks of 200 repeats — contention only ever
makes a measurement slower, so the minimum is the closest estimate of real cost. That brings the
spread across seeds to a median of 3%.

## Requirements

Python 3.10+, NumPy, SciPy, pandas, NetworkX, PyTorch, Matplotlib. See `requirements.txt`.

## License

MIT. See `LICENSE`.
