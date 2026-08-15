"""The governing equations and their numerical solution.

No existing simulator fitted. SERGIO exposes only settled states, BoolODE needs
a hand-written Boolean rule file that does not exist for an arbitrary CollecTRI
subnetwork, and SBML models from BioModels carry their own fixed wiring. All
three use the same functional form, so it is implemented directly here:
SERGIO's additive production term with shifted-Hill repression and BoolODE's
linear decay.
"""

import numpy as np
from scipy.integrate import solve_ivp

from . import config as C


def build_system(signs, rng):
    """Draw the parts of the system that stay fixed for the whole dataset.

    Link strengths must not vary per sample: if they did, the Laplacian
    eigenvectors would change per sample and there would be no fixed basis for a
    spectral operator to build on.
    """
    n = len(signs)
    k = C.KINETICS
    mag = rng.uniform(*k["k_edge"], size=(n, n)) * (signs != 0)
    in_deg = (signs != 0).sum(1)
    mag = mag / np.maximum(in_deg, 1)[:, None]        # so hubs do not saturate
    return dict(
        k_act=mag * (signs > 0),
        k_rep=mag * (signs < 0),
        half=rng.uniform(*k["half_response"], size=n),
        basal=np.where(in_deg == 0,
                       rng.uniform(*k["basal_master"], size=n),
                       rng.uniform(*k["basal"], size=n)),
        hill_n=k["hill_n"],
    )


def snapshot_times(t_max=None, t_points=None):
    """Log-spaced observation times, clipped into the span.

    10**log10(t_max) can land a ulp above t_max depending on the NumPy build,
    which solve_ivp rejects.
    """
    s = C.SAMPLING
    t_max = t_max or s["t_max"]
    t_points = t_points or s["t_points"]
    t = np.concatenate([[0.0], np.logspace(-1, np.log10(t_max), t_points - 1)])
    t = np.unique(np.clip(t, 0.0, t_max))
    assert len(t) == t_points
    return t


def rate_of_change(system, k, gamma, pert):
    half_pow = system["half"] ** system["hill_n"]
    k_act, k_rep, basal, n = system["k_act"], system["k_rep"], system["basal"], system["hill_n"]

    def rhs(t, x):
        x = np.maximum(x, 0.0)
        xn = x ** n
        hill = xn / (xn + half_pow)
        return pert * k * (basal + k_act @ hill + k_rep @ (1.0 - hill)) - gamma * x

    return rhs


def simulate(system, x0, k, gamma, target, level, t_eval):
    """One trajectory. Returns (steady state, trajectory, converged)."""
    s = C.SAMPLING
    pert = np.ones(len(x0))
    if target >= 0:
        pert[target] = 2.0 ** level
    rhs = rate_of_change(system, k, gamma, pert)
    sol = solve_ivp(rhs, (0.0, float(t_eval[-1])), np.maximum(x0, 0.0), t_eval=t_eval,
                    method="LSODA", rtol=s["rtol"], atol=s["atol"])
    if not sol.success or sol.y.shape[1] != len(t_eval):
        return None, None, False
    traj = np.maximum(sol.y, 0.0)
    steady = traj[:, -1]
    residual = np.linalg.norm(rhs(t_eval[-1], steady)) / max(np.linalg.norm(steady), 1e-9)
    return steady, traj, bool(residual < s["steady_tol"])
