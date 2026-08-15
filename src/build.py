"""Turning an arm name into a model and the geometry it needs.

Every arm shares the lift, the head and the training protocol; only the kernel
differs. Widths are found by binary search so that each lands within 1.6% of the
spectral operator's parameter count, with the neural ODEs the stated exception.
"""

import numpy as np
import torch

from . import config as C
from . import network, operators as ops


class Geometry:
    """Everything derived from the network that a model might need."""

    def __init__(self, graphs, n_genes, t_eval, norm, traj_norm=None):
        self.graphs = graphs                 # name -> signed adjacency
        self.n_genes = n_genes
        self.t_eval = t_eval
        self.eigs = {k: network.laplacian_eigs(v) for k, v in graphs.items()}
        self.order, self.inverse = network.gene_orderings(
            self.eigs["real"][1], n_genes)
        idx, attr, deg = network.edge_list(graphs["real"])
        self.edge_index = torch.tensor(idx, dtype=torch.long)
        self.edge_attr = torch.tensor(attr, dtype=torch.float32)
        self.degree = torch.tensor(deg, dtype=torch.float32)
        self.norm, self.traj_norm = norm, traj_norm

    def spectral(self, name, k=None, eigs=None):
        k = k or C.MODEL["k_modes"]
        w, V = eigs or self.eigs[name]
        k = min(k, V.shape[1])
        return (torch.tensor(V[:, :k], dtype=torch.float32),
                torch.tensor(w[:k], dtype=torch.float32))

    def wavelet(self, name, k=None, eigs=None):
        k = k or C.MODEL["k_modes"]
        w, V = eigs or self.eigs[name]
        profiles, _ = ops.sgwt_profiles(w[:k], C.MODEL["wavelet_scales"])
        return torch.tensor(V[:, :k], dtype=torch.float32), torch.tensor(profiles)

    def adjacency(self, name="real", signs=None):
        S = (self.graphs[name] if signs is None else signs).astype(np.float32)
        out = []
        for mask in [(S > 0).astype(np.float32), (S < 0).astype(np.float32)]:
            A = np.maximum(mask, mask.T) + np.eye(len(S), dtype=np.float32)
            di = 1.0 / np.sqrt(np.maximum(A.sum(1), 1e-12))
            out.append(torch.tensor(di[:, None] * A * di[None, :], dtype=torch.float32))
        return tuple(out)

    def for_arm(self, arm):
        if arm == "wno_real":
            return self.wavelet("real")
        if arm.startswith("psno_"):
            return self.spectral(arm.split("_", 1)[1])
        if arm == "gno_real":
            return self.edge_index, self.edge_attr, self.degree
        if arm in ("gcn", "node_graph"):
            return self.adjacency("real")
        return None


def spectral_operator(out=1):
    m = C.MODEL
    return ops.SpectralOperator(m["channels"], m["blocks"], m["head"], m["filter_basis"],
                                m["filter_hidden"], m["lambda_freqs"], out=out)


def target_params():
    return ops.n_params(spectral_operator())


def _neural_ode(kind, geom, task):
    o, m = C.ODE, C.MODEL
    steps, obs = ops.rk4_schedule(geom.t_eval, task)
    field = (ops.FieldDense(geom.n_genes, o["channels"] * 3, o["blocks"])
             if kind == "dense" else ops.FieldGraph(o["channels"], o["blocks"], o["head"]))
    # Start from a contraction: a field that is large at initialisation can blow
    # up before it learns anything, since RK4 is only stable while h|lambda| < 2.78.
    last = [mm for mm in field.modules() if isinstance(mm, torch.nn.Linear)][-1]
    torch.nn.init.zeros_(last.weight)
    if last.bias is not None:
        torch.nn.init.zeros_(last.bias)

    n = geom.norm
    in_mu = torch.tensor(n["x0_log_mu"], dtype=torch.float32)
    in_sd = torch.tensor(n["x0_log_sd"], dtype=torch.float32)
    if task == "B" and geom.traj_norm is not None:
        out_mu, out_sd = geom.traj_norm
    else:
        out_mu = torch.tensor(n["y_log_mu"], dtype=torch.float32)
        out_sd = torch.tensor(n["y_log_sd"], dtype=torch.float32)
    return ops.NeuralODE(field, steps, obs, kind, o["decay"], geom.n_genes,
                         in_mu, in_sd, out_mu, out_sd)


def build(arm, seed, geom, task="A"):
    torch.manual_seed(seed)
    m, target = C.MODEL, target_params()
    blocks, head, n = m["blocks"], m["head"], geom.n_genes
    out = len(geom.t_eval) if task == "B" else 1

    if arm.startswith("node_"):
        return _neural_ode("dense" if arm.endswith("mlp") else "graph", geom, task)

    if arm.startswith("psno_"):
        return spectral_operator(out)

    if arm == "wno_real":
        n_prof = m["wavelet_scales"] + 1
        w = ops.match_width(lambda c: ops.WaveletOperator(c, blocks, head, n_prof, out=out), target)
        return ops.WaveletOperator(w, blocks, head, n_prof, out=out)

    if arm.startswith("fno_"):
        key = arm.split("_", 1)[1]
        order, inverse = geom.order[key], geom.inverse[key]
        w = ops.match_width(
            lambda c: ops.FourierOperator1D(c, blocks, head, m["fno_modes"], order, inverse, out=out),
            target)
        return ops.FourierOperator1D(w, blocks, head, m["fno_modes"], order, inverse, out=out)

    if arm == "gno_real":
        edge_dim = geom.edge_attr.shape[1]
        w = ops.match_width(
            lambda c: ops.GraphNeuralOperator(c, blocks, head, edge_dim, m["gno_hidden"], out=out),
            target)
        return ops.GraphNeuralOperator(w, blocks, head, edge_dim, m["gno_hidden"], out=out)

    if arm == "gcn":
        w = ops.match_width(lambda c: ops.GraphConvNet(c, blocks, head, out=out), target)
        return ops.GraphConvNet(w, blocks, head, out=out)

    if arm == "mlp":
        w = ops.match_width(lambda c: ops.DenseNet(n, c, blocks, out_per_gene=out), target)
        return ops.DenseNet(n, w, blocks, out_per_gene=out)

    raise ValueError(arm)
