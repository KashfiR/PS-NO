"""Five operator families written as one kernel layer.

Following Kovachki et al. (2023), each model below except the neural ODEs is a
lift, a stack of kernel-integral layers and a projection, where the layer is

    v <- sigma( W v(x) + integral kappa(x, y) v(y) dy ).

The families differ only in kappa. Everything else -- lift, head, nonlinearity,
normalisation and the local path Wv -- is identical, which is what makes the
comparison a comparison.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C


def lambda_features(lam, n_freq):
    """Fourier features of the eigenvalue.

    lambda lies in [0, 2] for the normalised Laplacian on any graph of any size,
    which is what lets one filter serve networks it was not trained on.
    """
    f = (2.0 ** torch.arange(n_freq, dtype=torch.float32)) * math.pi
    z = lam[:, None] * f[None, :]
    return torch.cat([lam[:, None], torch.sin(z), torch.cos(z)], dim=-1)


class SpectralBlock(nn.Module):
    """Kernel diagonal in the graph eigenbasis.

    The filter reads the eigenvalue, not the mode index. An index-keyed filter
    would fail twice: mode number means nothing across networks, so it could
    never move to another gene set, and eigenvectors are defined only up to sign
    and up to rotation within a repeated eigenvalue, so the model would change
    if the solver returned an equally valid basis. Keying on lambda makes sign
    flips cancel and repeated eigenvalues receive equal treatment.
    """

    def __init__(self, channels, basis, hidden, n_freq):
        super().__init__()
        self.R = nn.Parameter(torch.randn(basis, channels, channels) / math.sqrt(channels))
        self.g = nn.Sequential(nn.Linear(1 + 2 * n_freq, hidden), nn.GELU(),
                               nn.Linear(hidden, basis))
        self.W = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def filter_matrices(self, lam_feat):
        return torch.einsum("km,mcd->kcd", self.g(lam_feat), self.R)

    def forward(self, v, phi, lam_feat):
        R = self.filter_matrices(lam_feat)
        vh = torch.einsum("bnc,nk->bkc", v, phi)
        vh = torch.einsum("bkc,kcd->bkd", vh, R)
        return self.norm(F.gelu(torch.einsum("bkc,nk->bnc", vh, phi) + self.W(v)))


class SpectralOperator(nn.Module):
    def __init__(self, channels, blocks, head, basis, hidden, n_freq, in_ch=5, out=1):
        super().__init__()
        self.n_freq = n_freq
        self.lift = nn.Linear(in_ch, channels)
        self.blocks = nn.ModuleList(
            [SpectralBlock(channels, basis, hidden, n_freq) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Linear(channels, head), nn.GELU(), nn.Linear(head, out))

    def forward(self, x, geom):
        phi, lam = geom
        lam_feat = lambda_features(lam, self.n_freq)
        v = self.lift(x)
        for b in self.blocks:
            v = b(v, phi, lam_feat)
        return self.head(v).squeeze(-1)


def sgwt_profiles(lam, n_scales):
    """Spectral graph wavelet kernels of Hammond et al. (2011).

    A wavelet at scale s is g(sL) = Phi g(s lambda) Phi^T. Returns one low-pass
    scaling function followed by n_scales band-pass profiles, using the design
    kernel g(x) = x exp(1 - x), which peaks at x = 1 and vanishes at zero, with
    scales log-spaced over the spectrum.
    """
    lam = np.asarray(lam, np.float64)
    lmax = float(lam.max())
    lmin = lmax / 20.0
    scales = np.exp(np.linspace(np.log(2.0 / lmax), np.log(2.0 / lmin), n_scales))[::-1]
    band = np.stack([(s * lam) * np.exp(1.0 - s * lam) for s in scales])
    low = np.exp(-((lam / (0.6 * lmin)) ** 4))
    return np.concatenate([low[None, :], band], 0).astype(np.float32), scales


class WaveletBlock(nn.Module):
    """Kernel built from a fixed multiscale dictionary.

    Only the per-scale channel mixings are learned; the profiles over lambda are
    set by the wavelet construction. That is the one structural difference from
    the spectral block, where the profile is itself learned.
    """

    def __init__(self, channels, n_profiles):
        super().__init__()
        self.A = nn.Parameter(torch.randn(n_profiles, channels, channels) / math.sqrt(channels))
        self.W = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, v, phi, profiles):
        vh = torch.einsum("bnc,nk->bkc", v, phi)
        mixed = torch.einsum("jk,bkc,jcd->bkd", profiles, vh, self.A)
        return self.norm(F.gelu(torch.einsum("bkc,nk->bnc", mixed, phi) + self.W(v)))


class WaveletOperator(nn.Module):
    def __init__(self, channels, blocks, head, n_profiles, in_ch=5, out=1):
        super().__init__()
        self.lift = nn.Linear(in_ch, channels)
        self.blocks = nn.ModuleList([WaveletBlock(channels, n_profiles) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Linear(channels, head), nn.GELU(), nn.Linear(head, out))

    def forward(self, x, geom):
        phi, profiles = geom
        v = self.lift(x)
        for b in self.blocks:
            v = b(v, phi, profiles)
        return self.head(v).squeeze(-1)


class FourierBlock(nn.Module):
    """The standard FNO spectral convolution along the gene axis, via rfft."""

    def __init__(self, channels, modes):
        super().__init__()
        self.modes = modes
        self.wr = nn.Parameter(torch.randn(modes, channels, channels) / channels)
        self.wi = nn.Parameter(torch.randn(modes, channels, channels) / channels)
        self.W = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, v):
        n = v.shape[1]
        vh = torch.fft.rfft(v, dim=1)
        m = min(self.modes, vh.shape[1])
        w = torch.complex(self.wr[:m], self.wi[:m])
        out = torch.zeros_like(vh)
        out[:, :m] = torch.einsum("bmc,mcd->bmd", vh[:, :m], w)
        return self.norm(F.gelu(torch.fft.irfft(out, n=n, dim=1) + self.W(v)))


class FourierOperator1D(nn.Module):
    """Genes are permuted into the given order, run through FNO blocks, then
    permuted back so the output is aligned with the original gene index."""

    def __init__(self, channels, blocks, head, modes, order, inverse, in_ch=5, out=1):
        super().__init__()
        self.register_buffer("order", torch.tensor(order, dtype=torch.long))
        self.register_buffer("inverse", torch.tensor(inverse, dtype=torch.long))
        self.lift = nn.Linear(in_ch, channels)
        self.blocks = nn.ModuleList([FourierBlock(channels, modes) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Linear(channels, head), nn.GELU(), nn.Linear(head, out))

    def forward(self, x, geom=None):
        v = self.lift(x[:, self.order, :])
        for b in self.blocks:
            v = b(v)
        return self.head(v).squeeze(-1)[:, self.inverse]


class NeighbourhoodBlock(nn.Module):
    """Kernel integrated over graph neighbourhoods (Li et al. 2020).

    kappa is a network on edge features returning a full C x C matrix, so the
    kernel genuinely depends on the edge rather than being a scalar weight.
    """

    def __init__(self, channels, edge_dim, hidden):
        super().__init__()
        self.channels = channels
        self.kernel = nn.Sequential(nn.Linear(edge_dim, hidden), nn.GELU(),
                                    nn.Linear(hidden, channels * channels))
        self.W = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, v, edge_index, edge_attr, degree):
        src, dst = edge_index
        K = self.kernel(edge_attr).view(-1, self.channels, self.channels)
        msg = torch.einsum("bec,ecd->bed", v[:, src, :], K)
        agg = torch.zeros_like(v).index_add_(1, dst, msg) / degree[None, :, None]
        return self.norm(F.gelu(agg + self.W(v)))


class GraphNeuralOperator(nn.Module):
    def __init__(self, channels, blocks, head, edge_dim, hidden, in_ch=5, out=1):
        super().__init__()
        self.lift = nn.Linear(in_ch, channels)
        self.blocks = nn.ModuleList(
            [NeighbourhoodBlock(channels, edge_dim, hidden) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Linear(channels, head), nn.GELU(), nn.Linear(head, out))

    def forward(self, x, geom):
        edge_index, edge_attr, degree = geom
        v = self.lift(x)
        for b in self.blocks:
            v = b(v, edge_index, edge_attr, degree)
        return self.head(v).squeeze(-1)


class MessagePassingBlock(nn.Module):
    """Activating and repressing adjacencies get separate weights, so this
    baseline sees the sign information the symmetrised eigenbasis discards."""

    def __init__(self, channels):
        super().__init__()
        self.Wa = nn.Linear(channels, channels, bias=False)
        self.Wr = nn.Linear(channels, channels, bias=False)
        self.Ws = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, v, adj_act, adj_rep):
        return self.norm(F.gelu(torch.einsum("ij,bjc->bic", adj_act, self.Wa(v))
                                + torch.einsum("ij,bjc->bic", adj_rep, self.Wr(v))
                                + self.Ws(v)))


class GraphConvNet(nn.Module):
    def __init__(self, channels, blocks, head, in_ch=5, out=1):
        super().__init__()
        self.lift = nn.Linear(in_ch, channels)
        self.blocks = nn.ModuleList([MessagePassingBlock(channels) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Linear(channels, head), nn.GELU(), nn.Linear(head, out))

    def forward(self, x, geom):
        adj_act, adj_rep = geom
        v = self.lift(x)
        for b in self.blocks:
            v = b(v, adj_act, adj_rep)
        return self.head(v).squeeze(-1)


class DenseNet(nn.Module):
    """No graph, and structurally tied to one gene count: it cannot be evaluated
    on another domain at all, which is itself a result."""

    def __init__(self, n_genes, width, blocks, in_ch=5, out_per_gene=1):
        super().__init__()
        self.n_genes, self.out_per_gene = n_genes, out_per_gene
        layers = [nn.Linear(n_genes * in_ch, width), nn.GELU()]
        for _ in range(blocks - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers += [nn.Linear(width, n_genes * out_per_gene)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, geom=None):
        if x.shape[1] != self.n_genes:
            raise RuntimeError("dense baseline is fixed to one gene count by construction")
        out = self.net(x.flatten(1))
        return out if self.out_per_gene == 1 else out.view(-1, self.n_genes, self.out_per_gene)


def rk4_schedule(t_eval, task="B"):
    """Step sizes and the indices where an observation falls.

    Steps grow with time: the transient is over within a few time units and the
    system then relaxes to a fixed point, so fine early and coarse late costs far
    less at no real accuracy loss. Task A needs only the final state and takes
    the cheaper schedule; Task B must land on all sixteen observation times.

    The cap is set by stability. Fixed-step RK4 holds while h |lambda| < 2.78,
    so h_max = 2 is safe for contraction rates up to 1.39, above the decay rates
    in the data.
    """
    o = C.ODE
    horizon = o.get("t_max_task_a") or float(t_eval[-1])

    def fill(a, b, out):
        t = float(a)
        while t < b - 1e-9:
            h = min(o["h0"] * (1 + o["growth"] * t), o["h_max"], b - t)
            out.append(h)
            t += h

    steps, obs = [], [0]
    if task == "A":
        fill(0.0, horizon, steps)
        obs.append(len(steps))
    else:
        for a, b in zip(t_eval[:-1], t_eval[1:]):
            fill(a, b, steps)
            obs.append(len(steps))
    return np.array(steps, np.float32), np.array(obs, np.int64)


class FieldDense(nn.Module):
    def __init__(self, n_genes, width, blocks, in_ch=5):
        super().__init__()
        layers = [nn.Linear(n_genes * in_ch, width), nn.GELU()]
        for _ in range(blocks - 1):
            layers += [nn.Linear(width, width), nn.GELU()]
        layers += [nn.Linear(width, n_genes)]
        self.net = nn.Sequential(*layers)

    def forward(self, z, ctx):
        return self.net(torch.cat([z.unsqueeze(-1), ctx], -1).flatten(1))


class FieldGraph(nn.Module):
    def __init__(self, channels, blocks, head, in_ch=5):
        super().__init__()
        self.lift = nn.Linear(in_ch, channels)
        self.blocks = nn.ModuleList([MessagePassingBlock(channels) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Linear(channels, head), nn.GELU(), nn.Linear(head, 1))

    def forward(self, z, ctx, adj_act, adj_rep):
        v = self.lift(torch.cat([z.unsqueeze(-1), ctx], -1))
        for b in self.blocks:
            v = b(v, adj_act, adj_rep)
        return self.head(v).squeeze(-1)


class NeuralODE(nn.Module):
    """dz/dt = f(z, context) - decay * z, integrated by fixed-step RK4 with
    gradients taken straight through the solver.

    Discretise-then-optimise rather than the adjoint: the adjoint saves memory,
    not time, and memory is not the binding constraint at 64 states.

    The initial condition is rescaled into the target's coordinates. The input
    channel is standardised with the statistics of the starting state while the
    target has its own, and since the trajectory target at t = 0 *is* the
    starting state, an unrescaled state is forced wrong at t = 0 by a fixed
    affine factor it cannot correct. Every other model has a head between
    features and output and is unaffected; here the state is the output.

    nfe is counted analytically, four per RK4 step.
    """

    def __init__(self, field, steps, obs_idx, kind, use_decay, n_genes,
                 in_mu, in_sd, out_mu, out_sd):
        super().__init__()
        self.field, self.kind, self.use_decay = field, kind, use_decay
        self.register_buffer("steps", torch.as_tensor(steps))
        self.register_buffer("obs_idx", torch.as_tensor(obs_idx))
        for name, v in [("in_mu", in_mu), ("in_sd", in_sd),
                        ("out_mu", out_mu), ("out_sd", out_sd)]:
            self.register_buffer(name, v.clone())
        if use_decay:
            self.decay = nn.Parameter(torch.zeros(n_genes))
        self.nfe = 4 * len(steps)

    def _f(self, z, ctx, geom):
        out = self.field(z, ctx) if self.kind == "dense" else self.field(z, ctx, *geom)
        return out - F.softplus(self.decay) * z if self.use_decay else out

    def _step(self, z, ctx, geom, h):
        k1 = self._f(z, ctx, geom)
        k2 = self._f(z + 0.5 * h * k1, ctx, geom)
        k3 = self._f(z + 0.5 * h * k2, ctx, geom)
        k4 = self._f(z + h * k3, ctx, geom)
        return z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def forward(self, x, geom=None, return_traj=False):
        z = (x[..., 0] * self.in_sd + self.in_mu - self.out_mu) / self.out_sd
        ctx = x[..., 1:]
        obs = set(self.obs_idx.tolist())
        out = [z] if return_traj else None
        for i, h in enumerate(self.steps):
            z = self._step(z, ctx, geom, h)
            if return_traj and (i + 1) in obs:
                out.append(z)
        return torch.stack(out, -1) if return_traj else z


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def match_width(builder, target, lo=8, hi=1024):
    """Smallest hidden width whose parameter count is closest to the target."""
    best, best_w = None, lo
    while lo <= hi:
        mid = (lo + hi) // 2
        n = n_params(builder(mid))
        if best is None or abs(n - target) < abs(best - target):
            best, best_w = n, mid
        lo, hi = (mid + 1, hi) if n < target else (lo, mid - 1)
    return best_w
