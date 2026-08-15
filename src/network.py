"""Building the regulatory network and its damaged variants.

Three selection choices, each forced by a diagnostic failure rather than chosen
up front:

  TF->TF links only     in the full network almost every gene is a leaf, nothing
                        feeds back, and the Laplacian spectrum is degenerate
  a community, not hubs the 64 highest-degree genes form a dense clump whose
                        spectrum is indistinguishable from a degree-matched
                        random graph, leaving the damage controls nothing to detect
  a density cap         pruning weakest-evidence links first, never disconnecting
"""

import os
import numpy as np
import networkx as nx
import pandas as pd

from . import config as C

CACHE = "collectri_raw.tsv"


def fetch_collectri():
    """Signed TF-target links from CollecTRI via the OmniPath REST endpoint.

    Deliberately not through decoupler: that package pulls NumPy past what Numba
    supports and breaks the runtime. The REST response is the same table.
    """
    p = C.path(CACHE)
    if os.path.exists(p):
        return pd.read_csv(p, sep="\t")
    url = ("https://omnipathdb.org/interactions?datasets=collectri"
           "&genesymbols=1&organisms=9606&fields=sources,references&format=tsv")
    df = pd.read_csv(url, sep="\t")
    df.to_csv(C.ensure(p), sep="\t", index=False)
    return df


def normalize_links(df, min_evidence):
    cols = {c.lower(): c for c in df.columns}
    src, tgt = cols.get("source_genesymbol", cols.get("source")), cols.get("target_genesymbol", cols.get("target"))
    if "mor" in cols:
        mor = df[cols["mor"]].astype(float)
    else:
        mor = df[cols["is_stimulation"]].astype(float) - df[cols["is_inhibition"]].astype(float)
    ev = (df[cols["sources"]].fillna("").astype(str).str.count(";") + 1
          if "sources" in cols else pd.Series(1, index=df.index))

    out = pd.DataFrame({"source": df[src].astype(str), "target": df[tgt].astype(str),
                        "mor": mor, "ev": ev})
    out = out[(out.mor != 0) & (out.source != out.target)]
    out["mor"] = np.sign(out.mor)
    g = out.groupby(["source", "target"]).agg(mor=("mor", "mean"), ev=("ev", "max")).reset_index()
    g = g[(np.abs(g.mor) == 1.0) & (g.ev >= min_evidence)]      # drop sign-conflicted
    g["mor"] = g.mor.astype(int)
    return g[["source", "target", "mor", "ev"]]


def select_subnetwork(links, n_genes, max_density, seed):
    tfs = set(links.source)
    G = nx.DiGraph()
    for s, t, m, ev in links[links.target.isin(tfs)].itertuples(index=False):
        G.add_edge(s, t, mor=int(m), ev=int(ev))
    G = G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()

    comms = sorted(nx.community.greedy_modularity_communities(nx.Graph(G)), key=len, reverse=True)
    big = [c for c in comms if len(c) >= n_genes]
    if big:
        nodes = set(min(big, key=len))
    else:
        nodes = set()
        for c in comms:
            nodes |= set(c)
            if len(nodes) >= n_genes:
                break

    H = G.subgraph(nodes).copy()
    H = H.subgraph(max(nx.weakly_connected_components(H), key=len)).copy()

    # Drop a node only if the result stays connected and stays large enough;
    # removing blindly and taking the largest component can undershoot n_genes.
    while H.number_of_nodes() > n_genes:
        cands = sorted(H.degree, key=lambda kv: (kv[1], kv[0]))[:40]
        for node, _ in cands:
            T = H.copy()
            T.remove_node(node)
            if T.number_of_nodes() >= n_genes and nx.is_weakly_connected(T):
                H = T
                break
        else:
            H.remove_node(cands[0][0])
            H = H.subgraph(max(nx.weakly_connected_components(H), key=len)).copy()
            if H.number_of_nodes() < n_genes:
                raise RuntimeError(f"community too small ({H.number_of_nodes()})")

    target_edges = int(max_density * n_genes * (n_genes - 1))
    for u, v, d in sorted(H.edges(data=True), key=lambda x: (x[2]["ev"], x[0], x[1])):
        if H.number_of_edges() <= target_edges:
            break
        H.remove_edge(u, v)
        if not nx.is_weakly_connected(H):
            H.add_edge(u, v, **d)

    genes = sorted(H.nodes)
    idx = {g: i for i, g in enumerate(genes)}
    signs = np.zeros((len(genes), len(genes)), np.int8)      # signs[i, j]: j -> i
    for s, t, d in H.edges(data=True):
        signs[idx[t], idx[s]] = d["mor"]
    return genes, signs


def rewire(signs, target_overlap, rng, max_attempts=400000, patience=200):
    """Degree-preserving rewiring by directed double-edge swaps.

    In-degree, out-degree and edge count are all preserved exactly, so density
    and degree sequence are identical across arms and only the wiring pattern
    changes. Swapping cannot randomise past a floor set by the degrees
    themselves, so callers should read the achieved overlap rather than assume
    the target was reached.
    """
    tgt, src = np.nonzero(signs)
    sign_vals = signs[tgt, src].copy()
    n_edges = len(src)
    tgt = tgt.copy()
    original = set(zip(tgt.tolist(), src.tolist()))
    current = set(original)
    if target_overlap >= 1.0:
        return signs.copy(), 1.0

    chunk, attempts, best, stall = max(1, n_edges // 50), 0, 1.0, 0
    min_attempts = 50 * n_edges
    overlap = 1.0
    while attempts < max_attempts:
        for _ in range(chunk):
            attempts += 1
            a, b = int(rng.integers(n_edges)), int(rng.integers(n_edges))
            if a == b:
                continue
            ta, tb = tgt[a], tgt[b]
            if ta == tb or src[a] == src[b]:
                continue
            if src[a] == tb or src[b] == ta:
                continue
            if (tb, src[a]) in current or (ta, src[b]) in current:
                continue
            current.discard((ta, src[a]))
            current.discard((tb, src[b]))
            tgt[a], tgt[b] = tb, ta
            current.add((tb, src[a]))
            current.add((ta, src[b]))
        overlap = len(original & current) / n_edges
        if overlap <= target_overlap:
            break
        if overlap < best - 1e-9:
            best, stall = overlap, 0
        else:
            stall += 1
            if stall >= patience and attempts >= min_attempts:
                break

    out = np.zeros_like(signs)
    out[tgt, src] = sign_vals
    assert (out != 0).sum() == n_edges
    assert np.array_equal((out != 0).sum(0), (signs != 0).sum(0))
    assert np.array_equal(np.sort((out != 0).sum(1)), np.sort((signs != 0).sum(1)))
    return out, overlap


def erdos_renyi(signs, rng):
    """Density-matched but degree-unmatched control.

    Against the degree-preserving rewiring this separates "the degree sequence
    matters" from "the wiring pattern matters", which a degree-preserving
    control alone cannot do.
    """
    n = len(signs)
    n_edges = int((signs != 0).sum())
    vals = signs[signs != 0]
    off = np.array([(i, j) for i in range(n) for j in range(n) if i != j])
    while True:
        out = np.zeros_like(signs)
        pick = rng.choice(len(off), size=n_edges, replace=False)
        out[off[pick, 0], off[pick, 1]] = rng.permutation(vals)
        A = np.maximum(np.abs(out) > 0, (np.abs(out) > 0).T).astype(float)
        if (A.sum(1) > 0).all():
            return out


def laplacian_eigs(signs, k=None):
    """Eigenvectors of the symmetric normalised Laplacian, sign-canonicalised.

    Eigenvectors are only defined up to a sign flip, so the largest-magnitude
    component of each is forced positive to make the basis reproducible.
    """
    A = (np.abs(signs) > 0).astype(float)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    deg = A.sum(1)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    L = np.eye(len(A)) - dinv[:, None] * A * dinv[None, :]
    w, V = np.linalg.eigh((L + L.T) / 2.0)
    if k:
        w, V = w[:k], V[:, :k]
    flip = np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(V.shape[1])])
    flip[flip == 0] = 1.0
    return w, V * flip[None, :]


def modularity_z_score(signs, rng, n_null=20):
    """Algebraic connectivity of the real network against a degree-matched null.

    A single rewiring is noise: the null spread is around 0.015, so the test has
    to be against a distribution.
    """
    lam_real = laplacian_eigs(signs, 2)[0][1]
    null = np.array([laplacian_eigs(rewire(signs, 0.0, rng, max_attempts=60000)[0], 2)[0][1]
                     for _ in range(n_null)])
    return lam_real, null.mean(), null.std(), (lam_real - null.mean()) / (null.std() + 1e-12)


def gene_orderings(eigvec, n_genes, seed=7):
    """Layouts for the 1-D Fourier arms.

    Fiedler sorts genes by the second Laplacian eigenvector so that adjacent
    positions tend to be adjacent in the network; the shuffled ordering is the
    control that separates "the Fourier basis helps" from "the ordering was
    sensible".
    """
    order = {"fiedler": np.argsort(eigvec[:, 1]),
             "shuffled": np.random.default_rng(seed).permutation(n_genes)}
    return order, {k: np.argsort(v) for k, v in order.items()}


def edge_list(signs):
    """Edge index and features for the graph neural operator."""
    tgt, src = np.nonzero(signs)
    din = np.maximum((signs != 0).sum(1), 1)
    dout = (signs != 0).sum(0)
    index = np.stack([src, tgt])
    attr = np.stack([signs[tgt, src].astype(np.float32),
                     np.log1p(dout[src]).astype(np.float32),
                     np.log1p(din[tgt]).astype(np.float32)], axis=1)
    return index, attr, din
