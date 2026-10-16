"""Which pairs of coordinates a generator may couple. docs/DESIGN_GRAPH.md.

    A_a = ( U diag(c_a) V + P_a Q_a ) * G         G fixed, 0/1, never learned

WHY THIS EXISTS. Arm E of the stage 2 ablation swapped the KEGG pathway rows for random
gene sets of the same sizes and L2 did not move (FINDINGS 21). KEGG sat at the entrance,
deciding which genes are averaged together, and the dense generator and readout behind it
mixed every coordinate again - a wrong grouping was simply undone. Here KEGG decides the
generator's SUPPORT, so an effect, and the interaction between two effects, can only
travel between coordinates the graph connects. B^n then sums the walks of length n on
the graph and exp(B) weights them by 1/n!.

THE GRAPH, on the K observable coordinates (pathway block first, then anchors):

    pathway i <- pathway j    the two share at least m genes
    pathway <-> anchor g      gene g belongs to the pathway
    anchor  <-> anchor        never; two genes meet through a pathway, two steps
    diagonal                  always

and on the constant coordinate, when there is one: its COLUMN is open - where a
perturbation first pushes is not restricted, DESIGN_GRAPH.md 2.3 measures why - and its
row is closed by the generator's own row mask, not here.

The pathway-pathway block has two sources (`generator_mask_source`). `shared` connects
pathways that share genes - a relation derived from the gene sets, not curated crosstalk,
and never called the latter; at m = 1 it lost to its own permutation (FINDINGS 22).
`maplink` connects a pathway to the maps its KEGG map draws an arrow to, read from the
KGML files - the curated relation, at 3.2 % density on norman fold 3 against 36.4 %.

THE CONTROLS. `permuted` relabels the kegg graph inside each block - pathways among
pathways, anchors among anchors - so the edge count and every degree are unchanged and
only WHICH pathways connect is wrong. `correlation` keeps the kegg graph's edge count per
block and fills it with the most correlated coordinate pairs of the training control
cells, which asks whether KEGG adds anything a correlation matrix of the data does not.
"""

from __future__ import annotations

import numpy as np
import torch

KINDS = ("none", "kegg", "permuted", "correlation")
# Where the pathway-pathway block of the KEGG graph comes from.
#   shared   two pathways share >= m genes - derived from the gene sets (FINDINGS 22)
#   maplink  one pathway's KEGG map draws a link to the other - the curated relation,
#            read from the KGML files (scripts/download_kegg.py --kgml)
SOURCES = ("shared", "maplink")


def coordinate_graph(support: np.ndarray, n_pathways: int,
                     min_shared: int) -> np.ndarray:
    """[K, K] bool, symmetric, diagonal on. `support` is M != 0, pathways first."""
    if min_shared < 1:
        raise ValueError(f"generator_mask_min_shared must be >= 1 (got {min_shared})")
    pathways = support[:n_pathways].astype(np.int64)
    return _with_anchors(pathways @ pathways.T >= min_shared, support, n_pathways)


def maplink_graph(pathways: list[str], links: dict[str, set[str]],
                  support: np.ndarray, n_pathways: int) -> np.ndarray:
    """The same graph with its pathway block taken from the maps' links.

    Symmetric: a map that draws an arrow to its neighbour connects the two both ways,
    since nothing in this model distinguishes upstream from downstream. Links to maps
    outside the observable space are dropped - they have no coordinate to land on.
    """
    if len(pathways) != n_pathways:
        raise ValueError(f"{len(pathways)} pathway ids for {n_pathways} pathway rows")
    index = {pathway: i for i, pathway in enumerate(pathways)}
    block = np.zeros((n_pathways, n_pathways), dtype=bool)
    for pathway, targets in links.items():
        if pathway not in index:
            continue
        for target in targets:
            if target in index:
                block[index[pathway], index[target]] = True
                block[index[target], index[pathway]] = True
    return _with_anchors(block, support, n_pathways)


def _with_anchors(pathway_block: np.ndarray, support: np.ndarray,
                  n_pathways: int) -> np.ndarray:
    """A pathway-pathway block, plus anchor-pathway membership and the diagonal."""
    support = support.astype(np.int64)
    k = support.shape[0]
    pathways = support[:n_pathways]
    graph = np.zeros((k, k), dtype=bool)
    graph[:n_pathways, :n_pathways] = pathway_block

    # An anchor row reads exactly one gene; it connects to the pathways holding it.
    anchor_genes = support[n_pathways:].argmax(axis=1)
    membership = pathways[:, anchor_genes] > 0                    # [P, A]
    graph[:n_pathways, n_pathways:] = membership
    graph[n_pathways:, :n_pathways] = membership.T

    np.fill_diagonal(graph, True)
    return graph


def permute_within_blocks(graph: np.ndarray, n_pathways: int,
                          seed: int) -> np.ndarray:
    """The same graph with its labels shuffled inside each block."""
    rng = np.random.default_rng(seed)
    k = graph.shape[0]
    order = np.concatenate([rng.permutation(n_pathways),
                            n_pathways + rng.permutation(k - n_pathways)])
    return graph[np.ix_(order, order)]


def _top_pairs(scores: np.ndarray, count: int) -> np.ndarray:
    """Indices of the `count` largest entries of a flat score array, ties broken by index."""
    if count <= 0:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort((np.arange(scores.size), -scores))
    return order[:count]


def correlation_graph(phi: np.ndarray, reference: np.ndarray,
                      n_pathways: int) -> np.ndarray:
    """As many edges per block as `reference`, chosen by |corr| over the rows of `phi`.

    `phi` is the training CONTROL cells' coordinates. Perturbed cells are left out on
    purpose: the question is whether KEGG knows something a correlation matrix of the
    unperturbed population does not.
    """
    k = reference.shape[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(phi.astype(np.float64), rowvar=False)
    strength = np.abs(np.nan_to_num(corr, nan=0.0))
    graph = np.zeros((k, k), dtype=bool)

    # Pathway-pathway, counted as unordered pairs so the result stays symmetric.
    upper_i, upper_j = np.triu_indices(n_pathways, k=1)
    wanted = int(reference[upper_i, upper_j].sum())
    pick = _top_pairs(strength[upper_i, upper_j], wanted)
    graph[upper_i[pick], upper_j[pick]] = True
    graph[upper_j[pick], upper_i[pick]] = True

    # Anchor-pathway.
    block = strength[:n_pathways, n_pathways:]
    wanted = int(reference[:n_pathways, n_pathways:].sum())
    pick = _top_pairs(block.ravel(), wanted)
    rows, cols = np.unravel_index(pick, block.shape)
    graph[rows, n_pathways + cols] = True
    graph[n_pathways + cols, rows] = True

    np.fill_diagonal(graph, True)
    return graph


def generator_mask(config: dict, observables, control_phi: np.ndarray | None,
                   constant: bool) -> tuple[torch.Tensor, dict]:
    """The [D, D] float mask a generator is multiplied by, and a summary of it.

    D = K + 1 with the constant coordinate. `none` returns all ones, so a model built
    with it is the stage 2 model to the bit.
    """
    cfg = config["model"]
    kind = str(cfg.get("generator_mask", "none"))
    if kind not in KINDS:
        raise ValueError(f"unknown model.generator_mask {kind!r} ({' | '.join(KINDS)})")
    k = int(observables.dim)
    d = k + int(constant)
    if kind == "none":
        return torch.ones(d, d), {"kind": "none"}

    # A mask means "pathway i and pathway j" only when coordinate i IS pathway i.
    if cfg.get("observables") != "kegg":
        raise ValueError(f"model.generator_mask={kind} needs model.observables=kegg "
                         f"(got {cfg.get('observables')!r})")
    if cfg.get("observable_whiten", "none") != "none":
        raise ValueError(f"model.generator_mask={kind} needs model.observable_whiten=none:"
                         f" whitening mixes the coordinates, and a mask on mixed "
                         f"coordinates is not the KEGG graph (DESIGN_GRAPH.md 2.4)")

    n_pathways = int(observables.n_pathways)
    support = (observables.matrix.detach().cpu().numpy() != 0)
    source = str(cfg.get("generator_mask_source", "shared"))
    if source not in SOURCES:
        raise ValueError(f"unknown model.generator_mask_source {source!r} "
                         f"({' | '.join(SOURCES)})")
    min_shared = int(cfg.get("generator_mask_min_shared", 5))
    if source == "shared":
        kegg = coordinate_graph(support, n_pathways, min_shared)
    else:
        from ..data.kegg import load_map_links, pathway_ids

        kegg_dir = config["data"]["kegg_dir"]
        pathways = pathway_ids(kegg_dir, list(observables.names[:n_pathways]))
        kegg = maplink_graph(pathways, load_map_links(kegg_dir, pathways), support,
                             n_pathways)
        min_shared = None                     # not a parameter of this graph
    if kind == "kegg":
        graph = kegg
    elif kind == "permuted":
        seed = cfg.get("generator_mask_seed")
        seed = int(config["train"]["seed"] if seed is None else seed)
        graph = permute_within_blocks(kegg, n_pathways, seed)
    else:
        if control_phi is None:
            raise ValueError("model.generator_mask=correlation needs the training "
                             "control cells' coordinates")
        graph = correlation_graph(control_phi, kegg, n_pathways)

    mask = torch.zeros(d, d)
    mask[:k, :k] = torch.from_numpy(graph.astype(np.float32))
    if constant:
        # The constant's column - each perturbation's own push - is open; its row is
        # left open here and closed by the generator's row mask, so the two masks
        # each say one thing.
        mask[:, k] = 1.0
        mask[k, :] = 1.0
    off = graph.copy()
    np.fill_diagonal(off, False)
    summary = {"kind": kind, "source": source, "min_shared": min_shared,
               "edges": int(off.sum()),
               "density": float(off.sum() / max(k * (k - 1), 1)),
               "pathway_pathway_edges": int(off[:n_pathways, :n_pathways].sum()),
               "anchor_pathway_edges": int(off[n_pathways:, :n_pathways].sum()),
               "isolated": int((off.sum(axis=1) == 0).sum())}
    return mask, summary
