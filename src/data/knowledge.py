"""What is known about a perturbation, as a fixed vector k_a - docs/DESIGN_ENCODER.md 2.

    k_a  STRING spectral embedding of the target gene     model.perturbation_knowledge=string
         one-hot of the drug's annotated mechanism         model.perturbation_knowledge=moa
         a single zero (the encoder with nothing to say)   model.perturbation_knowledge=zero

The encoder (models/operator.py) turns k_a into the perturbation's generator
coefficients, c_a = c_bar + E k_a + delta_a. Nothing here is learned and nothing reads
cells: k_a is the same whatever split, fold or seed it is used under, which is what lets
a perturbation the training data never contained still have one.

A perturbation the source does not cover gets k_a = 0, so its generator is the shared
c_bar - the average perturbation. That is the honest fallback, and it is also what the
"no knowledge" arm gives every perturbation.

THE CONTROL (model.knowledge_shuffle). The rows of K are permuted among the perturbations
that HAVE knowledge: the same set of vectors, attached to the wrong perturbations, and
the zero rows left where they are. For the mechanism one-hots that is a relabelling that
keeps every mechanism's size.
"""

from __future__ import annotations

import gzip
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

KINDS = ("none", "zero", "string", "moa")


def string_embedding(string_dir: str, min_score: int, dim: int
                     ) -> tuple[list[str], np.ndarray]:
    """(gene symbols, [n, dim]) - the top non-trivial eigenvectors of STRING's
    normalised adjacency at combined_score >= min_score.

    Deterministic: ARPACK starts from a fixed vector and every eigenvector's sign is
    fixed by its largest entry, so two machines given the same files agree. Each column
    is standardised over the proteins. Nothing here sees expression data.
    """
    info_path = os.path.join(string_dir, "string_info.txt.gz")
    links_path = os.path.join(string_dir, "string_links.txt.gz")
    for path in (info_path, links_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} is missing (data.string_dir)")
    with gzip.open(info_path, "rt", encoding="utf-8") as handle:
        info = pd.read_csv(handle, sep="\t")
    name_column = next(c for c in info.columns if "preferred_name" in c)
    name_of = dict(zip(info[info.columns[0]], info[name_column]))

    links = pd.read_csv(links_path, sep=" ")
    links = links[links["combined_score"] >= min_score]
    proteins = sorted(set(links["protein1"]) | set(links["protein2"]))
    index = {p: i for i, p in enumerate(proteins)}
    rows = links["protein1"].map(index).to_numpy()
    cols = links["protein2"].map(index).to_numpy()
    n = len(proteins)
    adjacency = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    adjacency = ((adjacency + adjacency.T) > 0).astype(np.float64)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    scale = sp.diags(1.0 / np.sqrt(np.maximum(degree, 1.0)))
    normalised = scale @ adjacency @ scale

    values, vectors = eigsh(normalised, k=dim + 1, which="LA",
                            v0=np.ones(n) / np.sqrt(n), tol=1e-8)
    order = np.argsort(-values)
    vectors = vectors[:, order[1:]]                    # drop the trivial top one
    signs = np.sign(vectors[np.abs(vectors).argmax(axis=0), np.arange(vectors.shape[1])])
    vectors = vectors * np.where(signs == 0, 1.0, signs)
    vectors = (vectors - vectors.mean(axis=0)) / np.maximum(vectors.std(axis=0), 1e-12)

    names, keep, seen = [], [], set()
    for i, protein in enumerate(proteins):
        name = name_of.get(protein)
        if name is not None and name not in seen:  # first protein wins a repeated name
            seen.add(name)
            names.append(name)
            keep.append(i)
    return names, vectors[keep].astype(np.float32)


def drug_mechanisms(raw_h5ad: str) -> dict[str, str]:
    """drug -> annotated mechanism, from ComboSciPlex's obs Drug1/Drug2/pathway1/2.

    Used as written. The annotation is coarse - four kinase inhibitors are all 'EGFR
    inhibitor', and the SRT compounds (SIRT1 activators) read 'Sirtuin inhibitor' - and
    correcting it by hand would be a choice made after looking at the data.
    """
    import anndata as ad

    obs = ad.read_h5ad(raw_h5ad, backed="r").obs
    needed = ("Drug1", "Drug2", "pathway1", "pathway2")
    if not all(c in obs.columns for c in needed):
        raise ValueError(f"{raw_h5ad} has no {needed} columns - not ComboSciPlex?")
    mechanism: dict[str, str] = {}
    for drug_column, path_column in (("Drug1", "pathway1"), ("Drug2", "pathway2")):
        pairs = obs[[drug_column, path_column]].astype(str).drop_duplicates()
        for drug, path in zip(pairs[drug_column], pairs[path_column]):
            if path == "Vehicle":
                continue
            if mechanism.get(drug, path) != path:
                raise ValueError(f"{drug} has two mechanisms: {mechanism[drug]}, {path}")
            mechanism[drug] = path
    return mechanism


def shuffle_rows(matrix: np.ndarray, seed: int) -> np.ndarray:
    """The control: rows permuted among the non-zero rows, zero rows left in place.

    The same vectors, attached to the wrong perturbations. Coverage does not change, so
    the control differs from the real knowledge in WHICH perturbation gets which vector
    and in nothing else.
    """
    covered = np.flatnonzero(np.abs(matrix).sum(axis=1) > 0)
    out = matrix.copy()
    out[covered] = matrix[np.random.default_rng(seed).permutation(covered)]
    return out


def knowledge_matrix(config: dict, perturbations: list[str]
                     ) -> tuple[np.ndarray | None, dict]:
    """K, [P, d], rows in `perturbations` order - or None for the lookup-table model."""
    cfg = config["model"]
    kind = str(cfg.get("perturbation_knowledge", "none"))
    if kind not in KINDS:
        raise ValueError(f"unknown model.perturbation_knowledge {kind!r} "
                         f"({' | '.join(KINDS)})")
    if kind == "none":
        return None, {"kind": "none"}
    if kind == "zero":
        matrix = np.zeros((len(perturbations), 1), dtype=np.float32)
    elif kind == "string":
        names, vectors = string_embedding(config["data"]["string_dir"],
                                          int(cfg.get("knowledge_min_score", 700)),
                                          int(cfg.get("knowledge_dim", 64)))
        row = {name: i for i, name in enumerate(names)}
        matrix = np.zeros((len(perturbations), vectors.shape[1]), dtype=np.float32)
        for i, gene in enumerate(perturbations):
            if gene in row:
                matrix[i] = vectors[row[gene]]
    else:
        mechanism = drug_mechanisms(config["data"]["raw_h5ad"])
        classes = sorted(set(mechanism.values()))
        matrix = np.zeros((len(perturbations), len(classes)), dtype=np.float32)
        for i, drug in enumerate(perturbations):
            if drug in mechanism:
                matrix[i, classes.index(mechanism[drug])] = 1.0

    covered = np.flatnonzero(np.abs(matrix).sum(axis=1) > 0)
    shuffled = bool(cfg.get("knowledge_shuffle", False))
    if shuffled:
        seed = cfg.get("knowledge_seed")
        matrix = shuffle_rows(matrix, int(config["train"]["seed"] if seed is None
                                          else seed))
    summary = {"kind": kind, "dim": int(matrix.shape[1]), "shuffled": shuffled,
               "covered": int(len(covered)), "perturbations": len(perturbations)}
    return matrix, summary
