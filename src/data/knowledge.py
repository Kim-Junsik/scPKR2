"""What is known about a perturbation, as a fixed vector k_a - docs/DESIGN_ENCODER.md 2.

    k_a  STRING spectral embedding of the target gene     model.perturbation_knowledge=string
         ESM-2 embedding of the target gene's protein      model.perturbation_knowledge=esm
         the two side by side                              model.perturbation_knowledge=string_esm
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

KINDS = ("none", "zero", "string", "moa", "esm", "string_esm")


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
            raise FileNotFoundError(
                f"{path} is missing (data.string_dir). The STRING files are not in git; "
                f"copy data/networks/string_info.txt.gz and string_links.txt.gz from the "
                f"machine that has them - the same files, so the embedding is the same")
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


def esm_embedding(table: str, dim: int) -> tuple[list[str], np.ndarray]:
    """(gene symbols, [n, dim]) - the top principal components of the ESM-2 table.

    The table is scripts/build_esm_knowledge.py's output: one mean-pooled ESM-2 vector
    per gene's protein, 1,280 numbers each. Reduced here the way the STRING embedding is
    built: principal components over the table's own genes (a fixed list, the same for
    every split, fold and seed), each component's sign fixed by its largest entry, and
    each column standardised. Nothing here sees expression data.

    THE SCALE. After standardising, the block is divided by sqrt(dim), so a gene's
    vector has a root-mean-square norm of 1. The STRING block's median norm over the
    Norman genes it covers is 1.06. Left at sqrt(64) = 8, the ESM half would move the
    encoder eight times faster than STRING under Adam and would decide string_esm alone.
    """
    if not os.path.exists(table):
        raise FileNotFoundError(
            f"{table} is missing (data.esm_table). Build it with "
            f"scripts/build_esm_knowledge.py, or git pull the committed one")
    saved = np.load(table, allow_pickle=False)
    genes = [str(g) for g in saved["genes"]]
    centred = saved["vectors"].astype(np.float64)
    centred = centred - centred.mean(axis=0)
    left, values, _ = np.linalg.svd(centred, full_matrices=False)
    # Centring costs one dimension: n genes give at most n - 1 components.
    dim = min(int(dim), len(genes) - 1)
    scores = left[:, :dim] * values[:dim]
    signs = np.sign(scores[np.abs(scores).argmax(axis=0), np.arange(dim)])
    scores = scores * np.where(signs == 0, 1.0, signs)
    scores = scores / np.maximum(scores.std(axis=0), 1e-12) / np.sqrt(dim)
    return genes, scores.astype(np.float32)


def _rows(names: list[str], vectors: np.ndarray, perturbations: list[str]) -> np.ndarray:
    """vectors re-ordered to `perturbations`, zero for a perturbation the source lacks."""
    row = {name: i for i, name in enumerate(names)}
    matrix = np.zeros((len(perturbations), vectors.shape[1]), dtype=np.float32)
    for i, gene in enumerate(perturbations):
        if gene in row:
            matrix[i] = vectors[row[gene]]
    return matrix


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


def read_mechanisms(config: dict) -> dict[str, str]:
    """drug -> mechanism: the committed table if it exists, else the raw ComboSciPlex obs.

    assets/knowledge/combosciplex_moa.csv is drug_mechanisms() run once on the raw file
    and committed - 17 rows - so a machine holding only the caches can still build the
    knowledge. It is the annotation as written, nothing corrected.
    """
    table = config["data"].get("moa_table")
    if table and os.path.exists(table):
        frame = pd.read_csv(table)
        if list(frame.columns) != ["drug", "mechanism"]:
            raise ValueError(f"{table}: expected columns drug, mechanism")
        return dict(zip(frame["drug"].astype(str), frame["mechanism"].astype(str)))
    raw = config["data"]["raw_h5ad"]
    if not os.path.exists(raw):
        raise FileNotFoundError(f"neither {table} nor {raw} exists - no mechanism table")
    return drug_mechanisms(raw)


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
    elif kind in ("string", "esm", "string_esm"):
        # string_esm puts the two blocks side by side; a gene one source lacks has zeros
        # in that block only.
        blocks = []
        if kind in ("string", "string_esm"):
            blocks.append(_rows(*string_embedding(config["data"]["string_dir"],
                                                  int(cfg.get("knowledge_min_score", 700)),
                                                  int(cfg.get("knowledge_dim", 64))),
                                perturbations))
        if kind in ("esm", "string_esm"):
            blocks.append(_rows(*esm_embedding(config["data"]["esm_table"],
                                               int(cfg.get("knowledge_esm_dim", 64))),
                                perturbations))
        matrix = np.concatenate(blocks, axis=1)
    else:
        mechanism = read_mechanisms(config)
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
