"""Build the Virtual Cell Challenge 2025 H1 cache - docs/DESIGN_ENCODER.md 3.1.

    python scripts/vcc_prepare.py --root data/vcc/2025
    python scripts/vcc_prepare.py --root nvcc/data/vcc/2025 --out-dir data/vcc

Input: the three released files (Arc Virtual Cell Atlas, CC0), raw counts as CSR,
obs['target_gene'] with 'non-targeting' for the control:

    <root>/train/adata_Training.h5ad        150 genes + controls
    <root>/validation/adata_Validation.h5ad  50 genes + controls
    <root>/test/adata_Test.h5ad             100 genes + controls

About 300,000 cells by 18,080 genes - too large to hold as one matrix, so every pass
reads row chunks straight from the HDF5 file:

  pass 1  TRAINING FILE ONLY. Library sizes, giving S = their median, and per-gene
          moments of count / library. scanpy's seurat-flavour highly-variable-gene rule
          on those moments picks n_hvg genes. Validation and test cells never reach the
          gene choice, as test conditions never reach it for Norman
          (data.exclude_test_from_hvg).
  pass 2  all three files. log1p(count / library * S), the recipe the other datasets use
          (normalize_total at the median library, then log1p). The cache keeps the chosen
          genes plus every target gene - the targets are part of the task statement,
          not of the test data. The evaluation file keeps per-condition means over ALL
          genes, because the challenge scores the whole transcriptome.

Writes:
    <out-dir>/vcc2025_cache.h5ad   X log1p-normalised, obs condition ('GENE+ctrl' |
                                   'ctrl'), vcc_set, target_gene, batch
    <out-dir>/vcc2025_means.npz    conditions, vcc_set, n_cells, means [C, G_all],
                                   gene_names, library_median

Control cells from all three files are pooled into one 'ctrl' condition: they carry no
perturbation, and the challenge hands its participants control cells to predict from.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

FILES = (("train", "train/adata_Training.h5ad"),
         ("validation", "validation/adata_Validation.h5ad"),
         ("test", "test/adata_Test.h5ad"))
CONTROL_SOURCE = "non-targeting"
CONTROL = "ctrl"


def _categorical(group: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    categories = np.array([c.decode() if isinstance(c, bytes) else str(c)
                           for c in group["categories"][:]])
    return categories, group["codes"][:]


def _strings(dataset: h5py.Dataset) -> np.ndarray:
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in dataset[:]])


class CountFile:
    """Row-chunked access to one released h5ad without loading X."""

    def __init__(self, path: str):
        self.path = path
        self.handle = h5py.File(path, "r")
        x = self.handle["X"]
        if dict(x.attrs).get("encoding-type") != "csr_matrix":
            raise ValueError(f"{path}: expected X as csr_matrix, found "
                             f"{dict(x.attrs).get('encoding-type')}")
        self.shape = tuple(int(v) for v in x.attrs["shape"])
        self.indptr = x["indptr"][:]
        self.data, self.indices = x["data"], x["indices"]
        obs = self.handle["obs"]
        categories, codes = _categorical(obs["target_gene"])
        self.target = categories[codes]
        if "batch" in obs:
            categories, codes = _categorical(obs["batch"])
            self.batch = categories[codes]
        else:
            self.batch = np.full(self.shape[0], "", dtype=object)
        var = self.handle["var"]
        self.genes = _strings(var[dict(var.attrs).get("_index", "_index")])

    def chunks(self, size: int):
        for start in range(0, self.shape[0], size):
            stop = min(start + size, self.shape[0])
            lo, hi = int(self.indptr[start]), int(self.indptr[stop])
            matrix = sp.csr_matrix(
                (self.data[lo:hi].astype(np.float32, copy=False),
                 self.indices[lo:hi], self.indptr[start:stop + 1] - lo),
                shape=(stop - start, self.shape[1]))
            yield start, stop, matrix

    def close(self) -> None:
        self.handle.close()


def seurat_hvg(mean: np.ndarray, var: np.ndarray, n_top: int) -> np.ndarray:
    """scanpy's flavor='seurat' rule from per-gene moments of normalised counts.

    The same arithmetic as scanpy.pp.highly_variable_genes(flavor='seurat'), which works
    on expm1 of the log data, i.e. on exactly these normalised counts: log dispersion,
    log1p mean, 20 mean bins, dispersion z-scored within its bin, top n_top.
    """
    mean = mean.copy()
    mean[mean == 0] = 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        dispersion = var / mean
        dispersion[dispersion == 0] = np.nan
        dispersion = np.log(dispersion)
    frame = pd.DataFrame({"mean": np.log1p(mean), "dispersion": dispersion})
    frame["bin"] = pd.cut(frame["mean"], bins=20)
    grouped = frame.groupby("bin", observed=False)["dispersion"]
    bin_mean, bin_std = grouped.mean(), grouped.std(ddof=1)
    single = bin_std.isnull()
    bin_std[single] = bin_mean[single]
    bin_mean[single] = 0.0
    normalised = ((frame["dispersion"] - bin_mean[frame["bin"]].to_numpy())
                  / bin_std[frame["bin"]].to_numpy()).to_numpy()
    normalised = np.where(np.isnan(normalised), -np.inf, normalised)
    return np.sort(np.argsort(-normalised, kind="stable")[:n_top])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="data/vcc/2025")
    parser.add_argument("--out-dir", default="data/vcc")
    parser.add_argument("--n-hvg", type=int, default=5000)
    parser.add_argument("--chunk", type=int, default=20000)
    args = parser.parse_args()

    files = []
    for name, relative in FILES:
        path = os.path.join(args.root, relative)
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}")
        files.append((name, CountFile(path)))
    genes = files[0][1].genes
    for name, handle in files[1:]:
        if not np.array_equal(handle.genes, genes):
            raise SystemExit(f"{name} has a different gene axis from train")
    n_genes = len(genes)
    print(f"{n_genes:,} genes; cells: "
          + ", ".join(f"{n} {h.shape[0]:,}" for n, h in files))

    # ---- pass 1: the training file decides S and the genes -------------------------
    t0 = time.time()
    train = files[0][1]
    libraries, first, second = [], np.zeros(n_genes), np.zeros(n_genes)
    for start, stop, matrix in train.chunks(args.chunk):
        library = np.asarray(matrix.sum(axis=1)).ravel()
        if (library <= 0).any():
            raise SystemExit(f"train rows {start}-{stop} hold a cell with no counts")
        libraries.append(library)
        scaled = sp.diags(1.0 / library) @ matrix
        first += np.asarray(scaled.sum(axis=0)).ravel()
        second += np.asarray(scaled.multiply(scaled).sum(axis=0)).ravel()
        print(f"  pass 1  {stop:,}/{train.shape[0]:,}  {time.time() - t0:.0f}s", flush=True)
    libraries = np.concatenate(libraries)
    median = float(np.median(libraries))
    n = train.shape[0]
    mean = first / n * median
    var = (second / n * median ** 2 - mean ** 2) * n / (n - 1)
    chosen = seurat_hvg(mean, np.maximum(var, 0.0), args.n_hvg)

    targets = sorted({t for _, h in files for t in np.unique(h.target)} - {CONTROL_SOURCE})
    column = {g: i for i, g in enumerate(genes)}
    missing = [t for t in targets if t not in column]
    keep = np.union1d(chosen, [column[t] for t in targets if t in column])
    print(f"  library median {median:.1f}; {len(chosen)} HVG + targets -> {len(keep)} genes"
          f"; {len(targets)} targets, {len(missing)} not measured {missing[:5]}")

    # ---- pass 2: every file, normalised, into the cache and the means --------------
    os.makedirs(args.out_dir, exist_ok=True)
    cache_path = os.path.join(args.out_dir, "vcc2025_cache.h5ad")
    means_path = os.path.join(args.out_dir, "vcc2025_means.npz")

    conditions = [CONTROL] + [f"{t}+{CONTROL}" for t in targets]
    index = {c: i for i, c in enumerate(conditions)}
    labels_of = {name: np.array([CONTROL if t == CONTROL_SOURCE else f"{t}+{CONTROL}"
                                 for t in handle.target]) for name, handle in files}
    set_of = {CONTROL: "control"}
    for name, handle in files:
        for t in np.unique(handle.target):
            if t != CONTROL_SOURCE:
                set_of[f"{t}+{CONTROL}"] = name
    obs = pd.concat([pd.DataFrame({"condition": labels_of[name], "vcc_set": name,
                                   "target_gene": handle.target, "batch": handle.batch})
                     for name, handle in files], ignore_index=True)
    obs.index = obs.index.astype(str)
    for column_name in obs.columns:
        obs[column_name] = obs[column_name].astype("category")
    n_cells = len(obs)
    build = {"source": os.path.abspath(args.root), "library_median": median,
             "recipe": "log1p(count / library * median library), median over the "
                       "training file",
             "n_hvg": args.n_hvg, "hvg": "seurat flavour on the training file only",
             "n_genes_kept": int(len(keep)), "targets": len(targets)}

    # The matrix never exists in memory as one object. anndata writes obs, var and uns
    # around an empty X; X is then rebuilt in place, chunk by chunk, in the csr_matrix
    # encoding anndata reads back.
    shell = ad.AnnData(X=sp.csr_matrix((n_cells, len(keep)), dtype=np.float32),
                       obs=obs, var=pd.DataFrame(index=genes[keep]))
    shell.uns["build_config"] = json.dumps(build)
    shell.write_h5ad(cache_path)
    del shell

    sums = np.zeros((len(conditions), n_genes))
    counts = np.zeros(len(conditions), dtype=np.int64)
    with h5py.File(cache_path, "r+") as out:
        del out["X"]
        x = out.create_group("X")
        x.attrs["encoding-type"] = "csr_matrix"
        x.attrs["encoding-version"] = "0.1.0"
        x.attrs["shape"] = np.array([n_cells, len(keep)])
        data = x.create_dataset("data", shape=(0,), maxshape=(None,), dtype=np.float32,
                                chunks=(1 << 20,))
        indices = x.create_dataset("indices", shape=(0,), maxshape=(None,),
                                   dtype=np.int32, chunks=(1 << 20,))
        indptr = x.create_dataset("indptr", shape=(n_cells + 1,), dtype=np.int64)
        indptr[0] = 0
        row, nnz = 0, 0
        for name, handle in files:
            labels = labels_of[name]
            for start, stop, matrix in handle.chunks(args.chunk):
                library = np.asarray(matrix.sum(axis=1)).ravel()
                normalised = (sp.diags(median / library) @ matrix).tocsr()
                normalised.data = np.log1p(normalised.data)
                codes = np.array([index[c] for c in labels[start:stop]])
                indicator = sp.csr_matrix(
                    (np.ones(len(codes)), (codes, np.arange(len(codes)))),
                    shape=(len(conditions), len(codes)))
                sums += np.asarray((indicator @ normalised).todense())
                counts += np.bincount(codes, minlength=len(conditions))
                subset = normalised[:, keep].tocsr()
                subset.sort_indices()
                added = subset.nnz
                data.resize((nnz + added,))
                indices.resize((nnz + added,))
                data[nnz:] = subset.data.astype(np.float32)
                indices[nnz:] = subset.indices.astype(np.int32)
                indptr[row + 1:row + 1 + (stop - start)] = subset.indptr[1:] + nnz
                row += stop - start
                nnz += added
                print(f"  pass 2  {name} {stop:,}/{handle.shape[0]:,}  "
                      f"{time.time() - t0:.0f}s", flush=True)
            handle.close()
        if row != n_cells:
            raise SystemExit(f"wrote {row:,} rows for {n_cells:,} cells")

    np.savez_compressed(
        means_path, conditions=np.array(conditions),
        vcc_set=np.array([set_of[c] for c in conditions]), n_cells=counts,
        means=(sums / np.maximum(counts, 1)[:, None]).astype(np.float32),
        gene_names=genes, library_median=median)
    check = ad.read_h5ad(cache_path, backed="r")
    print(f"\n-> {cache_path}: {check.shape[0]:,} cells x {check.shape[1]:,} genes, "
          f"{obs['condition'].nunique()} conditions "
          f"({(obs['condition'] == CONTROL).sum():,} control cells)")
    print(f"-> {means_path}: means over all {n_genes:,} genes for {len(conditions)} "
          f"conditions")
    print("   loading this cache holds X densely: about "
          f"{n_cells * len(keep) * 4 / 1e9:.1f} GB of memory")


if __name__ == "__main__":
    sys.exit(main())
