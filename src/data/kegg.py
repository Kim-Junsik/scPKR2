"""Assemble M_prior (K x G) from the KEGG REST download.

Measured on this gene space (docs/PIPELINE-v1.pdf 4.2):
  56/101 perturbation targets sit in some KEGG pathway, 48 survive the filters,
  K = 171 after dropping disease pathways and keeping sizes 10-300,
  density 0.87 %, pairwise Jaccard median 0.000.

53 of the 101 targets are in no usable pathway - HOX/FOX/DLX/LHX/POU3F2 and the
rest of the developmental TF block, which KEGG simply does not catalogue. Gene
selection cannot fix that.

scPKFM answered it with `n_free_tokens` all-zero rows appended here, for which its
attention mask reduced to act(alpha * M_residual) and the model invented its own
tokens. That answer does not transfer: this matrix IS the observable space, and a
row of zeros is an observable that reads nothing. Here an uncovered target gets an
ANCHOR coordinate in models/observables.py - the gene itself, read directly - so
the observable space can always see the quantity being perturbed. Everything this
module returns is therefore a real pathway.
"""

from __future__ import annotations

import collections
import os

import numpy as np

DISEASE_PREFIX = "hsa05"


def _read_gene_symbols(path: str) -> tuple[dict[str, str], dict[str, set[str]]]:
    """KEGG entry -> primary symbol, plus alias -> entries for the fallback."""
    primary: dict[str, str] = {}
    aliases: dict[str, set[str]] = collections.defaultdict(set)
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4 or ";" not in fields[3]:
                continue
            symbols = [s.strip() for s in fields[3].split(";")[0].split(",") if s.strip()]
            if not symbols:
                continue
            primary[fields[0]] = symbols[0]
            for symbol in symbols:
                aliases[symbol].add(fields[0])
    return primary, aliases


def load_pathways(kegg_dir: str) -> tuple[dict[str, set[str]], dict[str, str]]:
    """pathway id -> set of KEGG entries, and pathway id -> readable name."""
    members: dict[str, set[str]] = collections.defaultdict(set)
    with open(os.path.join(kegg_dir, "pathway_gene.tsv"), encoding="utf-8") as handle:
        for line in handle:
            pathway, entry = line.rstrip("\n").split("\t")
            members[pathway.replace("path:", "")].add(entry)

    names: dict[str, str] = {}
    with open(os.path.join(kegg_dir, "pathway_list.tsv"), encoding="utf-8") as handle:
        for line in handle:
            pathway, name = line.rstrip("\n").split("\t")
            names[pathway] = name.replace(" - Homo sapiens (human)", "")
    return members, names


def build_prior(config: dict, gene_names: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """M_prior in {0, 1}, rows aligned to the surviving pathways, columns to genes.

    Column order is `gene_names` as given - i.e. adata.var_names. That single
    convention replaces the old tokenizer/vocab alignment entirely, so there is no
    name-matching step that can silently misalign rows and columns.
    """
    kegg_dir = config["data"]["kegg_dir"]
    gene_set = {name: i for i, name in enumerate(gene_names)}

    primary, aliases = _read_gene_symbols(os.path.join(kegg_dir, "gene_list.tsv"))
    members, names = load_pathways(kegg_dir)

    # Invert alias -> entries once. Scanning the whole alias table for every KEGG
    # entry is ~9,400 x 50,000 Python-level comparisons and dominated the cost of
    # building the model.
    entry_aliases: dict[str, list[str]] = collections.defaultdict(list)
    for alias, entries in aliases.items():
        for entry in entries:
            entry_aliases[entry].append(alias)

    def to_column(entry: str) -> int | None:
        symbol = primary.get(entry)
        if symbol in gene_set:
            return gene_set[symbol]
        if symbol is None:
            return None
        for alias in entry_aliases.get(entry, ()):  # alias fallback, +97 genes
            if alias in gene_set:
                return gene_set[alias]
        return None

    kept_rows, kept_names = [], []
    for pathway, entries in sorted(members.items()):
        if config["data"]["drop_disease_pathways"] and pathway.startswith(DISEASE_PREFIX):
            continue
        columns = {c for c in (to_column(e) for e in entries) if c is not None}
        if not (config["data"]["pathway_min_genes"] <= len(columns)
                <= config["data"]["pathway_max_genes"]):
            continue
        row = np.zeros(len(gene_names), dtype=np.float32)
        row[sorted(columns)] = 1.0
        kept_rows.append(row)
        kept_names.append(names.get(pathway, pathway))

    prior = np.stack(kept_rows) if kept_rows else np.zeros((0, len(gene_names)), np.float32)

    # scPKFM appended model.n_free_tokens all-zero rows here, so its attention had
    # tokens with no prior to invent structure in. There are none here: this matrix
    # IS the observable space, and a row of zeros would be an observable that reads
    # nothing. Perturbation targets that no pathway covers - 53 of Norman's 101 -
    # get their own ANCHOR coordinate in models/observables.py, which is an explicit
    # gene rather than a free parameter.
    return prior, kept_names


def summarise(prior: np.ndarray, names: list[str], targets: list[str],
              gene_names: np.ndarray) -> dict:
    n_free = sum(1 for n in names if n.startswith("<free"))
    annotated = prior[:len(names) - n_free] if n_free else prior
    columns = {g: i for i, g in enumerate(gene_names)}
    covered = sum(1 for t in targets
                  if t in columns and annotated[:, columns[t]].sum() > 0)
    sizes = annotated.sum(axis=1)
    return {
        "K_total": int(prior.shape[0]),
        "K_annotated": int(annotated.shape[0]),
        "K_free": n_free,
        "density_percent": float(annotated.sum() / max(annotated.size, 1) * 100),
        "pathway_size_median": float(np.median(sizes)) if len(sizes) else 0.0,
        "targets_covered": covered,
        "targets_total": len(targets),
    }


def pathway_ids(kegg_dir: str, names: list[str]) -> list[str]:
    """KEGG ids for the readable names build_prior returns, in the same order.

    build_prior keeps names, not ids, and every consumer before the maplink graph only
    needed names. The two are one-to-one in this snapshot (checked: no name repeats), and
    a repeat would make this ambiguous, so it is refused rather than resolved silently.
    """
    _, readable = load_pathways(kegg_dir)
    by_name: dict[str, str] = {}
    for pathway, name in readable.items():
        if name in by_name:
            raise ValueError(f"two KEGG pathways share the name {name!r}")
        by_name[name] = pathway
    return [by_name[name] for name in names]


def load_map_links(kegg_dir: str, pathways: list[str]) -> dict[str, set[str]]:
    """pathway -> the other pathways its KEGG map draws a link to (KGML `map` entries).

    These are the arrows KEGG's curators put on each map pointing at neighbouring maps -
    the curated pathway-pathway relation the gene sets alone cannot give. Read from
    <kegg_dir>/kgml/<id>.xml, which scripts/download_kegg.py --kgml writes; a pathway
    whose file is missing is an error, because a silently empty row would read as "this
    pathway links to nothing".
    """
    import xml.etree.ElementTree as ET

    directory = os.path.join(kegg_dir, "kgml")
    links: dict[str, set[str]] = {}
    for pathway in pathways:
        path = os.path.join(directory, f"{pathway}.xml")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} is missing - run: python scripts/download_kegg.py --kgml")
        targets: set[str] = set()
        for entry in ET.parse(path).getroot().findall("entry"):
            if entry.get("type") != "map":
                continue
            for name in entry.get("name", "").split():
                name = name.replace("path:", "")
                if name.startswith("map"):
                    name = "hsa" + name[3:]
                if name != pathway:
                    targets.add(name)
        links[pathway] = targets
    return links
