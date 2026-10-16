"""Download the KEGG snapshot the pathway mask is built from.

    python scripts/download_kegg.py
    python scripts/download_kegg.py --force
    python scripts/download_kegg.py --kgml     the pathway maps' KGML, for the maplink graph

Three files from KEGG's REST API (free for academic use, no login):

    list/pathway/hsa   human pathway ids and names
    link/hsa/pathway   pathway <-> gene links
    list/hsa           gene entry -> symbol, for mapping onto adata.var_names

KEGG has no version endpoint, so the retrieval date is written next to the files.
Re-running later can therefore give a different snapshot; `manifest.json` records
line counts and byte sizes so a change is at least visible rather than silent.

For a fully pinned alternative, MSigDB ships the same content as a release-numbered
GMT in symbol form (c2.cp.kegg_medicus.v*.symbols.gmt), which also removes the
Entrez-to-symbol mapping step. It needs a login, so it cannot be fetched here.

--kgml fetches get/<pathway>/kgml for every pathway the observables of the given caches
actually use (default: every norman cache in assets/, 217 pathways on 2026-10-16), into
<kegg_dir>/kgml/. A KGML file is the pathway map itself; its `map` entries are the other
pathways the map links to, which is the curated pathway-pathway graph
docs/DESIGN_GRAPH.md section 9 deferred. One request every 0.5 s, so KEGG is not
hammered; a pathway with no KGML is recorded in the manifest rather than fatal.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config as config_module

ENDPOINTS = {
    "pathway_list.tsv": "https://rest.kegg.jp/list/pathway/hsa",
    "pathway_gene.tsv": "https://rest.kegg.jp/link/hsa/pathway",
    "gene_list.tsv": "https://rest.kegg.jp/list/hsa",
}


KGML_URL = "https://rest.kegg.jp/get/{}/kgml"


def pathways_in_use(config: dict, caches: list[str]) -> list[str]:
    """KEGG ids of the pathways the observables keep, over every cache given."""
    import anndata as ad

    from src.data.kegg import build_prior, load_pathways

    _, names = load_pathways(config["data"]["kegg_dir"])
    by_name = {name.replace(" - Homo sapiens (human)", ""): pid for pid, name in names.items()}
    used: set[str] = set()
    for cache in caches:
        genes = ad.read_h5ad(cache, backed="r").var_names.to_numpy()
        _, kept = build_prior(config, genes)
        used |= {by_name[name] for name in kept}
        print(f"  {cache}: {len(kept)} pathways")
    return sorted(used)


def download_kgml(config: dict, caches: list[str], force: bool) -> None:
    target = os.path.join(config["data"]["kegg_dir"], "kgml")
    os.makedirs(target, exist_ok=True)
    pathways = pathways_in_use(config, caches)
    print(f"  {len(pathways)} pathways in use -> {target}")
    fetched, missing, total = [], [], 0
    for i, pathway in enumerate(pathways, 1):
        path = os.path.join(target, f"{pathway}.xml")
        if not (os.path.exists(path) and os.path.getsize(path) > 0) or force:
            request = urllib.request.Request(KGML_URL.format(pathway),
                                             headers={"User-Agent": "scPKR2-kgml"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    body = response.read()
            except urllib.error.HTTPError as error:
                missing.append(f"{pathway} (HTTP {error.code})")
                print(f"  [{i:3d}/{len(pathways)}] {pathway}  no KGML (HTTP {error.code})")
                time.sleep(0.5)
                continue
            with open(path, "wb") as handle:
                handle.write(body)
            time.sleep(0.5)
        size = os.path.getsize(path)
        total += size
        fetched.append(pathway)
        if i % 25 == 0 or i == len(pathways):
            print(f"  [{i:3d}/{len(pathways)}] {total / 1e6:.1f} MB so far")
    manifest = {"retrieved": datetime.date.today().isoformat(),
                "source": KGML_URL.format("<pathway>"),
                "caches": caches, "pathways": len(pathways),
                "fetched": len(fetched), "bytes": total, "missing": missing}
    with open(os.path.join(target, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"\n-> {target}: {len(fetched)} files, {total / 1e6:.1f} MB, "
          f"{len(missing)} without KGML {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--kgml", action="store_true",
                        help="fetch the KGML maps of the pathways the observables use")
    parser.add_argument("--caches", nargs="*", default=None,
                        help="caches whose observables decide the pathways (--kgml); "
                             "default every assets/norman_scanpy5000*.h5ad")
    args = parser.parse_args()

    config = config_module.load(args.overrides)
    if args.kgml:
        caches = args.caches or sorted(glob.glob("assets/norman_scanpy5000*.h5ad"))
        if not caches:
            raise SystemExit("no caches given and none found in assets/")
        download_kgml(config, caches, args.force)
        return
    target = config["data"]["kegg_dir"]
    os.makedirs(target, exist_ok=True)

    manifest = {"retrieved": datetime.date.today().isoformat(), "files": {}}
    for name, url in ENDPOINTS.items():
        path = os.path.join(target, name)
        if os.path.exists(path) and os.path.getsize(path) > 0 and not args.force:
            print(f"  {name:20s} exists ({os.path.getsize(path):,} B) - use --force to refresh")
        else:
            print(f"  {name:20s} downloading from {url} ...")
            urllib.request.urlretrieve(url, path)
        with open(path, encoding="utf-8") as handle:
            lines = sum(1 for _ in handle)
        manifest["files"][name] = {"bytes": os.path.getsize(path), "lines": lines}
        print(f"  {'':20s} {lines:,} lines, {os.path.getsize(path):,} B")

    with open(os.path.join(target, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\n-> {target}")
    print("   reference snapshot (2026-08-12): pathway_list 372, pathway_gene 39,574, "
          "gene_list 24,252 lines")
    print("   a different count means KEGG changed; the mask K and its density will move")


if __name__ == "__main__":
    main()
