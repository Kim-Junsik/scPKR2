"""Our validation splits in scDFM's form: one JSON per family, read by the patched scDFM.

    python scripts/scdfm/export_splits.py

writes results/scdfm/splits/<family>.json for t1v, t2v and t3v0-2 (scripts/val_queue.py's
families), each {"test": [...], "exclude": [...]}. scDFM, patched with
scripts/scdfm/scdfm_val.patch and run with SCDFM_SPLIT_JSON=<that file>, drops the excluded
conditions before it computes anything from the data and scores the test list instead of
its own split - so it trains on exactly the conditions our validation runs train on, and
the real test conditions reach neither model.

The lists come from src/data/splits.py, the code our own runs use, never retyped. Norman
names are rewritten to scDFM's form ('ctrl' becomes 'control', as its loader does);
ComboSciPlex names are already the same in both.

What is NOT matched, by design: gene selection. scDFM chooses its genes its own way (an HVG
call over every remaining cell, the validation conditions included, as it does for its
test set); our runs exclude the scored conditions. Each model is run the way it was
published, on the same conditions.
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src import config as config_module  # noqa: E402
from src.data import splits  # noqa: E402
from val_queue import FAMILIES, cache_path  # noqa: E402


def scdfm_name(condition: str) -> str:
    return "+".join("control" if part == "ctrl" else part for part in condition.split("+"))


def export(family: str) -> dict:
    _, data, _, _ = FAMILIES[family]
    config = config_module.load(data + [f"data.cache_h5ad={cache_path(family)}"])
    fold = splits.folds(config)[int(config["split"]["fold"])]
    test, excluded = list(fold["test"]), list(fold.get("excluded", []))
    if set(test) & set(excluded):
        raise ValueError(f"{family}: conditions both scored and excluded")
    present = set(splits._read_obs(config)["condition"].astype(str))
    missing = [c for c in test + excluded if c not in present]
    if missing:
        raise ValueError(f"{family}: conditions absent from the data: {missing}")
    return {"family": family,
            "test": [scdfm_name(c) for c in test],
            "exclude": [scdfm_name(c) for c in excluded]}


def main() -> None:
    out_dir = os.path.join(ROOT, "results", "scdfm", "splits")
    os.makedirs(out_dir, exist_ok=True)
    for family in FAMILIES:
        split = export(family)
        path = os.path.join(out_dir, f"{family}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(split, handle, indent=1)
        print(f"{family}: {len(split['test'])} scored, {len(split['exclude'])} excluded -> {path}")


if __name__ == "__main__":
    main()
