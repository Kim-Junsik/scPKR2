#!/usr/bin/env bash
# Table 2: the final model (nEfm) on Norman's combinations split. Nothing else changes.
#
#   bash scripts/best_fm_table2.sh                   all five folds x 3 seeds, 15 runs
#   FOLDS="0 1 2" bash scripts/best_fm_table2.sh     part of it, one pane each
#   FOLDS="3 4" bash scripts/best_fm_table2.sh
#
# THE MODEL is nEfm exactly as scripts/best_fm.sh trains it for Table 1: stage 2's D
# settings, the STRING knowledge encoder, perturbation dropout 0.25, 250 epochs, seeds
# 0 1 2, and flow matching on (fm_weight 1.0, endpoint 0, fm_sigma 0, UOT coupling).
#
# THE ONE DIFFERENCE is the split: split.method=combinations, scDFM's, derived from the
# additive folds in src/data/splits.py. Each fold holds out its first 15 test doubles AND
# the singles of every gene in them (21-26 singles), so the test set has two blocks that
# Table 2 reports separately. Most held-out genes still appear in some training double;
# 2-4 per fold appear in no training condition at all, and those are predicted from the
# STRING knowledge alone (train.log says how many: "perturbations with training data").
#
# EACH FOLD NEEDS ITS OWN CACHE, because exclude_test_from_hvg makes the gene space depend
# on what is held out. The caches are built here, with the settings the additive caches
# were built with (read back from their uns["build_config"]): scanpy HVG, 5,000 genes,
# targets forced in, held-out cells excluded from the HVG choice. A cache that already
# exists is left alone. Panes given disjoint FOLDS build disjoint caches.
#
# SCORING. Single and Double are exported and scored as separate files, because
# de_spearman_sig and DS are computed over every perturbation in the scored file
# (FINDINGS 15). Same protocol as Table 1 otherwise:
#
#   for r in results/runs/nEfm_comb_f*_s*; do for g in single double; do python scripts/run_celleval.py $r --export-only --gate soft --bound-mean --infer-top-gene 1000 --pred-cells 128 --group $g || break 2; done; done
#   python scripts/paper_table.py results/runs/nEfm_comb_f*_s* --group single --group-export --gate soft --bound-mean --pred-cells 128 --results-tag ce0542 --infer-top-gene 1000 --device cuda --n-cells 4096
#   python scripts/paper_table.py results/runs/nEfm_comb_f*_s* --group double --group-export --gate soft --bound-mean --pred-cells 128 --results-tag ce0542 --infer-top-gene 1000 --device cuda --n-cells 4096
#
# THE ROWS TO BEAT, scDFM's Table 2 (arXiv 2602.07103, "Norman holdout split"):
#
#            L2       MSE     MAE     DE-Spearman  Pearson d  DS
#   Single   1.6186   0.0030  0.0190  0.6957       0.7127     0.8914   scDFM
#            1.6758   0.0035  0.0191  0.2860       0.7109     0.8072   CellFlow
#   Double   2.0309   0.0047  0.0235  0.5676       0.8357     0.9189   scDFM
#            2.1042   0.0049  0.0236  0.5074       0.8095     0.8622   CellFlow
#
# Every setting is written here and each run's SAVED config is printed at the end.

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0 1 2}"
FOLDS="${FOLDS:-0 1 2 3 4}"
# nEfm's length; overridable only for a local smoke test.
EPOCHS="${EPOCHS:-250}"

for fold in $FOLDS; do
  case "$fold" in
    0|1|2|3|4) ;;
    *) echo "unknown fold '$fold' (0 1 2 3 4)" >&2; exit 2 ;;
  esac
done

tags=()
for fold in $FOLDS; do
  cache="assets/norman_scanpy5000_combinations_fold${fold}.h5ad"
  if [ ! -f "$cache" ]; then
    echo ""
    echo "=== building ${cache} ==="
    python scripts/build_data.py --set \
      data.raw_h5ad=data/norman/norman.h5ad \
      data.n_hvg=5000 \
      data.hvg_criterion=scanpy \
      data.force_include_targets=true \
      data.exclude_test_from_hvg=true \
      split.source=reference \
      split.method=combinations \
      "split.fold=${fold}" \
      "data.cache_h5ad=${cache}"
  fi

  for seed in $SEEDS; do
    tag="nEfm_comb_f${fold}_s${seed}"
    echo ""
    echo "=== ${tag}  ->  results/runs/${tag}   [nEfm, combinations split] ==="
    python scripts/train.py --tag "$tag" --force --set \
      "train.device=${DEVICE}" \
      train.gene_endpoint_weight=10 \
      train.fm_weight=1.0 \
      train.endpoint_weight=0 \
      train.fm_sigma=0.0 \
      train.coupling=uot \
      model.observable_whiten=zca \
      model.response=generator \
      data.raw_h5ad=data/norman/norman.h5ad \
      split.source=reference \
      split.method=combinations \
      "split.fold=${fold}" \
      "data.cache_h5ad=${cache}" \
      "train.seed=${seed}" \
      "train.epochs=${EPOCHS}" \
      model.perturbation_dropout=0.25 \
      model.perturbation_knowledge=string
    tags+=("$tag")
  done
done

echo ""
echo "=== what each run was actually trained with ==="
for tag in "${tags[@]}"; do
  python - "$tag" <<'PY'
import sys, torch
tag = sys.argv[1]
saved = torch.load(f"results/runs/{tag}/checkpoint.pt", map_location="cpu",
                   weights_only=False)
c = saved["config"]
m, t, s = c["model"], c["train"], c["split"]
k = saved["model"].get("modulation.operators.knowledge")
seen = saved["model"].get("modulation.operators.seen")
covered = "-" if k is None else int((k.abs().sum(1) > 0).sum())
print(f"  {tag:20s} split={s['method']} fold={s['fold']} seed={t['seed']} "
      f"epochs={t['epochs']}  fm={t['fm_weight']} endpoint={t['endpoint_weight']} "
      f"sigma={t.get('fm_sigma')} coupling={t['coupling']} gene={t['gene_endpoint_weight']}  "
      f"whiten={m.get('observable_whiten')} response={m.get('response')}  "
      f"knowledge={m.get('perturbation_knowledge')} covered={covered} "
      f"seen={'-' if seen is None else int(seen.sum())}/{'-' if seen is None else seen.numel()} "
      f"p={m.get('perturbation_dropout')}  cache={c['data']['cache_h5ad'].split('/')[-1]}")
PY
done
