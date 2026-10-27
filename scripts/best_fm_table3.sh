#!/usr/bin/env bash
# Table 3: the final model on ComboSciPlex, scDFM's split. Ten seeds.
#
#   bash scripts/best_fm_table3.sh                        seeds 0-9, 10 runs
#   SEEDS="0 1 2 3 4" bash scripts/best_fm_table3.sh      part of it, one pane each
#   SEEDS="5 6 7 8 9" bash scripts/best_fm_table3.sh
#   ARM=N bash scripts/best_fm_table3.sh                  the encoder with k_a = 0 instead
#
# THE MODEL is nEfm (scripts/best_fm.sh, Tables 1 and 2) with the one part that cannot
# carry over replaced. Same generator, ZCA observables, gene endpoint 10, and flow matching
# on (fm_weight 1.0, endpoint 0, fm_sigma 0, UOT coupling).
#
# NO STRING. The perturbations are drugs, and a gene network says nothing about them. The
# drug knowledge there is - the annotated mechanisms - made things worse (FINDINGS 27: M
# 1.8813 against D 1.7248, worse in 7 of 7 conditions), so there is no knowledge to use.
# Two ways to have none, measured with FM off, 2,500 epochs, 3 seeds (FINDINGS 29):
#
#   ARM=D  the lookup table: each drug its own generator coefficients    1.7248 +/- 0.123
#   ARM=N  the encoder with k_a = 0: c_a = c_bar + delta_a, dropout 0.25 1.7397 +/- 0.206
#
# D is the default: lower and steadier, and the same within noise. Under D the dropout is
# not applied (models/generator.py switches it off without knowledge); it is written out
# anyway so the saved config matches the cD runs it is compared with.
#
# 2,500 EPOCHS, the length every ComboSciPlex encoder run used, chosen on the validation
# conditions (H0), not on these. It was chosen with FM off and is not re-selected here.
#
# TEN SEEDS, because ComboSciPlex has one split and only 7 test conditions, and its seed
# spread is large: 0.12-0.21 of L2 at 3 seeds, against a 0.07 gap to scDFM. Ten is fixed
# here, before any of these runs is seen, and the reported number is the mean of all ten.
# No EMA: FINDINGS 4 measured it changing nothing (the cosine schedule already ends at 1e-6).
#
# THE CACHE is the one the cD/cM/cMP/cN runs used: assets/combosciplex_scanpy5000_additive_
# scdfm7_fold0.h5ad. It is built here only if it is missing, with the settings it was built
# with (read back from its uns["build_config"]).
#
# SCORING. Table 3 is one block over all 7 test conditions (two of them, control+Dacinostat
# and control+Alvespimycin, are singles), so nothing is split by --group:
#
#   python scripts/compare_to_ridge.py results/runs/cDfm_s* results/runs/cD_s[012] --device cuda --group all --bound-mean
#   for r in results/runs/cDfm_s*; do python scripts/run_celleval.py $r --export-only --gate soft --bound-mean --infer-top-gene 1000 --pred-cells 128 || break; done
#   python scripts/paper_table.py results/runs/cDfm_s* --group all --gate soft --bound-mean --pred-cells 128 --results-tag ce0542 --infer-top-gene 1000 --device cuda --n-cells 4096
#
# THE ROWS TO BEAT, scDFM's Table 3 (arXiv 2602.07103). CellFlow did not report this block.
#
#            L2       MSE     MAE     DE-Spearman  Pearson d  DS
#   scDFM    1.6567   0.0028  0.0220   0.8289      0.8933     0.8776
#   CPA      1.6592   0.0029  0.0240   0.7906      0.8150     0.8980
#   scGPT    1.6934   0.0031  0.0251  -0.1261      0.8322     0.8571
#   ridge    1.8577   (compare_to_ridge.py, this repository's protocol)
#
# Every setting is written here and each run's SAVED config is printed at the end.

set -euo pipefail
cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
ARM="${ARM:-D}"
# ComboSciPlex's length; overridable only for a local smoke test.
EPOCHS="${EPOCHS:-2500}"
CACHE="assets/combosciplex_scanpy5000_additive_scdfm7_fold0.h5ad"

case "$ARM" in
  D) knowledge=none ;;
  N) knowledge=zero ;;
  *) echo "unknown ARM '$ARM' (D or N)" >&2; exit 2 ;;
esac

# Exactly what encoder_arms.sh passes the cD/cN runs (its COMBO and COMBO_TEST).
DATA=(
  data.raw_h5ad=data/combosciplex/combosciplex.h5ad
  data.control_label=control
  data.normalise_from_counts=counts
  data.n_hvg=5000
  data.hvg_criterion=scanpy
  split.source=list
  split.method=additive
  split.fold=0
  "data.cache_h5ad=${CACHE}"
)

if [ ! -f "$CACHE" ]; then
  echo ""
  echo "=== building ${CACHE} ==="
  # Two settings that only the build reads, as the existing cache records them.
  python scripts/build_data.py --set "${DATA[@]}" \
    data.force_include_targets=true \
    data.exclude_test_from_hvg=true
fi

tags=()
for seed in $SEEDS; do
  tag="c${ARM}fm_s${seed}"
  echo ""
  echo "=== ${tag}  ->  results/runs/${tag}   [final model, ComboSciPlex, knowledge=${knowledge}] ==="
  python scripts/train.py --tag "$tag" --force --set \
    "train.device=${DEVICE}" \
    train.gene_endpoint_weight=10 \
    train.fm_weight=1.0 \
    train.endpoint_weight=0 \
    train.fm_sigma=0.0 \
    train.coupling=uot \
    model.observable_whiten=zca \
    model.response=generator \
    "${DATA[@]}" \
    "train.seed=${seed}" \
    "train.epochs=${EPOCHS}" \
    model.perturbation_dropout=0.25 \
    "model.perturbation_knowledge=${knowledge}"
  tags+=("$tag")
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
seen = saved["model"].get("modulation.operators.seen")
print(f"  {tag:12s} split={s['source']}/{s['method']} fold={s['fold']} seed={t['seed']} "
      f"epochs={t['epochs']}  fm={t['fm_weight']} endpoint={t['endpoint_weight']} "
      f"sigma={t.get('fm_sigma')} coupling={t['coupling']} gene={t['gene_endpoint_weight']}  "
      f"whiten={m.get('observable_whiten')} response={m.get('response')}  "
      f"knowledge={m.get('perturbation_knowledge')} "
      f"seen={'-' if seen is None else int(seen.sum())}/{'-' if seen is None else seen.numel()} "
      f"p={m.get('perturbation_dropout')}  cache={c['data']['cache_h5ad'].split('/')[-1]}")
PY
done
