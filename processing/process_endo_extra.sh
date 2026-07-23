#!/usr/bin/env bash
###############################################################################
# process_endo_extra.sh
# Turn the ALREADY-DOWNLOADED endometriosis FASTQ (PRJNA424567 + PRJNA722289,
# in GD-ML-2026-07-12/endometriosis_extra/) into pipeline-schema taxa tables,
# reusing your existing QIIME2 templates so nothing is reinvented:
#     collected_dataset/04_processing_scripts/qiime2_process.sh
#     collected_dataset/04_processing_scripts/qiime2_to_pipeline.py
#
# It does NOT re-download anything (the download step already finished); it
# points QIIME2 straight at the FASTQ you already have.
#
# OUTPUT (into PEC-ML-pipeline/dataset/endometriosis/):
#     PRJNA424567_real_taxa_labeled.csv   (97 Endometriosis / 65 Control)
#     PRJNA722289_real_taxa_labeled.csv   (diagnosis = unknown until GEO labels
#                                          are supplied -- see note at bottom)
#
# RUN ON YOUR OWN MACHINE. Requires (same as your other runners):
#   * conda activate qiime2-amplicon-2024.10
#   * the SILVA 138 classifier .qza (path set below)
# This is the SLOW step (DADA2 denoise of ~183 samples): budget a few hours.
#
# Usage:
#   bash process_endo_extra.sh                 # both projects
#   bash process_endo_extra.sh PRJNA424567     # just one
###############################################################################
set -euo pipefail

# ---- paths (edit here if your layout differs) ----------------------------- #
MROP="${MROP:-$HOME/Documents/MROP/2026-07-06_MROP_work}"
SCRIPTS="$MROP/collected_dataset/04_processing_scripts"
FASTQ_ROOT="${FASTQ_ROOT:-$MROP/GD-ML-2026-07-12/endometriosis_extra}"
CLASSIFIER="${CLASSIFIER:-$SCRIPTS/silva-138-nb-classifier.qza}"
OUT_DATASET="${OUT_DATASET:-$MROP/PEC-ML-pipeline/dataset/endometriosis}"
QIIME_PROC="$SCRIPTS/qiime2_process.sh"
TO_PIPELINE="$SCRIPTS/qiime2_to_pipeline.py"

# rarefaction depth MUST match the other studies you intend to combine with
# (PRJNA1145097 etc.) so Shannon stays comparable -- see the long comment in
# qiime2_process.sh. Keep it identical across every study.
export RAREFY_DEPTH="${RAREFY_DEPTH:-5000}"

# ---- per-project config --------------------------------------------------- #
# trunc lengths: 424567 kept at your existing 250/220; 722289 reads are 2x251bp
# V3-V4, so 240/240 trims the low-quality tail without losing the overlap.
proj_trunc_f () { case "$1" in PRJNA424567) echo 250;; PRJNA722289) echo 240;; *) echo 240;; esac; }
proj_trunc_r () { case "$1" in PRJNA424567) echo 220;; PRJNA722289) echo 240;; *) echo 220;; esac; }

PROJECTS=("$@"); [ "${#PROJECTS[@]}" -eq 0 ] && PROJECTS=("PRJNA424567" "PRJNA722289")

echo "== step 0: prerequisites =="
command -v qiime >/dev/null || { echo "ERROR: qiime not found -- conda activate qiime2-amplicon-2024.10 first."; exit 1; }
[ -f "$CLASSIFIER" ]  || { echo "ERROR: SILVA classifier not found at $CLASSIFIER"; exit 1; }
[ -f "$QIIME_PROC" ]  || { echo "ERROR: $QIIME_PROC missing"; exit 1; }
[ -f "$TO_PIPELINE" ] || { echo "ERROR: $TO_PIPELINE missing"; exit 1; }
mkdir -p "$OUT_DATASET"

for proj in "${PROJECTS[@]}"; do
  echo
  echo "###############  $proj  ###############"
  fastq_dir="$FASTQ_ROOT/$proj"
  labels="$fastq_dir/sample_labels.csv"          # written by the download script
  out_dir="$SCRIPTS/qiime_out_${proj}"
  raw_csv="$OUT_DATASET/${proj}_real_taxa.csv"
  final_csv="$OUT_DATASET/${proj}_real_taxa_labeled.csv"

  [ -d "$fastq_dir" ] || { echo "  ERROR: $fastq_dir not found -- did the download finish?"; continue; }
  ls "$fastq_dir"/*_1.fastq.gz >/dev/null 2>&1 || { echo "  ERROR: no *_1.fastq.gz in $fastq_dir"; continue; }
  [ -f "$labels" ] || { echo "  ERROR: $labels missing (needed for diagnosis labels)"; continue; }

  tf="$(proj_trunc_f "$proj")"; tr="$(proj_trunc_r "$proj")"
  echo "== step 1 [$proj]: QIIME2 denoise + classify (trunc ${tf}/${tr}, rarefy ${RAREFY_DEPTH}) =="
  bash "$QIIME_PROC" "$fastq_dir" "$out_dir" "$CLASSIFIER" "$tf" "$tr"

  echo "== step 2 [$proj]: convert to pipeline schema (labels from sample_labels.csv 'group') =="
  # sample_labels.csv has run_accession + group (Endometriosis/Control/UNKNOWN),
  # which qiime2_to_pipeline.py auto-detects (its disease-column regex matches
  # 'group', its run-column detection matches 'run_accession').
  python3 "$TO_PIPELINE" \
    --phylum "$out_dir/table-L2.tsv" --genus "$out_dir/table-L6.tsv" \
    --shannon "$out_dir/exp-shannon/alpha-diversity.tsv" \
    --metadata "$labels" \
    --study "$proj" --sample-type mixed --out "$raw_csv"

  echo "== step 3 [$proj]: patch per-row sample_type (rectal_gut / vaginal) from labels =="
  python3 - "$raw_csv" "$labels" "$final_csv" <<'PYEOF'
import sys, pandas as pd
taxa_path, labels_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
taxa = pd.read_csv(taxa_path)
lab = pd.read_csv(labels_path).set_index("run_accession")
if "sample_type" in lab.columns:
    taxa["sample_type"] = taxa["sample_id"].map(lab["sample_type"]).fillna(taxa["sample_type"])
taxa.to_csv(out_path, index=False)
print(f"[patch] wrote {out_path}  shape={taxa.shape}")
print(taxa["diagnosis"].value_counts(dropna=False).to_string())
if "sample_type" in taxa.columns:
    print(taxa["sample_type"].value_counts(dropna=False).to_string())
PYEOF
  echo "  -> $final_csv"
done

echo
echo "=============================================================="
echo "DONE. Taxa tables are in: $OUT_DATASET"
echo
echo "PRJNA424567 -> labeled (Endometriosis/Control), ready to combine."
echo
echo "PRJNA722289 -> diagnosis will read 'unknown' for all 21 samples: its SRA"
echo "metadata carries only GEO IDs (GSM5243401-421), no disease field. To use"
echo "it for training you must map each GSM to Endometriosis/Control from the"
echo "GEO series page, write that into sample_labels.csv's 'group' column, and"
echo "re-run step 2/3 for this project. Until then, combine_endo_datasets.py"
echo "will (by default) DROP its unknown rows so they can't corrupt the model."
echo
echo "NEXT: python3 combine_endo_datasets.py   then   python3 run_endo_combined.py"
