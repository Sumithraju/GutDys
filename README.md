# PEC-ML-pipeline — PCOS & Endometriosis, real 16S data, two separate pipelines

Two independent, single-disease (vs Control) classifiers built from **real
sequenced 16S stool data** — split out on purpose so PCOS and Endometriosis
each have their own clean run, before any attempt at combining them into one
pooled real dataset.

```
PEC-ML-pipeline/
├── dataset/
│   ├── pcos/
│   │   ├── fastq/                                 (symlink -> MROP/datasets/PCOS_Lindheim_stool_16S)
│   │   ├── SRP077213_PCOS_stool_ENA_filereport.tsv
│   │   └── SRP077213_PCOS_stool_real_taxa.csv      (produced by QIIME2 step -- not there yet)
│   └── endometriosis/
│       ├── fastq/                                  (symlink -> collected_dataset/fastq_PRJNA1145097)
│       ├── PRJNA1145097_endo_stool_SRR_Acc_List.txt
│       ├── PRJNA1145097_endo_stool_SraRunTable.csv
│       └── PRJNA1145097_real_taxa_labeled.csv      (already processed -- ready to use)
├── src/
│   ├── pcos_pipeline.py            (PCOS vs Control)
│   └── endometriosis_pipeline.py   (Endometriosis vs Control)
├── results/
│   ├── pcos/            (eda/, models/, reports/ -- created when you run pcos_pipeline.py)
│   └── endometriosis/   (eda/, models/, reports/ -- created when you run endometriosis_pipeline.py)
└── README.md
```

`fastq/` under each dataset folder is a **symlink**, not a copy — the raw
FASTQ are several GB, so there's no point duplicating them on disk. Every
script here reads through the symlink transparently.

## Data sources (both genuinely real, not simulated)

- **PCOS**: SRP077213 — Lindheim/Insenser pilot study (PLOS ONE 2017,
  PMC5207627). 43 real stool 16S runs (24 PCOS / 19 Control). Case/control
  labels come directly from the study's own P0xx / C0xx sample aliases (see
  `dataset/pcos/fastq/sample_labels.csv`) — no manual inference needed.
- **Endometriosis**: PRJNA1145097 — Kommagani lab stool study. 50 real
  16S runs (31 Control / 19 Endometriosis), already fully processed through
  QIIME2/DADA2/SILVA138.

Both are small pilot-scale cohorts — treat any result here as preliminary,
not a definitive benchmark.

## Status right now

| | PCOS | Endometriosis |
|---|---|---|
| FASTQ downloaded | in progress (see `dataset/pcos/fastq/download_remaining_stool.sh`) | done |
| QIIME2/DADA2 processed | not yet | done |
| `src/*_pipeline.py` runnable today | no (needs the CSV below first) | **yes** |

### To finish the PCOS side
1. `cd dataset/pcos/fastq && bash download_remaining_stool.sh` (repeat until it reports `failed=0`)
2. `bash ../../../collected_dataset/04_processing_scripts/run_SRP077213_PCOS_stool.sh`
   (needs QIIME2 + the SILVA 138 classifier already set up for the endometriosis run)
3. That writes `dataset/pcos/SRP077213_PCOS_stool_real_taxa.csv`
4. `python3 src/pcos_pipeline.py`

### To run the Endometriosis side (works right now)
```
python3 src/endometriosis_pipeline.py
```

## What each pipeline does

Same shape in both files: `load_data` → `run_eda` → `taxa_group_comparison`
(which taxa are highest in which group) → `compare_to_literature` (sanity
check against published taxon-disease direction) → `engineer_features` +
`build_pipeline` (impute → ratio features → scale, all inside one sklearn
Pipeline so nothing leaks across CV folds) → `screen_models` (a 10-model zoo,
5-fold CV) → `tune_best` (GridSearchCV the winner) → `evaluate` (confusion
matrix, ROC/AUC, feature importance) → results saved under `results/<disease>/`.

## Next step (not done yet, on request)

Once both `results/pcos/` and `results/endometriosis/` have real taxa tables,
the plan is to merge them into one real pooled Control/Endometriosis/PCOS
dataset — processing each study's FASTQ separately through QIIME2 (different
studies can need different DADA2 truncation settings), classifying taxonomy
with the same SILVA 138 classifier so taxon names line up, then combining the
two resulting taxa tables with per-study normalization (to avoid the model
learning "which study" instead of "which disease").
