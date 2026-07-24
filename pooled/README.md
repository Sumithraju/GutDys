# Pooled 3-class demo (Control / Endometriosis / PCOS)

Self-contained multi-class arm, split out from the parent GD-ML project into the
same layout as the real PEC-ML-pipeline arms.

```
pooled/
├── src/pooled_pipeline.py                        # the pipeline (run this)
├── dataset/synthetic_demo_pooled_3class.csv      # 720 samples (346 C / 139 E / 235 P)
└── results/{eda,models,reports,explainability}/  # generated on run
```

## ⚠ Data honesty
`synthetic_demo_pooled_3class.csv` is **SYNTHETIC / DEMO data — not real sequenced
patients.** It exists only to demonstrate the multi-class version of the same
leakage-safe pipeline used in the real PCOS and endometriosis arms. **Do not
interpret its results biologically or clinically.**

## Run
```bash
cd src
python3 pooled_pipeline.py
```

Same design as the real arms — all preprocessing (impute → scale → select-K) is
inside the cross-validation folds, a model zoo is screened on CV macro-F1, the
winner is grid-searched, and results (leaderboard, confusion matrix, one-vs-rest
ROC, SHAP) are written to `results/`. Because this is a single synthetic table,
there is no leave-one-study-out or cross-cohort evaluation here.
