"""
================================================================================
 pooled_pipeline.py  —  Pooled 3-class classifier (Control / Endometriosis / PCOS)
================================================================================

DATA
----
../dataset/synthetic_demo_pooled_3class.csv   (720 samples)
  Control 346 / PCOS 235 / Endometriosis 139.

  *** SYNTHETIC / DEMO DATA — NOT real sequenced patients. ***
  This is a harmonised placeholder used only to demonstrate the multi-class
  version of the same pipeline used in the real PCOS and endometriosis arms.
  Do NOT interpret its results biologically or clinically.

WHY THIS FOLDER
---------------
Split out from the parent GD-ML project so the pooled 3-class demo lives on its
own, in the same layout as the real PEC-ML-pipeline arms:
    pooled/src/pooled_pipeline.py
    pooled/dataset/synthetic_demo_pooled_3class.csv
    pooled/results/{eda,models,reports,explainability}/

PIPELINE (same leakage-safe design as the real arms, adapted for 3 classes)
---------------------------------------------------------------------------
    load_data        -> read CSV, drop leakage cols, build X / y
    run_eda          -> class balance, missingness, correlation
    taxa_group_comparison -> mean taxon abundance per group (+ graph)
    build_pipeline   -> impute -> scale -> select-K -> classifier, all in-CV
    screen_models    -> CV macro-F1 over a model zoo, with overfit gap
    tune_best        -> GridSearchCV the winner
    evaluate + plots -> confusion, one-vs-rest ROC, feature importance
    explain_with_shap / _lime -> global + local explanations
    main             -> run everything, save results/

Run:  python3 pooled_pipeline.py
================================================================================
"""
from __future__ import annotations
import json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RepeatedStratifiedKFold, cross_val_score,
                                     GridSearchCV, learning_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, AdaBoostClassifier)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             accuracy_score, balanced_accuracy_score,
                             roc_curve, auc)

warnings.filterwarnings("ignore")

#  CONFIG                                                                      #
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATA_CSV = PROJECT_DIR / "dataset" / "synthetic_demo_pooled_3class.csv"
RESULTS_DIR = PROJECT_DIR / "results"
EDA_DIR = RESULTS_DIR / "eda"
MODELS_DIR = RESULTS_DIR / "models"
REPORTS_DIR = RESULTS_DIR / "reports"
EXPLAIN_DIR = RESULTS_DIR / "explainability"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
CV_REPEATS = 3
SCORING = "f1_macro"
N_JOBS = -1
SELECT_K = "all"          # only ~14 features here; n=720 is comfortable

TARGET_COL = "label"                                   # already 0/1/2
CLASS_NAMES = ("Control", "Endometriosis", "PCOS")     # label 0,1,2
LABEL_MAP = {i: n for i, n in enumerate(CLASS_NAMES)}
NON_FEATURE_COLS = ("sample_id", "study", "diagnosis", "sample_type",
                    "country", TARGET_COL)


def ensure_dirs() -> None:
    for d in (EDA_DIR, MODELS_DIR, REPORTS_DIR, EXPLAIN_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_fig(fig, folder: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(folder / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


#  STEP 1 — LOAD                                                               #
def load_data(path: Path = DATA_CSV):
    if not path.exists():
        raise FileNotFoundError(f"Expected pooled demo table at {path}.")
    raw = pd.read_csv(path)
    y = raw[TARGET_COL].to_numpy(dtype=int)
    all_missing = {c for c in raw.columns if raw[c].isna().all()}
    drop = set(NON_FEATURE_COLS) | all_missing
    feats = [c for c in raw.columns
             if c not in drop and pd.api.types.is_numeric_dtype(raw[c])]
    X = raw[feats].copy()
    print(f"[load] {len(raw)} samples (SYNTHETIC/DEMO), {len(feats)} features")
    print(f"[load] classes: {pd.Series([CLASS_NAMES[i] for i in y]).value_counts().to_dict()}")
    return raw, X, y, feats


#  STEP 2 — EDA                                                                #
def run_eda(raw: pd.DataFrame, features: list[str]) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = pd.Series(raw[TARGET_COL]).value_counts().reindex(range(len(CLASS_NAMES)), fill_value=0)
    ax.bar([CLASS_NAMES[i] for i in range(len(CLASS_NAMES))], counts.values, color="teal")
    ax.set_title("Class distribution — pooled (synthetic/demo)")
    for i, v in enumerate(counts.values):
        ax.annotate(int(v), (i, v), ha="center", va="bottom", fontsize=9)
    save_fig(fig, EDA_DIR, "01_class_distribution.png")

    miss = raw[features].isna().mean().sort_values(ascending=False) * 100
    miss = miss[miss > 0]
    if not miss.empty:
        fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(miss))))
        sns.barplot(x=miss.values, y=miss.index, hue=miss.index, palette="rocket", legend=False, ax=ax)
        ax.set_title("Missing data by feature"); ax.set_xlabel("% missing")
        save_fig(fig, EDA_DIR, "02_missingness.png")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(raw[features].corr(), cmap="coolwarm", center=0, square=True,
                cbar_kws={"shrink": .7}, ax=ax)
    ax.set_title("Feature correlation — pooled (synthetic/demo)")
    save_fig(fig, EDA_DIR, "03_correlation.png")


#  STEP 3 — per-group taxon comparison                                        #
def taxa_group_comparison(raw: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    grp = raw.copy()
    grp["_g"] = [CLASS_NAMES[i] for i in grp[TARGET_COL]]
    means = grp.groupby("_g")[features].mean().T[list(CLASS_NAMES)]
    tbl = means.copy(); tbl["highest_in"] = means.idxmax(axis=1)
    tbl.to_csv(REPORTS_DIR / "taxa_group_comparison.csv")
    long = means.reset_index().melt(id_vars="index", var_name="group", value_name="mean").rename(columns={"index": "taxon"})
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(data=long, x="taxon", y="mean", hue="group", palette="viridis", ax=ax)
    ax.set_title("Mean taxon abundance by group — pooled (synthetic/demo)")
    ax.set_xlabel(""); ax.tick_params(axis="x", rotation=40); ax.legend(title="group", fontsize=8)
    save_fig(fig, EDA_DIR, "04_taxa_group_comparison.png")
    return tbl


#  STEP 4 — MODELS                                                            #
def get_models() -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
        "GaussianNB": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
        "SVC_RBF": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
        "MLP": MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1.0, max_iter=2000, random_state=RANDOM_STATE),
    }


def build_pipeline(clf) -> Pipeline:
    """impute -> scale -> select-K -> classifier, all inside one Pipeline
    (re-fit per CV fold, so no leakage)."""
    return Pipeline([
        ("impute", KNNImputer(n_neighbors=5)),
        ("scale", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif, k=SELECT_K)),
        ("clf", clf),
    ])


def cv_splitter():
    return StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def cv_splitter_repeated():
    return RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)


def screen_models(X, y) -> pd.DataFrame:
    rows = []
    for name, base in get_models().items():
        pipe = build_pipeline(base)
        try:
            cv = cross_val_score(pipe, X, y, cv=cv_splitter_repeated(), scoring=SCORING, n_jobs=N_JOBS)
            pipe.fit(X, y)
            train_f1 = f1_score(y, pipe.predict(X), average="macro")
            rows.append({"name": name, "cv_f1_mean": float(cv.mean()), "cv_f1_std": float(cv.std()),
                         "train_f1": float(train_f1), "overfit_gap": float(train_f1 - cv.mean())})
        except Exception as exc:
            print(f"  ! {name} failed: {exc}")
    return pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False).reset_index(drop=True)


PARAM_GRIDS = {
    "RandomForest": {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 10, 16]},
    "ExtraTrees": {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 12, 18]},
    "GradientBoosting": {"clf__n_estimators": [150, 300], "clf__max_depth": [2, 3], "clf__learning_rate": [0.05, 0.1]},
    "LogisticRegression": {"clf__C": [0.1, 1.0, 10.0]},
    "SVC_RBF": {"clf__C": [1, 10], "clf__gamma": ["scale", 0.1]},
}


def tune_best(name, base, X, y):
    grid = PARAM_GRIDS.get(name)
    pipe = build_pipeline(base)
    if grid is None:
        pipe.fit(X, y)
        return pipe, {}, None
    gs = GridSearchCV(pipe, grid, scoring=SCORING, cv=cv_splitter_repeated(), n_jobs=N_JOBS)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_, float(gs.best_score_)


#  STEP 5 — EVALUATION + PLOTS                                                #
def plot_confusion(y_true, y_pred) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix — pooled (synthetic/demo)")
    save_fig(fig, MODELS_DIR, "confusion_matrix.png")


def plot_roc_ovr(model, X_te, y_te) -> dict:
    if not hasattr(model.named_steps["clf"], "predict_proba"):
        return {}
    proba = model.predict_proba(X_te)
    y_bin = label_binarize(y_te, classes=list(range(len(CLASS_NAMES))))
    aucs = {}
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for i, name in enumerate(CLASS_NAMES):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
        a = auc(fpr, tpr); aucs[name] = round(float(a), 3)
        ax.plot(fpr, tpr, label=f"{name} (AUC={a:.2f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=.4)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-Rest ROC — pooled (synthetic/demo)"); ax.legend(fontsize=8)
    save_fig(fig, MODELS_DIR, "roc_curves.png")
    return aucs


def plot_learning_curve(model, X, y) -> None:
    sizes, tr, va = learning_curve(model, X, y, cv=CV_FOLDS, scoring=SCORING,
                                   train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=N_JOBS)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, tr.mean(1), "o-", label="Training")
    ax.plot(sizes, va.mean(1), "s-", label="Cross-val")
    ax.set_xlabel("Training samples"); ax.set_ylabel(SCORING)
    ax.set_title("Learning curve — pooled (synthetic/demo)"); ax.legend()
    save_fig(fig, MODELS_DIR, "learning_curve.png")


def plot_feature_importance(model, feat_names) -> None:
    clf = model.named_steps["clf"]
    imp = None
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_).mean(0)
    if imp is None:
        return
    sel = model.named_steps["select"]
    kept = sel.get_support(indices=True) if hasattr(sel, "get_support") else range(len(imp))
    names = [feat_names[i] for i in kept][:len(imp)]
    s = pd.Series(imp[:len(names)], index=names).sort_values()
    fig, ax = plt.subplots(figsize=(7, max(4, 0.4 * len(s))))
    s.plot.barh(ax=ax, color="teal")
    ax.set_title("Feature importance — pooled (synthetic/demo)")
    save_fig(fig, MODELS_DIR, "feature_importance.png")


def evaluate(model, X_te, y_te) -> dict:
    y_pred = model.predict(X_te)
    plot_confusion(y_te, y_pred)
    return {"accuracy": accuracy_score(y_te, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_te, y_pred),
            "f1_macro": f1_score(y_te, y_pred, average="macro"),
            "f1_weighted": f1_score(y_te, y_pred, average="weighted"),
            "report": classification_report(y_te, y_pred, target_names=CLASS_NAMES,
                                            output_dict=True, zero_division=0)}


#  STEP 6 — EXPLAINABILITY                                                     #
def explain_with_shap(model, X_tr, X_te, feats):
    try:
        import shap
    except ImportError:
        print("[shap] not installed (pip install shap) — skipping")
        return None
    try:
        pre = Pipeline(model.steps[:-1])
        Xt = pre.transform(X_te)
        names = list(Xt.columns) if hasattr(Xt, "columns") else feats
        bg = shap.sample(np.asarray(pre.transform(X_tr)), 50, random_state=RANDOM_STATE)
        f = lambda d: model.named_steps["clf"].predict_proba(d)
        expl = shap.KernelExplainer(f, bg)
        sv = expl.shap_values(np.asarray(Xt)[:40], nsamples=100)
        # multiclass SHAP can be a list (one array per class) OR a 3-D array
        # (samples, features, classes). Collapse either to one value per feature.
        if isinstance(sv, list):
            vals = np.mean([np.abs(s) for s in sv], axis=0).mean(axis=0)
        else:
            arr = np.abs(np.asarray(sv))
            vals = arr.mean(axis=(0, 2)) if arr.ndim == 3 else arr.mean(axis=0)
        vals = np.ravel(vals)
        s = pd.Series(vals[:len(names)], index=names[:len(vals)]).sort_values()
        s.to_csv(EXPLAIN_DIR / "shap_mean_abs_importance.csv", header=["mean_abs_shap"])
        fig, ax = plt.subplots(figsize=(7, max(4, 0.4 * len(s))))
        s.plot.barh(ax=ax, color="purple"); ax.set_title("Mean |SHAP| — pooled (synthetic/demo)")
        save_fig(fig, EXPLAIN_DIR, "shap_summary_bar.png")
        return s.sort_values(ascending=False).to_dict()
    except Exception as exc:
        print(f"[shap] failed: {exc}")
        return None


#  MAIN                                                                        #
def main() -> None:
    ensure_dirs()
    print("========  Pooled 3-class demo — Control / Endometriosis / PCOS  ========")
    print("[note] SYNTHETIC/DEMO data — not real patients; results are illustrative only")

    raw, X, y, feats = load_data()
    run_eda(raw, feats)
    tbl = taxa_group_comparison(raw, feats)
    print("[biology] taxa peaking per group:"); print(tbl["highest_in"].to_string())

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    board = screen_models(X_tr, y_tr)
    board.to_csv(REPORTS_DIR / "model_leaderboard.csv", index=False)
    print("\n[screen] leaderboard:"); print(board.to_string(index=False))

    best = board.iloc[0]; base = get_models()[best["name"]]
    model, params, cv_score = tune_best(best["name"], base, X_tr, y_tr)
    print(f"\n[best] {best['name']}  tuned_params={params}")

    metrics = evaluate(model, X_te, y_te)
    aucs = plot_roc_ovr(model, X_te, y_te)
    plot_learning_curve(model, X_tr, y_tr)
    plot_feature_importance(model, feats)
    print(f"[test] acc={metrics['accuracy']:.3f}  f1_macro={metrics['f1_macro']:.3f}  "
          f"bal_acc={metrics['balanced_accuracy']:.3f}")
    print(f"[test] per-class ROC AUC (OvR): {aucs}")

    print("\n[explainability] SHAP (global)...")
    shap_imp = explain_with_shap(model, X_tr, X_te, feats)

    joblib.dump(model, MODELS_DIR / "best_model.joblib")
    summary = {
        "data_note": "SYNTHETIC/DEMO pooled 3-class data — NOT real patients; illustrative only.",
        "classes": list(CLASS_NAMES), "n_samples": int(len(raw)),
        "class_counts": {CLASS_NAMES[i]: int((y == i).sum()) for i in range(len(CLASS_NAMES))},
        "best_model": {"name": best["name"], "tuned_params": params,
                       "tuned_cv_f1_macro": None if cv_score is None else round(cv_score, 4),
                       "screen_cv_f1_macro": round(float(best["cv_f1_mean"]), 4),
                       "overfit_gap": round(float(best["overfit_gap"]), 4)},
        "test_metrics": {k: round(float(v), 4) for k, v in metrics.items() if k != "report"},
        "per_class_report": metrics["report"],
        "roc_auc_ovr": aucs,
        "shap_top": shap_imp,
        "taxa_highest_in": tbl["highest_in"].to_dict(),
    }
    (REPORTS_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\nDone. See results/ for leaderboard, plots, model and summary.")


if __name__ == "__main__":
    main()
