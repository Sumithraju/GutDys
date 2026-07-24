"""

 endometriosis_pipeline.py  —  Endometriosis vs Control, REAL 16S data

GOAL
----
Classify a stool sample as Control or Endometriosis from real 16S taxa
abundances (PRJNA1145097, Kommagani lab, processed via QIIME2/DADA2/SILVA138),
and report which taxa drive the split.

DATA
----
../dataset/endometriosis/PRJNA1145097_real_taxa_labeled.csv
  n=50 (31 Control / 19 Endometriosis) -- REAL sequenced data, not simulated.
  Small n: treat all results here as a pilot, not a definitive benchmark.

PIPELINE (each a def, same shape as the GD-ML project this was split out of)
-----------------------------------------------------------------------------
    load_data                 -> read CSV, drop leakage columns, build X / y
    run_eda                   -> class balance, missingness, correlation
    taxa_group_comparison     -> which taxa are high in which group (+ graph)
    compare_to_literature     -> sanity-check vs published taxon-disease direction
    engineer_features / build_pipeline -> impute + ratio features + scale, in-pipeline
    screen_models / tune_best -> CV sweep over a model zoo, then GridSearchCV the winner
    evaluate + plots          -> confusion matrix, ROC/AUC, feature importance
    explain_with_shap / explain_with_lime -> model-agnostic SHAP (global) + LIME (per-sample,
                                 local) explanations of the winning model, saved to
                                 results/endometriosis/explainability/
    main                      -> run everything, save results/

Run:  python3 endometriosis_pipeline.py


"""
from __future__ import annotations
import json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RepeatedStratifiedKFold, cross_val_score,
                                     cross_val_predict, GridSearchCV, learning_curve)
from sklearn.metrics import precision_recall_curve
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

try:
    import shap
    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    HAVE_LIME = True
except ImportError:
    HAVE_LIME = False

warnings.filterwarnings("ignore")

#  CONFIG                                                                      #
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
# Mirrors pcos_pipeline.py: a LIST of per-study taxa tables that load_data()
# concatenates. Studies not yet downloaded / QIIME2-processed are skipped with
# a note (not an error), and new studies are picked up automatically the moment
# their processed CSV appears -- no code change needed here.
#   PRJNA1145097 -> 50 stool  (Control/Endometriosis)         [READY]
#   PRJNA424567  -> 162 vaginal+rectal (Endometriosis/Control) [processed by
#                   processing/process_endo_extra.sh]
#   PRJNA722289  -> 21 samples, GEO IDs, diagnosis UNKNOWN until mapped from the
#                   GEO series; unknown-diagnosis rows are dropped automatically
#                   by the LABEL_MAP filter below, so listing it now is harmless.
DATA_CSVS = [
    PROJECT_DIR / "dataset" / "endometriosis" / "PRJNA1145097_real_taxa_labeled.csv",
    PROJECT_DIR / "dataset" / "endometriosis" / "PRJNA424567_real_taxa_labeled.csv",
    PROJECT_DIR / "dataset" / "endometriosis" / "PRJNA722289_real_taxa_labeled.csv",
]
RESULTS_DIR = PROJECT_DIR / "results" / "endometriosis"
EDA_DIR = RESULTS_DIR / "eda"
MODELS_DIR = RESULTS_DIR / "models"
REPORTS_DIR = RESULTS_DIR / "reports"
EXPLAIN_DIR = RESULTS_DIR / "explainability"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
CV_REPEATS = 5   # RepeatedStratifiedKFold for model screening/tuning: with n~40
                 # training samples, a single 5-fold split is noisy (which model
                 # "wins" can flip with the random fold assignment) -- repeating
                 # with different shuffles and averaging gives a much more
                 # stable ranking. NOT used for the out-of-fold threshold search
                 # (cross_val_predict requires a true one-shot partition).
SCORING = "f1"
N_JOBS = -1
EPS = 1e-6
SELECT_K = 10    # keep well below n_train (~40) to fight overfitting: rule of
                 # thumb is roughly >=4-5 samples per feature for stable fits

TARGET_COL = "diagnosis"
CLASS_NAMES = ("Control", "Endometriosis")
LABEL_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}
# shannon_diversity is EXCLUDED as a feature when combining studies: it is
# depth-dependent and here it behaves as a study/batch barcode rather than a
# disease signal -- PRJNA1145097 median 4.58 (50/50 present) vs PRJNA424567
# median 7.03 (only 13/161 present after rarefaction at depth 5000, the rest
# NaN). Imputing the 148 missing values would inject exactly that batch signal.
# Relative-abundance taxa (the real features) are unaffected and kept. Re-include
# it only if every combined study is rarefied to a common depth with good
# retention. (beta_glucuronidase is all-NaN in both studies -> auto-dropped by
# the all_missing check in load_data, no need to list it here.)
NON_FEATURE_COLS = ("sample_id", "study", "sample_type", "country",
                    "shannon_diversity", TARGET_COL)

# --- body-site filter (removes the vaginal/gut confound) -------------------- #
# PRJNA424567 mixes vaginal + rectal samples; PRJNA1145097 is stool. Vaginal and
# gut microbiomes differ FAR more than Endometriosis vs Control does, so pooling
# sites lets a model score by learning body site instead of disease (the
# Lactobacillus literature-mismatch in the all-sites run was that leak showing).
# Keeping only gut-type samples makes every sample comparable. Set GUT_ONLY=False
# to revert to all sites (not recommended for a disease model).
GUT_ONLY = True
GUT_SITES = {"stool", "rectal_gut", "rectal", "gut", "feces", "fecal"}

# --- beta_glucuronidase batch handling ------------------------------------- #
# PICRUSt2-predicted GUS (EC:3.2.1.31) abundance is depth/pipeline dependent, so
# its RAW magnitude differs ~13x between studies (PRJNA1145097 median ~322 vs
# PRJNA424567 ~25) -- it separates STUDY ~6x more than DIAGNOSIS. Fed raw it
# would be a study barcode (the Shannon problem). Z-scoring it WITHIN each study
# removes the between-study magnitude and keeps only the within-cohort
# Control-vs-Endometriosis contrast. Leakage-free: uses study membership only,
# never the diagnosis label. Set False to feed raw GUS (not recommended).
GUS_WITHIN_STUDY_Z = True

RATIO_FEATURES = ["FB_ratio", "BL_ratio", "Fae_Bact_ratio", "BG_Lacto_ratio", "PB_ratio"]

# Qualitative sanity check only (see compare_to_literature). Akkermansia and
# Escherichia_coli added for parity with pcos_pipeline.py now that both real
# taxa tables include them:
#   Akkermansia (muciniphila)     -> mucin-degrading commensal, generally
#                                     REDUCED in metabolic/inflammatory
#                                     dysbiosis states -> expect higher in Control
#   Escherichia_coli (Esch-Shig.) -> classic Proteobacteria-bloom / inflammatory
#                                     marker, same family as the existing
#                                     Proteobacteria row -> expect higher in disease
# Prevotella and Ruminococcus are deliberately NOT included here: published
# directions are inconsistent/study-dependent for Prevotella, and this
# pipeline's "Ruminococcus" column sums the true (often fiber-fermenting,
# health-associated) Ruminococcaceae genus together with SILVA's reclassified
# "[Ruminococcus]_gnavus/torques/gauvreauii_group" (Lachnospiraceae, often
# pro-inflammatory when elevated) -- two biologically distinct signals under
# one label, so asserting a single expected direction would overclaim.
#
# Collinsella/Roseburia/Desulfovibrio added (real, non-trivial-abundance genera
# confirmed present in the actual table-L6.tsv before adding). Endometriosis-
# specific evidence for these three is thinner than the PCOS-specific
# Collinsella/insulin-resistance literature, so these directions lean on the
# more general inflammatory-dysbiosis framework shared with the other entries
# above (Proteobacteria bloom, reduced butyrate producers):
#   Collinsella      -> general metabolic/inflammatory dysbiosis marker ->
#                        expect higher in Endometriosis
#   Roseburia        -> butyrate-producing, generally protective/anti-
#                        inflammatory, commonly reduced in dysbiosis -> expect
#                        higher in Control
#   Desulfovibrio    -> sulfate-reducing, H2S-producing, associated with gut
#                        barrier disruption/inflammation (grouped with
#                        Bilophila, see qiime2_to_pipeline.py) -> expect higher
#                        in Endometriosis
# Blautia/Alistipes/Parabacteroides/Sutterella are reported but NOT given an
# expected direction: literature on these in endometriosis specifically is
# thin/mixed, so a hardcoded expectation would overclaim -- their group means
# are still shown in taxa_group_comparison.csv for exploratory reference.
#
# REVISED against the 2019-2025 evidence base -- see ../LITERATURE_REFERENCES.md
# for full citations and the reasoning behind every entry and omission.
# Format: taxon -> (expected_higher_in, confidence, refs). Kept structurally
# identical to pcos_pipeline.py so the two disease pipelines stay comparable.
#
# CHANGES made after reading the source reviews properly:
#   1. Prevotella ADDED as Control-higher. Previously excluded as "mixed", but
#      Endobiota2019 reports Prevotella and Dialister decreased in
#      endometriosis, with Bacteroides/Prevotella dropping further in advanced
#      disease -- consistent enough to assert a direction now.
#   2. Firmicutes REMOVED. HROpen2025 flags pronounced cross-study
#      heterogeneity in taxonomic profiles at every anatomical site; a
#      phylum-level direction isn't defensible.
#   3. Bacteroides DOWNGRADED to not-asserted. Endobiota2019 finds it lower in
#      ADVANCED disease specifically, not consistently across endometriosis
#      overall -- and our labels have no stage information.
#   4. beta_glucuronidase REMOVED. It cannot be measured from 16S at all (the
#      column is all-NaN and dropped at load), so scoring it was meaningless.
#
# Confidence tiers: "strong" = meta-analysis/multi-study consistent;
# "moderate" = multiple primary studies; "weak" = single study or extrapolated
# from general dysbiosis biology; "contested" = studies actively disagree.
LITERATURE_EXPECTATIONS = {
    # --- higher in Endometriosis ---
    "Escherichia_coli": ("Endometriosis", "strong",   ["Endobiota2019", "HROpen2025"]),
    "Proteobacteria":   ("Endometriosis", "moderate", ["HROpen2025"]),
    "Collinsella":      ("Endometriosis", "weak",     ["inflammation-literature"]),
    "Desulfovibrio":    ("Endometriosis", "weak",     ["H2S-barrier-literature"]),
    # --- higher in Control ---
    "Prevotella":       ("Control", "moderate",  ["Endobiota2019"]),
    "Lactobacillus":    ("Control", "moderate",  ["Endobiota2019", "HROpen2025"]),
    "Faecalibacterium": ("Control", "moderate",  ["SCFA-literature"]),
    "Roseburia":        ("Control", "weak",      ["butyrate-literature"]),
    "Bifidobacterium":  ("Control", "weak",      ["general"]),
    "shannon_diversity":("Control", "contested", ["FrontMicro2025Endo"]),
    # DELIBERATELY OMITTED (still reported in taxa_group_comparison.csv, just
    # not scored): Firmicutes, Bacteroidetes, Bacteroides, Ruminococcus,
    # Blautia, Alistipes, Parabacteroides, Sutterella, Akkermansia -- see
    # LITERATURE_REFERENCES.md; HROpen2025 documents pronounced heterogeneity
    # across studies for these.
}


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, EDA_DIR, MODELS_DIR, REPORTS_DIR, EXPLAIN_DIR):
        d.mkdir(parents=True, exist_ok=True)


def save_fig(fig, folder: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(folder / name, dpi=130, bbox_inches="tight")
    plt.close(fig)


#  STEP 1 — LOAD                                                               #
def load_data(paths: list[Path] = DATA_CSVS):
    """
    Reads and concatenates every real taxa table in `paths` that exists yet --
    studies still mid-download or not-yet-QIIME2-processed are skipped with a
    note (not an error), so this pipeline always runs on whatever real data is
    currently available and automatically picks up new studies the moment their
    processed CSV appears. Identical loading logic to pcos_pipeline.py.
    """
    frames = []
    for p in paths:
        if not p.exists():
            print(f"[load] skipping {p.name} -- not found yet "
                  f"(not downloaded and/or not QIIME2-processed yet)")
            continue
        df = pd.read_csv(p)
        frames.append(df)
        print(f"[load] found {p.name}: {len(df)} samples")
    if not frames:
        raise FileNotFoundError(
            "No real endometriosis taxa tables found. At minimum you need "
            "PRJNA1145097_real_taxa_labeled.csv in dataset/endometriosis/, "
            "or run processing/process_endo_extra.sh to build the others."
        )
    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw[TARGET_COL] = raw[TARGET_COL].astype(str).str.strip()
    raw = raw[raw[TARGET_COL].isin(LABEL_MAP)].copy()   # drops unknown-diagnosis rows
    if GUT_ONLY and "sample_type" in raw.columns:
        before = len(raw)
        raw = raw[raw["sample_type"].astype(str).str.lower().isin(GUT_SITES)].copy()
        print(f"[load] GUT_ONLY filter: kept {len(raw)}/{before} gut-type samples "
              f"(dropped {before - len(raw)} non-gut, e.g. vaginal)")
    raw = raw.reset_index(drop=True)   # positional alignment for LOSO masks
    if (GUS_WITHIN_STUDY_Z and "beta_glucuronidase" in raw.columns
            and "study" in raw.columns and raw["beta_glucuronidase"].notna().any()):
        def _zscore(s):
            sd = s.std(ddof=0)
            return (s - s.mean()) / sd if (sd and np.isfinite(sd) and sd > 0) else s * 0.0
        raw["beta_glucuronidase"] = (raw.groupby("study")["beta_glucuronidase"]
                                        .transform(_zscore))
        print("[load] beta_glucuronidase z-scored WITHIN study "
              "(removes ~13x between-study batch magnitude; keeps within-cohort signal)")
    y = raw[TARGET_COL].map(LABEL_MAP).to_numpy(dtype=int)
    all_missing = {c for c in raw.columns if raw[c].isna().all()}
    drop = set(NON_FEATURE_COLS) | all_missing
    feats = [c for c in raw.columns if c not in drop and pd.api.types.is_numeric_dtype(raw[c])]
    X = raw[feats].copy()
    if "study" in raw.columns and raw["study"].nunique() > 1:
        print(f"[load] combined {raw['study'].nunique()} studies: "
              f"{raw['study'].value_counts().to_dict()}")
    return raw, X, y, feats


#  STEP 2 — EDA                                                                #
def run_eda(raw: pd.DataFrame, features: list[str]) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    order = [c for c in CLASS_NAMES if c in raw[TARGET_COL].unique()]
    # NOTE: legend=False is not accepted here on some seaborn/matplotlib
    # version pairings (raises AttributeError: Rectangle.set() got an
    # unexpected keyword argument 'legend'). Draw without it, then strip
    # the (redundant, since hue==x) legend manually if one was created.
    sns.countplot(data=raw, x=TARGET_COL, order=order, hue=TARGET_COL,
                  palette="viridis", ax=ax)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_title("Class distribution (REAL data, n=%d)" % len(raw))
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x()+p.get_width()/2, p.get_height()),
                    ha="center", va="bottom", fontsize=8)
    save_fig(fig, EDA_DIR, "01_class_distribution.png")

    miss = raw[features].isna().mean().sort_values(ascending=False) * 100
    miss = miss[miss > 0]
    if not miss.empty:
        fig, ax = plt.subplots(figsize=(7, max(3, 0.4*len(miss))))
        sns.barplot(x=miss.values, y=miss.index, hue=miss.index, palette="rocket", ax=ax)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_title("Missing data by taxon"); ax.set_xlabel("% missing")
        save_fig(fig, EDA_DIR, "02_missingness.png")

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(raw[features].corr(), cmap="coolwarm", center=0, square=True,
                cbar_kws={"shrink": .7}, ax=ax)
    ax.set_title("Taxon correlation")
    save_fig(fig, EDA_DIR, "03_correlation.png")


#  STEP 3 — BIOLOGY                                                            #
def taxa_group_comparison(raw: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    classes = [c for c in CLASS_NAMES if c in raw[TARGET_COL].unique()]
    means = raw.groupby(TARGET_COL)[features].mean().T[classes]
    table = means.copy()
    table["highest_in"] = means.idxmax(axis=1)

    # Mann-Whitney U per taxon (non-parametric -- appropriate for small-n,
    # non-normal relative-abundance data): turns the descriptive group means
    # above into an actual statistical comparison table, not just "which group
    # has the bigger average." Two-sided, no multiple-testing correction
    # applied given the small taxon count and pilot-scale n -- treat p-values
    # as exploratory signals, not confirmatory.
    if len(classes) == 2:
        g0 = raw.loc[raw[TARGET_COL] == classes[0]]
        g1 = raw.loc[raw[TARGET_COL] == classes[1]]
        pvals = []
        for feat in features:
            a, b = g0[feat].dropna(), g1[feat].dropna()
            try:
                _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            except ValueError:
                p = np.nan   # e.g. all-identical or empty after dropna
            pvals.append(p)
        table["mannwhitney_p"] = pvals
        table["significant_p<0.05"] = table["mannwhitney_p"].lt(0.05).fillna(False)
    table.to_csv(REPORTS_DIR / "taxa_group_comparison.csv")

    long = means.reset_index().melt(id_vars="index", var_name=TARGET_COL, value_name="mean") \
                .rename(columns={"index": "taxon"})
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=long, x="taxon", y="mean", hue=TARGET_COL, palette="viridis", ax=ax)
    if "significant_p<0.05" in table.columns:
        for i, taxon in enumerate(means.index):
            if table.loc[taxon, "significant_p<0.05"]:
                ymax = means.loc[taxon].max()
                ax.annotate("*", (i, ymax), ha="center", va="bottom", fontsize=14, color="red")
    ax.set_title("Mean taxon abundance by group (REAL data) -- * = Mann-Whitney p<0.05")
    ax.tick_params(axis="x", rotation=40); ax.set_xlabel("")
    save_fig(fig, EDA_DIR, "04_taxa_group_comparison.png")
    return table


def compare_to_literature(taxa_comparison: pd.DataFrame) -> pd.DataFrame:
    """
    Qualitative sanity check of observed group directions against the
    consolidated 2019-2025 evidence base (see ../LITERATURE_REFERENCES.md).

    Reports overall match rate AND a separate rate for high-confidence
    ("strong"/"moderate") expectations only -- a mismatch on a weak/contested
    taxon means little, but several mismatches on multi-study-backed taxa point
    to a data or processing problem rather than novel biology.
    """
    rows = []
    for taxon, (expected, confidence, refs) in LITERATURE_EXPECTATIONS.items():
        if taxon not in taxa_comparison.index:
            continue
        found = taxa_comparison.loc[taxon, "highest_in"]
        rows.append({"taxon": taxon, "literature_expected_highest_in": expected,
                     "model_found_highest_in": found, "matches_literature": found == expected,
                     "evidence_confidence": confidence, "refs": ";".join(refs)})
    report = pd.DataFrame(rows)
    if not report.empty:
        report.to_csv(REPORTS_DIR / "literature_comparison.csv", index=False)
        n_match = int(report["matches_literature"].sum())
        print(f"[literature-check] Endometriosis: {n_match}/{len(report)} taxa match published direction")
        high = report[report["evidence_confidence"].isin(["strong", "moderate"])]
        if not high.empty:
            n_hi = int(high["matches_literature"].sum())
            print(f"[literature-check] high-confidence subset (strong/moderate): "
                  f"{n_hi}/{len(high)} match")
            missed = high.loc[~high["matches_literature"], "taxon"].tolist()
            if missed:
                print(f"  ! high-confidence MISMATCHES: {missed}")
                print(f"    Multi-study backed. If several are wrong, suspect a data/processing "
                      f"problem (e.g. wrong DADA2 truncation for this study's read length) "
                      f"before reinterpreting the biology.")
    return report


#  STEP 4 — FEATURE ENGINEERING                                                #
def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    df = X.copy().clip(lower=0)
    have = set(df.columns)
    def has(*c): return all(x in have for x in c)
    def div(a, b): return a / (b + EPS)
    if has("Firmicutes", "Bacteroidetes"):    df["FB_ratio"] = div(df["Firmicutes"], df["Bacteroidetes"])
    if has("Bacteroides", "Lactobacillus"):   df["BL_ratio"] = div(df["Bacteroides"], df["Lactobacillus"])
    if has("Faecalibacterium", "Bacteroides"):df["Fae_Bact_ratio"] = div(df["Faecalibacterium"], df["Bacteroides"])
    if has("beta_glucuronidase", "Lactobacillus"): df["BG_Lacto_ratio"] = div(df["beta_glucuronidase"], df["Lactobacillus"])
    if has("Prevotella", "Bacteroides"):      df["PB_ratio"] = div(df["Prevotella"], df["Bacteroides"])
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# DATA LEAKAGE -- how this pipeline avoids it, spelled out explicitly:
#   1. train_test_split() in main() happens ONCE, before any screening/tuning.
#      X_te/y_te are never touched again until the final evaluate() calls.
#   2. Every step below (impute, engineer, scale, select) lives INSIDE this
#      sklearn Pipeline, not as a one-time transform on the whole dataset.
#      cross_val_score/cross_val_predict/GridSearchCV all refit the entire
#      pipeline fresh on each fold's training split, so imputation values,
#      ratio-feature stats, scaling mean/std, and SelectKBest's F-scores are
#      all computed from that fold's training rows only -- the held-out fold
#      never leaks into any of these fitted statistics.
#   3. find_best_threshold() is chosen from cross_val_predict() out-of-fold
#      probabilities on X_tr/y_tr ONLY -- X_te is never used to pick the
#      threshold, so a non-0.5 cutoff is not leakage.
def build_pipeline(clf) -> Pipeline:
    pipe = Pipeline([
        ("impute", KNNImputer(n_neighbors=5)),
        ("engineer", FunctionTransformer(engineer_features)),
        ("scale", StandardScaler()),
        # univariate ANOVA F-test feature selection, refit fresh inside every
        # CV fold (never sees the held-out fold, so no leakage) -- cuts
        # dimensionality from ~25 taxa+ratios down toward SELECT_K, which
        # directly fights the overfitting we saw (train_f1=1.0 vs cv_f1=0.65).
        # k is itself searched over in PARAM_GRIDS for the models that get
        # GridSearchCV-tuned, with SELECT_K as the fallback default otherwise.
        ("select", SelectKBest(score_func=f_classif, k=SELECT_K)),
        ("clf", clf),
    ])
    pipe.set_output(transform="pandas")
    return pipe


#  STEP 5 — MODELS, SCREEN, TUNE                                               #
def get_models() -> dict:
    # class_weight="balanced" on every model that supports it: reweights the loss
    # so errors on the minority class (Endometriosis, 19/50) cost as much as errors
    # on the majority class (Control, 31/50), instead of the optimizer being free
    # to lean on the majority class to rack up cheap accuracy. GradientBoosting,
    # AdaBoost, GaussianNB, KNN and MLP have no class_weight knob in sklearn --
    # left as-is for those.
    #
    # OVERFITTING: tree ensembles previously ran fully unconstrained during
    # screen_models() (only PARAM_GRIDS-tuned versions had depth limits), so
    # the screening leaderboard itself showed train_f1=1.0 for every one of
    # them -- trivially memorizing ~40 training rows. min_samples_leaf/
    # max_depth defaults below make even the untuned screening pass realistic,
    # not just the final GridSearchCV'd model.
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                                 class_weight="balanced"),
        "KNN": KNeighborsClassifier(),
        "GaussianNB": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=3,
                                               random_state=RANDOM_STATE, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3,
                                               random_state=RANDOM_STATE,
                                               n_jobs=N_JOBS, class_weight="balanced"),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3,
                                           random_state=RANDOM_STATE,
                                           n_jobs=N_JOBS, class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(max_depth=3, min_samples_leaf=3,
                                                       subsample=0.8, random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=RANDOM_STATE),
        "SVC_RBF": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE,
                       class_weight="balanced"),
        # UNDERFITTING: this model was collapsing to train_f1~0 in both
        # pipelines -- a 2-layer (32,16) network with early_stopping's default
        # validation_fraction=0.1 leaves ~3 samples for internal validation
        # inside an already-tiny ~32-40 row CV fold, an unusably noisy
        # stopping signal that was halting training almost immediately.
        # Simplified to a single small hidden layer, disabled early_stopping
        # (nothing left to validate against at this n), added L2 (alpha) and
        # more iterations so it can actually converge. If it still underperforms
        # after this, that itself is a legitimate finding -- neural nets
        # generally need far more than n~40-50 samples to be competitive on
        # tabular data, not a sign this specific config is still broken.
        "MLP": MLPClassifier(hidden_layer_sizes=(8,), alpha=1.0, max_iter=3000,
                             early_stopping=False, random_state=RANDOM_STATE),
    }


# --- decision-threshold policy (controls the sensitivity/specificity trade) --#
#   "youden"      : balances sensitivity & specificity (default, neutral)
#   "f2"          : maximises F2 (recall weighted 2x precision) -> higher sensitivity
#   "sensitivity" : lowest cutoff that still reaches TARGET_SENSITIVITY on the
#                   training out-of-fold predictions -> tune how many cases you catch
# Lowering the threshold ALWAYS raises sensitivity and lowers specificity; it
# moves along the ROC curve, it does NOT increase AUC. Chosen on training
# out-of-fold predictions only, so it is never leakage.
THRESHOLD_MODE = "youden"
TARGET_SENSITIVITY = 0.85


def find_best_threshold(y_true, probs) -> float:
    """
    Pick the decision threshold on training out-of-fold predictions according to
    THRESHOLD_MODE. The test set never influences it, so a non-0.5 cutoff is not
    leakage. See THRESHOLD_MODE notes above for the sensitivity trade-off.
    """
    from sklearn.metrics import roc_curve, fbeta_score
    if THRESHOLD_MODE == "sensitivity":
        # highest threshold that still achieves >= TARGET_SENSITIVITY (keeps the
        # best possible specificity for that sensitivity floor)
        fpr, tpr, thr = roc_curve(y_true, probs)
        ok = np.where(tpr >= TARGET_SENSITIVITY)[0]
        if len(ok):
            t = float(thr[ok[0]])          # roc_curve thresholds are descending
            return t if np.isfinite(t) else 0.5
        return 0.5
    if THRESHOLD_MODE == "f2":
        # sweep every candidate cutoff, keep the one with the best F2 (favours recall)
        best_t, best_f = 0.5, -1.0
        for t in np.unique(probs):
            pred = (probs >= t).astype(int)
            f = fbeta_score(y_true, pred, beta=2, zero_division=0)
            if f > best_f:
                best_f, best_t = f, float(t)
        return best_t
    # default: Youden's J (balanced)
    fpr, tpr, thr = roc_curve(y_true, probs)
    best_i = int(np.argmax(tpr - fpr))
    t = float(thr[best_i])
    return t if np.isfinite(t) else 0.5   # roc_curve's first thr can be +inf


def cv_splitter():
    """Single-partition splitter -- required by cross_val_predict (threshold
    tuning), which needs every sample predicted exactly once."""
    return StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def cv_splitter_repeated():
    """Repeated splitter for model screening/tuning only -- averages scores
    over CV_REPEATS different fold assignments for a much more stable model
    ranking on this small n, at the cost of CV_REPEATS x more fit time (still
    fast at n~40)."""
    return RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)


def screen_models(X, y) -> pd.DataFrame:
    rows = []
    for name, base in get_models().items():
        pipe = build_pipeline(base)
        try:
            cv = cross_val_score(pipe, X, y, cv=cv_splitter_repeated(), scoring=SCORING, n_jobs=N_JOBS)
            pipe.fit(X, y)
            train_f1 = f1_score(y, pipe.predict(X))
            rows.append({"name": name, "cv_f1_mean": float(cv.mean()), "cv_f1_std": float(cv.std()),
                        "train_f1": float(train_f1), "overfit_gap": float(train_f1 - cv.mean())})
        except Exception as exc:
            print(f"  ! {name} failed: {exc}")
    return pd.DataFrame(rows).sort_values("cv_f1_mean", ascending=False).reset_index(drop=True)


# "select__k" searched alongside each model's own hyperparameters (including
# k="all" = selection effectively off) rather than fixed at SELECT_K, so
# GridSearchCV can find the dimensionality that actually works best per model.
_SELECT_K_GRID = [6, 8, 10, "all"]
PARAM_GRIDS = {
    "RandomForest": {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 8, 12],
                     "select__k": _SELECT_K_GRID},
    "ExtraTrees": {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 8, 12],
                   "select__k": _SELECT_K_GRID},
    "GradientBoosting": {"clf__n_estimators": [100, 200], "clf__max_depth": [2, 3],
                         "clf__learning_rate": [0.05, 0.1], "select__k": _SELECT_K_GRID},
    "LogisticRegression": {"clf__C": [0.1, 1.0, 10.0], "select__k": _SELECT_K_GRID},
    "SVC_RBF": {"clf__C": [1, 10], "clf__gamma": ["scale", 0.1], "select__k": _SELECT_K_GRID},
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


#  STEP 6 — EVALUATION                                                         #
def plot_confusion(y_true, y_pred, suffix: str = "") -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (REAL data){' - ' + suffix if suffix else ''}")
    save_fig(fig, MODELS_DIR, f"confusion_matrix{('_' + suffix) if suffix else ''}.png")


def plot_roc(model, X_te, y_te) -> float | None:
    if not hasattr(model.named_steps["clf"], "predict_proba"):
        return None
    proba = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, proba)
    a = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, label=f"AUC={a:.2f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=.4)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC — Endometriosis vs Control"); ax.legend()
    save_fig(fig, MODELS_DIR, "roc_curve.png")
    return round(float(a), 3)


def plot_learning_curve(model, X, y) -> None:
    try:
        sizes, tr, va = learning_curve(model, X, y, cv=CV_FOLDS, scoring=SCORING,
                                       train_sizes=np.linspace(0.3, 1.0, 5), n_jobs=N_JOBS)
    except Exception as exc:
        print(f"  ! learning curve skipped: {exc}"); return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, tr.mean(1), "o-", label="Training")
    ax.plot(sizes, va.mean(1), "s-", label="Cross-val")
    ax.set_xlabel("Training samples"); ax.set_ylabel(SCORING)
    ax.set_title("Learning curve (small n -> expect noise)"); ax.legend()
    save_fig(fig, MODELS_DIR, "learning_curve.png")


def plot_feature_importance(model, feat_names) -> None:
    clf = model.named_steps["clf"]
    imp = None
    if hasattr(clf, "feature_importances_"): imp = clf.feature_importances_
    elif hasattr(clf, "coef_"): imp = np.abs(clf.coef_).ravel()
    if imp is None: return
    # feat_names is the FULL engineered feature list (original taxa + ratio
    # features), but the pipeline's "select" step (SelectKBest) may have kept
    # only a subset -- map importances back to the surviving names via its
    # boolean mask, otherwise names and values silently misalign.
    if "select" in model.named_steps:
        mask = model.named_steps["select"].get_support()
        names = [n for n, keep in zip(feat_names, mask) if keep]
    else:
        names = list(feat_names)
    n = min(len(imp), len(names))
    s = pd.Series(imp[:n], index=names[:n]).sort_values()
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35*n)))
    s.plot.barh(ax=ax, color="crimson")
    ax.set_title("Feature importance — Endometriosis vs Control")
    save_fig(fig, MODELS_DIR, "feature_importance.png")


#  STEP 6b — EXPLAINABILITY (SHAP + LIME)                                     #
def explain_with_shap(model, X_tr: pd.DataFrame, X_te: pd.DataFrame, feat_names: list[str]) -> dict | None:
    """
    Model-agnostic SHAP explanation of the *whole pipeline* (impute + engineer +
    scale + classifier treated as one black-box function via model.predict_proba),
    so this works regardless of which model won screen_models(). Uses
    KernelExplainer, which is slower than TreeExplainer/LinearExplainer but is
    the one SHAP algorithm guaranteed to work for every model type in get_models()
    (tree ensembles, linear, SVM, MLP, KNN, NB, boosting) -- fine here since n<=50.

    Positive class = index 1 = "Endometriosis" (see CLASS_NAMES / LABEL_MAP).
    """
    if not HAVE_SHAP:
        print("  ! shap not installed -- skipping SHAP explanations. "
              "Install with: pip install shap")
        return None
    try:
        # Wrap predict_proba in a plain function rather than passing the bound
        # method directly: SHAP's KernelExplainer tries to tag the callable's
        # underlying object with a feature-names attribute, but sklearn's
        # Pipeline exposes feature_names_in_ as a read-only property on newer
        # versions ("property 'feature_names_in_' of 'Pipeline' object has no
        # setter"). A plain wrapper function has no such property to collide
        # with, and rebuilding the DataFrame keeps column names intact for
        # engineer_features() inside the pipeline.
        def _predict_fn(arr):
            return model.predict_proba(pd.DataFrame(np.asarray(arr), columns=feat_names))

        # background = summarised training set (kmeans centroids) to keep
        # KernelExplainer fast; falls back to raw rows if n is tiny.
        bg = shap.kmeans(X_tr, min(10, len(X_tr))) if len(X_tr) > 15 else X_tr
        explainer = shap.KernelExplainer(_predict_fn, bg)
        raw_sv = explainer.shap_values(X_te, nsamples="auto")

        # Normalise across shap versions: list-of-arrays (older) vs single
        # (n, features, n_classes) array (newer) -- both handled here.
        if isinstance(raw_sv, list):
            sv_pos = raw_sv[1]
        elif np.asarray(raw_sv).ndim == 3:
            sv_pos = np.asarray(raw_sv)[..., 1]
        else:
            sv_pos = np.asarray(raw_sv)

        mean_abs = pd.Series(np.abs(sv_pos).mean(axis=0), index=feat_names) \
                     .sort_values(ascending=False)
        mean_abs.to_csv(EXPLAIN_DIR / "shap_mean_abs_importance.csv", header=["mean_abs_shap"])

        # beeswarm-style summary plot (per-sample spread + direction)
        shap.summary_plot(sv_pos, X_te, feature_names=feat_names, show=False)
        save_fig(plt.gcf(), EXPLAIN_DIR, "shap_summary_beeswarm.png")

        # plain bar plot (global mean |SHAP|) -- easiest one to read at a glance
        shap.summary_plot(sv_pos, X_te, feature_names=feat_names, plot_type="bar", show=False)
        save_fig(plt.gcf(), EXPLAIN_DIR, "shap_summary_bar.png")

        # one worked example: waterfall for the first test-set sample, showing
        # exactly how each taxon pushed that one prediction away from baseline
        try:
            base_val = explainer.expected_value[1] if hasattr(explainer.expected_value, "__len__") \
                       else explainer.expected_value
            expl = shap.Explanation(values=sv_pos[0], base_values=base_val,
                                    data=X_te.iloc[0].values, feature_names=feat_names)
            shap.plots.waterfall(expl, show=False)
            save_fig(plt.gcf(), EXPLAIN_DIR, "shap_waterfall_example.png")
        except Exception as exc:
            print(f"  ! shap waterfall example skipped: {exc}")

        print(f"[shap] top taxa by mean |SHAP|: {mean_abs.head(5).to_dict()}")
        return mean_abs.round(6).to_dict()
    except Exception as exc:
        print(f"  ! SHAP explanation failed: {exc}")
        return None


def explain_with_lime(model, X_tr: pd.DataFrame, X_te: pd.DataFrame, y_pred, feat_names: list[str]) -> list[str]:
    """
    LIME complements SHAP with a per-sample, easy-to-read local explanation:
    "for THIS patient's sample, these taxa pushed the prediction toward
    Endometriosis / Control, by roughly this much." Saved for one representative
    Endometriosis-predicted and one Control-predicted test sample (whichever
    exist in the test set).
    """
    if not HAVE_LIME:
        print("  ! lime not installed -- skipping LIME explanations. "
              "Install with: pip install lime")
        return []
    saved = []
    try:
        explainer = LimeTabularExplainer(
            X_tr.values, feature_names=feat_names, class_names=list(CLASS_NAMES),
            mode="classification", discretize_continuous=True, random_state=RANDOM_STATE)
        y_pred = np.asarray(y_pred)
        targets = []
        endo_idx = np.where(y_pred == 1)[0]
        ctrl_idx = np.where(y_pred == 0)[0]
        if len(endo_idx): targets.append(("endometriosis_example", int(endo_idx[0])))
        if len(ctrl_idx): targets.append(("control_example", int(ctrl_idx[0])))

        for tag, i in targets:
            exp = explainer.explain_instance(X_te.iloc[i].values, model.predict_proba,
                                             num_features=min(10, len(feat_names)))
            exp.save_to_file(str(EXPLAIN_DIR / f"lime_{tag}.html"))
            fig = exp.as_pyplot_figure()
            fig.suptitle(f"LIME — test sample #{i} ({tag})")
            save_fig(fig, EXPLAIN_DIR, f"lime_{tag}.png")
            saved.append(tag)
            print(f"[lime] saved explanation for {tag} (test row {i})")
        return saved
    except Exception as exc:
        print(f"  ! LIME explanation failed: {exc}")
        return saved


def evaluate(model, X_te, y_te, y_pred=None, suffix: str = "") -> dict:
    if y_pred is None:
        y_pred = model.predict(X_te)   # fallback: default 0.5 threshold
    plot_confusion(y_te, y_pred, suffix=suffix)
    return {"accuracy": accuracy_score(y_te, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_te, y_pred),
            "f1": f1_score(y_te, y_pred),
            "report": classification_report(y_te, y_pred, target_names=CLASS_NAMES,
                                            output_dict=True, zero_division=0)}


#  MAIN                                                                        #
def study_crosstab(raw: pd.DataFrame) -> None:
    """Print & save study x diagnosis, so any label imbalance across studies
    (a batch-confound risk) is explicit and reviewable."""
    if "study" not in raw.columns:
        return
    ct = pd.crosstab(raw["study"], raw[TARGET_COL])
    ct.to_csv(REPORTS_DIR / "study_diagnosis_crosstab.csv")
    print("\n[confound-check] study x diagnosis:")
    print(ct.to_string())


def leave_one_study_out(raw: pd.DataFrame, X: pd.DataFrame, y, base) -> dict:
    """
    HONEST cross-cohort generalization (the number to cite in a paper).

    For each study: train on ALL OTHER studies, test on the held-out study --
    which the model has never seen. Random-split CV can look good by learning
    batch/study effects shared between train and test; LOSO cannot, because the
    test cohort is entirely external. If LOSO AUC ~ 0.5 while random-CV looks
    higher, the pooled signal was batch, not biology -- and that itself is the
    publishable finding.
    """
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    if "study" not in raw.columns or raw["study"].nunique() < 2:
        print("[LOSO] <2 studies present -- cross-cohort test not possible")
        return {}
    studies = sorted(raw["study"].unique())
    results = {}
    print("\n[LOSO] leave-one-study-out (train on others, test on the held-out cohort):")
    for held in studies:
        te = (raw["study"].values == held)
        tr = ~te
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            print(f"  {held}: skipped (train or test set is single-class)")
            continue
        pipe = build_pipeline(clone(base))
        pipe.fit(X.iloc[tr], y[tr])
        proba = pipe.predict_proba(X.iloc[te])[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = float(roc_auc_score(y[te], proba))
        bal = float(balanced_accuracy_score(y[te], pred))
        results[held] = {"n_test": int(te.sum()), "auc": round(auc, 3),
                         "balanced_accuracy": round(bal, 3)}
        print(f"  test on {held} (n={int(te.sum())}): AUC={auc:.3f}  bal_acc={bal:.3f}")
    if results:
        macro_auc = float(np.mean([r["auc"] for r in results.values()]))
        print(f"  --> mean cross-study AUC = {macro_auc:.3f}  (HONEST generalization; "
              f"~0.5 means batch, not biology)")
        results["mean_cross_study_auc"] = round(macro_auc, 3)
    return results


def main() -> None:
    ensure_dirs()
    print("========  Endometriosis vs Control — REAL 16S data (multi-study, gut-only)  ========")

    raw, X, y, feats = load_data()
    print(f"[load] {len(raw)} samples, {len(feats)} taxa")
    study_crosstab(raw)

    run_eda(raw, feats)
    taxa_tbl = taxa_group_comparison(raw, feats)
    print(taxa_tbl["highest_in"].to_string())
    compare_to_literature(taxa_tbl)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                              random_state=RANDOM_STATE)

    board = screen_models(X_tr, y_tr)
    board.to_csv(REPORTS_DIR / "model_leaderboard.csv", index=False)
    print("\n[screen] leaderboard:"); print(board.to_string(index=False))

    best = board.iloc[0]
    base = get_models()[best["name"]]
    model, params, cv_score = tune_best(best["name"], base, X_tr, y_tr)
    print(f"\n[best] {best['name']}  tuned_params={params}")

    # --- threshold tuning: find the F1-best cutoff using ONLY out-of-fold
    # predictions on the training set, then apply it (never touches X_te/y_te
    # until the final evaluate() call below) ---
    oof_proba = cross_val_predict(model, X_tr, y_tr, cv=cv_splitter(),
                                  method="predict_proba", n_jobs=N_JOBS)[:, 1]
    best_thresh = find_best_threshold(y_tr, oof_proba)
    print(f"\n[threshold] default 0.5 -> tuned {best_thresh:.3f} "
          f"(chosen from training-set out-of-fold predictions only)")

    test_proba = model.predict_proba(X_te)[:, 1]
    y_pred_default = (test_proba >= 0.5).astype(int)
    y_pred_tuned = (test_proba >= best_thresh).astype(int)

    metrics_default = evaluate(model, X_te, y_te, y_pred=y_pred_default, suffix="default_threshold")
    metrics = evaluate(model, X_te, y_te, y_pred=y_pred_tuned, suffix="tuned_threshold")
    plot_learning_curve(model, X_tr, y_tr)
    auc_score = plot_roc(model, X_te, y_te)
    plot_feature_importance(model, list(X_tr.columns) + RATIO_FEATURES)
    print(f"[test @0.5   ] acc={metrics_default['accuracy']:.3f}  f1={metrics_default['f1']:.3f}  "
          f"bal_acc={metrics_default['balanced_accuracy']:.3f}")
    print(f"[test @tuned ] acc={metrics['accuracy']:.3f}  f1={metrics['f1']:.3f}  "
          f"bal_acc={metrics['balanced_accuracy']:.3f}  auc={auc_score}")

    # --- honest cross-cohort generalization (train on one study, test on the
    # other). This is the number to report in a paper; the random-split test
    # above can be inflated by batch effects shared across the split. ---
    loso_results = leave_one_study_out(raw, X, y, base)

    print("\n[explainability] SHAP (global, model-agnostic via KernelExplainer)...")
    shap_importance = explain_with_shap(model, X_tr, X_te, feats)
    print("[explainability] LIME (per-sample, local)...")
    lime_saved = explain_with_lime(model, X_tr, X_te, y_pred_tuned, feats)

    joblib.dump(model, MODELS_DIR / "best_model.joblib")
    summary = {
        "data_note": "GENUINELY REAL 16S data (PRJNA1145097). n=50 is small -- pilot results.",
        "classes": list(CLASS_NAMES), "n_samples": int(len(raw)),
        "class_counts": {k: int(v) for k, v in raw[TARGET_COL].value_counts().items()},
        "bias_mitigation": {
            "class_weight": "balanced (applied to models that support it -- "
                            "reweights minority-class errors instead of resampling)",
            "decision_threshold_default": 0.5,
            "decision_threshold_tuned": round(best_thresh, 4),
            "threshold_chosen_from": "training-set out-of-fold predictions only (test set untouched)",
        },
        "best_model": {"name": best["name"], "tuned_params": params,
                       "tuned_cv_f1": None if cv_score is None else round(cv_score, 4),
                       "screen_cv_f1": round(float(best["cv_f1_mean"]), 4),
                       "overfit_gap": round(float(best["overfit_gap"]), 4)},
        "test_metrics_at_0.5": {k: round(float(v), 4) for k, v in metrics_default.items() if k != "report"},
        "test_metrics_at_tuned_threshold": {k: round(float(v), 4) for k, v in metrics.items() if k != "report"},
        "roc_auc_random_split": auc_score,
        "leave_one_study_out": loso_results,
        "generalization_note": "Cite leave_one_study_out (cross-cohort) as the "
            "honest generalization metric. roc_auc_random_split can be inflated "
            "by batch effects shared across a random train/test split when "
            "multiple studies are pooled.",
        "per_class_report": metrics["report"],
        "taxa_highest_in": taxa_tbl["highest_in"].to_dict(),
        "explainability": {
            "shap": {
                "ran": shap_importance is not None,
                "method": "KernelExplainer on the full pipeline (model-agnostic)",
                "mean_abs_shap_by_taxon": shap_importance,
                "files": ["shap_summary_beeswarm.png", "shap_summary_bar.png",
                         "shap_waterfall_example.png", "shap_mean_abs_importance.csv"]
                        if shap_importance is not None else [],
            },
            "lime": {
                "ran": len(lime_saved) > 0,
                "examples_saved": lime_saved,
                "files": [f"lime_{tag}.html / lime_{tag}.png" for tag in lime_saved],
            },
        },
    }
    (REPORTS_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\nDone. See results/endometriosis/ for plots, leaderboard, model, summary.")


if __name__ == "__main__":
    main()
