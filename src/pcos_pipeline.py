"""
===============================================================================
 pcos_pipeline.py  —  PCOS vs Control, REAL 16S data
===============================================================================
GOAL
----
Classify a stool sample as Control or PCOS from real 16S taxa abundances,
combined across every real PCOS stool study currently downloaded + QIIME2-
processed (see DATA_CSVS below), and report which taxa drive the split.

DATA
----
../dataset/pcos/SRP077213_PCOS_stool_real_taxa.csv   [READY]
  n=43 (24 PCOS / 19 Control). SRP077213, Lindheim/Insenser pilot study,
  PLOS ONE 2017 / PMC5207627, processed via QIIME2/DADA2/SILVA138. See
  ../../collected_dataset/04_processing_scripts/run_SRP077213_PCOS_stool.sh
  and ../dataset/pcos/fastq/download_remaining_stool.sh.
../dataset/pcos/SRP085887_PCOS_stool_real_taxa.csv   [NOT READY -- labels resolved]
  Zhang et al. 2017, n=48 (33 PCOS / 15 Control). Downloading now (see
  ../../../datasets/PCOS_Zhang2017_stool_16S/). Labels WERE unresolvable from
  SraRunTable/runinfo, but each sample's individual NCBI BioSample record has
  a "host disease" attribute (PCOS/control) not exposed in those flattened
  exports -- fetched all 48 directly, counts (33/15) match the paper exactly.
  See sample_labels.csv in that folder. Still needs QIIME2 processing before
  its real taxa CSV exists here.
../dataset/pcos/PRJNA694729_PCOS_stool_real_taxa.csv  [NOT READY]
  Shengjing Hospital, n=82 (45 PCOS/37 Control), labels already clean. Needs
  downloading (see ../../../datasets/PCOS_Shengjing_stool_16S/) and a
  single-end QIIME2 run (denoise-single, not denoise-paired).
All REAL sequenced data, not simulated. load_data() combines whatever subset
of the above actually exists at run time -- no code change needed as more
studies come online. Small n even combined: treat all results here as a
pilot, not a definitive benchmark.

PIPELINE (each a def, same shape as endometriosis_pipeline.py / the parent
GD-ML project this was split out of, so the two disease pipelines stay
directly comparable)
-----------------------------------------------------------------------------
    load_data                    -> read + combine every available real taxa CSV
    check_study_confounding      -> flag single-diagnosis studies before batch effects get mistaken for biology
    run_eda                      -> class balance, missingness, correlation
    taxa_group_comparison        -> which taxa are high in which group (+ graph, Mann-Whitney)
    compare_to_literature        -> sanity-check vs published taxon-disease direction
    engineer_features / build_pipeline -> impute + ratio features + scale + SelectKBest, in-pipeline
    screen_models / tune_best    -> CV sweep over a model zoo (incl. XGBoost if installed), then GridSearchCV the winner
    find_best_threshold          -> tune the decision cutoff from training-set out-of-fold predictions only
    explain_with_shap / explain_with_lime -> model-agnostic SHAP (global) + LIME (per-sample) if installed
    evaluate + plots             -> confusion matrix, ROC/AUC, feature importance
    main                         -> run everything, save results/

Run:  python3 pcos_pipeline.py
===============================================================================
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
                             roc_curve, auc, precision_recall_curve)

# Optional extras -- gradient-boosted trees often the strongest tabular model,
# and SHAP/LIME give model-agnostic interpretability. All three are genuinely
# optional: the pipeline runs fine without them (XGBoost just won't appear on
# the leaderboard; SHAP/LIME print a one-line "not installed" note and skip),
# matching the existing pattern already used for shap/lime in
# endometriosis_pipeline.py (this file was missing that parity until now).
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

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

# Multiple real 16S PCOS studies, combined when available. load_data() checks
# each path at run time and silently uses whichever ones already exist --
# nothing here needs editing again as more studies finish processing, this
# list just grows to what's true "today":
#   SRP077213    Lindheim/Insenser pilot, n=43 (24 PCOS/19 Control) -- READY,
#                already downloaded + QIIME2-processed (the only one in as of
#                this edit).
#   SRP085887    Zhang et al. 2017, n=48 (33 PCOS/15 Control) -- download in
#                progress (../../../datasets/PCOS_Zhang2017_stool_16S/).
#                Labels ARE resolved now: SraRunTable/runinfo has no usable
#                disease column, but each sample's individual NCBI BioSample
#                record has a "host disease" attribute (PCOS/control) that
#                isn't exposed in those flattened exports -- fetched all 48
#                directly, counts match the paper exactly. See
#                sample_labels.csv in that folder. Still needs QIIME2
#                processing via qiime2_to_pipeline.py before its real taxa
#                CSV exists here.
#   PRJNA694729  Shengjing Hospital, n=82 (45 PCOS/37 Control) -- labels ARE
#                already clean (PCOS_N/control_N in SampleName, captured in
#                sample_labels.csv by the download script), but this is
#                SINGLE-END data needing a separate denoise-single QIIME2
#                script (qiime2_process_single_end.sh) before
#                qiime2_to_pipeline.py can run on it. Fastest path to a real
#                second combinable PCOS study.
DATA_CSVS = [
    PROJECT_DIR / "dataset" / "pcos" / "SRP077213_PCOS_stool_real_taxa.csv",
    PROJECT_DIR / "dataset" / "pcos" / "SRP085887_PCOS_stool_real_taxa.csv",
    PROJECT_DIR / "dataset" / "pcos" / "PRJNA694729_PCOS_stool_real_taxa.csv",
]
RESULTS_DIR = PROJECT_DIR / "results" / "pcos"
EDA_DIR = RESULTS_DIR / "eda"
MODELS_DIR = RESULTS_DIR / "models"
REPORTS_DIR = RESULTS_DIR / "reports"
EXPLAIN_DIR = RESULTS_DIR / "explainability"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
CV_REPEATS = 5   # RepeatedStratifiedKFold for model screening/tuning: with a
                 # small training set, a single 5-fold split is noisy (which
                 # model "wins" can flip with the random fold assignment) --
                 # repeating with different shuffles and averaging gives a
                 # much more stable ranking.
SCORING = "f1"
N_JOBS = -1
EPS = 1e-6
SELECT_K = 10    # keep well below n_train to fight overfitting: rule of
                 # thumb is roughly >=4-5 samples per feature for stable fits

TARGET_COL = "diagnosis"
CLASS_NAMES = ("Control", "PCOS")
LABEL_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}
NON_FEATURE_COLS = ("sample_id", "study", "sample_type", "country", "subject_id", TARGET_COL)

RATIO_FEATURES = ["FB_ratio", "BL_ratio", "Fae_Bact_ratio", "BG_Lacto_ratio", "PB_ratio"]

# Qualitative sanity check only (see compare_to_literature) -- grounded in the
# general PCOS gut-dysbiosis literature (lower diversity/Firmicutes/Lactobacillus
# in PCOS, altered Bacteroidetes/Proteobacteria), same framework as the parent
# GD-ML project's LITERATURE_EXPECTATIONS table. Akkermansia and Escherichia_coli
# added for parity with endometriosis_pipeline.py now that both real taxa
# tables include them:
#   Akkermansia (muciniphila)     -> mucin-degrading commensal, generally
#                                     REDUCED in metabolic/inflammatory
#                                     dysbiosis states (incl. insulin resistance,
#                                     a core PCOS feature) -> expect higher in Control
#   Escherichia_coli (Esch-Shig.) -> classic Proteobacteria-bloom / inflammatory
#                                     marker, same family as the existing
#                                     Proteobacteria row -> expect higher in PCOS
# Prevotella and Ruminococcus are deliberately NOT included here: published
# directions are inconsistent/study-dependent for Prevotella in PCOS, and this
# pipeline's "Ruminococcus" column sums the true (often fiber-fermenting,
# health-associated) Ruminococcaceae genus together with SILVA's reclassified
# "[Ruminococcus]_gnavus/torques/gauvreauii_group" (Lachnospiraceae, often
# pro-inflammatory when elevated) -- two biologically distinct signals under
# one label, so asserting a single expected direction would overclaim.
#
# Collinsella/Roseburia/Desulfovibrio added (real, non-trivial-abundance genera
# confirmed present in the actual table-L6.tsv before adding):
#   Collinsella      -> repeatedly linked to insulin resistance/metabolic
#                        syndrome specifically (a core PCOS feature) -> expect
#                        higher in PCOS
#   Roseburia        -> butyrate-producing, generally protective/anti-
#                        inflammatory, commonly reduced in dysbiosis -> expect
#                        higher in Control
#   Desulfovibrio    -> sulfate-reducing, H2S-producing, associated with gut
#                        barrier disruption/inflammation in multiple dysbiosis
#                        contexts (grouped with Bilophila, see qiime2_to_pipeline.py)
#                        -> expect higher in PCOS
# Blautia/Alistipes/Parabacteroides/Sutterella are reported but NOT given an
# expected direction: literature on these in PCOS specifically is thin/mixed,
# so a hardcoded expectation would overclaim -- their group means are still
# shown in taxa_group_comparison.csv for exploratory reference.
#
# REVISED against the 2023-2025 evidence base -- see ../LITERATURE_REFERENCES.md
# for full citations, quotes, and the reasoning behind every entry AND every
# deliberate omission. Format: taxon -> (expected_higher_in, confidence, refs).
#
# TWO IMPORTANT CHANGES made after reading the source papers properly:
#
#   1. Firmicutes and Bacteroidetes were REMOVED as expectations. Lindheim2017
#      -- the paper behind our own primary dataset (SRP077213) -- states
#      directly: "No statistically significant differences were observed
#      between PCOS patients and controls in bacterial taxa with a relative
#      abundance >1% or in the Firmicutes:Bacteroidetes ratio." Asserting a
#      direction for these was scoring the pipeline against a difference the
#      source study explicitly reported as absent, which made the
#      literature-match score misleading rather than informative.
#
#   2. shannon_diversity is now marked "contested", not asserted-strong.
#      Lindheim2017 found ~15% lower alpha diversity in PCOS (p=0.027/0.030)
#      and Zhang2017 found lower richness, but the much larger RepSci2024
#      meta-analysis (14 studies, 513 PCOS / 435 controls) found NO significant
#      biodiversity change. Pooled, the effect washes out.
#
# Confidence tiers: "strong" = supported by meta-analysis or MR; "moderate" =
# multiple consistent primary studies; "weak" = single study or extrapolated
# from general dysbiosis biology; "contested" = studies actively disagree.
LITERATURE_EXPECTATIONS = {
    # --- higher in PCOS ---
    "Proteobacteria":   ("PCOS", "strong",    ["RepSci2024", "Zhang2017"]),
    "Escherichia_coli": ("PCOS", "strong",    ["MR2023", "Zhang2017"]),
    "Bacteroides":      ("PCOS", "moderate",  ["MR2023", "Zhang2017"]),
    "Parabacteroides":  ("PCOS", "moderate",  ["MR2023", "Zhang2017"]),
    "Blautia":          ("PCOS", "weak",      ["Zhang2017"]),
    "Collinsella":      ("PCOS", "weak",      ["Zhang2017"]),
    "Desulfovibrio":    ("PCOS", "weak",      ["general-dysbiosis"]),
    # --- higher in Control ---
    "Lactobacillus":    ("Control", "moderate", ["Zhang2017"]),
    "Akkermansia":      ("Control", "moderate", ["Zhang2017"]),
    "Faecalibacterium": ("Control", "moderate", ["SCFA-literature"]),
    "Bifidobacterium":  ("Control", "weak",     ["general"]),
    "Roseburia":        ("Control", "weak",     ["butyrate-literature"]),
    "shannon_diversity":("Control", "contested",["Lindheim2017", "Zhang2017", "RepSci2024"]),
    # DELIBERATELY OMITTED (reported in taxa_group_comparison.csv, but NOT
    # scored): Firmicutes, Bacteroidetes (see note 1 above); Prevotella
    # (inconsistent across PCOS studies); Ruminococcus (our column merges true
    # Ruminococcaceae with SILVA's reclassified [Ruminococcus]_gnavus_group --
    # MR2023 implicates gnavus specifically but we cannot isolate it);
    # Alistipes, Sutterella (thin/mixed evidence).
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
    Reads and concatenates every real taxa table in `paths` that actually
    exists yet -- studies still mid-download or not-yet-QIIME2-processed are
    skipped with a note, not treated as an error, so this pipeline always
    runs on whatever real data is currently available and automatically picks
    up new studies the moment their processed CSV appears (no code change
    needed here when that happens).
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
            "No real taxa tables found at all. At minimum run:\n"
            "  bash ../../collected_dataset/04_processing_scripts/run_SRP077213_PCOS_stool.sh\n"
            "(needs conda + the qiime2-amplicon env active; FASTQ for all 43 stool\n"
            "samples is already downloaded in ../../../datasets/PCOS_Lindheim_stool_16S/)"
        )
    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw[TARGET_COL] = raw[TARGET_COL].astype(str).str.strip()
    raw = raw[raw[TARGET_COL].isin(LABEL_MAP)].copy()
    y = raw[TARGET_COL].map(LABEL_MAP).to_numpy(dtype=int)
    all_missing = {c for c in raw.columns if raw[c].isna().all()}
    drop = set(NON_FEATURE_COLS) | all_missing
    feats = [c for c in raw.columns if c not in drop and pd.api.types.is_numeric_dtype(raw[c])]
    X = raw[feats].copy()
    if "study" in raw.columns and raw["study"].nunique() > 1:
        print(f"[load] combined {raw['study'].nunique()} studies: "
              f"{raw['study'].value_counts().to_dict()}")
    return raw, X, y, feats


def check_study_confounding(raw: pd.DataFrame) -> None:
    """
    When multiple studies are combined, a model can trivially "cheat" by
    learning batch/study effects (different labs, DNA extraction kits, primer
    sets, sequencing runs, geography) instead of real PCOS biology --
    especially if a study is single-diagnosis (all-PCOS or all-Control),
    since then study identity and diagnosis are perfectly confounded for
    those samples. "study" itself is already excluded from the feature set
    (see NON_FEATURE_COLS), but that alone does NOT protect against this --
    systematic batch effects can leak into every OTHER feature as a
    consistent offset that correlates with diagnosis purely because it
    correlates with which study a sample came from. Not something modeling
    choices alone can fix -- flagged here so results are read with the right
    skepticism, and to make an informed call about excluding a confounded
    study rather than silently keeping or dropping it.
    """
    if "study" not in raw.columns or raw["study"].nunique() < 2:
        return
    ct = pd.crosstab(raw["study"], raw[TARGET_COL])
    print("\n[batch-effect check] samples by study x diagnosis:")
    print(ct.to_string())
    for study in ct.index:
        row = ct.loc[study]
        if (row > 0).sum() == 1:
            print(f"  ! WARNING: study '{study}' is single-diagnosis only "
                  f"({row[row > 0].index[0]}) -- study identity and diagnosis "
                  f"are perfectly confounded for these samples. Any accuracy "
                  f"gain from adding this study could reflect batch effects, "
                  f"not real PCOS biology. Treat feature importance/SHAP "
                  f"results with extra caution while this study is single-"
                  f"diagnosis; consider re-running with it excluded as a "
                  f"sensitivity check.")
    ct.to_csv(REPORTS_DIR / "study_diagnosis_crosstab.csv")

    # --- how much can study membership ALONE predict diagnosis? ---
    # If the studies have very different class balance, a model can gain
    # apparent accuracy just by recognising which cohort a sample came from.
    # Quantifying it means we can state plainly how much of any performance
    # could be confounding rather than biology.
    s = (raw["study"] == ct.index[-1]).to_numpy()
    y = (raw[TARGET_COL] == CLASS_NAMES[1]).to_numpy()
    if s.any() and (~s).any() and y.any() and (~y).any():
        p1, p0 = s[y].mean(), s[~y].mean()
        a = p1 * (1 - p0) + 0.5 * (p1 * p0 + (1 - p1) * (1 - p0))
        print(f"  study-membership alone predicts diagnosis with AUC = {max(a, 1-a):.3f} "
              f"(0.5 = no confounding). Any model AUC should be judged against this "
              f"floor, not against 0.5.")


def check_batch_markers(raw: pd.DataFrame, features: list[str]) -> pd.DataFrame | None:
    """
    Flag features that identify the STUDY better than they identify the
    DISEASE -- i.e. technical/batch artifacts masquerading as biology.

    Motivation (a real failure this caught): with two studies sequenced at
    ~43,900 vs ~6,300 reads/sample, unrarefied shannon_diversity separated the
    two studies with AUC = 0.998 while separating PCOS from Control at only
    0.730. It was effectively a batch barcode, yet SHAP ranked it the #2 most
    important feature. Feature importance on such a variable describes the
    sequencing run, not the patient.

    Uses a rank-based AUC (equivalent to the Mann-Whitney U statistic) and is
    reported symmetrically, so 0.5 = uninformative and 1.0 = perfect separator
    regardless of direction.
    """
    if "study" not in raw.columns or raw["study"].nunique() < 2:
        return None

    def _auc(x: pd.Series, g: np.ndarray) -> float:
        ok = x.notna().to_numpy()
        xv, gv = x[ok], g[ok]
        n1, n0 = int(gv.sum()), int((~gv).sum())
        if n1 == 0 or n0 == 0:
            return np.nan
        r = xv.rank().to_numpy()
        u = r[gv].sum() - n1 * (n1 + 1) / 2
        a = u / (n1 * n0)
        return float(max(a, 1 - a))

    study = (raw["study"] == raw["study"].unique()[0]).to_numpy()
    dis = (raw[TARGET_COL] == CLASS_NAMES[1]).to_numpy()
    rows = [{"feature": f,
             "auc_predicts_study": _auc(raw[f], study),
             "auc_predicts_diagnosis": _auc(raw[f], dis)} for f in features]
    tbl = pd.DataFrame(rows)
    tbl["batch_dominated"] = tbl["auc_predicts_study"] > tbl["auc_predicts_diagnosis"]
    tbl["risk"] = np.where(tbl["auc_predicts_study"] > 0.90, "SEVERE",
                    np.where(tbl["auc_predicts_study"] > 0.75, "high", ""))
    tbl = tbl.sort_values("auc_predicts_study", ascending=False)
    tbl.to_csv(REPORTS_DIR / "batch_marker_check.csv", index=False)

    severe = tbl.loc[tbl["risk"] == "SEVERE", "feature"].tolist()
    n_bad = int(tbl["batch_dominated"].sum())
    print(f"\n[batch-marker check] {n_bad}/{len(tbl)} features identify the study "
          f"better than the diagnosis")
    if severe:
        print(f"  ! SEVERE batch markers (study-AUC > 0.90): {severe}")
        print(f"    Treat these features' importance/SHAP values as untrustworthy -- "
              f"they may describe the sequencing run rather than the patient. If "
              f"shannon_diversity is listed, the studies were likely NOT rarefied to "
              f"a common depth (see RAREFY_DEPTH in qiime2_process.sh).")
    print(f"  full table -> {(REPORTS_DIR / 'batch_marker_check.csv').name}")
    return tbl


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
    consolidated 2017-2025 evidence base (see ../LITERATURE_REFERENCES.md).

    Reports the overall match rate AND a separate match rate restricted to
    high-confidence ("strong"/"moderate") expectations. That split matters: a
    mismatch on a "weak" or "contested" taxon is unremarkable and says little,
    whereas a mismatch on a meta-analysis-backed taxon like Proteobacteria is a
    genuine signal that something may be wrong with the data or processing.
    Scoring them all in one undifferentiated number hides that distinction.
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
        print(f"[literature-check] PCOS: {n_match}/{len(report)} taxa match published direction")
        high = report[report["evidence_confidence"].isin(["strong", "moderate"])]
        if not high.empty:
            n_hi = int(high["matches_literature"].sum())
            print(f"[literature-check] high-confidence subset (strong/moderate): "
                  f"{n_hi}/{len(high)} match")
            missed = high.loc[~high["matches_literature"], "taxon"].tolist()
            if missed:
                print(f"  ! high-confidence MISMATCHES: {missed}")
                print(f"    These are meta-analysis/multi-study backed. If several are wrong, "
                      f"suspect a data/processing problem (e.g. wrong DADA2 truncation "
                      f"lengths for a study's read length) before reinterpreting the biology.")
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
#      X_te/y_te are never touched again until the final evaluate() call.
#   2. Every step below (impute, engineer, scale, select) lives INSIDE this
#      sklearn Pipeline, not as a one-time transform on the whole dataset.
#      cross_val_score/GridSearchCV both refit the entire pipeline fresh on
#      each fold's training split, so imputation values, ratio-feature stats,
#      scaling mean/std, and SelectKBest's F-scores are all computed from that
#      fold's training rows only -- the held-out fold never leaks into any of
#      these fitted statistics.
def build_pipeline(clf) -> Pipeline:
    pipe = Pipeline([
        ("impute", KNNImputer(n_neighbors=5)),
        ("engineer", FunctionTransformer(engineer_features)),
        ("scale", StandardScaler()),
        # univariate ANOVA F-test feature selection, refit fresh inside every
        # CV fold (never sees the held-out fold, so no leakage) -- cuts
        # dimensionality from ~26 taxa+ratios down toward SELECT_K to fight
        # overfitting. k is searched in PARAM_GRIDS for GridSearchCV-tuned
        # models, with SELECT_K as the fallback default otherwise.
        ("select", SelectKBest(score_func=f_classif, k=SELECT_K)),
        ("clf", clf),
    ])
    pipe.set_output(transform="pandas")
    return pipe


#  STEP 5 — MODELS, SCREEN, TUNE                                               #
def get_models() -> dict:
    # class_weight="balanced" on every model that supports it, matching
    # endometriosis_pipeline.py: reweights the loss so errors on the minority
    # class (Control, 19/43) cost as much as errors on the majority class
    # (PCOS, 24/43). GradientBoosting, AdaBoost, GaussianNB, KNN and MLP have
    # no class_weight knob in sklearn -- left as-is for those.
    #
    # OVERFITTING: tree ensembles previously ran fully unconstrained during
    # screen_models() (only PARAM_GRIDS-tuned versions had depth limits), so
    # the screening leaderboard itself showed train_f1=1.0 for every one of
    # them -- trivially memorizing ~34 training rows. min_samples_leaf/
    # max_depth defaults below make even the untuned screening pass realistic,
    # not just the final GridSearchCV'd model.
    models = {
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
        # UNDERFITTING: this model was collapsing to train_f1~0.19 (and 0.0 in
        # the endometriosis pipeline) -- a 2-layer (32,16) network with
        # early_stopping's default validation_fraction=0.1 leaves only a
        # couple of samples for internal validation inside an already-tiny
        # ~34-row CV fold, an unusably noisy stopping signal that was halting
        # training almost immediately. Simplified to a single small hidden
        # layer, disabled early_stopping (nothing left to validate against at
        # this n), added L2 (alpha) and more iterations so it can actually
        # converge. If it still underperforms after this, that itself is a
        # legitimate finding -- neural nets generally need far more than
        # n~40-50 samples to be competitive on tabular data, not a sign this
        # specific config is still broken.
        "MLP": MLPClassifier(hidden_layer_sizes=(8,), alpha=1.0, max_iter=3000,
                             early_stopping=False, random_state=RANDOM_STATE),
    }
    # XGBoost is often the strongest tabular-data model, but n<=43 (until more
    # studies combine in) means it needs the same overfitting guardrails as
    # the other tree ensembles above: shallow max_depth, a real min_child_weight
    # (XGBoost's equivalent of min_samples_leaf), subsample/colsample<1, and a
    # small learning_rate with n_estimators capped rather than left to grow
    # unchecked. Optional -- only added if the xgboost package is installed.
    if HAVE_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_lambda=1.0, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=N_JOBS,
        )
    return models


def cv_splitter():
    return StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def cv_splitter_repeated():
    """Repeated splitter for model screening/tuning -- averages scores over
    CV_REPEATS different fold assignments for a much more stable model
    ranking on this small n (n=43), at the cost of CV_REPEATS x more fit time
    (still fast at this scale)."""
    return RepeatedStratifiedKFold(n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)


def find_best_threshold(y_true, probs) -> float:
    """
    Sweep decision thresholds and return the one maximizing F1 on the
    POSITIVE (PCOS) class. Only ever called on training-set out-of-fold
    predictions (see main()) -- the test set never influences this choice, so
    picking a non-0.5 threshold here is not a form of leakage. Ported from
    endometriosis_pipeline.py for parity: the default 0.5 cutoff is arbitrary,
    not something that should be assumed optimal for an imbalanced small-n
    problem.
    """
    prec, rec, thresh = precision_recall_curve(y_true, probs)
    f1s = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec + 1e-12), 0.0)
    best_i = int(np.argmax(f1s[:-1])) if len(thresh) else int(np.argmax(f1s))
    return float(thresh[best_i]) if len(thresh) else 0.5


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
    "XGBoost": {"clf__n_estimators": [100, 200], "clf__max_depth": [2, 3],
               "clf__learning_rate": [0.03, 0.05, 0.1], "select__k": _SELECT_K_GRID},
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
    ax.set_title("ROC — PCOS vs Control"); ax.legend()
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
    s.plot.barh(ax=ax, color="teal")
    ax.set_title("Feature importance — PCOS vs Control")
    save_fig(fig, MODELS_DIR, "feature_importance.png")


#  STEP 6b — EXPLAINABILITY (SHAP + LIME)                                     #
def explain_with_shap(model, X_tr: pd.DataFrame, X_te: pd.DataFrame, feat_names: list[str]) -> dict | None:
    """
    Model-agnostic SHAP explanation of the *whole pipeline* (impute + engineer
    + scale + classifier treated as one black-box function via
    model.predict_proba), so this works regardless of which model won
    screen_models(). Uses KernelExplainer, which is slower than
    TreeExplainer/LinearExplainer but is the one SHAP algorithm guaranteed to
    work for every model type in get_models() (tree ensembles, linear, SVM,
    MLP, KNN, NB, boosting, XGBoost) -- fine here since n is small.

    Positive class = index 1 = "PCOS" (see CLASS_NAMES / LABEL_MAP).
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
    PCOS / Control, by roughly this much." Saved for one representative
    PCOS-predicted and one Control-predicted test sample (whichever exist in
    the test set).
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
        pcos_idx = np.where(y_pred == 1)[0]
        ctrl_idx = np.where(y_pred == 0)[0]
        if len(pcos_idx): targets.append(("pcos_example", int(pcos_idx[0])))
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
def main() -> None:
    ensure_dirs()
    print("========  PCOS vs Control — REAL 16S data (combined studies)  ========")
    print("[note] SRP077213 (n=43, 24 PCOS/19 Control) is the only study fully "
          "processed as of this run -- Zhang2017/Shengjing join automatically "
          "once their real taxa CSVs exist (see DATA_CSVS comment above). "
          "Small n -- pilot results, not a final benchmark.")

    raw, X, y, feats = load_data()
    print(f"[load] {len(raw)} samples, {len(feats)} taxa")
    check_study_confounding(raw)
    batch_tbl = check_batch_markers(raw, feats)

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
    # until the final evaluate() call below). Ported from
    # endometriosis_pipeline.py for parity -- 0.5 is an arbitrary default, not
    # something to assume is optimal on an imbalanced, small-n problem. ---
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

    print("\n[explainability] SHAP (global, model-agnostic via KernelExplainer)...")
    shap_importance = explain_with_shap(model, X_tr, X_te, feats)
    print("[explainability] LIME (per-sample, local)...")
    lime_saved = explain_with_lime(model, X_tr, X_te, y_pred_tuned, feats)

    joblib.dump(model, MODELS_DIR / "best_model.joblib")
    summary = {
        "data_note": "GENUINELY REAL 16S data. Small n -- pilot results, not a final benchmark.",
        "classes": list(CLASS_NAMES), "n_samples": int(len(raw)),
        "class_counts": {k: int(v) for k, v in raw[TARGET_COL].value_counts().items()},
        "studies_combined": (raw["study"].value_counts().to_dict()
                             if "study" in raw.columns else None),
        "batch_markers": (None if batch_tbl is None else {
            "n_features_study_dominated": int(batch_tbl["batch_dominated"].sum()),
            "n_features_total": int(len(batch_tbl)),
            "severe_batch_markers": batch_tbl.loc[batch_tbl["risk"] == "SEVERE",
                                                  "feature"].tolist(),
            "note": "Features whose study-AUC exceeds their diagnosis-AUC may encode "
                    "sequencing/lab differences rather than biology; their SHAP and "
                    "feature-importance values should not be interpreted clinically.",
        }),
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
        "roc_auc": auc_score,
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
    print("\nDone. See results/pcos/ for plots, leaderboard, model, summary, explainability/.")


if __name__ == "__main__":
    main()