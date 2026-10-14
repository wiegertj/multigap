#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyses for investigating the diagnostic and clinic associations between the computed gaps and
respective measures. Generates the association graph figures as well as the attenuation analysis figure.
"""

import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch, Rectangle

from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================

OUT_DIR = ...
RESULTS_DIR = os.path.join(OUT_DIR, "_main_MAIN_groups")
CSV_INTERNAL_ALL_OTHER_GAPS = ...

CSV_DIAGNOSES = (...)
CSV_MEAN_MOTION = ...

INPUT_FILES = {
    "freesurfer": (
        ...
    ),
    "connectivity": (
        ...
    ),
}

FIG2_OUT = os.path.join(
    RESULTS_DIR,
    ...,
)
FIG2_OUT_PDF = os.path.join(
    RESULTS_DIR,
    ...,
)

INDIVIDUAL_PNG_DPI = 450

FIG2_PANELS_C_VERTICAL_OUT = os.path.join(
    RESULTS_DIR,
    ...,
)
FIG2_PANEL_C1_OUT = os.path.join(
    RESULTS_DIR,
    ...,
)
FIG2_PANEL_C2_OUT = os.path.join(
    RESULTS_DIR,
    ...,
)

FIG4_MAIN_OUT = os.path.join(
    RESULTS_DIR,
    ...,
)
FIG4_MAIN_OUT_PDF = os.path.join(
    RESULTS_DIR,
    ...,
)
FIGS_ALL_OUT = os.path.join(
    RESULTS_DIR,
    ...,
)
FIGS_ALL_OUT_PDF = os.path.join(
    RESULTS_DIR,
    ...,
)
FIG4_ALL_SUMMARY_CSV = os.path.join(
    RESULTS_DIR,
    ...,
)
FIG4_MAIN50_SUMMARY_CSV = os.path.join(
    RESULTS_DIR,
    ...,
)
FIG4_DETAILED_CSV = os.path.join(
    RESULTS_DIR,
    ...,
)

FIG4_DPI = 450
FIG4_MIN_MODEL_N = 30
FIG4_MAIN_MIN_MUTUAL_ATTENUATION_PCT = 33.0
FIG4_MAX_SEGMENTS_PER_ROW = 3
FIG4_XMAX = 100.0
FIG4_ALLGAPS_TEXT_X = 107.0
FIG4_ALLGAPS_MARKER_X = 113.0
FIG4_ROW_FONT_SIZE = 10.9
FIG4_XTICK_FONT_SIZE = 9.8
FIG4_TITLE_FONT_SIZE = 17.0
FIG4_LABEL_FONT_SIZE = 8.0
FIG4_LEGEND_FONT_SIZE = 9.1
FIG4_ALLGAPS_NUMBER_FONT_SIZE = 8.6
FIG4_OTHER_COLOR = "0.86"

CSV_MAIN_PREDICTOR_EDGES_WITH_CORRECTED_FLAG = os.path.join(
    RESULTS_DIR,
    ...,
)

ID_COL = "Participant ID"
SPLIT_GROUP_COL = "split_group"
TRAIN_HC_SPLIT_VALUE = "train_hc"

MEAN_MOTION_ID_COL = "eid"
MEAN_MOTION_VALUE_COL = "framewise_dis"
MEAN_MOTION_COVARIATE_COL = "mean_motion"

FEATURE_TARGET_EXCLUSIONS = {
    "freesurfer": {
        "Sleep_duration",
    },
    "connectivity": {
    },
}

REGRESSION_TARGETS = [
    "brain_age",
    "Fluid_intelligence_score",
    "body_fat_percentage",
    "Sleep_duration",
    "Hand_grip_strength_mean",
    "bmi",
    "diastolic_bp",
    "trail_making",
]

TARGET_ORDER = REGRESSION_TARGETS.copy()
FEATURE_SET_ORDER = ["freesurfer", "connectivity"]

GRAPH_EXCLUDE_TARGETS = {}

GROUP_ORDER_FULL = [
    "MAIN_AUD",
    "MAIN_MDD",
    "MAIN_ANX",
]

MDD_COLOR = "red"
ANX_COLOR = "green"
AUD_COLOR = "blue"

HC_COL = "MAIN_HC"
AUD_COL = "MAIN_AUD"
MDD_COL = "MAIN_MDD"
ANX_COL = "MAIN_ANX"

DIAGNOSIS_COLUMNS_FOR_PLOTTING = [
    HC_COL,
    AUD_COL,
    MDD_COL,
    ANX_COL,
]

BIP_SCZ_EXCLUSION_COLUMNS = []

COVARIATE_CANDIDATES = [
    "sex",
    "age_assessment_center_2",
    "site",
    "etiv",
    "mean_motion",
]

COVARIATE_SOURCE_ALIASES = {
    "etiv": ["etiv", "eTIV", "tiv", "TIV", "ICV", "EstimatedTotalIntraCranialVol"],
}

SCORE_PREDICTOR_COLS = [
    "audit_obs",
    "phq9_sum",
    "clinical_antidepressant_n_answered",
]

SCORE_PREDICTOR_LABELS = {
    "audit_obs": "AUDIT",
    "phq9_sum": "PHQ-9",
    "clinical_antidepressant_n_answered": "#Antidepressants",
}

RANK_NORMALIZE_SCORE_PREDICTORS = True

FDR_ALPHA = 0.05
GAP_GRAPH_MIN_ABS_BETA_STD = 0.03
GAP_GRAPH_LABEL_MIN_ABS_BETA_STD = 0.03

RANDOM_SEED = 123

GRAPH_TITLE_Y = 1.045
GRAPH_TITLE_PAD = 12
GRAPH_TITLE_HEADROOM_FRAC = 0.22
GRAPH_NODE_MARGIN_FRAC = 0.10

PANEL_LABEL_SIZE = 46
TITLE_SIZE = 34
AXIS_LABEL_SIZE = 29
TICK_LABEL_SIZE = 24
LEGEND_SIZE = 23

GRAPH_GAP_FONT_SIZE = 19.5
GRAPH_PRED_FONT_SIZE = 20.0
GRAPH_EDGE_LABEL_SIZE = 16.5

GRAPH_PRED_EDGE_MIN_LW = 1.8
GRAPH_PRED_EDGE_MAX_LW = 4.2
GRAPH_PRED_EDGE_ALPHA = 0.88

GRAPH_EDGE_LABEL_FRACTION_FROM_PREDICTOR = 1.0 / 3.0
GRAPH_PRED_GAP_MIN_DIST = 3.00
GRAPH_EDGE_LABEL_BBOX_ALPHA = 0.82
GRAPH_EDGE_LABEL_BBOX_PAD = 0.06
GRAPH_CORRECTED_LABEL_BBOX_PAD = 0.12

GRAPH_CORRECTED_LABEL_EDGE_COLOR = "black"
GRAPH_CORRECTED_LABEL_EDGE_WIDTH = 1.8
GRAPH_CORRECTED_LABEL_FACE_ALPHA = 0.96

GRAPH_NODE_LABEL_MAX_CHARS = 5
GRAPH_LEGEND_SIZE = 18.0
GRAPH_PANEL_TITLE_SIZE = 28.0
GRAPH_NOTE_SIZE = 15.0
GRAPH_DRAW_GAP_GAP_EDGES = False
GRAPH_PRIMARY_MODEL_NOTE = ""

plt.rcParams.update({
    "font.size": TICK_LABEL_SIZE,
    "axes.titlesize": TITLE_SIZE,
    "axes.labelsize": AXIS_LABEL_SIZE,
    "xtick.labelsize": TICK_LABEL_SIZE,
    "ytick.labelsize": TICK_LABEL_SIZE,
    "legend.fontsize": LEGEND_SIZE,
})

GRAPH_GAP_CIRCLE_PAD = 0.48
GRAPH_GAP_CIRCLE_LINEWIDTH = 1.45
GRAPH_GAP_NODE_ALPHA = 0.96
GRAPH_GAP_HATCH_FREESURFER = None
GRAPH_GAP_HATCH_CONNECTIVITY = "///"

GRAPH_PRED_BOX_PAD = 0.42
GRAPH_PRED_BOX_ROUNDING = 0.13
GRAPH_PRED_BOX_LINEWIDTH = 1.45
GRAPH_EDGE_LENGTH_SCALE = 1.38

GRAPH_LAYOUT_SCALE = 2.70
GRAPH_LAYOUT_K = 4.65
GRAPH_GAP_MIN_DIST = 1.02
GRAPH_PRED_OUTWARD_PUSH = 1.42
GRAPH_PRED_MIN_DIST_FRAC = 0.35
GRAPH_COMBINED_MIN_DIST = 0.98
GRAPH_LIMIT_PAD = 1.34

GRAPH_USE_HORIZONTAL_MODALITY_LAYOUT = True
GRAPH_TARGET_PAIRS_PER_ROW = 2
GRAPH_TARGET_ROW_SPACING = 1.42
GRAPH_MODALITY_INNER_X = 2.65
GRAPH_MODALITY_OUTER_X = 5.15

GRAPH_PREDICTOR_AXIS_MAX_PULL = 1.30
GRAPH_PREDICTOR_AXIS_Y_MARGIN_FRAC = 0.06
GRAPH_PREDICTOR_AXIS_EPS = 1e-12


# =============================================================================
# BASIC HELPERS
# =============================================================================

def canonicalize_id_series(x):
    return (
        x.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def drop_train_hc(df, label="dataframe"):
    if SPLIT_GROUP_COL not in df.columns:
        print(f"[INFO] {label}: no {SPLIT_GROUP_COL!r} column; no train-HC filtering.")
        return df.copy()

    before = len(df)
    split = df[SPLIT_GROUP_COL].astype(str).str.strip()
    out = df.loc[~split.eq(TRAIN_HC_SPLIT_VALUE)].copy()

    print(
        f"[INFO] {label}: removed {before - len(out):,} rows with "
        f"{SPLIT_GROUP_COL} == {TRAIN_HC_SPLIT_VALUE!r}; remaining n={len(out):,}"
    )

    return out


def safe_numeric(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def binary_col_positive(df, col):
    return pd.to_numeric(df[col], errors="coerce").fillna(0).eq(1)


def add_panel_label(ax, label, x=-0.045, y=1.045):
    if hasattr(ax, "text2D"):
        ax.text2D(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
            va="top",
            ha="right",
        )
    else:
        ax.text(
            x,
            y,
            label,
            transform=ax.transAxes,
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
            va="top",
            ha="right",
        )


def group_mask(df, group):
    if group == "heldout_hc":
        if HC_COL not in df.columns:
            return pd.Series(False, index=df.index)
        return binary_col_positive(df, HC_COL)
    if group in {AUD_COL, MDD_COL, ANX_COL}:
        if group not in df.columns:
            return pd.Series(False, index=df.index)
        return binary_col_positive(df, group)
    if "analysis_label" in df.columns:
        return df["analysis_label"].astype(str).str.strip().eq(str(group))
    return pd.Series(False, index=df.index)


def gap_to_feature_target(gap):
    if "__" in gap:
        fs, target = gap.split("__", 1)
    else:
        fs, target = "", gap
    return fs, target


def limit_graph_node_label(label, max_chars=GRAPH_NODE_LABEL_MAX_CHARS):
    text = str(label).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def short_gap_name(name):
    if "__" in name:
        _, target = name.split("__", 1)
    else:
        target = name

    replacements = {
        "brain_age": "Age",
        "Fluid_intelligence_score": "Intel",
        "body_fat_percentage": "BodyF",
        "Sleep_duration": "Sleep",
        "Systolic_blood_pressure": "SBP",
        "Hand_grip_strength_mean": "Grip",
        "Pulse_rate_automated_reading": "Pulse",
        "bmi": "BMI",
        "diastolic_bp": "DBP",
        "trail_making": "Trail",
    }

    return limit_graph_node_label(replacements.get(target, target))


def short_predictor_name(label):
    replacements = {
        "AUD vs HC": "AUD",
        "MDD vs HC": "MDD",
        "ANX vs HC": "ANX",
        "AUDIT": "AUDIT-C",
        "PHQ-9": "PHQ-9",
        "#Antidepressants": "ADcnt",
    }
    text = replacements.get(str(label), label)
    if text == "AUDIT-C":
        return text
    return limit_graph_node_label(text)


def ranknorm_series(x):
    x = pd.to_numeric(x, errors="coerce").astype(float)
    out = pd.Series(np.nan, index=x.index, dtype=float)
    valid = x.notna()
    n = int(valid.sum())

    if n < 3 or x.loc[valid].nunique(dropna=True) < 2:
        return out

    ranks = x.loc[valid].rank(method="average")
    quantiles = (ranks - 0.5) / n
    out.loc[valid] = norm.ppf(quantiles)

    return out


def zscore_series(x):
    x = pd.to_numeric(x, errors="coerce").astype(float)
    sd = x.std(ddof=0)

    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=x.index, dtype=float)

    return (x - x.mean()) / sd


def is_valid_graph_label(x):
    if x is None or pd.isna(x):
        return False

    s = str(x).strip()

    if s == "":
        return False
    if s.lower() in {"none", "nan", "null"}:
        return False

    return True


def is_graph_excluded_gap(gap):
    if gap is None or pd.isna(gap):
        return False

    _, target = gap_to_feature_target(str(gap).strip())
    return target in GRAPH_EXCLUDE_TARGETS


def graph_gap_cols(gap_cols):
    return [
        g for g in gap_cols
        if is_valid_graph_label(g) and not is_graph_excluded_gap(g)
    ]


# =============================================================================
# DATA LOADING
# =============================================================================

def get_gap_specs(df, feature_set):
    specs = []

    excluded_targets = FEATURE_TARGET_EXCLUSIONS.get(feature_set, set())

    for t in REGRESSION_TARGETS:
        if t in excluded_targets:
            print(f"[INFO] Excluding {feature_set} target: {t}")
            continue

        c = f"{t}_gap_z"
        if c in df.columns:
            specs.append((feature_set, t, c))

    return specs


def resolve_covariate_source_columns(header, subject_md_columns):
    header_set = set(header)
    subject_cols = set(subject_md_columns)

    read_cols = []
    rename_map = {}

    for canonical in COVARIATE_CANDIDATES:
        if canonical in subject_cols:
            continue

        aliases = COVARIATE_SOURCE_ALIASES.get(canonical, [canonical])

        for source_col in aliases:
            if source_col in header_set:
                read_cols.append(source_col)

                if source_col != canonical:
                    rename_map[source_col] = canonical

                break

    return read_cols, rename_map


def add_diagnosis_columns(subject_md):
    header = pd.read_csv(CSV_DIAGNOSES, nrows=0).columns.tolist()

    optional_covariates, rename_map = resolve_covariate_source_columns(
        header=header,
        subject_md_columns=subject_md.columns,
    )

    optional_scores = [
        c for c in SCORE_PREDICTOR_COLS
        if c in header and c not in subject_md.columns
    ]

    optional_bip_scz_exclusion = [
        c for c in BIP_SCZ_EXCLUSION_COLUMNS
        if c in header and c not in subject_md.columns
    ]

    required = [
        ID_COL,
        *DIAGNOSIS_COLUMNS_FOR_PLOTTING,
        *optional_bip_scz_exclusion,
        *optional_covariates,
        *optional_scores,
    ]
    required = list(dict.fromkeys(required))

    diag = pd.read_csv(CSV_DIAGNOSES, usecols=required)
    diag[ID_COL] = canonicalize_id_series(diag[ID_COL])

    if rename_map:
        diag = diag.rename(columns=rename_map)

    if diag[ID_COL].duplicated().any():
        n_dup = int(diag[ID_COL].duplicated().sum())
        raise ValueError(f"Diagnosis file contains duplicated IDs: n={n_dup}")

    subject_md = subject_md.copy()
    subject_md[ID_COL] = canonicalize_id_series(subject_md[ID_COL])

    stale = [
        c for c in (DIAGNOSIS_COLUMNS_FOR_PLOTTING + BIP_SCZ_EXCLUSION_COLUMNS)
        if c in subject_md.columns
    ]
    if stale:
        subject_md = subject_md.drop(columns=stale)

    merge_cols = [
        c for c in diag.columns
        if c == ID_COL or c not in subject_md.columns
    ]

    diag = diag[merge_cols].copy()

    subject_md = subject_md.merge(
        diag,
        on=ID_COL,
        how="left",
        validate="many_to_one",
    )

    print("\nDiagnosis indicator counts")
    print("=" * 80)

    count_cols = [
        c for c in (DIAGNOSIS_COLUMNS_FOR_PLOTTING + BIP_SCZ_EXCLUSION_COLUMNS)
        if c in subject_md.columns
    ]
    for col in count_cols:
        x = pd.to_numeric(subject_md[col], errors="coerce")
        print(
            f"{col:30s}: n == 1: {int((x == 1).sum()):8,d} | "
            f"non-missing: {int(x.notna().sum()):8,d}"
        )

    return subject_md


def add_mean_motion_covariate(subject_md):
    if not os.path.exists(CSV_MEAN_MOTION):
        print(f"[WARN] Mean-motion file not found; skipping: {CSV_MEAN_MOTION}")
        return subject_md.copy()

    mm = pd.read_csv(
        CSV_MEAN_MOTION,
        usecols=[MEAN_MOTION_ID_COL, MEAN_MOTION_VALUE_COL],
    )

    mm = mm.rename(
        columns={
            MEAN_MOTION_ID_COL: ID_COL,
            MEAN_MOTION_VALUE_COL: MEAN_MOTION_COVARIATE_COL,
        }
    )

    mm[ID_COL] = canonicalize_id_series(mm[ID_COL])
    mm[MEAN_MOTION_COVARIATE_COL] = pd.to_numeric(
        mm[MEAN_MOTION_COVARIATE_COL],
        errors="coerce",
    )

    if mm[ID_COL].duplicated().any():
        mm = mm.drop_duplicates(subset=[ID_COL], keep="first")

    out = subject_md.copy()
    out[ID_COL] = canonicalize_id_series(out[ID_COL])

    stale = [
        c for c in [MEAN_MOTION_COVARIATE_COL, MEAN_MOTION_VALUE_COL]
        if c in out.columns
    ]

    if stale:
        out = out.drop(columns=stale)

    out = out.merge(mm, on=ID_COL, how="left", validate="many_to_one")

    print("\nMean-motion merge")
    print("=" * 80)
    print(f"Non-missing mean_motion: {int(out[MEAN_MOTION_COVARIATE_COL].notna().sum()):,}")

    return out


def apply_plotting_diagnosis_labels(subject_md):
    subject_md = subject_md.copy()
    subject_md["analysis_label"] = np.nan

    subject_md.loc[group_mask(subject_md, "heldout_hc"), "analysis_label"] = "heldout_hc"
    subject_md.loc[group_mask(subject_md, AUD_COL), "analysis_label"] = AUD_COL
    subject_md.loc[group_mask(subject_md, ANX_COL), "analysis_label"] = ANX_COL
    subject_md.loc[group_mask(subject_md, MDD_COL), "analysis_label"] = MDD_COL

    print("\nFinal MAIN group counts (independent membership; overlap allowed)")
    print("=" * 80)
    print(f"{'heldout_hc':30s}: n={int(group_mask(subject_md, 'heldout_hc').sum()):8,d}")
    for g in GROUP_ORDER_FULL:
        print(f"{g:30s}: n={int(group_mask(subject_md, g).sum()):8,d}")
    print(
        f"{'MAIN_MDD & MAIN_ANX':30s}: "
        f"n={int((group_mask(subject_md, MDD_COL) & group_mask(subject_md, ANX_COL)).sum()):8,d}"
    )
    return subject_md


def load_gap_space(subject_md):
    frames = []
    specs_all = []

    for feature_set, path in INPUT_FILES.items():
        dfi = pd.read_csv(path)
        dfi[ID_COL] = canonicalize_id_series(dfi[ID_COL])
        specs = get_gap_specs(dfi, feature_set)

        keep = [ID_COL]
        if SPLIT_GROUP_COL in dfi.columns:
            keep.append(SPLIT_GROUP_COL)

        keep += [s[2] for s in specs]
        dfi = dfi[keep].copy()

        rename = {}

        for fs_name, target, old in specs:
            new = f"{fs_name}__{target}"
            rename[old] = new
            specs_all.append((fs_name, target, new))

        dfi = dfi.rename(columns=rename)
        frames.append(dfi)

    if not frames:
        raise ValueError("No gap files loaded.")

    gap_df_all = frames[0].copy()

    for f in frames[1:]:
        merge_cols = [ID_COL]

        if SPLIT_GROUP_COL in gap_df_all.columns and SPLIT_GROUP_COL in f.columns:
            merge_cols.append(SPLIT_GROUP_COL)

        gap_df_all = gap_df_all.merge(
            f,
            on=merge_cols,
            how="inner",
            validate="one_to_one",
        )

    gap_cols = [s[2] for s in specs_all]
    gap_df_all = safe_numeric(gap_df_all, gap_cols)

    gap_df = drop_train_hc(gap_df_all, label="merged gap_df")

    extra_cols = [
        c for c in (
                DIAGNOSIS_COLUMNS_FOR_PLOTTING
                + COVARIATE_CANDIDATES
                + SCORE_PREDICTOR_COLS
        )
        if c in subject_md.columns
    ]

    label_cols = [ID_COL, "analysis_label", *extra_cols]
    label_cols = list(dict.fromkeys(label_cols))

    labels = subject_md[label_cols].copy()
    labels[ID_COL] = canonicalize_id_series(labels[ID_COL])

    gap_df[ID_COL] = canonicalize_id_series(gap_df[ID_COL])
    gap_df = gap_df.merge(labels, on=ID_COL, how="inner")

    return gap_df, gap_cols


# =============================================================================
# PANEL A
# =============================================================================


# =============================================================================
# OLS-HC3 MODEL HELPERS
# =============================================================================

def available_covariates(df):
    return [c for c in COVARIATE_CANDIDATES if c in df.columns]


def prepare_covariate_design(df, covariates):
    if not covariates:
        return pd.DataFrame(index=df.index)

    pieces = []

    for c in covariates:
        x = df[c]
        numeric = pd.to_numeric(x, errors="coerce")
        observed = x.notna()
        numeric_fraction = numeric[observed].notna().mean() if observed.sum() else 0.0

        if numeric_fraction >= 0.90:
            z = numeric.astype(float)
            if z.notna().sum() >= 10 and z.nunique(dropna=True) > 1:
                pieces.append(z.rename(c).to_frame())
        else:
            dummies = pd.get_dummies(x.astype("category"), prefix=c, drop_first=True)
            if dummies.shape[1] > 0:
                pieces.append(dummies.astype(float))

    if not pieces:
        return pd.DataFrame(index=df.index)

    X_cov = pd.concat(pieces, axis=1)

    keep = []
    for c in X_cov.columns:
        vals = pd.to_numeric(X_cov[c], errors="coerce")
        if vals.notna().sum() >= 10 and vals.nunique(dropna=True) > 1:
            keep.append(c)

    return X_cov[keep].astype(float)


def drop_constant_predictors(X):
    keep = []

    for c in X.columns:
        vals = pd.to_numeric(X[c], errors="coerce")
        if vals.notna().sum() >= 10 and vals.nunique(dropna=True) > 1:
            keep.append(c)

    return X[keep].astype(float)


def standardize_model_dataframe(model_df, outcome_col, predictor_of_interest):
    out = model_df.copy()

    for c in out.columns:
        out[c] = zscore_series(out[c])

    keep_predictors = []

    for c in out.columns:
        if c == outcome_col:
            continue

        vals = pd.to_numeric(out[c], errors="coerce")
        if vals.notna().sum() >= 10 and vals.nunique(dropna=True) > 1:
            keep_predictors.append(c)

    if predictor_of_interest not in keep_predictors:
        return pd.DataFrame(columns=[outcome_col, predictor_of_interest])

    return out[[outcome_col, *keep_predictors]].dropna().copy()


def fit_ols_hc3_predictor(
        df,
        outcome_col,
        predictor_col,
        covariates,
        extra_predictors=None,
):
    extra_predictors = extra_predictors or []

    needed = [outcome_col, predictor_col, *covariates, *extra_predictors]
    needed = list(dict.fromkeys([c for c in needed if c in df.columns]))

    if outcome_col not in needed or predictor_col not in needed:
        return None

    tmp = df[needed].copy()
    tmp[outcome_col] = pd.to_numeric(tmp[outcome_col], errors="coerce")
    tmp[predictor_col] = pd.to_numeric(tmp[predictor_col], errors="coerce")

    for c in extra_predictors:
        if c in tmp.columns:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    tmp = tmp.dropna(subset=[outcome_col, predictor_col]).copy()

    if len(tmp) < 30:
        return None

    X_cov = prepare_covariate_design(tmp, covariates)

    pieces = [
        tmp[[outcome_col, predictor_col]].astype(float),
        X_cov,
    ]

    if extra_predictors:
        available_extra = [c for c in extra_predictors if c in tmp.columns]
        if available_extra:
            pieces.append(tmp[available_extra].astype(float))

    model_df = pd.concat(pieces, axis=1).dropna().copy()

    if len(model_df) < 30:
        return None

    model_df_std = standardize_model_dataframe(
        model_df=model_df,
        outcome_col=outcome_col,
        predictor_of_interest=predictor_col,
    )

    if len(model_df_std) < 30:
        return None

    y = model_df_std[outcome_col].astype(float)
    X = model_df_std.drop(columns=[outcome_col])
    X = drop_constant_predictors(X)

    if predictor_col not in X.columns:
        return None

    X = sm.add_constant(X, has_constant="add")

    try:
        fit = sm.OLS(y, X).fit(cov_type="HC3")
        return {
            "beta_std": float(fit.params.get(predictor_col, np.nan)),
            "se_hc3": float(fit.bse.get(predictor_col, np.nan)),
            "t_hc3": float(fit.tvalues.get(predictor_col, np.nan)),
            "p_hc3": float(fit.pvalues.get(predictor_col, np.nan)),
            "n": int(len(model_df_std)),
        }
    except Exception:
        return None


def add_binary_diagnosis_flags(df):
    out = df.copy()
    for c in DIAGNOSIS_COLUMNS_FOR_PLOTTING:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    out["is_hc"] = out[HC_COL].eq(1)
    out["is_aud"] = out[AUD_COL].eq(1)
    out["is_mdd"] = out[MDD_COL].eq(1)
    out["is_anx"] = out[ANX_COL].eq(1)
    return out


def make_contrast_subset(df, contrast_name):
    if contrast_name == "AUD vs HC":
        case_mask = df["is_aud"]
        ref_mask = df["is_hc"]
    elif contrast_name == "MDD vs HC":
        case_mask = df["is_mdd"]
        ref_mask = df["is_hc"]
    elif contrast_name == "ANX vs HC":
        case_mask = df["is_anx"]
        ref_mask = df["is_hc"]
    else:
        raise ValueError(f"Unknown contrast: {contrast_name}")

    keep = case_mask | ref_mask
    work = df.loc[keep].copy()
    work["diagnosis_indicator"] = case_mask.loc[work.index].astype(int)

    n_case = int(work["diagnosis_indicator"].eq(1).sum())
    n_ref = int(work["diagnosis_indicator"].eq(0).sum())
    if n_case < 10 or n_ref < 10:
        print(f"[WARN] Too few subjects for {contrast_name}: case={n_case}, ref={n_ref}")
    return work


def fit_standard_diagnosis_gap_edges(gap_df, gap_cols):
    df = add_binary_diagnosis_flags(gap_df)
    covariates = available_covariates(df)

    contrasts = [
        "AUD vs HC",
        "MDD vs HC",
        "ANX vs HC",
    ]

    rows = []

    for contrast in contrasts:
        sub = make_contrast_subset(df, contrast)

        for gap in gap_cols:
            stats = fit_ols_hc3_predictor(
                df=sub,
                outcome_col=gap,
                predictor_col="diagnosis_indicator",
                covariates=covariates,
                extra_predictors=None,
            )

            if stats is None:
                continue

            rows.append({
                "predictor": contrast,
                "gap": gap,
                "source": "diagnosis",
                **stats,
            })

    out = pd.DataFrame(rows)

    if len(out) == 0:
        return out

    out["q_fdr_bh"] = np.nan

    for predictor, idx in out.groupby("predictor").groups.items():
        idx = list(idx)
        p = pd.to_numeric(out.loc[idx, "p_hc3"], errors="coerce")
        ok = p.notna()

        if ok.sum() > 0:
            valid_idx = out.loc[idx].index[ok]
            out.loc[valid_idx, "q_fdr_bh"] = multipletests(
                p.loc[ok],
                method="fdr_bh",
            )[1]

    return out


def fit_standard_score_gap_edges(gap_df, gap_cols):
    df = gap_df.copy()
    covariates = available_covariates(df)

    score_cols = [c for c in SCORE_PREDICTOR_COLS if c in df.columns]

    if RANK_NORMALIZE_SCORE_PREDICTORS:
        for c in score_cols:
            df[c] = ranknorm_series(df[c])

    rows = []

    for score_col in score_cols:
        for gap in gap_cols:
            stats = fit_ols_hc3_predictor(
                df=df,
                outcome_col=gap,
                predictor_col=score_col,
                covariates=covariates,
                extra_predictors=None,
            )

            if stats is None:
                continue

            rows.append({
                "predictor": SCORE_PREDICTOR_LABELS.get(score_col, score_col),
                "score_predictor": score_col,
                "gap": gap,
                "source": "clinical_score",
                **stats,
            })

    out = pd.DataFrame(rows)

    if len(out) == 0:
        return out

    out["q_fdr_bh"] = np.nan

    for predictor, idx in out.groupby("predictor").groups.items():
        idx = list(idx)
        p = pd.to_numeric(out.loc[idx, "p_hc3"], errors="coerce")
        ok = p.notna()

        if ok.sum() > 0:
            valid_idx = out.loc[idx].index[ok]
            out.loc[valid_idx, "q_fdr_bh"] = multipletests(
                p.loc[ok],
                method="fdr_bh",
            )[1]

    return out


def compute_gap_adjusted_ols_hc3_edges(gap_df, gap_cols):
    graph_cols = graph_gap_cols(gap_cols)
    covariates = available_covariates(gap_df)

    rows = []

    for a, b in combinations(graph_cols, 2):
        stats_ab = fit_ols_hc3_predictor(
            df=gap_df,
            outcome_col=a,
            predictor_col=b,
            covariates=covariates,
            extra_predictors=None,
        )
        stats_ba = fit_ols_hc3_predictor(
            df=gap_df,
            outcome_col=b,
            predictor_col=a,
            covariates=covariates,
            extra_predictors=None,
        )

        if stats_ab is None or stats_ba is None:
            continue

        beta_ab = stats_ab["beta_std"]
        beta_ba = stats_ba["beta_std"]

        if not np.isfinite(beta_ab) or not np.isfinite(beta_ba):
            continue

        beta_sym = 0.5 * (beta_ab + beta_ba)
        p_pair = max(stats_ab["p_hc3"], stats_ba["p_hc3"])
        n_pair = min(stats_ab["n"], stats_ba["n"])

        rows.append({
            "gap_a": a,
            "gap_b": b,
            "gap_a_short": short_gap_name(a),
            "gap_b_short": short_gap_name(b),
            "beta_std_symmetric": float(beta_sym),
            "p_pair": float(p_pair),
            "n_complete": int(n_pair),
        })

    out = pd.DataFrame(rows)

    if len(out) == 0:
        return []

    out["q_fdr_bh_unique_pairs"] = multipletests(
        out["p_pair"],
        method="fdr_bh",
    )[1]

    out = out[
        out["q_fdr_bh_unique_pairs"].lt(FDR_ALPHA)
        & out["beta_std_symmetric"].abs().ge(GAP_GRAPH_MIN_ABS_BETA_STD)
        ].copy()

    out = out.sort_values(
        "beta_std_symmetric",
        key=lambda x: x.abs(),
        ascending=False,
    )

    edge_csv = os.path.join(
        RESULTS_DIR,
        "figure2_adaptive_nodes_gap_gap_edges.csv",
    )
    out.to_csv(edge_csv, index=False)
    print(f"Saved gap-gap edge table:\n{edge_csv}")

    return out.to_dict("records")


FIGURE3_EDGE_KEY = ["predictor", "source", "gap"]


def select_unique_figure3_predictor_edges(edges, context="Figure 3"):
    if edges is None or len(edges) == 0:
        return pd.DataFrame(
            columns=[
                "predictor",
                "gap",
                "source",
                "beta_std",
                "q_fdr_bh",
            ]
        )

    required = {"predictor", "gap", "source", "beta_std", "q_fdr_bh"}
    missing = required.difference(edges.columns)
    if missing:
        raise ValueError(
            f"{context} edge table is missing columns: "
            + ", ".join(sorted(missing))
        )

    out = edges.copy()
    out["beta_std"] = pd.to_numeric(out["beta_std"], errors="coerce")
    out["q_fdr_bh"] = pd.to_numeric(out["q_fdr_bh"], errors="coerce")

    out = out[
        out["q_fdr_bh"].lt(FDR_ALPHA)
        & out["beta_std"].notna()
        & out["beta_std"].abs().ge(GAP_GRAPH_MIN_ABS_BETA_STD)
        & out["predictor"].map(is_valid_graph_label)
        & out["gap"].map(is_valid_graph_label)
        & ~out["gap"].map(is_graph_excluded_gap)
        ].copy()

    if len(out) == 0:
        return out.reset_index(drop=True)

    duplicate_mask = out.duplicated(FIGURE3_EDGE_KEY, keep=False)
    if duplicate_mask.any():
        audit_cols = [
            c for c in [
                *FIGURE3_EDGE_KEY,
                "beta_std",
                "p_hc3",
                "q_fdr_bh",
                "n",
            ]
            if c in out.columns
        ]
        print(f"\n[{context}] WARNING: duplicate graphical edge rows detected")
        print("=" * 80)
        print(
            out.loc[duplicate_mask, audit_cols]
            .sort_values(FIGURE3_EDGE_KEY + ["q_fdr_bh"])
            .to_string(index=False)
        )

    out["_abs_beta_for_dedup"] = out["beta_std"].abs()
    out = (
        out
        .sort_values(
            FIGURE3_EDGE_KEY + ["q_fdr_bh", "_abs_beta_for_dedup"],
            ascending=[True, True, True, True, False],
            kind="mergesort",
        )
        .drop_duplicates(FIGURE3_EDGE_KEY, keep="first")
        .drop(columns=["_abs_beta_for_dedup"])
        .reset_index(drop=True)
    )

    if out.duplicated(FIGURE3_EDGE_KEY).any():
        raise AssertionError(
            f"{context}: graphical edge keys are still duplicated after deduplication."
        )

    print(
        f"[{context}] unique graphical predictor-gap edges: {len(out):,}"
    )

    return out


def collect_predictor_gap_edges(diagnosis_edges, score_edges):
    frames = []

    for df in [diagnosis_edges, score_edges]:
        if df is not None and len(df) > 0:
            frames.append(df.copy())

    if not frames:
        return pd.DataFrame(
            columns=[
                "predictor",
                "gap",
                "source",
                "beta_std",
                "q_fdr_bh",
            ]
        )

    combined = pd.concat(frames, axis=0, ignore_index=True)

    out = select_unique_figure3_predictor_edges(
        combined,
        context="Figure 3",
    )

    edge_csv = os.path.join(
        RESULTS_DIR,
        "figure2_adaptive_nodes_predictor_gap_edges.csv",
    )
    out.to_csv(edge_csv, index=False)
    print(f"Saved predictor-gap edge table:\n{edge_csv}")

    return out


def node_set_from_gap_edges(edges):
    nodes = []

    for e in edges:
        a = e.get("gap_a")
        b = e.get("gap_b")

        if is_valid_graph_label(a):
            nodes.append(a)
        if is_valid_graph_label(b):
            nodes.append(b)

    return set(nodes)


def gap_node_color(gap):
    fs, _ = gap_to_feature_target(gap)

    if fs == "freesurfer":
        return "C0"
    if fs == "connectivity":
        return "C1"

    return "0.45"


def gap_node_hatch(gap):
    fs, _ = gap_to_feature_target(gap)

    if fs == "connectivity":
        return GRAPH_GAP_HATCH_CONNECTIVITY
    if fs == "freesurfer":
        return GRAPH_GAP_HATCH_FREESURFER

    return None


def predictor_node_color(name):
    name = str(name)

    if name in {"PHQ-9", "#Antidepressants"}:
        return MDD_COLOR
    if name == "AUDIT":
        return AUD_COLOR
    if "ANX" in name or "Anxiety" in name or "anxiety" in name:
        return ANX_COLOR
    if "AUD" in name or "Alcohol" in name or "alcohol" in name:
        return AUD_COLOR
    if "MDD" in name or "Depression" in name or "depression" in name:
        return MDD_COLOR

    return "0.45"


def decramp_positions(
        pos,
        min_dist=GRAPH_GAP_MIN_DIST,
        iterations=1200,
        anchor_strength=0.040,
):
    if not pos:
        return pos

    adjusted = {k: np.asarray(v, dtype=float).copy() for k, v in pos.items()}
    anchors = {k: v.copy() for k, v in adjusted.items()}
    keys = list(adjusted.keys())

    for _ in range(iterations):
        disp = {k: np.zeros(2, dtype=float) for k in keys}
        moved = False

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                delta = adjusted[a] - adjusted[b]
                d = float(np.linalg.norm(delta))

                if not np.isfinite(d) or d < 1e-12:
                    angle = 2.399963229728653 * (i + j + 1)
                    delta = np.array([np.cos(angle), np.sin(angle)], dtype=float)
                    d = 1.0

                if d < min_dist:
                    push = 0.5 * (min_dist - d) * delta / d
                    disp[a] += push
                    disp[b] -= push
                    moved = True

        for k in keys:
            adjusted[k] += 0.76 * disp[k]
            adjusted[k] += anchor_strength * (anchors[k] - adjusted[k])

        if not moved:
            break

    arr = np.vstack([adjusted[k] for k in keys])
    center = arr.mean(axis=0)

    for k in keys:
        adjusted[k] = adjusted[k] - center

    return adjusted


def layout_gap_graph(nodes, edges):
    nodes = [n for n in list(nodes) if is_valid_graph_label(n)]

    if len(nodes) == 0:
        return {}

    try:
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(nodes)

        for e in edges:
            a = e.get("gap_a")
            b = e.get("gap_b")

            if a in G and b in G:
                G.add_edge(
                    a,
                    b,
                    weight=max(abs(float(e["beta_std_symmetric"])), 1e-4),
                )

        if len(nodes) == 1:
            return {nodes[0]: np.array([0.0, 0.0])}

        pos = nx.spring_layout(
            G,
            weight="weight",
            seed=RANDOM_SEED,
            k=GRAPH_LAYOUT_K / np.sqrt(max(len(nodes), 2)),
            iterations=2200,
            scale=GRAPH_LAYOUT_SCALE,
            center=(0.0, 0.0),
        )

        pos = {k: np.asarray(v, dtype=float) for k, v in pos.items()}

        return decramp_positions(
            pos,
            min_dist=GRAPH_GAP_MIN_DIST,
            iterations=1400,
            anchor_strength=0.038,
        )

    except Exception:
        angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)

        pos = {
            node: GRAPH_LAYOUT_SCALE * np.array([np.cos(a), np.sin(a)], dtype=float)
            for node, a in zip(nodes, angles)
        }

        return decramp_positions(
            pos,
            min_dist=GRAPH_GAP_MIN_DIST,
            iterations=1400,
            anchor_strength=0.038,
        )


def arrange_gap_modalities_horizontally(gap_pos):
    if not gap_pos:
        return {}

    target_to_nodes = {}
    target_y_anchor = {}

    for node, xy in gap_pos.items():
        _, target = gap_to_feature_target(node)
        target_to_nodes.setdefault(target, []).append(node)
        target_y_anchor.setdefault(target, []).append(float(xy[1]))

    target_y_anchor = {
        target: float(np.mean(values))
        for target, values in target_y_anchor.items()
    }

    ordered_targets = sorted(
        target_to_nodes,
        key=lambda target: (-target_y_anchor[target], str(target)),
    )

    per_row = max(int(GRAPH_TARGET_PAIRS_PER_ROW), 1)
    n_rows = int(np.ceil(len(ordered_targets) / per_row))

    if per_row == 1:
        free_x_slots = [-GRAPH_MODALITY_INNER_X]
        conn_x_slots = [GRAPH_MODALITY_INNER_X]
    elif per_row == 2:
        free_x_slots = [-GRAPH_MODALITY_OUTER_X, -GRAPH_MODALITY_INNER_X]
        conn_x_slots = [GRAPH_MODALITY_OUTER_X, GRAPH_MODALITY_INNER_X]
    else:
        free_x_slots = np.linspace(
            -GRAPH_MODALITY_OUTER_X,
            -GRAPH_MODALITY_INNER_X,
            per_row,
        ).tolist()
        conn_x_slots = np.linspace(
            GRAPH_MODALITY_OUTER_X,
            GRAPH_MODALITY_INNER_X,
            per_row,
        ).tolist()

    out = {}

    for index, target in enumerate(ordered_targets):
        row = index // per_row
        slot = index % per_row
        y = (0.5 * (n_rows - 1) - row) * GRAPH_TARGET_ROW_SPACING

        for node in target_to_nodes[target]:
            feature_set, _ = gap_to_feature_target(node)

            if feature_set == "freesurfer":
                x = free_x_slots[slot]
            elif feature_set == "connectivity":
                x = conn_x_slots[slot]
            else:
                # Unknown modalities remain near the center without colliding
                # exactly with either of the two recognized modality columns.
                x = 0.0 + 0.45 * (slot - 0.5 * (per_row - 1))

            out[node] = np.array([float(x), float(y)], dtype=float)

    return out


def layout_predictors_on_central_vertical_axis(predictor_edges, gap_pos):
    if predictor_edges is None or len(predictor_edges) == 0 or not gap_pos:
        return {}

    anchors = []

    for predictor in dict.fromkeys(predictor_edges["predictor"].tolist()):
        sub = predictor_edges[predictor_edges["predictor"].eq(predictor)]

        weighted_y = []
        weighted_y_weights = []
        fs_strength = 0.0
        conn_strength = 0.0

        for _, row in sub.iterrows():
            gap = row["gap"]
            if gap not in gap_pos:
                continue

            try:
                beta_abs = abs(float(row["beta_std"]))
            except (TypeError, ValueError):
                continue

            if not np.isfinite(beta_abs):
                continue

            weight = beta_abs + GRAPH_PREDICTOR_AXIS_EPS
            weighted_y.append(float(gap_pos[gap][1]))
            weighted_y_weights.append(weight)

            feature_set, _ = gap_to_feature_target(gap)
            feature_key = str(feature_set).strip().lower()

            if feature_key == "freesurfer":
                fs_strength += beta_abs
            elif feature_key == "connectivity":
                conn_strength += beta_abs

        if weighted_y:
            anchor_y = float(
                np.average(
                    np.asarray(weighted_y, dtype=float),
                    weights=np.asarray(weighted_y_weights, dtype=float),
                )
            )
        else:
            anchor_y = 0.0

        total_strength = fs_strength + conn_strength

        if total_strength > GRAPH_PREDICTOR_AXIS_EPS:
            modality_balance = (conn_strength - fs_strength) / total_strength
        else:
            modality_balance = 0.0

        modality_balance = float(np.clip(modality_balance, -1.0, 1.0))
        x = GRAPH_PREDICTOR_AXIS_MAX_PULL * modality_balance

        anchors.append(
            {
                "predictor": predictor,
                "anchor_y": anchor_y,
                "x": float(x),
                "fs_strength": float(fs_strength),
                "connectivity_strength": float(conn_strength),
                "modality_balance": modality_balance,
            }
        )

    if not anchors:
        return {}

    anchors = sorted(
        anchors,
        key=lambda d: (-d["anchor_y"], str(d["predictor"])),
    )

    gap_y = np.asarray([float(v[1]) for v in gap_pos.values()], dtype=float)
    y_min = float(gap_y.min())
    y_max = float(gap_y.max())
    y_span = max(y_max - y_min, 1e-6)
    margin = GRAPH_PREDICTOR_AXIS_Y_MARGIN_FRAC * y_span

    top = y_max - margin
    bottom = y_min + margin

    n = len(anchors)

    if n == 1:
        assigned_y = np.array(
            [float(np.clip(anchors[0]["anchor_y"], bottom, top))],
            dtype=float,
        )
    else:
        assigned_y = np.linspace(top, bottom, n, dtype=float)

    pred_pos = {
        item["predictor"]: np.array(
            [item["x"], float(y)],
            dtype=float,
        )
        for item, y in zip(anchors, assigned_y)
    }

    print("\nPanel-C central predictor-axis layout")
    print("=" * 80)
    for item, y in zip(anchors, assigned_y):
        direction = (
            "FreeSurfer"
            if item["modality_balance"] < -1e-12
            else "Connectivity"
            if item["modality_balance"] > 1e-12
            else "center"
        )
        print(
            f"{str(item['predictor']):24s} "
            f"S_FS={item['fs_strength']:.3f} | "
            f"S_CONN={item['connectivity_strength']:.3f} | "
            f"balance={item['modality_balance']:+.3f} | "
            f"pull={direction:12s} | "
            f"x={item['x']:+.3f}, y={float(y):+.3f}"
        )

    return pred_pos


def layout_predictors_near_connected_gaps(predictor_edges, gap_pos):
    if predictor_edges is None or len(predictor_edges) == 0 or not gap_pos:
        return {}

    anchors = {}

    for p in dict.fromkeys(predictor_edges["predictor"].tolist()):
        sub = predictor_edges[predictor_edges["predictor"].eq(p)].copy()

        coords = []
        weights = []

        for _, r in sub.iterrows():
            g = r["gap"]
            if g not in gap_pos:
                continue

            coords.append(np.asarray(gap_pos[g], dtype=float))
            weights.append(abs(float(r["beta_std"])) + 1e-4)

        if not coords:
            continue

        coords = np.vstack(coords)
        weights = np.asarray(weights, dtype=float)
        anchor = np.average(coords, axis=0, weights=weights)

        radial = anchor.copy()
        norm_radial = float(np.linalg.norm(radial))

        if not np.isfinite(norm_radial) or norm_radial < 1e-8:
            idx = len(anchors)
            angle = 2 * np.pi * idx / max(1, predictor_edges["predictor"].nunique())
            radial = np.array([np.cos(angle), np.sin(angle)], dtype=float)
            norm_radial = 1.0

        radial = radial / norm_radial
        anchors[p] = anchor + GRAPH_PRED_OUTWARD_PUSH * radial

    if not anchors:
        return {}

    adjusted = {k: v.copy() for k, v in anchors.items()}
    keys = list(adjusted.keys())

    all_gap_coords = np.vstack([np.asarray(v, dtype=float) for v in gap_pos.values()])
    span = max(
        all_gap_coords[:, 0].max() - all_gap_coords[:, 0].min(),
        all_gap_coords[:, 1].max() - all_gap_coords[:, 1].min(),
        1e-6,
    )

    min_dist = GRAPH_PRED_MIN_DIST_FRAC * span
    anchor_strength = 0.25

    for _ in range(700):
        disp = {k: np.zeros(2, dtype=float) for k in keys}
        moved = False

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                delta = adjusted[a] - adjusted[b]
                d = float(np.linalg.norm(delta))

                if not np.isfinite(d) or d < 1e-12:
                    angle = 2.399963229728653 * (i + j + 1)
                    delta = np.array([np.cos(angle), np.sin(angle)], dtype=float)
                    d = 1.0

                if d < min_dist:
                    push = 0.5 * (min_dist - d) * delta / d
                    disp[a] += push
                    disp[b] -= push
                    moved = True

        for k in keys:
            adjusted[k] += 0.68 * disp[k]
            adjusted[k] += anchor_strength * (anchors[k] - adjusted[k])

        if not moved:
            break

    return adjusted


def decramp_combined_positions(gap_pos, pred_pos):
    if not pred_pos:
        return gap_pos, pred_pos

    combined = {}
    kind = {}

    for k, v in gap_pos.items():
        key = ("gap", k)
        combined[key] = np.asarray(v, dtype=float).copy()
        kind[key] = "gap"

    for k, v in pred_pos.items():
        key = ("pred", k)
        combined[key] = np.asarray(v, dtype=float).copy()
        kind[key] = "pred"

    keys = list(combined.keys())
    anchors = {k: combined[k].copy() for k in keys}

    for _ in range(1500):
        disp = {k: np.zeros(2, dtype=float) for k in keys}
        moved = False

        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                delta = combined[a] - combined[b]
                d = float(np.linalg.norm(delta))

                if not np.isfinite(d) or d < 1e-12:
                    angle = 2.399963229728653 * (i + j + 1)
                    delta = np.array([np.cos(angle), np.sin(angle)], dtype=float)
                    d = 1.0

                local_min = GRAPH_COMBINED_MIN_DIST

                if kind[a] == "pred" or kind[b] == "pred":
                    local_min *= 1.42

                if d < local_min:
                    push = 0.5 * (local_min - d) * delta / d
                    disp[a] += push
                    disp[b] -= push
                    moved = True

        for k in keys:
            anchor_strength = 0.052 if kind[k] == "gap" else 0.018
            combined[k] += 0.80 * disp[k]
            combined[k] += anchor_strength * (anchors[k] - combined[k])

        if not moved:
            break

    out_gap = {k[1]: combined[k] for k in keys if k[0] == "gap"}
    out_pred = {k[1]: combined[k] for k in keys if k[0] == "pred"}

    return out_gap, out_pred


def scale_graph_positions(gap_pos, pred_pos=None, scale=GRAPH_EDGE_LENGTH_SCALE):
    pred_pos = pred_pos or {}

    if not gap_pos and not pred_pos:
        return gap_pos, pred_pos

    coords = []

    for v in gap_pos.values():
        coords.append(np.asarray(v, dtype=float))

    for v in pred_pos.values():
        coords.append(np.asarray(v, dtype=float))

    center = np.vstack(coords).mean(axis=0)

    def _scale_one(pos_dict):
        out = {}
        for k, v in pos_dict.items():
            v = np.asarray(v, dtype=float)
            out[k] = center + scale * (v - center)
        return out

    return _scale_one(gap_pos), _scale_one(pred_pos)


def draw_gap_edges(ax, edges, pos, show_labels=False, alpha=0.18, zorder=1):
    if not edges:
        return

    max_abs_beta = max([abs(e["beta_std_symmetric"]) for e in edges] + [1e-6])

    for e in sorted(edges, key=lambda x: abs(x["beta_std_symmetric"])):
        a = e["gap_a"]
        b = e["gap_b"]

        if a not in pos or b not in pos:
            continue

        beta_val = float(e["beta_std_symmetric"])

        x0, y0 = pos[a]
        x1, y1 = pos[b]

        lw = 1.2 + 5.2 * abs(beta_val) / max_abs_beta
        edge_color = "0.20" if beta_val >= 0 else "0.62"
        line_style = "solid" if beta_val >= 0 else (0, (5, 2.5))

        ax.plot(
            [x0, x1],
            [y0, y1],
            color=edge_color,
            linewidth=lw,
            alpha=alpha,
            linestyle=line_style,
            solid_capstyle="round",
            zorder=zorder,
        )

        if show_labels and abs(beta_val) >= GAP_GRAPH_LABEL_MIN_ABS_BETA_STD:
            xm = 0.5 * (x0 + x1)
            ym = 0.5 * (y0 + y1)

            ax.text(
                xm,
                ym,
                f"{beta_val:+.2f}",
                ha="center",
                va="center",
                fontsize=GRAPH_EDGE_LABEL_SIZE,
                color="black",
                fontweight="bold",
                path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
                zorder=6,
            )


def draw_adaptive_gap_circle(ax, x, y, label, color, hatch=None):
    txt = short_gap_name(label)

    bbox_kwargs = dict(
        boxstyle=f"circle,pad={GRAPH_GAP_CIRCLE_PAD}",
        facecolor=color,
        edgecolor="black",
        linewidth=GRAPH_GAP_CIRCLE_LINEWIDTH,
        alpha=GRAPH_GAP_NODE_ALPHA,
    )

    if hatch:
        bbox_kwargs["hatch"] = hatch

    ax.text(
        x,
        y,
        txt,
        ha="center",
        va="center",
        fontsize=GRAPH_GAP_FONT_SIZE,
        color="white",
        fontweight="bold",
        linespacing=0.88,
        bbox=bbox_kwargs,
        path_effects=[pe.withStroke(linewidth=2.6, foreground="black")],
        zorder=5,
    )


def draw_gap_nodes(ax, nodes, pos):
    for n in nodes:
        if n not in pos:
            continue

        x, y = pos[n]

        draw_adaptive_gap_circle(
            ax=ax,
            x=x,
            y=y,
            label=n,
            color=gap_node_color(n),
            hatch=gap_node_hatch(n),
        )


def draw_adaptive_predictor_box(ax, x, y, label, color):
    txt = short_predictor_name(label)

    if txt == "ADcnt":
        measure = ax.text(
            x,
            y,
            txt,
            ha="center",
            va="center",
            fontsize=GRAPH_PRED_FONT_SIZE,
            fontweight="bold",
            linespacing=0.88,
            alpha=0.0,
            zorder=-100,
        )
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        text_bbox = measure.get_window_extent(renderer=renderer)
        measure.remove()

        pad_px = GRAPH_PRED_BOX_PAD * GRAPH_PRED_FONT_SIZE * ax.figure.dpi / 72.0
        x0_px = text_bbox.x0 - pad_px
        x1_px = text_bbox.x1 + pad_px
        y0_px = text_bbox.y0 - pad_px
        y1_px = text_bbox.y1 + pad_px

        inv = ax.transData.inverted()
        (x0, y0) = inv.transform((x0_px, y0_px))
        (x1, y1) = inv.transform((x1_px, y1_px))

        width = float(x1 - x0)
        height = float(y1 - y0)
        rounding = 0.20 * min(abs(width), abs(height))

        rounded_clip = FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            facecolor=MDD_COLOR,
            edgecolor="none",
            linewidth=0.0,
            alpha=0.97,
            zorder=6.8,
        )
        ax.add_patch(rounded_clip)

        green_half = Rectangle(
            (x0 + 0.5 * width, y0),
            0.5 * width,
            height,
            facecolor=ANX_COLOR,
            edgecolor="none",
            alpha=0.97,
            zorder=6.9,
        )
        green_half.set_clip_path(rounded_clip)
        ax.add_patch(green_half)

        outline = FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={rounding}",
            facecolor="none",
            edgecolor="black",
            linewidth=GRAPH_PRED_BOX_LINEWIDTH,
            zorder=7.0,
        )
        ax.add_patch(outline)

        ax.text(
            x,
            y,
            txt,
            ha="center",
            va="center",
            fontsize=GRAPH_PRED_FONT_SIZE,
            color="white",
            fontweight="bold",
            linespacing=0.88,
            path_effects=[pe.withStroke(linewidth=2.4, foreground="black")],
            zorder=7.2,
        )
        return

    ax.text(
        x,
        y,
        txt,
        ha="center",
        va="center",
        fontsize=GRAPH_PRED_FONT_SIZE,
        color="white",
        fontweight="bold",
        linespacing=0.88,
        bbox=dict(
            boxstyle=f"round,pad={GRAPH_PRED_BOX_PAD},rounding_size={GRAPH_PRED_BOX_ROUNDING}",
            facecolor=color,
            edgecolor="black",
            linewidth=GRAPH_PRED_BOX_LINEWIDTH,
            alpha=0.97,
        ),
        path_effects=[pe.withStroke(linewidth=2.4, foreground="black")],
        zorder=7,
    )


def set_graph_limits(ax, gap_pos, pred_pos=None):
    xs = []
    ys = []

    for v in gap_pos.values():
        xs.append(v[0])
        ys.append(v[1])

    if pred_pos:
        for v in pred_pos.values():
            xs.append(v[0])
            ys.append(v[1])

    if not xs:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        return

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    x_span = max(float(x_max - x_min), 1e-6)
    y_span = max(float(y_max - y_min), 1e-6)

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)

    effective_pad = GRAPH_LIMIT_PAD / max(float(GRAPH_EDGE_LENGTH_SCALE), 1e-6)
    effective_pad = max(effective_pad, 1.02)

    x_half = 0.5 * x_span * effective_pad
    y_half = 0.5 * y_span * effective_pad

    node_margin_x = GRAPH_NODE_MARGIN_FRAC * x_span
    node_margin_y = GRAPH_NODE_MARGIN_FRAC * y_span
    title_headroom = GRAPH_TITLE_HEADROOM_FRAC * y_span

    ax.set_xlim(
        x_mid - x_half - node_margin_x,
        x_mid + x_half + node_margin_x,
    )
    ax.set_ylim(
        y_mid - y_half - node_margin_y,
        y_mid + y_half + node_margin_y + title_headroom,
    )


def make_graph_legend_handles():
    """Legend entries for the refined Figure 3 network panels."""
    return [
        Patch(facecolor="C0", edgecolor="black", label="FreeSurfer gap"),
        Patch(
            facecolor="C1",
            edgecolor="black",
            hatch=GRAPH_GAP_HATCH_CONNECTIVITY,
            label="Connectivity gap",
        ),
        Patch(
            facecolor=MDD_COLOR,
            edgecolor="black",
            label="MDD / PHQ-9 / AD count",
        ),
        Patch(facecolor=ANX_COLOR, edgecolor="black", label="ANX"),
        Patch(facecolor=AUD_COLOR, edgecolor="black", label="AUD / AUDIT"),
        Line2D(
            [0], [0],
            color="0.20",
            linewidth=2.8,
            linestyle="solid",
            label=r"Positive standardized $\beta$",
        ),
        Line2D(
            [0], [0],
            color="0.62",
            linewidth=2.8,
            linestyle=(0, (5, 2.5)),
            label=r"Negative standardized $\beta$",
        ),
        Line2D(
            [0], [0],
            color="black",
            marker="s",
            markersize=8.0,
            markerfacecolor="white",
            markeredgewidth=1.6,
            linewidth=0,
            label="Boxed coefficient: remains FDR-significant\nafter adjustment for all other gaps",
        ),
    ]


def add_graph_model_note(fig, y=0.020):
    if not GRAPH_PRIMARY_MODEL_NOTE:
        return
    fig.text(
        0.5,
        y,
        GRAPH_PRIMARY_MODEL_NOTE,
        ha="center",
        va="bottom",
        fontsize=GRAPH_NOTE_SIZE,
        linespacing=1.25,
    )


def enforce_predictor_gap_min_distance(
        predictor_edges,
        gap_pos,
        pred_pos,
        min_dist=GRAPH_PRED_GAP_MIN_DIST,
):
    if (
            predictor_edges is None
            or len(predictor_edges) == 0
            or not gap_pos
            or not pred_pos
    ):
        return pred_pos

    out = {
        p: np.asarray(xy, dtype=float).copy()
        for p, xy in pred_pos.items()
    }

    edge_rows = []
    for _, row in predictor_edges.iterrows():
        p = row["predictor"]
        g = row["gap"]
        if p in out and g in gap_pos:
            edge_rows.append((p, g))

    if not edge_rows:
        return out

    for _ in range(250):
        corrections = {p: np.zeros(2, dtype=float) for p in out}
        counts = {p: 0 for p in out}
        any_short = False

        for p, g in edge_rows:
            pred_xy = out[p]
            gap_xy = np.asarray(gap_pos[g], dtype=float)
            delta = pred_xy - gap_xy
            dist = float(np.linalg.norm(delta))

            if not np.isfinite(dist):
                continue

            if dist < 1e-12:
                # Deterministic fallback direction if the nodes coincide.
                direction = np.array([1.0, 0.0], dtype=float)
                dist = 0.0
            else:
                direction = delta / dist

            if dist < min_dist:
                any_short = True
                corrections[p] += (min_dist - dist) * direction
                counts[p] += 1

        if not any_short:
            break

        max_move = 0.0
        for p in out:
            if counts[p] == 0:
                continue
            step = corrections[p] / float(counts[p])
            out[p] += step
            max_move = max(max_move, float(np.linalg.norm(step)))

        if max_move < 1e-8:
            break

    return out


def choose_edge_label_position(
        x0,
        y0,
        x1,
        y1,
        fraction_from_predictor=GRAPH_EDGE_LABEL_FRACTION_FROM_PREDICTOR,
):
    t = float(np.clip(fraction_from_predictor, 0.0, 1.0))
    return (
        float(x0 + t * (x1 - x0)),
        float(y0 + t * (y1 - y0)),
    )


def plot_gap_graph_with_predictor_nodes(
        ax,
        gap_edges,
        predictor_edges,
        title,
        show_predictor_legend=True,
        all_gap_nodes=None,
        draw_gap_gap_edges=GRAPH_DRAW_GAP_GAP_EDGES,
):
    if title:
        ax.set_title(
            title,
            fontsize=GRAPH_PANEL_TITLE_SIZE,
            y=GRAPH_TITLE_Y,
            pad=GRAPH_TITLE_PAD,
        )
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")

    if len(predictor_edges) == 0:
        ax.text(
            0.5,
            0.5,
            (
                "No predictor--gap associations\n"
                f"q_FDR < {FDR_ALPHA:g}, |beta_std| >= {GAP_GRAPH_MIN_ABS_BETA_STD:g}"
            ),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=TITLE_SIZE,
        )
        return

    gap_nodes_from_layout_edges = {
        g for g in node_set_from_gap_edges(gap_edges)
        if is_valid_graph_label(g)
    }

    gap_nodes_from_predictors = {
        str(g).strip()
        for g in predictor_edges["gap"].dropna().tolist()
        if is_valid_graph_label(g) and not is_graph_excluded_gap(g)
    }

    gap_nodes_requested = {
        str(g).strip()
        for g in (all_gap_nodes or [])
        if is_valid_graph_label(g) and not is_graph_excluded_gap(g)
    }

    ordered_cols = [
        f"{fs}__{target}"
        for fs in FEATURE_SET_ORDER
        for target in TARGET_ORDER
    ]
    order_rank = {g: i for i, g in enumerate(ordered_cols)}

    nodes = sorted(
        gap_nodes_requested | gap_nodes_from_layout_edges | gap_nodes_from_predictors,
        key=lambda g: order_rank.get(g, 999),
    )

    pos = layout_gap_graph(nodes, gap_edges)

    if GRAPH_USE_HORIZONTAL_MODALITY_LAYOUT:
        pos = arrange_gap_modalities_horizontally(pos)
        pred_pos = layout_predictors_on_central_vertical_axis(
            predictor_edges,
            pos,
        )
    else:
        pred_pos = layout_predictors_near_connected_gaps(predictor_edges, pos)
        pos, pred_pos = decramp_combined_positions(pos, pred_pos)

    pred_pos = enforce_predictor_gap_min_distance(
        predictor_edges=predictor_edges,
        gap_pos=pos,
        pred_pos=pred_pos,
        min_dist=GRAPH_PRED_GAP_MIN_DIST,
    )

    pos, pred_pos = scale_graph_positions(
        gap_pos=pos,
        pred_pos=pred_pos,
        scale=GRAPH_EDGE_LENGTH_SCALE,
    )

    if draw_gap_gap_edges:
        draw_gap_edges(
            ax,
            gap_edges,
            pos,
            show_labels=False,
            alpha=0.14,
            zorder=1,
        )

    max_abs_beta = max(
        [abs(float(v)) for v in predictor_edges["beta_std"].tolist()]
        + [1e-6]
    )

    for _, e in predictor_edges.sort_values(
            "beta_std",
            key=lambda x: x.abs(),
    ).iterrows():
        p = e["predictor"]
        g = e["gap"]

        if p not in pred_pos or g not in pos:
            continue

        beta_val = float(e["beta_std"])
        x0, y0 = pred_pos[p]
        x1, y1 = pos[g]

        magnitude = min(abs(beta_val) / max_abs_beta, 1.0)
        lw = (
                GRAPH_PRED_EDGE_MIN_LW
                + (GRAPH_PRED_EDGE_MAX_LW - GRAPH_PRED_EDGE_MIN_LW) * magnitude
        )
        edge_color = predictor_node_color(p)
        line_style = "solid" if beta_val >= 0 else (0, (5, 2.5))

        ax.plot(
            [x0, x1],
            [y0, y1],
            color=edge_color,
            linewidth=lw,
            alpha=GRAPH_PRED_EDGE_ALPHA,
            linestyle=line_style,
            solid_capstyle="round",
            zorder=2,
        )

        xm, ym = choose_edge_label_position(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
        )

        edge_label = f"{beta_val:+.2f}"

        survives_corrected = bool(e.get("survives_corrected", False))
        label_bbox = dict(
            boxstyle=f"round,pad={GRAPH_CORRECTED_LABEL_BBOX_PAD if survives_corrected else GRAPH_EDGE_LABEL_BBOX_PAD}",
            facecolor="white",
            edgecolor=(GRAPH_CORRECTED_LABEL_EDGE_COLOR if survives_corrected else "none"),
            linewidth=(GRAPH_CORRECTED_LABEL_EDGE_WIDTH if survives_corrected else 0.0),
            alpha=(GRAPH_CORRECTED_LABEL_FACE_ALPHA if survives_corrected else GRAPH_EDGE_LABEL_BBOX_ALPHA),
        )

        ax.text(
            xm,
            ym,
            edge_label,
            ha="center",
            va="center",
            fontsize=GRAPH_EDGE_LABEL_SIZE,
            color="black",
            fontweight="bold",
            bbox=label_bbox,
            path_effects=[pe.withStroke(linewidth=1.2, foreground="white")],
            zorder=6,
        )

    draw_gap_nodes(ax, nodes, pos)

    for p, xy in pred_pos.items():
        x, y = xy
        draw_adaptive_predictor_box(
            ax=ax,
            x=x,
            y=y,
            label=p,
            color=predictor_node_color(p),
        )

    set_graph_limits(ax, pos, pred_pos)

    if show_predictor_legend:
        ax.legend(
            handles=make_graph_legend_handles(),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.060),
            frameon=False,
            fontsize=GRAPH_LEGEND_SIZE,
            borderaxespad=0.0,
            handlelength=2.0,
            labelspacing=0.55,
            ncol=2,
        )


def figure4_predictor_order(label):
    order = [
        "AUD vs HC",
        "MDD vs HC",
        "ANX vs HC",
        "AUDIT",
        "PHQ-9",
        "#Antidepressants",
    ]
    try:
        return order.index(str(label))
    except ValueError:
        return len(order)


def figure4_predictor_label(label):
    replacements = {
        "AUD vs HC": "AUD",
        "MDD vs HC": "MDD",
        "ANX vs HC": "ANX",
        "AUDIT": "AUDIT",
        "PHQ-9": "PHQ-9",
        "#Antidepressants": "AD count",
    }
    return replacements.get(str(label), str(label))


def figure4_gap_label(gap):
    fs, target = gap_to_feature_target(gap)
    modality = "FS" if fs == "freesurfer" else "FC"

    target_labels = {
        "brain_age": "Age",
        "Fluid_intelligence_score": "Intel",
        "body_fat_percentage": "Body fat",
        "Sleep_duration": "Sleep",
        "Hand_grip_strength_mean": "Grip",
        "bmi": "BMI",
        "diastolic_bp": "DBP",
        "trail_making": "Trail",
    }

    return f"{modality} {target_labels.get(target, target)}"


def figure4_row_label(predictor, gap):
    return f"{figure4_predictor_label(predictor)} \u2192 {figure4_gap_label(gap)}"


def figure4_gap_modality(gap):
    fs, _ = gap_to_feature_target(gap)
    if fs == "freesurfer":
        return "structural"
    if fs == "connectivity":
        return "functional"
    return "other"


def figure4_segment_color(modality, rank):
    blues = plt.cm.Blues([0.72, 0.62, 0.52, 0.42, 0.34])
    oranges = plt.cm.Oranges([0.74, 0.64, 0.54, 0.44, 0.36])

    idx = max(0, min(rank, len(blues) - 1))

    if modality == "structural":
        return blues[idx]
    if modality == "functional":
        return oranges[idx]
    return FIG4_OTHER_COLOR


def prepare_figure4_predictor_dataset(gap_df, predictor, source):
    if source == "diagnosis":
        df = add_binary_diagnosis_flags(gap_df)
        df = make_contrast_subset(df, predictor)
        predictor_col = "diagnosis_indicator"

    elif source == "clinical_score":
        reverse = {
            label: col
            for col, label in SCORE_PREDICTOR_LABELS.items()
        }

        if predictor not in reverse:
            raise ValueError(
                f"Cannot map Figure-4 clinical predictor: {predictor}"
            )

        predictor_col = reverse[predictor]
        df = gap_df.copy()

        if predictor_col not in df.columns:
            raise ValueError(
                f"Missing clinical predictor column: {predictor_col}"
            )

        if RANK_NORMALIZE_SCORE_PREDICTORS:
            df[predictor_col] = ranknorm_series(df[predictor_col])

    else:
        raise ValueError(f"Unknown Figure-4 source: {source}")

    covariates = available_covariates(df)
    return df, predictor_col, covariates


def _figure4_standardized_matched_dataframe(
        df,
        outcome_gap,
        predictor_col,
        covariates,
        extra_gaps,
):
    extra_gaps = list(extra_gaps)

    needed = [
        outcome_gap,
        predictor_col,
        *covariates,
        *extra_gaps,
    ]
    needed = list(dict.fromkeys([c for c in needed if c in df.columns]))

    required = {outcome_gap, predictor_col, *extra_gaps}
    if not required.issubset(set(needed)):
        return None, []

    tmp = df[needed].copy()
    tmp[outcome_gap] = pd.to_numeric(tmp[outcome_gap], errors="coerce")
    tmp[predictor_col] = pd.to_numeric(tmp[predictor_col], errors="coerce")

    for g in extra_gaps:
        tmp[g] = pd.to_numeric(tmp[g], errors="coerce")

    tmp = tmp.dropna(subset=[outcome_gap, predictor_col]).copy()

    if len(tmp) < FIG4_MIN_MODEL_N:
        return None, []

    X_cov = prepare_covariate_design(tmp, covariates)

    pieces = [
        tmp[[outcome_gap, predictor_col]].astype(float),
        X_cov,
    ]

    if extra_gaps:
        pieces.append(tmp[extra_gaps].astype(float))

    model_df = pd.concat(pieces, axis=1).dropna().copy()

    if len(model_df) < FIG4_MIN_MODEL_N:
        return None, []

    model_df_std = standardize_model_dataframe(
        model_df=model_df,
        outcome_col=outcome_gap,
        predictor_of_interest=predictor_col,
    )

    if len(model_df_std) < FIG4_MIN_MODEL_N:
        return None, []

    cov_design_cols = [
        c for c in X_cov.columns
        if c in model_df_std.columns
    ]

    return model_df_std, cov_design_cols


def _figure4_fit_from_standardized_dataframe(
        model_df_std,
        outcome_gap,
        predictor_col,
        cov_design_cols,
        extra_gaps,
):
    if model_df_std is None or len(model_df_std) < FIG4_MIN_MODEL_N:
        return None

    extra_gaps = [
        g for g in extra_gaps
        if g in model_df_std.columns
    ]

    X_cols = [
        predictor_col,
        *cov_design_cols,
        *extra_gaps,
    ]
    X_cols = list(dict.fromkeys(X_cols))

    X = model_df_std[X_cols].copy()
    X = drop_constant_predictors(X)

    if predictor_col not in X.columns:
        return None

    y = model_df_std[outcome_gap].astype(float)
    X = sm.add_constant(X, has_constant="add")

    try:
        fit = sm.OLS(y, X).fit(cov_type="HC3")
    except Exception:
        return None

    return {
        "beta_std": float(fit.params.get(predictor_col, np.nan)),
        "se_hc3": float(fit.bse.get(predictor_col, np.nan)),
        "p_hc3": float(fit.pvalues.get(predictor_col, np.nan)),
        "n": int(len(model_df_std)),
    }


def fit_figure4_matched_models(
        df,
        outcome_gap,
        predictor_col,
        covariates,
        extra_gaps,
):
    model_df_std, cov_design_cols = (
        _figure4_standardized_matched_dataframe(
            df=df,
            outcome_gap=outcome_gap,
            predictor_col=predictor_col,
            covariates=covariates,
            extra_gaps=extra_gaps,
        )
    )

    if model_df_std is None:
        return None, None

    paired_primary = _figure4_fit_from_standardized_dataframe(
        model_df_std=model_df_std,
        outcome_gap=outcome_gap,
        predictor_col=predictor_col,
        cov_design_cols=cov_design_cols,
        extra_gaps=[],
    )

    adjusted = _figure4_fit_from_standardized_dataframe(
        model_df_std=model_df_std,
        outcome_gap=outcome_gap,
        predictor_col=predictor_col,
        cov_design_cols=cov_design_cols,
        extra_gaps=extra_gaps,
    )

    return paired_primary, adjusted


def compute_internal_all_other_gap_correction(
        gap_df,
        gap_cols,
        predictor_edges,
):
    graph_cols = graph_gap_cols(gap_cols)

    primary_edges = select_unique_figure3_predictor_edges(
        predictor_edges,
        context="Internal all-other-gaps correction input",
    )

    if len(primary_edges) == 0:
        raise ValueError(
            "No significant Figure-3 predictor edges are available to define "
            "predictors for the internal all-other-gaps correction."
        )

    predictor_specs = (
        primary_edges[["predictor", "source"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    frames = []

    for _, spec in predictor_specs.iterrows():
        predictor = spec["predictor"]
        source = spec["source"]

        print(
            f"[Internal all-other-gaps] {predictor} ({source})"
        )

        work, predictor_col, covariates = prepare_figure4_predictor_dataset(
            gap_df=gap_df,
            predictor=predictor,
            source=source,
        )

        rows = []

        for target_gap in graph_cols:
            other_gaps = [g for g in graph_cols if g != target_gap]

            paired, adjusted = fit_figure4_matched_models(
                df=work,
                outcome_gap=target_gap,
                predictor_col=predictor_col,
                covariates=covariates,
                extra_gaps=other_gaps,
            )

            if paired is None or adjusted is None:
                continue

            beta0 = float(paired["beta_std"])
            beta1 = float(adjusted["beta_std"])

            if (
                    np.isfinite(beta0)
                    and abs(beta0) > 1e-12
                    and np.isfinite(beta1)
            ):
                attenuation = 100.0 * max(
                    0.0,
                    1.0 - abs(beta1) / abs(beta0),
                )
            else:
                attenuation = np.nan

            rows.append({
                "predictor": predictor,
                "source": source,
                "target_gap": target_gap,
                "added_gap": "__ALL_OTHER_GAPS__",
                "adjustment_type": "all_other_gaps",
                "beta_paired_primary": beta0,
                "beta_adjusted": beta1,
                "se_adjusted_hc3": float(adjusted.get("se_hc3", np.nan)),
                "p_adjusted_hc3": float(adjusted["p_hc3"]),
                "n_matched": int(adjusted["n"]),
                "raw_positive_attenuation_pct": float(attenuation),
            })

        if not rows:
            continue

        corr = pd.DataFrame(rows)
        corr["q_adjusted_fdr_bh"] = np.nan

        pvals = pd.to_numeric(corr["p_adjusted_hc3"], errors="coerce")
        ok = pvals.notna()

        if ok.sum() > 0:
            corr.loc[ok, "q_adjusted_fdr_bh"] = multipletests(
                pvals.loc[ok].values,
                method="fdr_bh",
            )[1]

        corr["adjusted_fdr_significant"] = (
            corr["q_adjusted_fdr_bh"].lt(FDR_ALPHA)
        )

        frames.append(corr)

    if not frames:
        raise ValueError(
            "No internal all-other-gaps corrected models could be estimated."
        )

    corrected = pd.concat(frames, axis=0, ignore_index=True)

    corrected.to_csv(CSV_INTERNAL_ALL_OTHER_GAPS, index=False)

    print("\nInternal all-other-gaps correction")
    print("=" * 80)
    print(f"Corrected models estimated: {len(corrected):,}")
    print(
        "FDR-significant after all-other-gaps adjustment: "
        f"{int(corrected['adjusted_fdr_significant'].sum()):,}"
    )
    print(f"Saved:\n{CSV_INTERNAL_ALL_OTHER_GAPS}")

    return corrected


def attach_internal_all_other_correction_to_primary_edges(
        predictor_edges,
        corrected_models,
):
    edges = select_unique_figure3_predictor_edges(
        predictor_edges,
        context="Internal corrected-edge annotation",
    )

    if len(edges) == 0:
        edges["survives_corrected"] = pd.Series(dtype=bool)
        return edges

    corr = corrected_models.copy().rename(
        columns={"target_gap": "gap"}
    )

    keep = [
        "predictor",
        "source",
        "gap",
        "beta_paired_primary",
        "beta_adjusted",
        "se_adjusted_hc3",
        "p_adjusted_hc3",
        "q_adjusted_fdr_bh",
        "n_matched",
        "raw_positive_attenuation_pct",
        "adjusted_fdr_significant",
    ]
    keep = [c for c in keep if c in corr.columns]
    corr = corr[keep].copy()

    if corr.duplicated(["predictor", "source", "gap"]).any():
        raise ValueError(
            "Internal all-other-gaps table contains duplicate predictor/source/gap rows."
        )

    out = edges.merge(
        corr,
        on=["predictor", "source", "gap"],
        how="left",
        validate="one_to_one",
    )

    out["survives_corrected"] = (
        out["adjusted_fdr_significant"]
        .fillna(False)
        .astype(bool)
    )

    out["beta_all_other_gaps"] = pd.to_numeric(
        out.get("beta_adjusted"), errors="coerce"
    )
    out["p_all_other_gaps_hc3"] = pd.to_numeric(
        out.get("p_adjusted_hc3"), errors="coerce"
    )
    out["q_all_other_gaps_fdr_bh"] = pd.to_numeric(
        out.get("q_adjusted_fdr_bh"), errors="coerce"
    )
    out["n_all_other_gaps_matched"] = pd.to_numeric(
        out.get("n_matched"), errors="coerce"
    )

    out.to_csv(
        CSV_MAIN_PREDICTOR_EDGES_WITH_CORRECTED_FLAG,
        index=False,
    )

    print("\nFigure-3 edges after internal all-other-gaps correction")
    print("=" * 80)
    print(f"Primary graphical edges: {len(out):,}")
    print(
        "Boxed / survives all-other-gaps FDR: "
        f"{int(out['survives_corrected'].sum()):,}/{len(out):,}"
    )
    print(
        f"Saved annotated edge audit:\n"
        f"{CSV_MAIN_PREDICTOR_EDGES_WITH_CORRECTED_FLAG}"
    )

    return out


def compute_figure4_relative_profiles(
        gap_df,
        gap_cols,
        predictor_edges,
        all_other_corrected_models,
):
    graph_cols = graph_gap_cols(gap_cols)

    if predictor_edges is None or len(predictor_edges) == 0:
        raise ValueError(
            "Figure 4 requires the significant Figure-3 predictor edges."
        )

    required = {
        "predictor",
        "gap",
        "source",
        "beta_std",
        "q_fdr_bh",
        "survives_corrected",
    }
    missing = required.difference(predictor_edges.columns)
    if missing:
        raise ValueError(
            "Figure-4 predictor-edge table is missing columns: "
            + ", ".join(sorted(missing))
        )

    primary_edges = select_unique_figure3_predictor_edges(
        predictor_edges,
        context="Figure 4 input",
    )

    if len(primary_edges) == 0:
        raise ValueError(
            "Figure 4 has no unique Figure-3 graphical edges after filtering."
        )

    if "survives_corrected" not in primary_edges.columns:
        raise ValueError(
            "Figure-4 input is missing 'survives_corrected'. Run the internal "
            "all-other-gaps correction before computing Figure 4."
        )

    primary_edges["survives_corrected"] = (
        primary_edges["survives_corrected"]
        .fillna(False)
        .astype(bool)
    )

    print("\nFigure-3 / Figure-4 unique-edge audit")
    print("=" * 80)
    print(f"Incoming predictor-edge rows: {len(predictor_edges):,}")
    print(f"Unique graphical edges used: {len(primary_edges):,}")
    print("Edges by source:")
    print(primary_edges.groupby("source").size().to_string())
    print("Edges by predictor:")
    print(primary_edges.groupby(["source", "predictor"]).size().to_string())

    predictor_specs = (
        primary_edges[["predictor", "source"]]
        .drop_duplicates()
        .copy()
    )
    predictor_specs["_order"] = predictor_specs["predictor"].map(
        figure4_predictor_order
    )
    predictor_specs = predictor_specs.sort_values("_order")

    one_gap_frames = []

    for _, spec in predictor_specs.iterrows():
        predictor = spec["predictor"]
        source = spec["source"]

        print(
            f"[Figure 4] relative attenuation profiles: "
            f"{predictor} ({source})"
        )

        work, predictor_col, covariates = (
            prepare_figure4_predictor_dataset(
                gap_df=gap_df,
                predictor=predictor,
                source=source,
            )
        )

        for added_gap in graph_cols:
            condition_rows = []

            for target_gap in graph_cols:
                if target_gap == added_gap:
                    continue

                paired, adjusted = fit_figure4_matched_models(
                    df=work,
                    outcome_gap=target_gap,
                    predictor_col=predictor_col,
                    covariates=covariates,
                    extra_gaps=[added_gap],
                )

                if paired is None or adjusted is None:
                    continue

                beta0 = float(paired["beta_std"])
                beta1 = float(adjusted["beta_std"])

                if (
                        np.isfinite(beta0)
                        and abs(beta0) > 1e-12
                        and np.isfinite(beta1)
                ):
                    raw_positive_attenuation_pct = (
                            100.0
                            * max(
                        0.0,
                        1.0 - abs(beta1) / abs(beta0)
                    )
                    )
                else:
                    raw_positive_attenuation_pct = np.nan

                condition_rows.append({
                    "predictor": predictor,
                    "source": source,
                    "target_gap": target_gap,
                    "added_gap": added_gap,
                    "adjustment_type": "single_gap",
                    "beta_paired_primary": beta0,
                    "beta_adjusted": beta1,
                    "p_adjusted_hc3": float(adjusted["p_hc3"]),
                    "n_matched": int(adjusted["n"]),
                    "raw_positive_attenuation_pct": float(
                        raw_positive_attenuation_pct
                    ),
                })

            if not condition_rows:
                continue

            condition_df = pd.DataFrame(condition_rows)
            condition_df["q_adjusted_fdr_bh"] = np.nan

            p = pd.to_numeric(
                condition_df["p_adjusted_hc3"],
                errors="coerce",
            )
            ok = p.notna()

            if ok.sum() > 0:
                condition_df.loc[
                    ok,
                    "q_adjusted_fdr_bh",
                ] = multipletests(
                    p.loc[ok].values,
                    method="fdr_bh",
                )[1]

            condition_df["adjusted_fdr_significant"] = (
                condition_df["q_adjusted_fdr_bh"].lt(FDR_ALPHA)
            )

            one_gap_frames.append(condition_df)

    if not one_gap_frames:
        raise ValueError(
            "No Figure-4 one-gap models could be estimated."
        )

    one_gap_df = pd.concat(one_gap_frames, axis=0, ignore_index=True)

    if all_other_corrected_models is None or len(all_other_corrected_models) == 0:
        raise ValueError(
            "Figure 4 requires the internally computed all-other-gaps correction table."
        )

    mutual_df = all_other_corrected_models.copy()

    required_mutual_cols = {
        "predictor",
        "source",
        "target_gap",
        "added_gap",
        "adjustment_type",
        "beta_paired_primary",
        "beta_adjusted",
        "p_adjusted_hc3",
        "q_adjusted_fdr_bh",
        "n_matched",
        "raw_positive_attenuation_pct",
        "adjusted_fdr_significant",
    }
    missing_mutual = required_mutual_cols.difference(mutual_df.columns)
    if missing_mutual:
        raise ValueError(
            "Internal all-other-gaps table is missing columns: "
            + ", ".join(sorted(missing_mutual))
        )

    valid_specs = primary_edges[["predictor", "source"]].drop_duplicates()
    mutual_df = mutual_df.merge(
        valid_specs.assign(_keep=True),
        on=["predictor", "source"],
        how="inner",
        validate="many_to_one",
    ).drop(columns=["_keep"])

    detailed = pd.concat(
        [one_gap_df, mutual_df],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    edge_status = primary_edges[
        [
            "predictor",
            "source",
            "gap",
            "beta_std",
            "q_fdr_bh",
            "survives_corrected",
        ]
    ].rename(
        columns={
            "gap": "target_gap",
            "beta_std": "beta_figure3_primary",
            "q_fdr_bh": "q_figure3_primary",
        }
    )

    detailed = detailed.merge(
        edge_status,
        on=["predictor", "source", "target_gap"],
        how="inner",
        validate="many_to_one",
    )

    detailed["row_label"] = [
        figure4_row_label(p, g)
        for p, g in zip(
            detailed["predictor"],
            detailed["target_gap"],
        )
    ]
    detailed["added_gap_label"] = detailed["added_gap"].map(
        lambda g: (
            "All other gaps"
            if g == "__ALL_OTHER_GAPS__"
            else figure4_gap_label(g)
        )
    )
    detailed["added_gap_modality"] = detailed["added_gap"].map(
        lambda g: (
            "all"
            if g == "__ALL_OTHER_GAPS__"
            else figure4_gap_modality(g)
        )
    )
    detailed["single_gap_removes_fdr"] = (
            detailed["adjustment_type"].eq("single_gap")
            & ~detailed["adjusted_fdr_significant"].fillna(False)
    )

    detailed["normalized_profile_pct"] = np.nan

    single_gap_mask = detailed["adjustment_type"].eq("single_gap")

    grouped = detailed.loc[single_gap_mask].groupby(
        ["predictor", "source", "target_gap"],
        sort=False,
    )

    for key, idx in grouped.groups.items():
        idx = list(idx)
        vals = pd.to_numeric(
            detailed.loc[idx, "raw_positive_attenuation_pct"],
            errors="coerce",
        ).fillna(0.0)

        total = float(vals.sum())

        if total > 0:
            detailed.loc[idx, "normalized_profile_pct"] = (
                    100.0 * vals / total
            )
        else:
            detailed.loc[idx, "normalized_profile_pct"] = 0.0

    detailed.to_csv(FIG4_DETAILED_CSV, index=False)
    print(f"Saved Figure-4 detailed table:\n{FIG4_DETAILED_CSV}")

    graph_order = {g: i for i, g in enumerate(graph_cols)}
    edges_for_plot = primary_edges.copy()
    edges_for_plot["_predictor_order"] = edges_for_plot["predictor"].map(
        figure4_predictor_order
    )
    edges_for_plot["_gap_order"] = edges_for_plot["gap"].map(
        lambda g: graph_order.get(g, 999)
    )
    edges_for_plot = edges_for_plot.sort_values(
        ["_predictor_order", "_gap_order"]
    ).reset_index(drop=True)

    summary_rows = []

    for _, edge in edges_for_plot.iterrows():
        predictor = edge["predictor"]
        source = edge["source"]
        target_gap = edge["gap"]

        sub_single = detailed.loc[
            detailed["predictor"].eq(predictor)
            & detailed["source"].eq(source)
            & detailed["target_gap"].eq(target_gap)
            & detailed["adjustment_type"].eq("single_gap")
            ].copy()

        # Explicitly exclude self-adjustment if present.
        sub_single = sub_single.loc[
            sub_single["added_gap"].ne(target_gap)
        ].copy()

        sub_single["raw_positive_attenuation_pct"] = pd.to_numeric(
            sub_single["raw_positive_attenuation_pct"],
            errors="coerce",
        ).fillna(0.0)
        sub_single["normalized_profile_pct"] = pd.to_numeric(
            sub_single["normalized_profile_pct"],
            errors="coerce",
        ).fillna(0.0)

        total_positive = float(
            sub_single["raw_positive_attenuation_pct"].sum()
        )

        display = sub_single.loc[
            sub_single["raw_positive_attenuation_pct"] > 0
            ].copy()

        display = display.sort_values(
            ["normalized_profile_pct", "raw_positive_attenuation_pct"],
            ascending=[False, False],
        ).head(FIG4_MAX_SEGMENTS_PER_ROW)

        displayed_added_gaps = set(display["added_gap"].tolist())

        other_share = float(
            sub_single.loc[
                ~sub_single["added_gap"].isin(displayed_added_gaps),
                "normalized_profile_pct",
            ].sum()
        )

        mutual_row = detailed.loc[
            detailed["predictor"].eq(predictor)
            & detailed["source"].eq(source)
            & detailed["target_gap"].eq(target_gap)
            & detailed["adjustment_type"].eq("all_other_gaps")
            ].copy()
        mutual_row = mutual_row.iloc[0] if len(mutual_row) > 0 else None

        row = {
            "predictor": predictor,
            "source": source,
            "target_gap": target_gap,
            "row_label": figure4_row_label(predictor, target_gap),
            "beta_figure3_primary": float(edge["beta_std"]),
            "survives_corrected": bool(edge["survives_corrected"]),
            "mutual_raw_positive_attenuation_pct": (
                float(
                    pd.to_numeric(
                        mutual_row["raw_positive_attenuation_pct"],
                        errors="coerce",
                    )
                )
                if mutual_row is not None
                else np.nan
            ),
            "other_share_pct": other_share,
            "n_total_single_gap_models": int(len(sub_single)),
            "n_display_segments": int(len(display)),
            "total_positive_attenuation_pct": total_positive,
        }

        for rank in range(FIG4_MAX_SEGMENTS_PER_ROW):
            if rank < len(display):
                d = display.iloc[rank]

                row[f"seg{rank + 1}_gap"] = d["added_gap"]
                row[f"seg{rank + 1}_gap_label"] = figure4_gap_label(
                    d["added_gap"]
                )
                row[f"seg{rank + 1}_modality"] = figure4_gap_modality(
                    d["added_gap"]
                )
                row[f"seg{rank + 1}_raw_positive_attenuation_pct"] = float(
                    d["raw_positive_attenuation_pct"]
                )
                row[f"seg{rank + 1}_normalized_profile_pct"] = float(
                    d["normalized_profile_pct"]
                )
            else:
                row[f"seg{rank + 1}_gap"] = np.nan
                row[f"seg{rank + 1}_gap_label"] = np.nan
                row[f"seg{rank + 1}_modality"] = np.nan
                row[f"seg{rank + 1}_raw_positive_attenuation_pct"] = np.nan
                row[f"seg{rank + 1}_normalized_profile_pct"] = np.nan

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    if len(summary_df) == 0:
        raise ValueError("Figure-4 summary is empty.")

    summary_edge_key = ["predictor", "source", "target_gap"]

    expected_n_rows = len(primary_edges)
    actual_n_rows = len(summary_df)

    if summary_df.duplicated(summary_edge_key).any():
        duplicated_summary = summary_df.loc[
            summary_df.duplicated(summary_edge_key, keep=False),
            summary_edge_key,
        ].sort_values(summary_edge_key)
        raise AssertionError(
            "Figure 4 summary contains duplicated horizontal rows:\n"
            + duplicated_summary.to_string(index=False)
        )

    if actual_n_rows != expected_n_rows:
        raise AssertionError(
            "Figure 3 / Figure 4 row mismatch: "
            f"{expected_n_rows} unique graphical edges but "
            f"{actual_n_rows} attenuation rows."
        )

    expected_edges = {
        (str(r["predictor"]), str(r["source"]), str(r["gap"]))
        for _, r in primary_edges.iterrows()
    }
    actual_edges = {
        (str(r["predictor"]), str(r["source"]), str(r["target_gap"]))
        for _, r in summary_df.iterrows()
    }

    missing_from_figure4 = sorted(expected_edges - actual_edges)
    extra_in_figure4 = sorted(actual_edges - expected_edges)

    if missing_from_figure4 or extra_in_figure4:
        parts = ["Figure 3 / Figure 4 edge-identity mismatch."]
        if missing_from_figure4:
            parts.append(
                "Missing from Figure 4:\n"
                + "\n".join(
                    f"  {pred} | {source} | {gap}"
                    for pred, source, gap in missing_from_figure4
                )
            )
        if extra_in_figure4:
            parts.append(
                "Extra in Figure 4:\n"
                + "\n".join(
                    f"  {pred} | {source} | {gap}"
                    for pred, source, gap in extra_in_figure4
                )
            )
        raise AssertionError("\n".join(parts))

    print(
        f"[Figure 4] row-count and edge-identity checks passed: "
        f"{actual_n_rows:,} attenuation rows for "
        f"{expected_n_rows:,} unique Figure-3 edges."
    )

    summary_df.to_csv(FIG4_ALL_SUMMARY_CSV, index=False)
    print(f"Saved Figure-4 all-rows summary table:\n{FIG4_ALL_SUMMARY_CSV}")

    return summary_df, detailed


def _figure4_panel_rows(summary_df, source):
    sub = summary_df.loc[summary_df["source"].eq(source)].copy()

    if len(sub) == 0:
        return pd.DataFrame()

    sub["_mutual_attenuation_sort"] = pd.to_numeric(
        sub["mutual_raw_positive_attenuation_pct"],
        errors="coerce",
    ).fillna(-np.inf)

    sub["_predictor_order"] = sub["predictor"].map(
        figure4_predictor_order
    )

    return sub.sort_values(
        [
            "_mutual_attenuation_sort",
            "_predictor_order",
            "row_label",
        ],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def draw_figure4_profile_panel(
        ax,
        panel_df,
        panel_label,
        panel_title,
        show_x_label,
        show_attenuation_number=True,
        show_all_gaps_marker=True,
):
    if len(panel_df) == 0:
        ax.axis("off")
        return

    n_rows = len(panel_df)
    y_positions = np.arange(n_rows)[::-1]

    for i, (_, row) in enumerate(panel_df.iterrows()):
        y = y_positions[i]

        ax.barh(
            y=y,
            width=FIG4_XMAX,
            left=0.0,
            height=0.72,
            color="white",
            edgecolor="0.82",
            linewidth=0.8,
            zorder=0,
        )

        start = 0.0

        seg_rank_drawn = 0
        for rank in range(FIG4_MAX_SEGMENTS_PER_ROW):
            width = pd.to_numeric(
                row.get(f"seg{rank + 1}_normalized_profile_pct"),
                errors="coerce",
            )

            if not np.isfinite(width) or width <= 0:
                continue

            modality = row.get(f"seg{rank + 1}_modality")
            label = row.get(f"seg{rank + 1}_gap_label")

            color = figure4_segment_color(modality, seg_rank_drawn)

            ax.barh(
                y=y,
                width=float(width),
                left=start,
                height=0.72,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=2,
            )

            if float(width) >= 8.0 and pd.notna(label):
                ax.text(
                    start + float(width) / 2.0,
                    y,
                    str(label),
                    ha="center",
                    va="center",
                    fontsize=FIG4_LABEL_FONT_SIZE,
                    color="black",
                    zorder=3,
                )

            start += float(width)
            seg_rank_drawn += 1

        other_width = pd.to_numeric(
            row.get("other_share_pct"),
            errors="coerce",
        )

        if np.isfinite(other_width) and other_width > 0:
            ax.barh(
                y=y,
                width=float(other_width),
                left=start,
                height=0.72,
                color=FIG4_OTHER_COLOR,
                edgecolor="white",
                linewidth=0.8,
                zorder=1,
            )

            if float(other_width) >= 9.0:
                ax.text(
                    start + float(other_width) / 2.0,
                    y,
                    "Other",
                    ha="center",
                    va="center",
                    fontsize=FIG4_LABEL_FONT_SIZE,
                    color="black",
                    zorder=3,
                )

        total_positive = pd.to_numeric(
            row.get("total_positive_attenuation_pct"),
            errors="coerce",
        )

        if not np.isfinite(total_positive) or float(total_positive) <= 0:
            ax.text(
                2.0,
                y,
                "no positive attenuation",
                ha="left",
                va="center",
                fontsize=FIG4_LABEL_FONT_SIZE,
                color="0.45",
                style="italic",
                zorder=3,
            )

        mutual_raw_att = pd.to_numeric(
            row.get("mutual_raw_positive_attenuation_pct"),
            errors="coerce",
        )

        if show_attenuation_number and np.isfinite(mutual_raw_att):
            ax.text(
                FIG4_ALLGAPS_TEXT_X,
                y,
                f"{float(mutual_raw_att):.0f}%",
                ha="right",
                va="center",
                fontsize=FIG4_ALLGAPS_NUMBER_FONT_SIZE,
                color="black",
                zorder=4,
            )

        if show_all_gaps_marker:
            survives = bool(row.get("survives_corrected", False))

            ax.scatter(
                [FIG4_ALLGAPS_MARKER_X],
                [y],
                s=46,
                marker="s",
                facecolors=("black" if survives else "white"),
                edgecolors="black",
                linewidths=1.15,
                zorder=4,
            )

        ax.hlines(
            y,
            0.0,
            100.0,
            linewidth=0.3,
            color="0.94",
            zorder=0,
        )

    if show_all_gaps_marker:
        ax.set_xlim(0.0, 118.0)
    elif show_attenuation_number:
        ax.set_xlim(0.0, 111.0)
    else:
        ax.set_xlim(0.0, 100.0)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        panel_df["row_label"].tolist(),
        fontsize=FIG4_ROW_FONT_SIZE,
    )
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(
        ["0", "20", "40", "60", "80", "100"],
        fontsize=FIG4_XTICK_FONT_SIZE,
    )
    ax.tick_params(axis="y", length=0)

    if show_x_label:
        ax.set_xlabel(
            "Relative contribution to one-gap attenuation profile (%)",
            fontsize=11.4,
            labelpad=9,
        )

    ax.text(
        -0.03,
        1.03,
        panel_label,
        transform=ax.transAxes,
        fontsize=15.5,
        fontweight="bold",
        ha="right",
        va="bottom",
    )

    ax.set_title(
        panel_title,
        fontsize=FIG4_TITLE_FONT_SIZE,
        pad=10,
    )

    if show_attenuation_number:
        ax.text(
            FIG4_ALLGAPS_TEXT_X,
            y_positions[0] + 0.90,
            "Atten.",
            ha="right",
            va="bottom",
            fontsize=FIG4_LABEL_FONT_SIZE,
            fontweight="bold",
        )

    if show_all_gaps_marker:
        ax.text(
            FIG4_ALLGAPS_MARKER_X,
            y_positions[0] + 0.90,
            "All\ngaps",
            ha="center",
            va="bottom",
            fontsize=FIG4_LABEL_FONT_SIZE,
            fontweight="bold",
        )

    ax.grid(
        axis="x",
        linewidth=0.38,
        alpha=0.23,
        zorder=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def plot_figure4_relative_profiles_version(
        summary_df,
        out_png,
        out_pdf,
        figure_title_mode,
):
    if summary_df is None or len(summary_df) == 0:
        print(
            f"[INFO] No rows available for "
            f"{'main-text' if figure_title_mode == 'main' else 'supplementary'} "
            f"attenuation profile figure."
        )
        return

    diag_df = _figure4_panel_rows(summary_df, "diagnosis")
    clin_df = _figure4_panel_rows(summary_df, "clinical_score")

    if len(diag_df) == 0 and len(clin_df) == 0:
        print(
            f"[INFO] No diagnosis/clinical rows available for "
            f"{'main-text' if figure_title_mode == 'main' else 'supplementary'} "
            f"attenuation profile figure."
        )
        return

    n_diag = max(len(diag_df), 1)
    n_clin = max(len(clin_df), 1)

    fig_height = max(
        8.4,
        min(15.8, 2.8 + 0.43 * (n_diag + n_clin)),
    )

    fig = plt.figure(figsize=(12.8, fig_height))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[n_diag, n_clin],
        hspace=0.25,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])

    draw_figure4_profile_panel(
        ax=ax_a,
        panel_df=diag_df,
        panel_label="A",
        panel_title="Diagnosis associations",
        show_x_label=False,
        show_attenuation_number=True,
        show_all_gaps_marker=(figure_title_mode != "main"),
    )

    draw_figure4_profile_panel(
        ax=ax_b,
        panel_df=clin_df,
        panel_label="B",
        panel_title="Clinical associations",
        show_x_label=True,
        show_attenuation_number=True,
        show_all_gaps_marker=(figure_title_mode != "main"),
    )

    legend_handles = [
        Patch(
            facecolor=figure4_segment_color("structural", 0),
            edgecolor="white",
            label="Structural attenuator",
        ),
        Patch(
            facecolor=figure4_segment_color("functional", 0),
            edgecolor="white",
            label="Functional-connectivity attenuator",
        ),
        Patch(
            facecolor=FIG4_OTHER_COLOR,
            edgecolor="white",
            label="Pooled remaining attenuators",
        ),
    ]

    if figure_title_mode != "main":
        legend_handles.extend([
            Line2D(
                [0],
                [0],
                marker="s",
                color="black",
                markerfacecolor="black",
                markersize=6.6,
                linewidth=0,
                label="All-other-gaps result survives mutual adjustment",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="black",
                markerfacecolor="white",
                markersize=6.6,
                linewidth=0,
                label="All-other-gaps result no longer FDR-significant",
            ),
        ])

    # Keep the legend clearly below panel B's x-axis label.
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        frameon=False,
        ncol=(3 if figure_title_mode == "main" else 2),
        fontsize=FIG4_LEGEND_FONT_SIZE,
        handletextpad=0.55,
        columnspacing=1.55,
        labelspacing=0.75,
    )

    fig.subplots_adjust(
        left=0.30,
        right=0.965,
        top=0.94,
        bottom=(0.17 if figure_title_mode == "main" else 0.19),
    )

    fig.savefig(
        out_png,
        dpi=FIG4_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        out_pdf,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    label = "main-text Figure 4" if figure_title_mode == "main" else "supplementary attenuation figure"
    print(f"Saved {label} PNG:\\n{out_png}")
    print(f"Saved {label} PDF:\\n{out_pdf}")


def save_figure4_main_subset(summary_df):
    """
    Save the filtered main-text Figure 4 subset:
    only rows with >= 50% attenuation under the all-other-gaps model.
    """
    if summary_df is None or len(summary_df) == 0:
        empty = pd.DataFrame()
        empty.to_csv(FIG4_MAIN50_SUMMARY_CSV, index=False)
        print(f"Saved empty main-text Figure-4 summary table:\\n{FIG4_MAIN50_SUMMARY_CSV}")
        return empty

    main_df = summary_df.copy()
    main_df["_mutual_att"] = pd.to_numeric(
        main_df["mutual_raw_positive_attenuation_pct"],
        errors="coerce",
    )

    main_df = main_df.loc[
        main_df["_mutual_att"].ge(FIG4_MAIN_MIN_MUTUAL_ATTENUATION_PCT)
    ].copy()

    main_df = main_df.drop(columns=["_mutual_att"], errors="ignore")

    main_df.to_csv(FIG4_MAIN50_SUMMARY_CSV, index=False)
    print(f"Saved main-text Figure-4 summary table:\\n{FIG4_MAIN50_SUMMARY_CSV}")

    return main_df


def make_all_figure4_versions(summary_df):
    main_df = save_figure4_main_subset(summary_df)

    plot_figure4_relative_profiles_version(
        summary_df=main_df,
        out_png=FIG4_MAIN_OUT,
        out_pdf=FIG4_MAIN_OUT_PDF,
        figure_title_mode="main",
    )

    plot_figure4_relative_profiles_version(
        summary_df=summary_df,
        out_png=FIGS_ALL_OUT,
        out_pdf=FIGS_ALL_OUT_PDF,
        figure_title_mode="supp",
    )


def save_and_close_figure(fig, path, dpi=INDIVIDUAL_PNG_DPI):
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved standalone PNG:\n{path}")


def save_standalone_panel_pngs(
        gap_cols,
        gap_edges,
        diagnosis_predictor_edges,
        score_predictor_edges,
):
    """Save graph panels individually and as a vertically stacked figure."""
    graph_nodes = graph_gap_cols(gap_cols)

    fig_c = plt.figure(figsize=(22.0, 20.5))
    gs_c = fig_c.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.30)
    ax_c1 = fig_c.add_subplot(gs_c[0, 0])
    ax_c2 = fig_c.add_subplot(gs_c[1, 0])

    plot_gap_graph_with_predictor_nodes(
        ax=ax_c1, gap_edges=gap_edges,
        predictor_edges=diagnosis_predictor_edges,
        title="Significant adjusted diagnosis associations",
        show_predictor_legend=False, all_gap_nodes=graph_nodes,
        draw_gap_gap_edges=False,
    )
    plot_gap_graph_with_predictor_nodes(
        ax=ax_c2, gap_edges=gap_edges,
        predictor_edges=score_predictor_edges,
        title="Significant adjusted clinical associations",
        show_predictor_legend=False, all_gap_nodes=graph_nodes,
        draw_gap_gap_edges=False,
    )
    add_panel_label(ax_c1, "A", x=-0.025, y=1.030)
    add_panel_label(ax_c2, "B", x=-0.025, y=1.030)
    fig_c.legend(
        handles=make_graph_legend_handles(), loc="lower center",
        bbox_to_anchor=(0.5, 0.064), frameon=False,
        fontsize=GRAPH_LEGEND_SIZE, ncol=4, borderaxespad=0.0,
        handlelength=1.8, handletextpad=0.45,
        columnspacing=1.10, labelspacing=0.45,
    )
    add_graph_model_note(fig_c, y=0.010)
    fig_c.subplots_adjust(left=0.035, right=0.965, top=0.965, bottom=0.120)
    save_and_close_figure(fig_c, FIG2_PANELS_C_VERTICAL_OUT)

    for edges, title, label, path in [
        (diagnosis_predictor_edges, "Significant adjusted diagnosis associations", "A", FIG2_PANEL_C1_OUT),
        (score_predictor_edges, "Significant adjusted clinical associations", "B", FIG2_PANEL_C2_OUT),
    ]:
        fig, ax = plt.subplots(figsize=(20.5, 11.5))
        plot_gap_graph_with_predictor_nodes(
            ax=ax, gap_edges=gap_edges, predictor_edges=edges,
            title=title, show_predictor_legend=False,
            all_gap_nodes=graph_nodes, draw_gap_gap_edges=False,
        )
        add_panel_label(ax, label, x=-0.025, y=1.030)
        fig.legend(
            handles=make_graph_legend_handles(), loc="lower center",
            bbox_to_anchor=(0.5, 0.088), frameon=False,
            fontsize=GRAPH_LEGEND_SIZE, ncol=4, handlelength=1.8,
            handletextpad=0.45, columnspacing=1.10,
        )
        add_graph_model_note(fig, y=0.014)
        fig.subplots_adjust(left=0.035, right=0.965, top=0.955, bottom=0.170)
        save_and_close_figure(fig, path)


def create_figure2(
        gap_cols,
        gap_edges,
        diagnosis_predictor_edges,
        score_predictor_edges,
):
    """Create the combined two-panel association graph figure."""
    fig = plt.figure(figsize=(22.0, 22.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.43)
    ax_c1 = fig.add_subplot(gs[0, 0])
    ax_c2 = fig.add_subplot(gs[1, 0])
    graph_nodes = graph_gap_cols(gap_cols)
    plot_gap_graph_with_predictor_nodes(
        ax=ax_c1, gap_edges=gap_edges,
        predictor_edges=diagnosis_predictor_edges,
        title="Significant adjusted diagnosis associations",
        show_predictor_legend=False, all_gap_nodes=graph_nodes,
        draw_gap_gap_edges=False,
    )
    plot_gap_graph_with_predictor_nodes(
        ax=ax_c2, gap_edges=gap_edges,
        predictor_edges=score_predictor_edges,
        title="Significant adjusted clinical associations",
        show_predictor_legend=True, all_gap_nodes=graph_nodes,
        draw_gap_gap_edges=False,
    )
    add_panel_label(ax_c1, "A", x=-0.018, y=1.030)
    add_panel_label(ax_c2, "B", x=-0.018, y=1.030)
    fig.subplots_adjust(left=0.080, right=0.955, top=0.985, bottom=0.045)
    fig.savefig(FIG2_OUT, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG2_OUT_PDF, bbox_inches="tight", facecolor="white")
    print(f"Saved Figure 2 PNG:\n{FIG2_OUT}")
    print(f"Saved Figure 2 PDF:\n{FIG2_OUT_PDF}")
    save_standalone_panel_pngs(
        gap_cols=gap_cols, gap_edges=gap_edges,
        diagnosis_predictor_edges=diagnosis_predictor_edges,
        score_predictor_edges=score_predictor_edges,
    )
    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(
        "Main figures with internally computed all-other-gaps predictor-edge "
        f"boxes will be written to:\n{RESULTS_DIR}"
    )

    # Build subject metadata from the IDs in the first gap input.
    first_gap_path = next(iter(INPUT_FILES.values()))
    subject_md = pd.read_csv(first_gap_path, usecols=[ID_COL])
    subject_md[ID_COL] = canonicalize_id_series(subject_md[ID_COL])
    subject_md = subject_md.drop_duplicates(subset=[ID_COL])
    subject_md = add_diagnosis_columns(subject_md)
    subject_md = add_mean_motion_covariate(subject_md)
    subject_md = apply_plotting_diagnosis_labels(subject_md)
    gap_df, gap_cols = load_gap_space(subject_md)

    print("\nFitting predictor--gap associations and layout aids")
    print("=" * 80)

    graph_cols = graph_gap_cols(gap_cols)

    # Gap--gap associations are retained only as a hidden layout aid. They are
    # not drawn in the refined manuscript Figure 3.
    gap_edges = compute_gap_adjusted_ols_hc3_edges(
        gap_df=gap_df,
        gap_cols=gap_cols,
    )

    diagnosis_edges = fit_standard_diagnosis_gap_edges(
        gap_df=gap_df,
        gap_cols=graph_cols,
    )

    score_edges = fit_standard_score_gap_edges(
        gap_df=gap_df,
        gap_cols=graph_cols,
    )

    predictor_edges = collect_predictor_gap_edges(
        diagnosis_edges=diagnosis_edges,
        score_edges=score_edges,
    )

    print("\nComputing internal all-other-gaps correction")
    print("=" * 80)

    all_other_corrected_models = compute_internal_all_other_gap_correction(
        gap_df=gap_df,
        gap_cols=graph_cols,
        predictor_edges=predictor_edges,
    )

    predictor_edges = attach_internal_all_other_correction_to_primary_edges(
        predictor_edges=predictor_edges,
        corrected_models=all_other_corrected_models,
    )

    diagnosis_predictor_edges = predictor_edges[
        predictor_edges["source"].eq("diagnosis")
    ].copy()

    score_predictor_edges = predictor_edges[
        predictor_edges["source"].eq("clinical_score")
    ].copy()

    print("\nSplit graph edge counts")
    print("=" * 80)
    print(f"Diagnosis-vs-HC predictor edges: {len(diagnosis_predictor_edges):,}")
    print(f"Score / antidepressant predictor edges: {len(score_predictor_edges):,}")

    print("\nComputing Figure 4 relative attenuation profiles")
    print("=" * 80)

    figure4_summary, figure4_detailed = compute_figure4_relative_profiles(
        gap_df=gap_df,
        gap_cols=gap_cols,
        predictor_edges=predictor_edges,
        all_other_corrected_models=all_other_corrected_models,
    )

    make_all_figure4_versions(
        summary_df=figure4_summary,
    )

    create_figure2(
        gap_cols=gap_cols,
        gap_edges=gap_edges,
        diagnosis_predictor_edges=diagnosis_predictor_edges,
        score_predictor_edges=score_predictor_edges,
    )


if __name__ == "__main__":
    main()

