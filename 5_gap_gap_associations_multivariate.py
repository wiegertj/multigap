#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone MAIN-primary analysis of the current 13-gap space.

This script performs TWO analyses using the current MAIN definitions:

1) Pairwise PRIMARY partial correlations among every gap in the 13-gap space.

   Each tested gap pair is adjusted only for the ordinary MAIN covariates:
       - sex
       - age_assessment_center_2
       - site
       - etiv
       - mean_motion

2) Joint / multivariate association of ALL 13 gaps with each current
   clinical or diagnosis measure, adjusting for the same available ordinary
   covariates.
"""

from __future__ import annotations

import os
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t as student_t
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

OUT_DIR = ...

RESULTS_DIR = os.path.join(OUT_DIR, "_gap_correlations_and_multivariate_R_MAIN_primary")

CSV_DIAGNOSES = ...

CSV_MEAN_MOTION = ...

INPUT_FILES = {"freesurfer": ..., "connectivity": ...}

ID_COL = "Participant ID"
SPLIT_GROUP_COL = "split_group"
TRAIN_HC_SPLIT_VALUE = "train_hc"
HELDOUT_HC_SPLIT_VALUE = "heldout_hc"

MEAN_MOTION_ID_COL = "eid"
MEAN_MOTION_VALUE_COL = "framewise_dis"
MEAN_MOTION_COVARIATE_COL = "mean_motion"

# Same continuous target list as the supplied Figure-2 script.
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

# Final MAIN-primary 13-gap space.
# FreeSurfer contributes 7 targets; connectivity contributes 6 targets.
FEATURE_TARGET_EXCLUSIONS = {
    "freesurfer": set(),
    "connectivity": {
        "Hand_grip_strength_mean",
    },
}

FEATURE_SET_ORDER = ["freesurfer", "connectivity"]

# Current MAIN diagnosis/control variables.
HC_COL = "MAIN_HC"
AUD_COL = "MAIN_AUD"
MDD_COL = "MAIN_MDD"
ANX_COL = "MAIN_ANX"

DIAGNOSIS_COLUMNS = [
    HC_COL,
    AUD_COL,
    MDD_COL,
    ANX_COL,
]

# Clinical score variables used by Figure 2.
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

RANK_NORMALIZE_CLINICAL_MEASURES = True

# Same candidate adjustment covariates as Figure 2.
COVARIATE_CANDIDATES = [
    "sex",
    "age_assessment_center_2",
    "site",
    "etiv",
    "mean_motion",
]

COVARIATE_SOURCE_ALIASES = {
    "etiv": [
        "etiv",
        "eTIV",
        "tiv",
        "TIV",
        "ICV",
        "EstimatedTotalIntraCranialVol",
    ],
}

# Statistics.
FDR_ALPHA = 0.05
CI_LEVEL = 0.95
RANDOM_SEED = 123

# Bootstrap for the multivariate partial-R confidence intervals.
# 1000 is a reasonable manuscript setting; use 2000+ if you want a smoother CI
# and can tolerate the extra runtime.
N_BOOT_MULTIVARIATE_R = 1000

# Minimum sample requirements.
MIN_N_CORRELATION = 10
MIN_N_MULTIVARIATE = 30
MIN_DIAGNOSIS_CASES = 10
MIN_DIAGNOSIS_CONTROLS = 10

# Heatmap appearance.
HEATMAP_DPI = 350
HEATMAP_ANNOTATE = True
HEATMAP_SHOW_Q_STARS = True
HEATMAP_FIGSIZE = (15.5, 13.5)

# Thick divider between imaging modalities in the correlation heatmap.
# The gap columns are ordered by FEATURE_SET_ORDER, so this border cleanly
# separates the FreeSurfer block from the connectivity block on both axes.
HEATMAP_MODALITY_BORDER_LINEWIDTH = 3.0
HEATMAP_MODALITY_BORDER_COLOR = "black"

# Multivariate-R forest plot.
MULTIVARIATE_FIGSIZE = (11.5, 7.2)
MULTIVARIATE_DPI = 350

# Outputs.
PAIRWISE_LONG_CSV = os.path.join(
    RESULTS_DIR, "pairwise_gap_partial_correlations_MAIN_primary_long.csv"
)
PAIRWISE_R_MATRIX_CSV = os.path.join(
    RESULTS_DIR, "pairwise_gap_partial_correlation_MAIN_primary_matrix.csv"
)
PAIRWISE_Q_MATRIX_CSV = os.path.join(
    RESULTS_DIR, "pairwise_gap_partial_correlation_MAIN_primary_q_matrix.csv"
)
PAIRWISE_HEATMAP_PNG = os.path.join(
    RESULTS_DIR, "pairwise_gap_partial_correlation_MAIN_primary_heatmap.png"
)
PAIRWISE_HEATMAP_PDF = os.path.join(
    RESULTS_DIR, "pairwise_gap_partial_correlation_MAIN_primary_heatmap.pdf"
)

MULTIVARIATE_R_CSV = os.path.join(
    RESULTS_DIR, "multivariate_R_all_gaps_to_clinical_and_diagnosis.csv"
)
MULTIVARIATE_R_PNG = os.path.join(
    RESULTS_DIR, "multivariate_R_all_gaps_forest.png"
)
MULTIVARIATE_R_PDF = os.path.join(
    RESULTS_DIR, "multivariate_R_all_gaps_forest.pdf"
)


# =============================================================================
# BASIC HELPERS
# =============================================================================

def canonicalize_id_series(x: pd.Series) -> pd.Series:
    return (
        x.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def ranknorm_series(x: pd.Series) -> pd.Series:
    """Rank-based inverse-normal transform, matching the Figure-2 logic."""
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


def zscore_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0, ddof=0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    return (x - mu) / sd


def target_short_label(target: str) -> str:
    labels = {
        "brain_age": "Age",
        "Fluid_intelligence_score": "Fluid intelligence",
        "body_fat_percentage": "Body fat",
        "Sleep_duration": "Sleep",
        "Hand_grip_strength_mean": "Grip",
        "bmi": "BMI",
        "diastolic_bp": "DBP",
        "trail_making": "Trail Making",
    }
    return labels.get(target, target)


def gap_display_label(gap: str) -> str:
    if "__" not in gap:
        return gap
    feature_set, target = gap.split("__", 1)
    prefix = "FS" if feature_set == "freesurfer" else "Conn"
    return f"{prefix} | {target_short_label(target)}"


def available_covariates(df: pd.DataFrame) -> list[str]:
    return [c for c in COVARIATE_CANDIDATES if c in df.columns]


def resolve_covariate_source_columns(
    header: list[str],
    existing_columns: list[str],
) -> tuple[list[str], dict[str, str]]:
    """Find source columns for canonical covariate names."""
    header_set = set(header)
    existing = set(existing_columns)

    read_cols = []
    rename_map = {}

    for canonical in COVARIATE_CANDIDATES:
        if canonical in existing:
            continue

        aliases = COVARIATE_SOURCE_ALIASES.get(canonical, [canonical])
        for source_col in aliases:
            if source_col in header_set:
                read_cols.append(source_col)
                if source_col != canonical:
                    rename_map[source_col] = canonical
                break

    return read_cols, rename_map


def get_gap_specs(df: pd.DataFrame, feature_set: str) -> list[tuple[str, str, str]]:
    """Return (feature_set, target, source_gap_column) for Figure-2 gaps."""
    specs = []
    excluded = FEATURE_TARGET_EXCLUSIONS.get(feature_set, set())

    for target in REGRESSION_TARGETS:
        if target in excluded:
            continue

        source_col = f"{target}_gap_z"
        if source_col in df.columns:
            specs.append((feature_set, target, source_col))
        else:
            print(
                f"[WARN] {feature_set}: expected gap column not found and will be "
                f"skipped: {source_col}"
            )

    return specs


# =============================================================================
# LOAD THE SAME GAP SPACE + METADATA
# =============================================================================

def load_gap_space() -> tuple[pd.DataFrame, list[str]]:
    """
    Merge FreeSurfer and connectivity gap files, rename gap columns as
    feature_set__target, and remove train_hc.
    """
    frames = []
    gap_cols = []

    for feature_set in FEATURE_SET_ORDER:
        path = INPUT_FILES[feature_set]

        df = pd.read_csv(path)


        df[ID_COL] = canonicalize_id_series(df[ID_COL])
        df[SPLIT_GROUP_COL] = df[SPLIT_GROUP_COL].astype(str).str.strip()


        specs = get_gap_specs(df, feature_set)
        keep = [ID_COL, SPLIT_GROUP_COL] + [x[2] for x in specs]
        df = df[keep].copy()

        rename = {}
        for fs_name, target, old_col in specs:
            new_col = f"{fs_name}__{target}"
            rename[old_col] = new_col
            gap_cols.append(new_col)

        df = df.rename(columns=rename)
        for c in rename.values():
            df[c] = pd.to_numeric(df[c], errors="coerce")

        frames.append(df)


    merged = frames[0].copy()
    for other in frames[1:]:
        merged = merged.merge(
            other,
            on=[ID_COL, SPLIT_GROUP_COL],
            how="inner",
            validate="one_to_one",
        )

    before = len(merged)
    merged = merged.loc[
        ~merged[SPLIT_GROUP_COL].eq(TRAIN_HC_SPLIT_VALUE)
    ].copy()

    print("\nLoaded Figure-2 gap space")
    print("=" * 80)
    print(f"Rows before train-HC exclusion: {before:,}")
    print(f"Rows after train-HC exclusion:  {len(merged):,}")
    print(f"Number of gap variables:       {len(gap_cols):,}")
    for c in gap_cols:
        print(f"  - {c}")

    return merged, gap_cols


def add_diagnosis_clinical_and_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """Merge diagnosis indicators, clinical scores, covariates, and mean motion."""

    header = pd.read_csv(CSV_DIAGNOSES, nrows=0).columns.tolist()

    missing_diag = [c for c in DIAGNOSIS_COLUMNS if c not in header]

    optional_scores = [c for c in SCORE_PREDICTOR_COLS if c in header]
    missing_scores = [c for c in SCORE_PREDICTOR_COLS if c not in header]
    if missing_scores:
        print(
            "[WARN] Missing clinical score columns; these outcomes will be skipped: "
            + ", ".join(missing_scores)
        )

    cov_source_cols, rename_map = resolve_covariate_source_columns(
        header=header,
        existing_columns=df.columns.tolist(),
    )

    usecols = [
        ID_COL,
        *DIAGNOSIS_COLUMNS,
        *optional_scores,
        *cov_source_cols,
    ]
    usecols = list(dict.fromkeys(usecols))

    meta = pd.read_csv(CSV_DIAGNOSES, usecols=usecols)
    meta[ID_COL] = canonicalize_id_series(meta[ID_COL])

    if rename_map:
        meta = meta.rename(columns=rename_map)


    out = df.copy()
    out[ID_COL] = canonicalize_id_series(out[ID_COL])
    out = out.merge(meta, on=ID_COL, how="left", validate="one_to_one")

    # Mean motion is stored separately in the Figure-2 workflow.
    if os.path.exists(CSV_MEAN_MOTION):
        mm = pd.read_csv(
            CSV_MEAN_MOTION,
            usecols=[MEAN_MOTION_ID_COL, MEAN_MOTION_VALUE_COL],
        ).rename(
            columns={
                MEAN_MOTION_ID_COL: ID_COL,
                MEAN_MOTION_VALUE_COL: MEAN_MOTION_COVARIATE_COL,
            }
        )
        mm[ID_COL] = canonicalize_id_series(mm[ID_COL])
        mm[MEAN_MOTION_COVARIATE_COL] = pd.to_numeric(
            mm[MEAN_MOTION_COVARIATE_COL], errors="coerce"
        )
        mm = mm.drop_duplicates(subset=[ID_COL], keep="first")

        # Replace a stale same-name column if one somehow already exists.
        out = out.drop(columns=[MEAN_MOTION_COVARIATE_COL], errors="ignore")
        out = out.merge(mm, on=ID_COL, how="left", validate="one_to_one")
    else:
        print(f"[WARN] Mean-motion file not found; skipping: {CSV_MEAN_MOTION}")

    for c in DIAGNOSIS_COLUMNS + SCORE_PREDICTOR_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    print("\nMerged diagnosis / clinical / covariate data")
    print("=" * 80)
    print(f"Rows: {len(out):,}")
    print("Available covariates:", ", ".join(available_covariates(out)) or "none")

    return out


# =============================================================================
# ANALYSIS 1: MODALITY-ADJUSTED PAIRWISE GAP PARTIAL CORRELATIONS
# =============================================================================

def partial_r_fisher_ci(
    r: float,
    n: int,
    effective_covariate_df: int,
    ci_level: float = CI_LEVEL,
) -> tuple[float, float]:
    """
    Approximate Fisher-z confidence interval for a partial correlation.

    If k independent adjustment dimensions are partialled out, the standard
    Fisher-z approximation uses

        SE[z] = 1 / sqrt(n - k - 3).

    Here k is obtained from the rank of the actual adjustment design after
    dummy coding and after adding the two leave-pair-out modality PC1 scores.
    """
    if not np.isfinite(r):
        return np.nan, np.nan

    fisher_df = int(n - effective_covariate_df - 3)
    if fisher_df <= 0:
        return np.nan, np.nan

    r_clip = float(np.clip(r, -0.999999999, 0.999999999))
    z = np.arctanh(r_clip)
    alpha = 1.0 - ci_level
    zcrit = norm.ppf(1.0 - alpha / 2.0)
    se = 1.0 / np.sqrt(float(fisher_df))

    lo = np.tanh(z - zcrit * se)
    hi = np.tanh(z + zcrit * se)
    return float(lo), float(hi)


def _residualize_vector(y: np.ndarray, X_with_intercept: np.ndarray) -> np.ndarray:
    """OLS residuals of one vector after adjustment for X_with_intercept."""
    beta, *_ = np.linalg.lstsq(X_with_intercept, y, rcond=None)
    return y - X_with_intercept @ beta










def compute_pairwise_gap_correlations(
    df: pd.DataFrame,
    gap_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute the MAIN-primary covariate-adjusted partial correlations.

    For each unique gap pair A, B:
      1. use subjects complete for A, B, and the ordinary covariates;
      2. build the ordinary covariate design;
      3. residualize A and B on intercept + covariates;
      4. correlate the residuals;
      5. compute the two-sided partial-correlation t test using
             df = n - k - 2
         where k is the effective rank of the covariate design minus intercept;
      6. compute a Fisher-z 95% CI;
      7. apply BH-FDR across all unique gap-gap tests.

    No modality-PC nuisance variables are used in this primary analysis.
    """
    rows = []
    covariates = available_covariates(df)

    print(
        "MAIN-primary partial-correlation covariates: "
        + (", ".join(covariates) if covariates else "none")
    )
    print("Additional modality-PC adjustment: none (primary analysis)")

    all_gap_cols = [g for g in gap_cols if g in df.columns]

    for a, b in combinations(all_gap_cols, 2):
        needed = [a, b, *covariates]
        needed = list(dict.fromkeys([c for c in needed if c in df.columns]))
        tmp = df[needed].copy()

        tmp[a] = pd.to_numeric(tmp[a], errors="coerce")
        tmp[b] = pd.to_numeric(tmp[b], errors="coerce")
        tmp = tmp.replace([np.inf, -np.inf], np.nan)
        tmp = tmp.dropna(subset=[c for c in needed if c in tmp.columns]).copy()

        X_cov = prepare_covariate_design(tmp, covariates)
        model_df = pd.concat([tmp[[a, b]], X_cov], axis=1)
        model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna().copy()

        n = int(len(model_df))
        r = p = ci_low = ci_high = np.nan
        effective_covariate_df = 0
        partial_test_df = np.nan
        design_rank_with_intercept = np.nan

        if n >= MIN_N_CORRELATION:
            x = model_df[a].astype(float).values
            y = model_df[b].astype(float).values

            if (
                np.unique(x[np.isfinite(x)]).size >= 2
                and np.unique(y[np.isfinite(y)]).size >= 2
            ):
                cov_cols = [c for c in X_cov.columns if c in model_df.columns]
                if cov_cols:
                    C = model_df[cov_cols].astype(float).values
                    X_adjust = np.column_stack([np.ones(n, dtype=float), C])
                else:
                    X_adjust = np.ones((n, 1), dtype=float)

                rank_adjust = int(np.linalg.matrix_rank(X_adjust))
                design_rank_with_intercept = rank_adjust
                effective_covariate_df = max(0, rank_adjust - 1)
                partial_test_df = int(n - effective_covariate_df - 2)

                x_resid = _residualize_vector(x, X_adjust)
                y_resid = _residualize_vector(y, X_adjust)

                sx = float(np.std(x_resid, ddof=0))
                sy = float(np.std(y_resid, ddof=0))

                if (
                    np.isfinite(sx)
                    and np.isfinite(sy)
                    and sx > 0
                    and sy > 0
                    and partial_test_df > 0
                ):
                    r = float(np.corrcoef(x_resid, y_resid)[0, 1])
                    r = float(np.clip(r, -1.0, 1.0))

                    if abs(r) >= 1.0:
                        p = 0.0
                    else:
                        t_stat = r * np.sqrt(
                            float(partial_test_df) / max(1.0 - r * r, 1e-15)
                        )
                        p = float(
                            2.0 * student_t.sf(abs(t_stat), df=partial_test_df)
                        )

                    ci_low, ci_high = partial_r_fisher_ci(
                        r=r,
                        n=n,
                        effective_covariate_df=effective_covariate_df,
                        ci_level=CI_LEVEL,
                    )

        fs_a, target_a = a.split("__", 1)
        fs_b, target_b = b.split("__", 1)

        rows.append({
            "gap_1": a,
            "gap_2": b,
            "feature_set_1": fs_a,
            "target_1": target_a,
            "feature_set_2": fs_b,
            "target_2": target_b,
            "partial_r": float(r) if np.isfinite(r) else np.nan,
            "r": float(r) if np.isfinite(r) else np.nan,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": float(p) if np.isfinite(p) else np.nan,
            "n": n,
            "covariates_requested": ";".join(covariates),
            "adjustment": "ordinary MAIN covariates only",
            "n_covariate_design_columns": int(X_cov.shape[1]),
            "design_rank_with_intercept": (
                int(design_rank_with_intercept)
                if np.isfinite(design_rank_with_intercept)
                else np.nan
            ),
            "effective_covariate_df": int(effective_covariate_df),
            "partial_test_df": (
                int(partial_test_df) if np.isfinite(partial_test_df) else np.nan
            ),
            "ci_level": CI_LEVEL,
        })

    long_df = pd.DataFrame(rows)
    long_df["q_fdr_bh"] = np.nan

    ok = long_df["p_value"].notna()
    if ok.sum() > 0:
        long_df.loc[ok, "q_fdr_bh"] = multipletests(
            long_df.loc[ok, "p_value"].values,
            method="fdr_bh",
        )[1]

    long_df["significant_fdr_0p05"] = long_df["q_fdr_bh"].lt(FDR_ALPHA)
    long_df["abs_r"] = long_df["partial_r"].abs()
    long_df = long_df.sort_values(
        ["q_fdr_bh", "abs_r"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)

    rmat = pd.DataFrame(
        np.eye(len(all_gap_cols)), index=all_gap_cols, columns=all_gap_cols
    )
    qmat = pd.DataFrame(
        np.nan, index=all_gap_cols, columns=all_gap_cols
    )
    np.fill_diagonal(qmat.values, 0.0)

    for _, row in long_df.iterrows():
        a = row["gap_1"]
        b = row["gap_2"]
        rmat.loc[a, b] = row["partial_r"]
        rmat.loc[b, a] = row["partial_r"]
        qmat.loc[a, b] = row["q_fdr_bh"]
        qmat.loc[b, a] = row["q_fdr_bh"]

    return long_df, rmat, qmat

def plot_pairwise_gap_heatmap(
    rmat: pd.DataFrame,
    qmat: pd.DataFrame,
    out_png: str,
    out_pdf: str,
) -> None:
    """Lower-triangle MAIN-primary partial-correlation heatmap."""
    labels = [gap_display_label(c) for c in rmat.columns]
    values = rmat.values.astype(float)
    qvals = qmat.values.astype(float)
    n_vars = len(labels)

    # Mask the redundant upper triangle only; retain the diagonal.
    mask = np.triu(np.ones_like(values, dtype=bool), k=1)
    plot_values = np.ma.array(values, mask=mask)

    fig, ax = plt.subplots(figsize=HEATMAP_FIGSIZE)
    im = ax.imshow(
        plot_values,
        vmin=-1.0,
        vmax=1.0,
        cmap="RdBu_r",
        interpolation="none",
        aspect="equal",
    )

    ax.set_xticks(np.arange(n_vars))
    ax.set_yticks(np.arange(n_vars))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=10.5)
    ax.set_yticklabels(labels, fontsize=10.5)

    # Fine cell grid.
    ax.set_xticks(np.arange(-0.5, n_vars, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_vars, 1), minor=True)
    ax.grid(which="minor", linewidth=0.45, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    # -------------------------------------------------------------------------
    # VISUAL MODALITY BORDER: FreeSurfer vs connectivity
    # -------------------------------------------------------------------------
    # rmat preserves gap_cols order from load_gap_space(), which follows
    # FEATURE_SET_ORDER = ["freesurfer", "connectivity"].  Find every point
    # where the feature-set prefix changes and draw a thick boundary between
    # the corresponding rows and columns.  With the current Figure-2 gap set
    # this produces one clear FS/CONN divider.
    feature_sets = [str(c).split("__", 1)[0] for c in rmat.columns]
    modality_boundaries = [
        i - 0.5
        for i in range(1, n_vars)
        if feature_sets[i] != feature_sets[i - 1]
    ]

    for boundary in modality_boundaries:
        ax.axvline(
            boundary,
            color=HEATMAP_MODALITY_BORDER_COLOR,
            linewidth=HEATMAP_MODALITY_BORDER_LINEWIDTH,
            zorder=8,
            solid_capstyle="butt",
        )
        ax.axhline(
            boundary,
            color=HEATMAP_MODALITY_BORDER_COLOR,
            linewidth=HEATMAP_MODALITY_BORDER_LINEWIDTH,
            zorder=8,
            solid_capstyle="butt",
        )

    if HEATMAP_ANNOTATE:
        for i in range(n_vars):
            for j in range(n_vars):
                if mask[i, j]:
                    continue

                r = values[i, j]
                if not np.isfinite(r):
                    text = "NA"
                elif i == j:
                    text = "1.00"
                else:
                    star = ""
                    if (
                        HEATMAP_SHOW_Q_STARS
                        and np.isfinite(qvals[i, j])
                        and qvals[i, j] < FDR_ALPHA
                    ):
                        star = "*"
                    text = f"{r:.2f}{star}"

                # Choose text color from cell magnitude for readability.
                color = "white" if np.isfinite(r) and abs(r) >= 0.55 else "black"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    color=color,
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("Partial r", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=HEATMAP_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================================
# ANALYSIS 2: ALL-GAPS MULTIVARIATE R
# =============================================================================

def prepare_covariate_design(
    df: pd.DataFrame,
    covariates: list[str],
) -> pd.DataFrame:
    """
    Match the Figure-2 covariate handling:
      - >=90% numeric-parsable -> numeric covariate
      - otherwise -> categorical dummy coding with drop_first=True
    """
    if not covariates:
        return pd.DataFrame(index=df.index)

    pieces = []

    for c in covariates:
        if c not in df.columns:
            continue

        x = df[c]
        numeric = pd.to_numeric(x, errors="coerce")
        observed = x.notna()
        numeric_fraction = (
            numeric[observed].notna().mean() if observed.sum() else 0.0
        )

        if numeric_fraction >= 0.90:
            z = numeric.astype(float)
            if z.notna().sum() >= 10 and z.nunique(dropna=True) > 1:
                pieces.append(z.rename(c).to_frame())
        else:
            dummies = pd.get_dummies(
                x.astype("category"),
                prefix=c,
                drop_first=True,
            )
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


def build_complete_model_arrays(
    df: pd.DataFrame,
    outcome_col: str,
    gap_cols: list[str],
    covariates: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], int]:
    """
    Return y, covariate matrix C, gap matrix G, used gap names, and n.

    C does NOT include the intercept; it is added inside the fitting functions.
    G and numeric C columns are z-scored after complete-case selection for
    numerical stability. This does not change R^2 or the omnibus model span.
    """
    needed = [outcome_col, *gap_cols, *covariates]
    needed = list(dict.fromkeys([c for c in needed if c in df.columns]))

    if outcome_col not in needed:
        return (
            np.empty(0),
            np.empty((0, 0)),
            np.empty((0, 0)),
            [],
            0,
        )

    tmp = df[needed].copy()
    tmp[outcome_col] = pd.to_numeric(tmp[outcome_col], errors="coerce")
    for g in gap_cols:
        if g in tmp.columns:
            tmp[g] = pd.to_numeric(tmp[g], errors="coerce")

    X_cov = prepare_covariate_design(tmp, covariates)

    pieces = [tmp[[outcome_col]], tmp[gap_cols].astype(float), X_cov]
    model_df = pd.concat(pieces, axis=1)
    model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna().copy()

    if len(model_df) < MIN_N_MULTIVARIATE:
        return (
            np.empty(0),
            np.empty((0, 0)),
            np.empty((0, 0)),
            [],
            len(model_df),
        )

    # Drop any gap that is constant in this exact outcome-specific complete-case set.
    used_gaps = []
    for g in gap_cols:
        vals = model_df[g].astype(float)
        if vals.nunique(dropna=True) > 1 and np.isfinite(vals.std(ddof=0)):
            if vals.std(ddof=0) > 0:
                used_gaps.append(g)

    if not used_gaps:
        return (
            np.empty(0),
            np.empty((0, 0)),
            np.empty((0, 0)),
            [],
            len(model_df),
        )

    y = model_df[outcome_col].astype(float).values
    G = model_df[used_gaps].astype(float).values

    cov_cols = [c for c in X_cov.columns if c in model_df.columns]
    if cov_cols:
        C = model_df[cov_cols].astype(float).values
    else:
        C = np.empty((len(model_df), 0), dtype=float)

    # Standardization is purely numerical; model fit/R^2 is invariant to it.
    y_sd = np.std(y, ddof=0)
    if np.isfinite(y_sd) and y_sd > 0:
        y = (y - np.mean(y)) / y_sd

    G = zscore_array(G)
    if C.shape[1] > 0:
        C = zscore_array(C)

    return y.astype(float), C.astype(float), G.astype(float), used_gaps, len(y)


def add_intercept(X: np.ndarray, n: int) -> np.ndarray:
    if X.size == 0:
        return np.ones((n, 1), dtype=float)
    return np.column_stack([np.ones(n, dtype=float), X])


def ols_sse(y: np.ndarray, X: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(resid @ resid)


def partial_r_from_arrays(
    y: np.ndarray,
    C: np.ndarray,
    G: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return partial_R, partial_R2, full_R2, reduced_R2."""
    n = len(y)
    Xr = add_intercept(C, n)
    Xf = add_intercept(np.column_stack([C, G]) if C.shape[1] else G, n)

    sse_r = ols_sse(y, Xr)
    sse_f = ols_sse(y, Xf)

    y_centered = y - np.mean(y)
    sst = float(y_centered @ y_centered)

    if not np.isfinite(sse_r) or sse_r <= 0 or not np.isfinite(sse_f):
        return np.nan, np.nan, np.nan, np.nan

    partial_r2 = (sse_r - sse_f) / sse_r
    partial_r2 = float(np.clip(partial_r2, 0.0, 1.0))
    partial_r = float(np.sqrt(partial_r2))

    if sst > 0:
        full_r2 = float(np.clip(1.0 - sse_f / sst, 0.0, 1.0))
        reduced_r2 = float(np.clip(1.0 - sse_r / sst, 0.0, 1.0))
    else:
        full_r2 = reduced_r2 = np.nan

    return partial_r, partial_r2, full_r2, reduced_r2


def fit_multivariate_omnibus(
    y: np.ndarray,
    C: np.ndarray,
    G: np.ndarray,
) -> dict:
    """
    Fit the nested OLS models and return R statistics plus classic and HC3 p-values.
    """
    n = len(y)
    if n < MIN_N_MULTIVARIATE or G.shape[1] == 0:
        return {}

    Xr_raw = C
    Xf_raw = np.column_stack([C, G]) if C.shape[1] else G
    Xr = add_intercept(Xr_raw, n)
    Xf = add_intercept(Xf_raw, n)

    # Rank check is reported; statsmodels uses a pseudoinverse if needed.
    full_rank = int(np.linalg.matrix_rank(Xf))
    n_full_params = int(Xf.shape[1])

    try:
        fit_reduced = sm.OLS(y, Xr).fit()
        fit_full_classic = sm.OLS(y, Xf).fit()
        fit_full_hc3 = sm.OLS(y, Xf).fit(cov_type="HC3")
    except Exception as exc:
        print(f"[WARN] OLS fit failed: {exc}")
        return {}

    partial_r, partial_r2, full_r2, reduced_r2 = partial_r_from_arrays(y, C, G)

    # Standard nested-model F test, useful as a conventional reference.
    try:
        f_stat, p_classic, df_diff = fit_full_classic.compare_f_test(fit_reduced)
        f_stat = float(f_stat)
        p_classic = float(p_classic)
        df_diff = int(round(float(df_diff)))
    except Exception:
        f_stat = p_classic = np.nan
        df_diff = int(G.shape[1])

    # Primary p-value: HC3-robust omnibus Wald/F test of ALL gap coefficients.
    # Parameter layout is: intercept, C columns, G columns.
    p_hc3 = np.nan
    wald_stat = np.nan
    try:
        n_params = Xf.shape[1]
        n_gap = G.shape[1]
        gap_start = 1 + C.shape[1]
        R = np.zeros((n_gap, n_params), dtype=float)
        for j in range(n_gap):
            R[j, gap_start + j] = 1.0

        wald = fit_full_hc3.wald_test(R, use_f=True, scalar=True)
        wald_stat = float(np.asarray(wald.statistic).squeeze())
        p_hc3 = float(np.asarray(wald.pvalue).squeeze())
    except Exception as exc:
        print(f"[WARN] HC3 omnibus Wald test failed: {exc}")

    return {
        "partial_R": partial_r,
        "partial_R2": partial_r2,
        "full_R": float(np.sqrt(full_r2)) if np.isfinite(full_r2) else np.nan,
        "full_R2": full_r2,
        "covariate_only_R2": reduced_r2,
        "incremental_R2_all_gaps": partial_r2,
        "wald_F_hc3": wald_stat,
        "p_value": p_hc3,
        "classic_nested_F": f_stat,
        "p_classic_F": p_classic,
        "df_gap_block": df_diff,
        "full_design_rank": full_rank,
        "full_design_n_columns": n_full_params,
    }


def bootstrap_partial_r_ci(
    y: np.ndarray,
    C: np.ndarray,
    G: np.ndarray,
    n_boot: int,
    seed: int,
    stratify_binary: bool = False,
) -> tuple[float, float, int]:
    """
    Subject-level percentile bootstrap CI for partial multiple R.

    For diagnosis 0/1 outcomes, resampling is stratified so every bootstrap keeps
    the observed case/control sample sizes.
    """
    if n_boot <= 0:
        return np.nan, np.nan, 0

    rng = np.random.default_rng(seed)
    n = len(y)
    boot_r = np.full(n_boot, np.nan, dtype=float)

    if stratify_binary:
        # y may have been z-scored, so identify the two unique outcome levels.
        levels = np.unique(y[np.isfinite(y)])
        if len(levels) != 2:
            stratify_binary = False
        else:
            idx0 = np.where(y == levels[0])[0]
            idx1 = np.where(y == levels[1])[0]

    for b in range(n_boot):
        if stratify_binary:
            draw0 = rng.choice(idx0, size=len(idx0), replace=True)
            draw1 = rng.choice(idx1, size=len(idx1), replace=True)
            idx = np.concatenate([draw0, draw1])
            rng.shuffle(idx)
        else:
            idx = rng.integers(0, n, size=n)

        try:
            r, _, _, _ = partial_r_from_arrays(
                y=y[idx],
                C=C[idx, :] if C.shape[1] else C,
                G=G[idx, :],
            )
            boot_r[b] = r
        except Exception:
            continue

    valid = boot_r[np.isfinite(boot_r)]
    if len(valid) < max(50, int(0.25 * n_boot)):
        return np.nan, np.nan, int(len(valid))

    alpha = 1.0 - CI_LEVEL
    lo, hi = np.quantile(valid, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi), int(len(valid))


def diagnosis_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create current MAIN diagnosis flags and heldout-HC reference flag."""
    out = df.copy()

    for c in DIAGNOSIS_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    out["is_hc"] = (
        out[HC_COL].eq(1)
        & out[SPLIT_GROUP_COL].astype(str).str.strip().eq(HELDOUT_HC_SPLIT_VALUE)
    )
    out["is_aud"] = out[AUD_COL].eq(1)
    out["is_mdd"] = out[MDD_COL].eq(1)
    out["is_anx"] = out[ANX_COL].eq(1)

    return out


def make_diagnosis_contrast(
    df: pd.DataFrame,
    contrast_name: str,
) -> tuple[pd.DataFrame, str, int, int]:
    """
    Return the current MAIN case-vs-heldout-HC subset.

    MAIN_MDD and MAIN_ANX overlap is intentionally preserved. MAIN_AUD is used
    exactly as encoded in the current diagnosis table.
    """
    work = diagnosis_flags(df)

    if contrast_name == "AUD vs HC":
        case_mask = work["is_aud"]
    elif contrast_name == "MDD vs HC":
        case_mask = work["is_mdd"]
    elif contrast_name == "ANX vs HC":
        case_mask = work["is_anx"]
    else:
        raise ValueError(f"Unknown MAIN contrast: {contrast_name}")

    ref_mask = work["is_hc"]
    keep = case_mask | ref_mask
    work = work.loc[keep].copy()

    outcome_col = "__diagnosis_outcome__"
    work[outcome_col] = case_mask.loc[work.index].astype(int)

    n_case = int(work[outcome_col].eq(1).sum())
    n_ref = int(work[outcome_col].eq(0).sum())

    return work, outcome_col, n_case, n_ref

def analyze_one_multivariate_outcome(
    df: pd.DataFrame,
    gap_cols: list[str],
    outcome_col: str,
    outcome_label: str,
    outcome_type: str,
    covariates: list[str],
    seed: int,
    n_case: int | None = None,
    n_control: int | None = None,
) -> dict | None:
    y, C, G, used_gaps, n = build_complete_model_arrays(
        df=df,
        outcome_col=outcome_col,
        gap_cols=gap_cols,
        covariates=covariates,
    )

    if n < MIN_N_MULTIVARIATE or len(used_gaps) == 0:
        print(
            f"[WARN] Skipping {outcome_label}: insufficient complete-case data "
            f"(n={n}, gaps={len(used_gaps)})."
        )
        return None

    stats = fit_multivariate_omnibus(y=y, C=C, G=G)
    if not stats:
        return None

    stratify_binary = outcome_type == "diagnosis"
    ci_low, ci_high, n_boot_valid = bootstrap_partial_r_ci(
        y=y,
        C=C,
        G=G,
        n_boot=N_BOOT_MULTIVARIATE_R,
        seed=seed,
        stratify_binary=stratify_binary,
    )

    row = {
        "outcome": outcome_label,
        "outcome_column": outcome_col,
        "outcome_type": outcome_type,
        "n": int(n),
        "n_case_before_complete_case": n_case,
        "n_control_before_complete_case": n_control,
        "n_gaps_requested": len(gap_cols),
        "n_gaps_used": len(used_gaps),
        "gaps_used": " | ".join(used_gaps),
        "covariates_requested": " | ".join(covariates),
        "n_covariate_design_columns": int(C.shape[1]),
        "ci_level": CI_LEVEL,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_boot_requested": N_BOOT_MULTIVARIATE_R,
        "n_boot_valid": n_boot_valid,
        **stats,
    }

    return row


def compute_multivariate_R_table(
    df: pd.DataFrame,
    gap_cols: list[str],
) -> pd.DataFrame:
    rows = []
    covariates = available_covariates(df)
    seed_counter = 0

    # -------------------------------------------------------------------------
    # Clinical outcomes
    # -------------------------------------------------------------------------
    clinical_df = df.copy()

    for score_col in SCORE_PREDICTOR_COLS:
        if score_col not in clinical_df.columns:
            continue

        outcome_col = score_col
        if RANK_NORMALIZE_CLINICAL_MEASURES:
            transformed_col = f"__ranknorm__{score_col}"
            clinical_df[transformed_col] = ranknorm_series(clinical_df[score_col])
            outcome_col = transformed_col

        row = analyze_one_multivariate_outcome(
            df=clinical_df,
            gap_cols=gap_cols,
            outcome_col=outcome_col,
            outcome_label=SCORE_PREDICTOR_LABELS.get(score_col, score_col),
            outcome_type="clinical_score",
            covariates=covariates,
            seed=RANDOM_SEED + seed_counter,
        )
        seed_counter += 1

        if row is not None:
            row["original_outcome_column"] = score_col
            rows.append(row)

    # -------------------------------------------------------------------------
    # Diagnosis outcomes: same Figure-2 case-vs-HC contrast logic
    # -------------------------------------------------------------------------
    contrasts = [
        "AUD vs HC",
        "MDD vs HC",
        "ANX vs HC",
    ]

    for contrast in contrasts:
        sub, outcome_col, n_case, n_control = make_diagnosis_contrast(df, contrast)

        if n_case < MIN_DIAGNOSIS_CASES or n_control < MIN_DIAGNOSIS_CONTROLS:
            print(
                f"[WARN] Skipping {contrast}: too few subjects before complete-case "
                f"filtering (case={n_case}, control={n_control})."
            )
            continue

        row = analyze_one_multivariate_outcome(
            df=sub,
            gap_cols=gap_cols,
            outcome_col=outcome_col,
            outcome_label=contrast,
            outcome_type="diagnosis",
            covariates=covariates,
            seed=RANDOM_SEED + seed_counter,
            n_case=n_case,
            n_control=n_control,
        )
        seed_counter += 1

        if row is not None:
            row["original_outcome_column"] = contrast
            rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    # FDR over the requested family: all clinical + diagnosis omnibus tests.
    out["q_fdr_bh"] = np.nan
    ok = out["p_value"].notna()
    if ok.sum() > 0:
        out.loc[ok, "q_fdr_bh"] = multipletests(
            out.loc[ok, "p_value"].values,
            method="fdr_bh",
        )[1]

    out["significant_fdr_0p05"] = out["q_fdr_bh"].lt(FDR_ALPHA)

    # Organize columns for easy manuscript/table use.
    preferred = [
        "outcome",
        "original_outcome_column",
        "outcome_type",
        "n",
        "n_case_before_complete_case",
        "n_control_before_complete_case",
        "partial_R",
        "ci_low",
        "ci_high",
        "partial_R2",
        "incremental_R2_all_gaps",
        "full_R",
        "full_R2",
        "covariate_only_R2",
        "wald_F_hc3",
        "p_value",
        "q_fdr_bh",
        "significant_fdr_0p05",
        "classic_nested_F",
        "p_classic_F",
        "df_gap_block",
        "n_gaps_requested",
        "n_gaps_used",
        "gaps_used",
        "covariates_requested",
        "n_covariate_design_columns",
        "full_design_rank",
        "full_design_n_columns",
        "ci_level",
        "n_boot_requested",
        "n_boot_valid",
        "outcome_column",
    ]
    preferred = [c for c in preferred if c in out.columns]
    remaining = [c for c in out.columns if c not in preferred]
    out = out[preferred + remaining]

    return out


def format_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}".replace("0.", ".")


def plot_multivariate_R_forest(
    table: pd.DataFrame,
    out_png: str,
    out_pdf: str,
) -> None:
    if table.empty:
        print("[WARN] Multivariate-R table is empty; no forest plot created.")
        return

    # Clinical first, then diagnosis, preserving the intended analysis order.
    type_order = {"clinical_score": 0, "diagnosis": 1}
    plot_df = table.copy()
    plot_df["__type_order"] = plot_df["outcome_type"].map(type_order).fillna(99)
    plot_df["__row_order"] = np.arange(len(plot_df))
    plot_df = plot_df.sort_values(["__type_order", "__row_order"]).reset_index(drop=True)

    y = np.arange(len(plot_df))[::-1]

    fig, ax = plt.subplots(figsize=MULTIVARIATE_FIGSIZE)

    r = plot_df["partial_R"].astype(float).values
    lo = plot_df["ci_low"].astype(float).values
    hi = plot_df["ci_high"].astype(float).values

    xerr = np.vstack([
        np.maximum(r - lo, 0.0),
        np.maximum(hi - r, 0.0),
    ])
    xerr[~np.isfinite(xerr)] = 0.0

    ax.errorbar(
        r,
        y,
        xerr=xerr,
        fmt="o",
        markersize=7.5,
        capsize=4,
        linewidth=1.5,
        color="black",
        ecolor="black",
    )

    labels = []
    for _, row in plot_df.iterrows():
        q = row.get("q_fdr_bh", np.nan)
        suffix = " *" if np.isfinite(q) and q < FDR_ALPHA else ""
        labels.append(f"{row['outcome']}{suffix}")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Partial multiple R for all gaps (95% bootstrap CI)", fontsize=12.5)
    ax.axvline(0, linewidth=0.9, color="0.55")
    ax.grid(axis="x", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    finite_hi = hi[np.isfinite(hi)]
    xmax = max(0.10, float(np.max(finite_hi)) * 1.16) if len(finite_hi) else 1.0
    xmax = min(1.0, xmax)
    ax.set_xlim(0, xmax)

    # Add compact p/q text at the right side of each row.
    text_x = xmax * 0.98
    for yy, (_, row) in zip(y, plot_df.iterrows()):
        ptxt = format_p(float(row.get("p_value", np.nan)))
        qtxt = format_p(float(row.get("q_fdr_bh", np.nan)))
        ax.text(
            text_x,
            yy,
            f"p={ptxt}, q={qtxt}",
            ha="right",
            va="center",
            fontsize=9.3,
        )

    fig.tight_layout()
    fig.savefig(out_png, dpi=MULTIVARIATE_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Results directory:\n{RESULTS_DIR}")

    # Load exact Figure-2 gap space and metadata.
    gap_df, gap_cols = load_gap_space()


    analysis_df = add_diagnosis_clinical_and_covariates(gap_df)

    # -------------------------------------------------------------------------
    # 1) Pairwise MAIN-primary gap partial correlations
    # -------------------------------------------------------------------------
    print("\nComputing MAIN-primary pairwise gap partial correlations")
    print("=" * 80)

    corr_long, rmat, qmat = compute_pairwise_gap_correlations(
        df=analysis_df,
        gap_cols=gap_cols,
    )

    corr_long.to_csv(PAIRWISE_LONG_CSV, index=False)
    rmat.to_csv(PAIRWISE_R_MATRIX_CSV)
    qmat.to_csv(PAIRWISE_Q_MATRIX_CSV)

    plot_pairwise_gap_heatmap(
        rmat=rmat,
        qmat=qmat,
        out_png=PAIRWISE_HEATMAP_PNG,
        out_pdf=PAIRWISE_HEATMAP_PDF,
    )

    print(f"Saved:\n  {PAIRWISE_LONG_CSV}")
    print(f"  {PAIRWISE_R_MATRIX_CSV}")
    print(f"  {PAIRWISE_Q_MATRIX_CSV}")
    print(f"  {PAIRWISE_HEATMAP_PNG}")
    print(f"  {PAIRWISE_HEATMAP_PDF}")

    if not corr_long.empty:
        print("\nStrongest MAIN-primary gap partial correlations by |r|:")
        preview = corr_long.sort_values("abs_r", ascending=False).head(15)
        print(
            preview[
                [
                    "gap_1",
                    "gap_2",
                    "partial_r",
                    "ci_low",
                    "ci_high",
                    "p_value",
                    "q_fdr_bh",
                    "n",
                    "effective_covariate_df",
                    "partial_test_df",
                ]
            ].to_string(index=False)
        )

    # -------------------------------------------------------------------------
    # 2) All-gaps multivariate R to clinical + diagnosis measures
    # -------------------------------------------------------------------------
    print("\nComputing all-gaps multivariate R")
    print("=" * 80)
    print(
        "Primary p-value = HC3 robust omnibus Wald test; "
        "q = BH-FDR across all outcomes."
    )
    print(
        f"Bootstrap confidence intervals: {N_BOOT_MULTIVARIATE_R:,} resamples, "
        f"CI={CI_LEVEL:.0%}."
    )

    multi = compute_multivariate_R_table(
        df=analysis_df,
        gap_cols=gap_cols,
    )

    multi.to_csv(MULTIVARIATE_R_CSV, index=False)
    plot_multivariate_R_forest(
        table=multi,
        out_png=MULTIVARIATE_R_PNG,
        out_pdf=MULTIVARIATE_R_PDF,
    )

    print(f"Saved:\n  {MULTIVARIATE_R_CSV}")
    print(f"  {MULTIVARIATE_R_PNG}")
    print(f"  {MULTIVARIATE_R_PDF}")

    if not multi.empty:
        print("\nMultivariate-R results:")
        show_cols = [
            "outcome",
            "outcome_type",
            "n",
            "partial_R",
            "ci_low",
            "ci_high",
            "p_value",
            "q_fdr_bh",
            "n_gaps_used",
        ]
        print(multi[show_cols].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()

