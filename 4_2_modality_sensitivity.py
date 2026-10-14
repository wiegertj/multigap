#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for running the modality bias
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# =============================================================================
# CONSTANTS
# =============================================================================

EXPECTED_MODALITIES: Tuple[str, str] = ("freesurfer", "connectivity")
MIN_MODEL_N_DEFAULT = 30
FRAMEWISE_ID_COL = "eid"
FRAMEWISE_VALUE_COL = "framewise_dis"
FRAMEWISE_THRESHOLD = 0.5
CSV_FRAMEWISE_DIS = ...

GLOBAL_TARGET_EXCLUSIONS: set[str] = set()

FEATURE_TARGET_EXCLUSIONS: Dict[str, set[str]] = {
    "freesurfer": {
        "Sleep_duration",
    },
    "connectivity": {
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
    )
    parser.add_argument(
        "--base-script",
        type=Path,
        default=None
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None
    )
    parser.add_argument(
        "--allow-partial-summaries",
        action="store_true",
        help=(
        ),
    )
    parser.add_argument(
        "--min-remaining-axes",
        type=int,
        default=2,
        help=(
        ),
    )
    parser.add_argument(
        "--fdr-alpha",
        type=float,
        default=None)
    parser.add_argument(
        "--no-plots",
        action="store_true")
    parser.add_argument(
        "--diagnosis-only",
        action="store_true")
    parser.add_argument(
        "--scores-only",
        action="store_true")
    return parser.parse_args()


def resolve_base_script(explicit: Optional[Path]) -> Path:
    return Path(...)


def import_base_module(path: Path):
    module_name = "gap_primary_analysis_for_sensitivity"
    spec = importlib.util.spec_from_file_location(module_name, str(path))

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def apply_connectivity_framewise_exclusion(
        dfi: pd.DataFrame,
        base: Any,
        connectivity_gap_cols: Sequence[str],
) -> pd.DataFrame:
    if not connectivity_gap_cols:
        return dfi

    framewise_path = Path(CSV_FRAMEWISE_DIS)

    fw = pd.read_csv(
        framewise_path,
        usecols=[FRAMEWISE_ID_COL, FRAMEWISE_VALUE_COL],
    )
    fw = fw.rename(columns={FRAMEWISE_ID_COL: base.ID_COL})
    fw[base.ID_COL] = base.canonicalize_id_series(fw[base.ID_COL])
    fw[FRAMEWISE_VALUE_COL] = pd.to_numeric(
        fw[FRAMEWISE_VALUE_COL],
        errors="coerce",
    )
    fw = fw.drop_duplicates(subset=[base.ID_COL], keep="first")

    out = dfi.merge(
        fw[[base.ID_COL, FRAMEWISE_VALUE_COL]],
        on=base.ID_COL,
        how="left",
        validate="one_to_one",
    )

    high_motion = out[FRAMEWISE_VALUE_COL].gt(FRAMEWISE_THRESHOLD)

    out.loc[high_motion, list(connectivity_gap_cols)] = np.nan

    return out.drop(columns=[FRAMEWISE_VALUE_COL])


def load_gap_analysis_dataframe(base: Any) -> Tuple[pd.DataFrame, List[str]]:
    frames: List[pd.DataFrame] = []
    specs_all: List[Tuple[str, str, str]] = []

    for feature_set, raw_path in base.INPUT_FILES.items():
        path = Path(raw_path)

        dfi = pd.read_csv(path)

        dfi[base.ID_COL] = base.canonicalize_id_series(dfi[base.ID_COL])
        specs = base.get_gap_specs(dfi, feature_set)

        feature_key = str(feature_set).strip().lower()
        excluded_targets = (
                set(GLOBAL_TARGET_EXCLUSIONS)
                | set(FEATURE_TARGET_EXCLUSIONS.get(feature_key, set()))
        )
        specs = [spec for spec in specs if spec[1] not in excluded_targets]

        keep = [base.ID_COL]
        if base.SPLIT_GROUP_COL in dfi.columns:
            keep.append(base.SPLIT_GROUP_COL)
        keep.extend(spec[2] for spec in specs)
        keep = list(dict.fromkeys(keep))
        dfi = dfi[keep].copy()

        rename: Dict[str, str] = {}
        for fs_name, target, old_col in specs:
            new_col = f"{fs_name}__{target}"
            rename[old_col] = new_col
            specs_all.append((fs_name, target, new_col))

        dfi = dfi.rename(columns=rename)

        if feature_key == "connectivity":
            dfi = apply_connectivity_framewise_exclusion(
                dfi=dfi,
                base=base,
                connectivity_gap_cols=list(rename.values()),
            )

        merge_key = [base.ID_COL]
        if base.SPLIT_GROUP_COL in dfi.columns:
            merge_key.append(base.SPLIT_GROUP_COL)

        frames.append(dfi)

    gap_df = frames[0].copy()
    for frame in frames[1:]:
        merge_cols = [base.ID_COL]
        if (
                base.SPLIT_GROUP_COL in gap_df.columns
                and base.SPLIT_GROUP_COL in frame.columns
        ):
            merge_cols.append(base.SPLIT_GROUP_COL)

        gap_df = gap_df.merge(
            frame,
            on=merge_cols,
            how="inner",
            validate="one_to_one",
        )

    gap_cols = [spec[2] for spec in specs_all]
    gap_cols = list(dict.fromkeys(gap_cols))
    gap_df = base.safe_numeric(gap_df, gap_cols)
    gap_df = base.drop_train_hc(gap_df, label="sensitivity merged gap_df")

    # Attach diagnosis indicators, covariates, and group labels to gap participants.
    gap_df = base.add_diagnosis_columns(gap_df)
    gap_df = base.add_mean_motion_covariate(gap_df)
    gap_df = base.apply_plotting_diagnosis_labels(gap_df)

    if hasattr(base, "add_centered_age_squared_covariate"):
        gap_df = base.add_centered_age_squared_covariate(gap_df)

    print("\nSensitivity-analysis dataframe")
    print("=" * 80)
    print(f"Subjects: {len(gap_df):,}")
    print(f"Gap outcomes: {len(gap_cols):,}")
    print(f"Available primary covariates: {base.available_covariates(gap_df)}")

    return gap_df, gap_cols


# =============================================================================
# LEAVE-ONE-OUT MODALITY SUMMARIES
# =============================================================================


def split_gap_columns_by_modality(
        gap_cols: Sequence[str],
) -> Dict[str, List[str]]:
    modality_cols: Dict[str, List[str]] = {m: [] for m in EXPECTED_MODALITIES}

    for col in gap_cols:
        if "__" not in col:
            continue
        modality = col.split("__", 1)[0].strip().lower()
        if modality in modality_cols:
            modality_cols[modality].append(col)

    return modality_cols


def summary_column_name(modality: str, kind: str) -> str:
    return f"loo_{modality}_{kind}"


def compute_row_summary(
        df: pd.DataFrame,
        cols: Sequence[str],
        absolute: bool,
        require_complete: bool,
        min_remaining_axes: int,
) -> Tuple[pd.Series, pd.Series]:
    if not cols:
        empty = pd.Series(np.nan, index=df.index, dtype=float)
        counts = pd.Series(0, index=df.index, dtype=int)
        return empty, counts

    matrix = df[list(cols)].apply(pd.to_numeric, errors="coerce").astype(float)
    observed_counts = matrix.notna().sum(axis=1).astype(int)

    values = matrix.abs() if absolute else matrix
    summary = values.mean(axis=1, skipna=True).astype(float)

    if require_complete:
        valid = observed_counts.eq(len(cols))
    else:
        required = min(max(1, min_remaining_axes), len(cols))
        valid = observed_counts.ge(required)

    summary = summary.where(valid, np.nan)
    return summary, observed_counts


def add_loo_modality_summaries(
        df: pd.DataFrame,
        outcome_col: str,
        modality_cols: Mapping[str, Sequence[str]],
        require_complete: bool,
        min_remaining_axes: int,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    out = df.copy()
    summary_cols: List[str] = []
    diagnostics: Dict[str, Any] = {
        "gap": outcome_col,
        "n_total": int(len(df)),
        "outcome_modality": outcome_col.split("__", 1)[0] if "__" in outcome_col else "",
        "summary_mode": "complete" if require_complete else "partial",
    }

    for modality in EXPECTED_MODALITIES:
        original_cols = list(modality_cols[modality])
        remaining_cols = [c for c in original_cols if c != outcome_col]
        excluded = outcome_col if outcome_col in original_cols else None

        diagnostics[f"{modality}_n_axes_total"] = int(len(original_cols))
        diagnostics[f"{modality}_n_axes_in_summary"] = int(len(remaining_cols))
        diagnostics[f"{modality}_excluded_outcome"] = excluded or ""

        signed_name = summary_column_name(modality, "signed_mean")
        absolute_name = summary_column_name(modality, "mean_absolute")

        signed, observed_counts = compute_row_summary(
            df=out,
            cols=remaining_cols,
            absolute=False,
            require_complete=require_complete,
            min_remaining_axes=min_remaining_axes,
        )
        absolute, _ = compute_row_summary(
            df=out,
            cols=remaining_cols,
            absolute=True,
            require_complete=require_complete,
            min_remaining_axes=min_remaining_axes,
        )

        out[signed_name] = signed
        out[absolute_name] = absolute
        summary_cols.extend([signed_name, absolute_name])

        diagnostics[f"{modality}_observed_axes_min"] = int(observed_counts.min())
        diagnostics[f"{modality}_observed_axes_median"] = float(observed_counts.median())
        diagnostics[f"{modality}_observed_axes_max"] = int(observed_counts.max())
        diagnostics[f"{signed_name}_n_nonmissing"] = int(signed.notna().sum())
        diagnostics[f"{absolute_name}_n_nonmissing"] = int(absolute.notna().sum())

    all_summaries_complete = out[summary_cols].notna().all(axis=1)
    diagnostics["all_four_summaries_n_nonmissing"] = int(all_summaries_complete.sum())

    return out, summary_cols, diagnostics


def safe_sign_label(value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "missing"
    if not np.isfinite(x):
        return "missing"
    if x > 0:
        return "positive"
    if x < 0:
        return "negative"
    return "zero"


def fit_same_sample_model_pair(
        base: Any,
        df: pd.DataFrame,
        outcome_col: str,
        predictor_col: str,
        covariates: Sequence[str],
        summary_cols: Sequence[str],
) -> Optional[Dict[str, Any]]:
    required = [outcome_col, predictor_col, *summary_cols]
    required = list(dict.fromkeys(required))
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        return None

    paired_df = df.dropna(subset=required).copy()
    if len(paired_df) < MIN_MODEL_N_DEFAULT:
        return None

    before = base.fit_ols_hc3_predictor(
        df=paired_df,
        outcome_col=outcome_col,
        predictor_col=predictor_col,
        covariates=list(covariates),
        extra_predictors=None,
    )
    after = base.fit_ols_hc3_predictor(
        df=paired_df,
        outcome_col=outcome_col,
        predictor_col=predictor_col,
        covariates=list(covariates),
        extra_predictors=list(summary_cols),
    )

    if before is None or after is None:
        return None

    n_before = int(before.get("n", -1))
    n_after = int(after.get("n", -1))

    row: Dict[str, Any] = {
        "gap": outcome_col,
        "predictor_column": predictor_col,
        "n_paired": n_before,
    }
    for key, value in before.items():
        row[f"before_{key}"] = value
    for key, value in after.items():
        row[f"after_{key}"] = value
    return row


def diagnosis_contrasts() -> List[str]:
    return [
        "AUD vs HC",
        "MDD vs HC",
        "ANX vs HC",
    ]


def add_main_binary_diagnosis_flags(base: Any, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    required = [base.HC_COL, base.AUD_COL, base.MDD_COL, base.ANX_COL]
    missing = [c for c in required if c not in out.columns]

    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    is_hc = out[base.HC_COL].eq(1)
    if base.SPLIT_GROUP_COL in out.columns:
        heldout_value = str(getattr(base, "HELDOUT_HC_SPLIT_VALUE", "heldout_hc"))
        split = out[base.SPLIT_GROUP_COL].astype(str).str.strip()
        is_hc = is_hc & split.eq(heldout_value)

    out["is_hc"] = is_hc
    out["is_aud"] = out[base.AUD_COL].eq(1)
    out["is_mdd"] = out[base.MDD_COL].eq(1)
    out["is_anx"] = out[base.ANX_COL].eq(1)

    return out


def make_main_contrast_subset(df: pd.DataFrame, contrast_name: str) -> pd.DataFrame:
    if contrast_name == "AUD vs HC":
        case_mask = df["is_aud"]
    elif contrast_name == "MDD vs HC":
        case_mask = df["is_mdd"]
    elif contrast_name == "ANX vs HC":
        case_mask = df["is_anx"]
    else:
        raise ValueError(f"Unknown MAIN contrast: {contrast_name}")

    ref_mask = df["is_hc"]
    keep = case_mask | ref_mask
    work = df.loc[keep].copy()
    work["diagnosis_indicator"] = case_mask.loc[work.index].astype(int)

    n_case = int(work["diagnosis_indicator"].eq(1).sum())
    n_ref = int(work["diagnosis_indicator"].eq(0).sum())
    if n_case < 10 or n_ref < 10:
        print(
            f"[WARN] Too few subjects for {contrast_name}: "
            f"case={n_case}, ref={n_ref}"
        )

    return work


def run_diagnosis_models(
        base: Any,
        gap_df: pd.DataFrame,
        gap_cols: Sequence[str],
        modality_cols: Mapping[str, Sequence[str]],
        require_complete: bool,
        min_remaining_axes: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = add_main_binary_diagnosis_flags(base, gap_df)
    covariates = base.available_covariates(df)

    rows: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []

    for outcome_i, gap in enumerate(gap_cols, start=1):
        print(f"[diagnosis] outcome {outcome_i:02d}/{len(gap_cols):02d}: {gap}")
        with_summaries, summary_cols, diag = add_loo_modality_summaries(
            df=df,
            outcome_col=gap,
            modality_cols=modality_cols,
            require_complete=require_complete,
            min_remaining_axes=min_remaining_axes,
        )
        diag["source"] = "diagnosis"
        diagnostics.append(diag)

        for contrast in diagnosis_contrasts():
            sub = make_main_contrast_subset(with_summaries, contrast)
            fit = fit_same_sample_model_pair(
                base=base,
                df=sub,
                outcome_col=gap,
                predictor_col="diagnosis_indicator",
                covariates=covariates,
                summary_cols=summary_cols,
            )
            if fit is None:
                continue

            complete_mask = sub[
                [gap, "diagnosis_indicator", *summary_cols]
            ].notna().all(axis=1)
            complete_sub = sub.loc[complete_mask]

            fit.update(
                {
                    "source": "diagnosis",
                    "predictor": contrast,
                    "score_predictor": "",
                    "n_case_paired": int(
                        complete_sub["diagnosis_indicator"].eq(1).sum()
                    ),
                    "n_reference_paired": int(
                        complete_sub["diagnosis_indicator"].eq(0).sum()
                    ),
                    "summary_columns": ";".join(summary_cols),
                }
            )
            rows.append(fit)

    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def prepare_rank_normalized_score_dataframe(base: Any, gap_df: pd.DataFrame) -> pd.DataFrame:
    df = gap_df.copy()
    score_cols = [c for c in base.SCORE_PREDICTOR_COLS if c in df.columns]
    if bool(getattr(base, "RANK_NORMALIZE_SCORE_PREDICTORS", False)):
        for col in score_cols:
            df[col] = base.ranknorm_series(df[col])
    return df


def run_score_models(
        base: Any,
        gap_df: pd.DataFrame,
        gap_cols: Sequence[str],
        modality_cols: Mapping[str, Sequence[str]],
        require_complete: bool,
        min_remaining_axes: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = prepare_rank_normalized_score_dataframe(base, gap_df)
    covariates = base.available_covariates(df)
    score_cols = [c for c in base.SCORE_PREDICTOR_COLS if c in df.columns]

    rows: List[Dict[str, Any]] = []
    diagnostics: List[Dict[str, Any]] = []

    for outcome_i, gap in enumerate(gap_cols, start=1):
        print(f"[scores]    outcome {outcome_i:02d}/{len(gap_cols):02d}: {gap}")
        with_summaries, summary_cols, diag = add_loo_modality_summaries(
            df=df,
            outcome_col=gap,
            modality_cols=modality_cols,
            require_complete=require_complete,
            min_remaining_axes=min_remaining_axes,
        )
        diag["source"] = "clinical_score"
        diagnostics.append(diag)

        for score_col in score_cols:
            fit = fit_same_sample_model_pair(
                base=base,
                df=with_summaries,
                outcome_col=gap,
                predictor_col=score_col,
                covariates=covariates,
                summary_cols=summary_cols,
            )
            if fit is None:
                continue

            fit.update(
                {
                    "source": "clinical_score",
                    "predictor": base.SCORE_PREDICTOR_LABELS.get(score_col, score_col),
                    "score_predictor": score_col,
                    "n_case_paired": np.nan,
                    "n_reference_paired": np.nan,
                    "summary_columns": ";".join(summary_cols),
                }
            )
            rows.append(fit)

    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


# =============================================================================
# FDR, COMPARISON METRICS, AND SUMMARIES
# =============================================================================


def fdr_family_key(row: pd.Series) -> str:
    if row["source"] == "diagnosis":
        return f"diagnosis::{row['predictor']}"
    score = row.get("score_predictor", "") or row["predictor"]
    return f"clinical_score::{score}"


def add_fdr_columns(results: pd.DataFrame, alpha: float) -> pd.DataFrame:
    if results.empty:
        return results.copy()

    out = results.copy()
    out["fdr_family"] = out.apply(fdr_family_key, axis=1)
    out["before_q_fdr_bh"] = np.nan
    out["after_q_fdr_bh"] = np.nan

    for family, idx in out.groupby("fdr_family").groups.items():
        indices = list(idx)
        for phase in ("before", "after"):
            p_col = f"{phase}_p_hc3"
            q_col = f"{phase}_q_fdr_bh"
            p = pd.to_numeric(out.loc[indices, p_col], errors="coerce")
            valid = p.notna() & np.isfinite(p)
            if int(valid.sum()) == 0:
                continue
            valid_indices = p.index[valid]
            out.loc[valid_indices, q_col] = multipletests(
                p.loc[valid], method="fdr_bh"
            )[1]

    out["before_fdr_significant"] = out["before_q_fdr_bh"].lt(alpha)
    out["after_fdr_significant"] = out["after_q_fdr_bh"].lt(alpha)
    return out


def fdr_transition(before: bool, after: bool) -> str:
    if before and after:
        return "retained"
    if before and not after:
        return "lost"
    if not before and after:
        return "gained"
    return "not_significant"


def add_comparison_metrics(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results.copy()

    out = results.copy()
    before = pd.to_numeric(out["before_beta_std"], errors="coerce")
    after = pd.to_numeric(out["after_beta_std"], errors="coerce")

    out["before_sign"] = before.map(safe_sign_label)
    out["after_sign"] = after.map(safe_sign_label)
    out["sign_preserved"] = (
            out["before_sign"].eq(out["after_sign"])
            & ~out["before_sign"].isin(["missing", "zero"])
    )
    out["sign_flip"] = before.mul(after).lt(0)

    out["before_abs_beta_std"] = before.abs()
    out["after_abs_beta_std"] = after.abs()
    out["delta_beta_std"] = after - before
    out["delta_abs_beta_std"] = after.abs() - before.abs()

    denominator = before.abs().replace(0.0, np.nan)
    out["absolute_beta_ratio_after_vs_before"] = after.abs() / denominator
    out["attenuation_percent"] = 100.0 * (
            1.0 - out["absolute_beta_ratio_after_vs_before"]
    )

    out["fdr_transition"] = [
        fdr_transition(bool(b), bool(a))
        for b, a in zip(
            out["before_fdr_significant"], out["after_fdr_significant"]
        )
    ]

    return out


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    grouping = ["source", "predictor"]
    rows: List[Dict[str, Any]] = []

    for keys, sub in results.groupby(grouping, dropna=False):
        source, predictor = keys
        transitions = sub["fdr_transition"].value_counts()
        rows.append(
            {
                "source": source,
                "predictor": predictor,
                "n_models": int(len(sub)),
                "median_n_paired": float(pd.to_numeric(sub["n_paired"]).median()),
                "median_abs_beta_before": float(
                    pd.to_numeric(sub["before_abs_beta_std"]).median()
                ),
                "median_abs_beta_after": float(
                    pd.to_numeric(sub["after_abs_beta_std"]).median()
                ),
                "median_attenuation_percent": float(
                    pd.to_numeric(sub["attenuation_percent"]).median()
                ),
                "n_sign_flips": int(sub["sign_flip"].fillna(False).sum()),
                "n_fdr_before": int(sub["before_fdr_significant"].sum()),
                "n_fdr_after": int(sub["after_fdr_significant"].sum()),
                "n_fdr_retained": int(transitions.get("retained", 0)),
                "n_fdr_lost": int(transitions.get("lost", 0)),
                "n_fdr_gained": int(transitions.get("gained", 0)),
                "n_not_significant": int(transitions.get("not_significant", 0)),
            }
        )

    return pd.DataFrame(rows).sort_values(grouping).reset_index(drop=True)


def summarize_by_source(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    for source, sub in results.groupby("source"):
        transitions = sub["fdr_transition"].value_counts()
        rows.append(
            {
                "source": source,
                "n_models": int(len(sub)),
                "n_sign_flips": int(sub["sign_flip"].fillna(False).sum()),
                "median_abs_beta_before": float(sub["before_abs_beta_std"].median()),
                "median_abs_beta_after": float(sub["after_abs_beta_std"].median()),
                "median_attenuation_percent": float(sub["attenuation_percent"].median()),
                "n_fdr_before": int(sub["before_fdr_significant"].sum()),
                "n_fdr_after": int(sub["after_fdr_significant"].sum()),
                "n_fdr_retained": int(transitions.get("retained", 0)),
                "n_fdr_lost": int(transitions.get("lost", 0)),
                "n_fdr_gained": int(transitions.get("gained", 0)),
                "n_not_significant": int(transitions.get("not_significant", 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("source").reset_index(drop=True)


# =============================================================================
# PLOTS
# =============================================================================


def finite_limits(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    values = pd.concat([x, y], ignore_index=True)
    values = pd.to_numeric(values, errors="coerce")
    values = values[np.isfinite(values)]
    if values.empty:
        return -1.0, 1.0
    lim = float(max(abs(values.min()), abs(values.max())))
    lim = max(lim * 1.08, 0.05)
    return -lim, lim


def plot_beta_before_after(results: pd.DataFrame, output_path: Path) -> None:
    if results.empty:
        return

    sources = [s for s in ["diagnosis", "clinical_score"] if s in set(results["source"])]
    if not sources:
        return

    fig, axes = plt.subplots(1, len(sources), figsize=(7 * len(sources), 6.5), squeeze=False)
    axes_flat = axes.ravel()

    for ax, source in zip(axes_flat, sources):
        sub = results.loc[results["source"].eq(source)].copy()
        x = pd.to_numeric(sub["before_beta_std"], errors="coerce")
        y = pd.to_numeric(sub["after_beta_std"], errors="coerce")
        valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        sub = sub.loc[valid]
        x = x.loc[valid]
        y = y.loc[valid]

        marker_sizes = np.where(
            sub["before_fdr_significant"] | sub["after_fdr_significant"],
            52.0,
            24.0,
        )
        ax.scatter(x, y, s=marker_sizes, alpha=0.65, edgecolors="none")
        lo, hi = finite_limits(x, y)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)
        ax.axhline(0.0, linewidth=0.7)
        ax.axvline(0.0, linewidth=0.7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Standardized beta before modality adjustment")
        ax.set_ylabel("Standardized beta after modality adjustment")
        ax.set_title("Diagnosis contrasts" if source == "diagnosis" else "Clinical scores")

        n_flip = int(sub["sign_flip"].sum())
        n_before = int(sub["before_fdr_significant"].sum())
        n_after = int(sub["after_fdr_significant"].sum())
        ax.text(
            0.03,
            0.97,
            f"n={len(sub)}\nSign flips={n_flip}\nFDR sig: {n_before} → {n_after}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
        )

    fig.suptitle("Leave-one-out modality sensitivity: effect comparison", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fdr_transitions(results: pd.DataFrame, output_path: Path) -> None:
    if results.empty:
        return

    order = ["retained", "lost", "gained", "not_significant"]
    table = (
        results.groupby(["source", "fdr_transition"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=order, fill_value=0)
    )
    if table.empty:
        return

    ax = table.plot(kind="bar", stacked=True, figsize=(9, 6))
    ax.set_xlabel("")
    ax.set_ylabel("Number of tested associations")
    ax.set_title("FDR-significance transitions after modality adjustment")
    ax.legend(title="Transition", frameon=False)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# OUTPUT
# =============================================================================


def ordered_result_columns(results: pd.DataFrame) -> List[str]:
    preferred = [
        "source",
        "predictor",
        "score_predictor",
        "predictor_column",
        "gap",
        "n_paired",
        "n_case_paired",
        "n_reference_paired",
        "before_beta_std",
        "after_beta_std",
        "before_abs_beta_std",
        "after_abs_beta_std",
        "delta_beta_std",
        "delta_abs_beta_std",
        "absolute_beta_ratio_after_vs_before",
        "attenuation_percent",
        "before_sign",
        "after_sign",
        "sign_preserved",
        "sign_flip",
        "before_se_hc3",
        "after_se_hc3",
        "before_t_hc3",
        "after_t_hc3",
        "before_p_hc3",
        "after_p_hc3",
        "before_q_fdr_bh",
        "after_q_fdr_bh",
        "before_fdr_significant",
        "after_fdr_significant",
        "fdr_transition",
        "fdr_family",
        "summary_columns",
    ]
    return [c for c in preferred if c in results.columns] + [
        c for c in results.columns if c not in preferred
    ]


def write_outputs(
        output_dir: Path,
        results: pd.DataFrame,
        coverage: pd.DataFrame,
        summary_by_predictor: pd.DataFrame,
        summary_by_source: pd.DataFrame,
        metadata: Dict[str, Any],
        make_plots: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    results = results[ordered_result_columns(results)] if not results.empty else results

    all_path = output_dir / ...
    diagnosis_path = output_dir / ...
    score_path = output_dir / ...
    sig_path = output_dir / ...
    coverage_path = output_dir / ...
    predictor_summary_path = output_dir / ...
    source_summary_path = output_dir / ...
    metadata_path = output_dir / ...

    results.to_csv(all_path, index=False)
    results.loc[results["source"].eq("diagnosis")].to_csv(diagnosis_path, index=False)
    results.loc[results["source"].eq("clinical_score")].to_csv(score_path, index=False)

    fdr_relevant = results.loc[
        results["before_fdr_significant"] | results["after_fdr_significant"]
        ].copy()
    fdr_relevant.to_csv(sig_path, index=False)

    coverage.to_csv(coverage_path, index=False)
    summary_by_predictor.to_csv(predictor_summary_path, index=False)
    summary_by_source.to_csv(source_summary_path, index=False)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True, default=str)

    if make_plots:
        plot_beta_before_after(
            results,
            output_dir / "sensitivity_beta_before_vs_after.png",
        )
        plot_fdr_transitions(
            results,
            output_dir / "sensitivity_fdr_transitions.png",
        )

    print("\nSaved sensitivity outputs")
    print("=" * 80)
    for path in [
        all_path,
        diagnosis_path,
        score_path,
        sig_path,
        coverage_path,
        predictor_summary_path,
        source_summary_path,
        metadata_path,
    ]:
        print(path)


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    args = parse_args()

    base_script = resolve_base_script(args.base_script)
    base = import_base_module(base_script)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else Path(base.OUT_DIR).expanduser().resolve()
             / "sensitivity_leave_one_out_modality_MAIN_groups"
    )
    fdr_alpha = (
        float(args.fdr_alpha)
        if args.fdr_alpha is not None
        else float(getattr(base, "FDR_ALPHA", 0.05))
    )

    require_complete = not args.allow_partial_summaries

    print("Leave-one-out modality sensitivity analysis")
    print("=" * 80)
    print(f"Base analysis script: {base_script}")
    print(f"Output directory:    {output_dir}")
    print(f"Summary mode:        {'complete remaining axes' if require_complete else 'partial remaining axes'}")
    if not require_complete:
        print(f"Minimum axes:        {args.min_remaining_axes}")
    print(f"FDR alpha:           {fdr_alpha}")
    print("Diagnosis groups:    MAIN_AUD / MAIN_MDD / MAIN_ANX vs held-out MAIN_HC")
    print("Normative refit:     no (uses current MAIN gap files by design)")

    gap_df, gap_cols = load_gap_analysis_dataframe(base)
    # Explicit MAIN-group audit before fitting any sensitivity model.
    audited = add_main_binary_diagnosis_flags(base, gap_df)
    print("\nMAIN group counts in sensitivity analysis set")
    print("=" * 80)
    print(f"heldout MAIN_HC: {int(audited['is_hc'].sum()):,}")
    print(f"MAIN_AUD:        {int(audited['is_aud'].sum()):,}")
    print(f"MAIN_MDD:        {int(audited['is_mdd'].sum()):,}")
    print(f"MAIN_ANX:        {int(audited['is_anx'].sum()):,}")
    print(f"MDD+ANX overlap: {int((audited['is_mdd'] & audited['is_anx']).sum()):,}")

    modality_cols = split_gap_columns_by_modality(gap_cols)
    tested_gap_cols = (
        list(base.graph_gap_cols(gap_cols))
        if hasattr(base, "graph_gap_cols")
        else list(gap_cols)
    )

    print("\nGap axes by modality")
    print("=" * 80)
    for modality, cols in modality_cols.items():
        print(f"{modality:15s}: {len(cols):2d}")
        for col in cols:
            print(f"  - {col}")

    result_frames: List[pd.DataFrame] = []
    coverage_frames: List[pd.DataFrame] = []

    if not args.scores_only:
        diagnosis_results, diagnosis_coverage = run_diagnosis_models(
            base=base,
            gap_df=gap_df,
            gap_cols=tested_gap_cols,
            modality_cols=modality_cols,
            require_complete=require_complete,
            min_remaining_axes=args.min_remaining_axes,
        )
        result_frames.append(diagnosis_results)
        coverage_frames.append(diagnosis_coverage)

    if not args.diagnosis_only:
        score_results, score_coverage = run_score_models(
            base=base,
            gap_df=gap_df,
            gap_cols=tested_gap_cols,
            modality_cols=modality_cols,
            require_complete=require_complete,
            min_remaining_axes=args.min_remaining_axes,
        )
        result_frames.append(score_results)
        coverage_frames.append(score_coverage)

    nonempty_results = [df for df in result_frames if not df.empty]
    if not nonempty_results:
        raise RuntimeError("No sensitivity models were successfully fit.")

    results = pd.concat(nonempty_results, ignore_index=True, sort=False)
    coverage = pd.concat(
        [df for df in coverage_frames if not df.empty],
        ignore_index=True,
        sort=False,
    )

    results = add_fdr_columns(results, alpha=fdr_alpha)
    results = add_comparison_metrics(results)
    results = results.sort_values(
        ["source", "predictor", "gap"], kind="stable"
    ).reset_index(drop=True)

    summary_by_predictor = summarize_results(results)
    summary_by_source = summarize_by_source(results)

    metadata: Dict[str, Any] = {
        "base_script": str(base_script),
        "base_output_dir": str(getattr(base, "OUT_DIR", "")),
        "sensitivity_output_dir": str(output_dir),
        "n_subjects_loaded": int(len(gap_df)),
        "n_gap_axes_for_summaries": int(len(gap_cols)),
        "n_gap_outcomes_tested": int(len(tested_gap_cols)),
        "gap_axes_for_summaries": list(gap_cols),
        "gap_outcomes_tested": list(tested_gap_cols),
        "modality_axis_counts": {k: len(v) for k, v in modality_cols.items()},
        "modality_axes": {k: list(v) for k, v in modality_cols.items()},
        "global_target_exclusions": sorted(GLOBAL_TARGET_EXCLUSIONS),
        "feature_target_exclusions": {
            k: sorted(v) for k, v in FEATURE_TARGET_EXCLUSIONS.items()
        },
        "summary_mode": "complete" if require_complete else "partial",
        "minimum_remaining_axes_if_partial": int(args.min_remaining_axes),
        "fdr_alpha": float(fdr_alpha),
        "fdr_method": "Benjamini-Hochberg",
        "fdr_family_definition": (
            "separate family for each diagnosis contrast or clinical-score predictor, "
            "across all gap outcomes; corrected separately before and after adjustment"
        ),
        "paired_sample_comparison": True,
        "group_definition": "MAIN_HC / MAIN_AUD / MAIN_MDD / MAIN_ANX",
        "heldout_hc_reference_requires_MAIN_HC": True,
        "mdd_anx_overlap_allowed": True,
        "normative_models_refit_for_this_sensitivity": False,
        "normative_gap_source": "current MAIN normative-score files from base.INPUT_FILES",
        "sensitivity_perturbation": (
            "downstream adjustment for signed-mean and mean-absolute leave-one-out "
            "gap summaries within FreeSurfer and connectivity"
        ),
        "original_covariates": list(base.available_covariates(gap_df)),
        "rank_normalize_score_predictors": bool(
            getattr(base, "RANK_NORMALIZE_SCORE_PREDICTORS", False)
        ),
        "model": "OLS with HC3 robust covariance via base.fit_ols_hc3_predictor",
        "n_models": int(len(results)),
        "n_sign_flips": int(results["sign_flip"].sum()),
        "n_fdr_before": int(results["before_fdr_significant"].sum()),
        "n_fdr_after": int(results["after_fdr_significant"].sum()),
    }

    write_outputs(
        output_dir=output_dir,
        results=results,
        coverage=coverage,
        summary_by_predictor=summary_by_predictor,
        summary_by_source=summary_by_source,
        metadata=metadata,
        make_plots=not args.no_plots,
    )

    print("\nOverall comparison")
    print("=" * 80)
    print(summary_by_source.to_string(index=False))


if __name__ == "__main__":
    main()
