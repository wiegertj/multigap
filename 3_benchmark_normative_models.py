#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute pooled out-of-fold performance for the normative
models and generate coefficient-based feature-importance reports.
"""

import os
import sys
import re
import hashlib
import importlib.util
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages

from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


# =============================================================================
# LOAD BASE GAP PIPELINE
# =============================================================================

BASE_SCRIPT_PATH = ... # points to 1_compute_normative_gaps.py


_base_spec = importlib.util.spec_from_file_location(
    "gap_calculation_base",
    BASE_SCRIPT_PATH,
)

base = importlib.util.module_from_spec(_base_spec)
sys.modules[_base_spec.name] = base
_base_spec.loader.exec_module(base)

# =============================================================================
# CONFIG
# =============================================================================

OUT_DIR = base.OUT_DIR

FIG_DIR = os.path.join(OUT_DIR, ...)
os.makedirs(FIG_DIR, exist_ok=True)

OOF_PRED_OUT = os.path.join(
    FIG_DIR,
    ...
)

SUMMARY_OUT = os.path.join(
    FIG_DIR,
    ...
)

SUPPLEMENTARY_TABLE_TEX = os.path.join(
    FIG_DIR,
    ...
)

OUT_PNG = os.path.join(
    FIG_DIR,
    ...
)

OUT_PDF = os.path.join(
    FIG_DIR,
    ...
)

# Standalone no-title manuscript panels.
OUT_PANEL_A_PNG = os.path.join(
    FIG_DIR,
    ...
)

OUT_PANEL_B_PNG = os.path.join(
    FIG_DIR,
    ...
)

FEATURE_IMPORTANCE_DIR = os.path.join(FIG_DIR, ...)
os.makedirs(FEATURE_IMPORTANCE_DIR, exist_ok=True)

FEATURE_IMPORTANCE_FOLD_OUT = os.path.join(
    FEATURE_IMPORTANCE_DIR,
    ...
)

FEATURE_IMPORTANCE_SUMMARY_OUT = os.path.join(
    FEATURE_IMPORTANCE_DIR,
    ...
)

FEATURE_IMPORTANCE_TOP_OUT = os.path.join(
    FEATURE_IMPORTANCE_DIR,
    ...
)

FEATURE_IMPORTANCE_REPORT_PDF = os.path.join(
    FEATURE_IMPORTANCE_DIR,
    ...
)

FEATURE_IMPORTANCE_TOP_N = 20

FEATURE_ORDER = ["freesurfer", "connectivity"]

FEATURE_LABELS = {
    "freesurfer": "FreeSurfer",
    "connectivity": "Connectivity",
}

SELECTED_TARGETS = [
    "brain_age",
    "Fluid_intelligence_score",
    "Sleep_duration",
    "Hand_grip_strength_mean",
    "bmi",
    "diastolic_bp",
    "trail_making",
]

FEATURE_TARGET_EXCLUSIONS = {
    "freesurfer": set(),
    "connectivity": set(),
}


TARGET_LABELS = {
    "brain_age": "Age",
    "Fluid_intelligence_score": "Fluid intelligence",
    "Sleep_duration": "Sleep duration",
    "Hand_grip_strength_mean": "Grip strength",
    "bmi": "BMI",
    "diastolic_bp": "DBP",
    "trail_making": "Trail Making time",
}

TARGET_ORDER = SELECTED_TARGETS.copy()

N_BOOT = 2000
BOOT_SEED = 42
SAVE_DPI = 600


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": SAVE_DPI,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.28,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


COLORS = {
    "freesurfer": "#2E86AB",
    "connectivity": "#A23B72",
    "zero": "#222222",
    "regression_bg": "#F3F8FB",
}

def pretty_target(x):
    return TARGET_LABELS.get(x, str(x).replace("_", " "))


def pretty_feature(x):
    return FEATURE_LABELS.get(x, str(x).replace("_", " ").title())




def stable_seed_offset(*parts, modulo=100000):
    key = "||".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.md5(key).hexdigest()[:8], 16) % modulo


def safe_filename(text):
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_") or "target"


def wrap_feature_name(name, width=34):
    name = str(name)
    name = name.replace("connectivity__", "")
    name = name.replace("_", " ")
    return "\n".join(textwrap.wrap(name, width=width))


def safe_pearsonr(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 3:
        return np.nan

    if np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return np.nan

    return float(pearsonr(y_true, y_pred)[0])



def safe_pearson_p(y_true, y_pred):
    """Two-sided p-value for Pearson correlation."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 3:
        return np.nan

    if np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return np.nan

    return float(pearsonr(y_true, y_pred)[1])


def benjamini_hochberg(p_values):

    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)

    finite = np.isfinite(p)
    if not finite.any():
        return q

    p_fin = p[finite]
    m = len(p_fin)

    order = np.argsort(p_fin)
    ranked = p_fin[order]

    adjusted = ranked * m / np.arange(1, m + 1, dtype=float)

    # Enforce monotonicity from largest rank to smallest.
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    q_fin = np.empty_like(adjusted)
    q_fin[order] = adjusted
    q[finite] = q_fin

    return q

def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return np.nan

    if np.nanstd(y_true) == 0:
        return np.nan

    return float(r2_score(y_true, y_pred))

def make_selected_targets(feature_name):

    base_targets = base.make_targets_for_feature_set(feature_name)

    exclusions = FEATURE_TARGET_EXCLUSIONS.get(feature_name, set())
    wanted = [
        target
        for target in SELECTED_TARGETS
        if target not in exclusions
    ]


    selected = {
        target: base_targets[target]
        for target in wanted
    }




    return selected


def load_feature_set_data(feature_name):
    targets = make_selected_targets(feature_name)

    if feature_name == "freesurfer":
        df, feature_cols = base.prepare_dataframe_single(
            feature_file=base.FEATURE_FILES["freesurfer"],
            diag_file=base.DIAG_FILE,
            targets=targets,
            add_mean_motion=False,
        )

    elif feature_name == "connectivity":
        df, feature_cols = base.prepare_dataframe_single(
            feature_file=base.FEATURE_FILES["connectivity"],
            diag_file=base.DIAG_FILE,
            targets=targets,
            add_mean_motion=True,
        )


    return df, feature_cols, targets


def get_best_alpha_from_saved_cv(feature_name, target_name):
    cv_path = os.path.join(OUT_DIR, f"cv_results_{feature_name}.csv")

    cv_df = pd.read_csv(cv_path)

    sub = cv_df[cv_df["target"] == target_name].copy()

    best_alpha = (
        sub.groupby("alpha", as_index=False)["val_mse"]
        .mean()
        .sort_values("val_mse", ascending=True)
        .iloc[0]["alpha"]
    )

    return float(best_alpha)


def get_best_alpha(feature_name, target_name, hc_train, feature_cols, spec):
    try:
        return get_best_alpha_from_saved_cv(
            feature_name=feature_name,
            target_name=target_name,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(
            f"[WARN] Could not load saved alpha for {feature_name}/{target_name}: {exc}"
        )
        print("[WARN] Recomputing CV alpha on the current train-HC data.")

        best_alpha, cv_df = base.run_cv_for_target(
            df_train_hc=hc_train,
            feature_cols=feature_cols,
            target_name=target_name,
            spec=spec,
            alphas=base.ALPHAS,
        )

        cv_out = os.path.join(
            FIG_DIR,
            f"cv_results_{feature_name}_{target_name}_computed_in_performance_script.csv",
        )
        cv_df.to_csv(cv_out, index=False)
        print(f"[WARN] Saved fallback CV results: {cv_out}")

        return float(best_alpha)

def build_fold_importance_table(
    model,
    feature_cols,
    feature_name,
    target_name,
    target_col,
    fold_idx,
    alpha,
    confound_cols,
):
    coef = np.asarray(model.coef_).reshape(-1).astype(float)


    abs_coef = np.abs(coef)
    abs_sum = float(abs_coef.sum())

    if np.isfinite(abs_sum) and abs_sum > 0:
        normalized_abs = abs_coef / abs_sum
    else:
        normalized_abs = np.zeros_like(abs_coef)

    out = pd.DataFrame({
        "feature_set": feature_name,
        "target": target_name,
        "target_col": target_col,
        "task": "regression",
        "fold": int(fold_idx),
        "alpha": float(alpha),
        "feature": list(feature_cols),
        "coefficient": coef,
        "abs_coefficient": abs_coef,
        "normalized_abs_importance": normalized_abs,
        "confounds": ",".join(confound_cols),
    })

    out["rank_within_fold"] = (
        out["abs_coefficient"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    return out


def compute_regression_oof(
    df_train_hc,
    feature_cols,
    feature_name,
    target_name,
    spec,
    alpha,
):
    target_col = spec["target_col"]
    confound_cols = spec["confounds"]

    cv_iter = base.get_cv_iterator(
        df_train_hc=df_train_hc,
        task="regression",
        target_col=target_col,
    )

    prediction_rows = []
    importance_rows = []

    for fold_idx, (tr_idx, va_idx) in enumerate(cv_iter, start=1):
        tr_df = df_train_hc.iloc[tr_idx].reset_index(drop=True)
        va_df = df_train_hc.iloc[va_idx].reset_index(drop=True)

        y_train = tr_df[target_col].values.astype(float)
        y_val = va_df[target_col].values.astype(float)

        X_train = tr_df[feature_cols].values.astype(float)
        X_val = va_df[feature_cols].values.astype(float)

        # Fit feature deconfounding only on the fold-training subjects.
        feat_resid = base.FeatureResidualizer(
            confound_cols=confound_cols,
        ).fit(tr_df, X_train)

        X_train_resid = feat_resid.residualize(tr_df, X_train)
        X_val_resid = feat_resid.residualize(va_df, X_val)

        _, X_train_z, X_val_z = base.feature_scaler_fit_transform(
            X_train_resid,
            X_val_resid,
        )

        model = Ridge(
            alpha=alpha,
            random_state=base.RANDOM_STATE,
        )
        model.fit(X_train_z, y_train)

        pred_val = model.predict(X_val_z)

        prediction_rows.append(
            pd.DataFrame({
                "participant_id": va_df[base.ID_COL].values,
                "feature_set": feature_name,
                "target": target_name,
                "target_col": target_col,
                "task": "regression",
                "fold": int(fold_idx),
                "alpha": float(alpha),
                "y_true": y_val,
                "y_pred": pred_val,
                "confounds": ",".join(confound_cols),
            })
        )

        importance_rows.append(
            build_fold_importance_table(
                model=model,
                feature_cols=feature_cols,
                feature_name=feature_name,
                target_name=target_name,
                target_col=target_col,
                fold_idx=fold_idx,
                alpha=alpha,
                confound_cols=confound_cols,
            )
        )

    return (
        pd.concat(prediction_rows, axis=0, ignore_index=True),
        pd.concat(importance_rows, axis=0, ignore_index=True),
    )


def compute_all_oof_predictions():
    hc_split_map = base.make_global_hc_split_map(base.DIAG_FILE)
    all_oof = []
    all_importances = []

    for feature_name in FEATURE_ORDER:
        print("\n" + "=" * 80)
        print(f"Feature set: {feature_name}")
        print("=" * 80)

        df, feature_cols, targets = load_feature_set_data(feature_name)

        hc_train_all, hc_holdout_all, cases_all = base.split_hc_cases(df, hc_split_map=hc_split_map)

        print(f"Merged rows after feature/health filters: {len(df)}")
        print(f"Feature columns:                    {len(feature_cols)}")
        print(f"Stable train HC before target NA:   {len(hc_train_all)}")
        print(f"Stable heldout HC before target NA: {len(hc_holdout_all)}")
        print(f"Stable cases before target NA:      {len(cases_all)}")

        for target_name in SELECTED_TARGETS:
            if target_name not in targets:
                # Expected modality-specific exclusion (currently connectivity grip).
                continue

            spec = targets[target_name]
            target_col = spec["target_col"]
            confound_cols = spec["confounds"]

            print("\n--- Target:", target_name, "---")
            print(f"Target column: {target_col}")
            print(f"Confounds: {confound_cols}")

            hc_train = base.dropna_for_current_target(
                hc_train_all,
                target_name=target_name,
                spec=spec,
                stage_label="train_hc",
            )
            hc_holdout = base.dropna_for_current_target(
                hc_holdout_all,
                target_name=target_name,
                spec=spec,
                stage_label="heldout_hc",
            )
            cases = base.dropna_for_current_target(
                cases_all,
                target_name=target_name,
                spec=spec,
                stage_label="case",
            )

            base.assert_enough_current_target_data(
                target_name=target_name,
                task="regression",
                target_col=target_col,
                hc_train=hc_train,
                hc_holdout=hc_holdout,
            )

            alpha = get_best_alpha(
                feature_name=feature_name,
                target_name=target_name,
                hc_train=hc_train,
                feature_cols=feature_cols,
                spec=spec,
            )

            print(
                f"OOF train-HC n={len(hc_train)} | "
                f"heldout-HC complete n={len(hc_holdout)} | "
                f"case complete n={len(cases)} | "
                f"alpha={alpha}"
            )

            oof_df, importance_df = compute_regression_oof(
                df_train_hc=hc_train,
                feature_cols=feature_cols,
                feature_name=feature_name,
                target_name=target_name,
                spec=spec,
                alpha=alpha,
            )

            oof_df["n_train_hc_complete_current_target"] = int(len(hc_train))
            oof_df["n_heldout_hc_complete_current_target"] = int(len(hc_holdout))
            oof_df["n_cases_complete_current_target"] = int(len(cases))

            importance_df["n_train_hc_complete_current_target"] = int(len(hc_train))
            importance_df["n_heldout_hc_complete_current_target"] = int(len(hc_holdout))
            importance_df["n_cases_complete_current_target"] = int(len(cases))

            all_oof.append(oof_df)
            all_importances.append(importance_df)


    oof_all = pd.concat(all_oof, axis=0, ignore_index=True)
    importance_all = pd.concat(all_importances, axis=0, ignore_index=True)




    return oof_all, importance_all

def metric_from_oof(sub, metric):
    if metric == "pearson":
        return safe_pearsonr(sub["y_true"].values, sub["y_pred"].values)

    if metric == "r2":
        return safe_r2(sub["y_true"].values, sub["y_pred"].values)

    raise ValueError(f"Unknown metric: {metric}")


def bootstrap_ci(sub, metric, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    sub = sub.reset_index(drop=True)

    n = len(sub)
    if n < 5:
        return np.nan, np.nan

    values = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot = sub.iloc[idx]
        value = metric_from_oof(boot, metric)

        if np.isfinite(value):
            values.append(value)

    if len(values) < 20:
        return np.nan, np.nan

    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def summarize_oof_performance(oof_df):
    rows = []

    for (feature_set, target), sub in oof_df.groupby(
        ["feature_set", "target"],
        sort=False,
    ):
        sub = sub.reset_index(drop=True)

        alpha = float(sub["alpha"].iloc[0])
        confounds = sub["confounds"].iloc[0]
        n_oof = int(len(sub))

        seed_offset = stable_seed_offset(feature_set, target)

        pearson_value = metric_from_oof(sub, "pearson")
        pearson_p = safe_pearson_p(
            sub["y_true"].values,
            sub["y_pred"].values,
        )
        r2_value = metric_from_oof(sub, "r2")

        pearson_low, pearson_high = bootstrap_ci(
            sub,
            metric="pearson",
            seed=BOOT_SEED + seed_offset + 1,
        )
        r2_low, r2_high = bootstrap_ci(
            sub,
            metric="r2",
            seed=BOOT_SEED + seed_offset + 2,
        )

        rows.append({
            "feature_set": feature_set,
            "feature_label": pretty_feature(feature_set),
            "target": target,
            "target_label": pretty_target(target),
            "task": "regression",
            "n_oof": n_oof,
            "alpha": alpha,
            "confounds": confounds,
            "pearson": pearson_value,
            "pearson_p": pearson_p,
            "pearson_ci_low": pearson_low,
            "pearson_ci_high": pearson_high,
            "r2": r2_value,
            "r2_ci_low": r2_low,
            "r2_ci_high": r2_high,
        })

    summary = pd.DataFrame(rows)

    summary["pearson_q_fdr_bh"] = benjamini_hochberg(
        summary["pearson_p"].values
    )

    feature_order = {feature: i for i, feature in enumerate(FEATURE_ORDER)}
    target_order = {target: i for i, target in enumerate(TARGET_ORDER)}

    summary["feature_order"] = summary["feature_set"].map(feature_order)
    summary["target_order"] = (
        summary["target"].map(target_order).fillna(999).astype(int)
    )

    summary = summary.sort_values(
        ["target_order", "feature_order"],
        ascending=[True, True],
    ).reset_index(drop=True)

    return summary


def _fmt_latex_number(value, digits=3):
    if value is None or not np.isfinite(value):
        return "--"
    return f"{float(value):.{digits}f}"


def _fmt_latex_p(value):

    if value is None or not np.isfinite(value):
        return "--"

    value = float(value)

    if value == 0.0:
        # Numerical underflow from the underlying test.
        return "0.000e+00"

    if abs(value) < 0.001:
        return f"{value:.3e}"

    return f"{value:.6f}".rstrip("0").rstrip(".")


def _fmt_latex_ci(value, low, high, digits=3):
    if (
        value is None or low is None or high is None
        or not np.isfinite(value)
        or not np.isfinite(low)
        or not np.isfinite(high)
    ):
        return "--"
    return (
        f"{float(value):.{digits}f} "
        f"({float(low):.{digits}f}, {float(high):.{digits}f})"
    )


def write_performance_latex_table(summary_df, output_path):


    work = summary_df.copy()

    feature_order = {feature: i for i, feature in enumerate(FEATURE_ORDER)}
    target_order = {target: i for i, target in enumerate(TARGET_ORDER)}

    work["_feature_order"] = work["feature_set"].map(feature_order).fillna(999)
    work["_target_order"] = work["target"].map(target_order).fillna(999)

    work = work.sort_values(
        ["_feature_order", "_target_order"],
        kind="stable",
    ).reset_index(drop=True)


    lines = []
    lines.append(r"% Required packages: \usepackage{booktabs}, \usepackage{pdflscape}")
    lines.append(r"\begin{landscape}")
    lines.append(r"\begin{table}[p]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Out-of-fold predictive performance of the 14 candidate "
        r"modality--target normative models.}"
    )
    lines.append(r"\label{tab:supp_normative_performance}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{8pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.10}")
    lines.append(r"\begin{tabular}{llrrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Imaging modality & Target "
        r"& Pearson $r$ (95\% CI) "
        r"& $p$ "
        r"& $q_{\mathrm{FDR}}$ "
        r"& $R^2$ (95\% CI) "
        r"& Ridge $\alpha$ \\"
    )
    lines.append(r"\midrule")

    previous_feature = None

    for _, row in work.iterrows():
        feature = str(row["feature_set"])
        modality = pretty_feature(feature)
        target = pretty_target(row["target"])

        if previous_feature is not None and feature != previous_feature:
            lines.append(r"\addlinespace")

        modality_cell = modality if feature != previous_feature else ""

        pearson = _fmt_latex_ci(
            row["pearson"],
            row["pearson_ci_low"],
            row["pearson_ci_high"],
            digits=3,
        )
        pval = _fmt_latex_p(row["pearson_p"])
        qval = _fmt_latex_p(row["pearson_q_fdr_bh"])
        r2 = _fmt_latex_ci(
            row["r2"],
            row["r2_ci_low"],
            row["r2_ci_high"],
            digits=3,
        )
        alpha = _fmt_latex_number(row["alpha"], digits=3)

        lines.append(
            f"{modality_cell} & {target} & "
            f"{pearson} & {pval} & {qval} & {r2} & {alpha} \\\\"
        )

        previous_feature = feature

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{minipage}{0.98\textwidth}")
    lines.append(r"\end{minipage}")
    lines.append(r"\end{table}")
    lines.append(r"\end{landscape}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved Supplementary Table 1 LaTeX:\n{output_path}")


# =============================================================================
# FEATURE IMPORTANCE AGGREGATION
# =============================================================================

def aggregate_feature_importance(fold_importance_df):

    rows = []

    group_cols = [
        "feature_set",
        "target",
        "target_col",
        "feature",
    ]

    for keys, sub in fold_importance_df.groupby(group_cols, sort=False):
        feature_set, target, target_col, feature = keys
        sub = sub.reset_index(drop=True)

        coefs = sub["coefficient"].astype(float).values
        abs_coefs = sub["abs_coefficient"].astype(float).values
        norm_abs = sub["normalized_abs_importance"].astype(float).values
        ranks = sub["rank_within_fold"].astype(float).values

        finite_coef = coefs[np.isfinite(coefs)]
        if finite_coef.size == 0:
            positive_fraction = np.nan
            negative_fraction = np.nan
            zero_fraction = np.nan
            sign_consistency = np.nan
        else:
            positive_fraction = float(np.mean(finite_coef > 0))
            negative_fraction = float(np.mean(finite_coef < 0))
            zero_fraction = float(np.mean(finite_coef == 0))
            sign_consistency = float(max(positive_fraction, negative_fraction))

        rows.append({
            "feature_set": feature_set,
            "feature_label": pretty_feature(feature_set),
            "target": target,
            "target_label": pretty_target(target),
            "target_col": target_col,
            "task": "regression",
            "feature": feature,
            "n_folds": int(sub["fold"].nunique()),
            "alpha": float(sub["alpha"].iloc[0]),
            "confounds": sub["confounds"].iloc[0],
            "mean_coefficient": float(np.nanmean(coefs)),
            "sd_coefficient": float(np.nanstd(coefs, ddof=1)) if len(coefs) > 1 else 0.0,
            "mean_abs_coefficient": float(np.nanmean(abs_coefs)),
            "sd_abs_coefficient": float(np.nanstd(abs_coefs, ddof=1)) if len(abs_coefs) > 1 else 0.0,
            "mean_normalized_abs_importance": float(np.nanmean(norm_abs)),
            "sd_normalized_abs_importance": (
                float(np.nanstd(norm_abs, ddof=1)) if len(norm_abs) > 1 else 0.0
            ),
            "mean_rank_within_fold": float(np.nanmean(ranks)),
            "positive_fraction": positive_fraction,
            "negative_fraction": negative_fraction,
            "zero_fraction": zero_fraction,
            "sign_consistency": sign_consistency,
        })

    summary = pd.DataFrame(rows)

    if summary.empty:
        return summary

    summary["importance_rank"] = (
        summary.groupby(["feature_set", "target"])["mean_abs_coefficient"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    feature_order = {feature: i for i, feature in enumerate(FEATURE_ORDER)}
    target_order = {target: i for i, target in enumerate(TARGET_ORDER)}

    summary["feature_order"] = summary["feature_set"].map(feature_order)
    summary["target_order"] = (
        summary["target"].map(target_order).fillna(999).astype(int)
    )

    summary = summary.sort_values(
        ["target_order", "feature_order", "importance_rank"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    return summary


def _plot_importance_axis(ax, sub, feature_name, top_n):
    sub = sub.sort_values("importance_rank", ascending=True).head(top_n).copy()

    if sub.empty:
        ax.text(
            0.5,
            0.5,
            "No features available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    sub = sub.sort_values("mean_abs_coefficient", ascending=True)

    y = np.arange(len(sub))
    x = sub["mean_abs_coefficient"].astype(float).values
    xerr = sub["sd_abs_coefficient"].astype(float).values

    ax.barh(
        y,
        x,
        xerr=xerr,
        color=COLORS[feature_name],
        alpha=0.82,
        capsize=2,
        linewidth=0,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(
        [wrap_feature_name(feature) for feature in sub["feature"].values],
        fontsize=7,
    )
    ax.set_xlabel("Mean |standardized coefficient| across OOF folds")
    ax.set_title(pretty_feature(feature_name), fontweight="bold")
    ax.grid(axis="x")
    ax.set_axisbelow(True)

    x_max = float(np.nanmax(x + np.nan_to_num(xerr, nan=0.0))) if len(x) else 1.0
    if not np.isfinite(x_max) or x_max <= 0:
        x_max = 1.0
    text_pad = 0.025 * x_max

    for yi, (_, row) in enumerate(sub.iterrows()):
        mean_signed = row["mean_coefficient"]
        stability = row["sign_consistency"]
        label = f"β={mean_signed:+.3g}"
        if np.isfinite(stability):
            label += f" | sign={stability:.2f}"

        ax.text(
            float(row["mean_abs_coefficient"]) + float(row["sd_abs_coefficient"]) + text_pad,
            yi,
            label,
            va="center",
            ha="left",
            fontsize=6.5,
            color=COLORS[feature_name],
        )

    right = x_max * 1.42
    ax.set_xlim(0, right)


def plot_feature_importance_target(
    target,
    importance_summary,
    pdf=None,
    top_n=FEATURE_IMPORTANCE_TOP_N,
):
    set_style()

    target_sub = importance_summary[
        importance_summary["target"] == target
    ].copy()

    if target_sub.empty:
        return

    fig, axes = plt.subplots(
        1,
        len(FEATURE_ORDER),
        figsize=(14.0, max(6.2, 0.36 * top_n + 1.8)),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, feature_name in zip(axes, FEATURE_ORDER):
        sub = target_sub[
            target_sub["feature_set"] == feature_name
        ].copy()
        _plot_importance_axis(
            ax=ax,
            sub=sub,
            feature_name=feature_name,
            top_n=top_n,
        )

    fig.suptitle(
        f"Feature importance — {pretty_target(target)}",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    fig.text(
        0.5,
        0.005,
        "Bars: mean absolute coefficient across OOF folds; error bars: SD across folds. "
        "Annotations show signed mean coefficient and sign consistency.",
        ha="center",
        va="bottom",
        fontsize=7.5,
    )

    fig.tight_layout(rect=[0.0, 0.025, 1.0, 0.965])

    png_path = os.path.join(
        FEATURE_IMPORTANCE_DIR,
        f"feature_importance_{safe_filename(target)}.png",
    )
    fig.savefig(png_path, bbox_inches="tight")

    if pdf is not None:
        pdf.savefig(fig, bbox_inches="tight")

    plt.close(fig)
    print(f"Saved feature-importance plot: {png_path}")


def make_feature_importance_report(importance_summary):
    if importance_summary.empty:
        print("[WARN] Feature-importance summary is empty; no report created.")
        return

    with PdfPages(FEATURE_IMPORTANCE_REPORT_PDF) as pdf:
        for target in TARGET_ORDER:
            if target not in set(importance_summary["target"]):
                continue

            plot_feature_importance_target(
                target=target,
                importance_summary=importance_summary,
                pdf=pdf,
                top_n=FEATURE_IMPORTANCE_TOP_N,
            )

        extra_targets = [
            target for target in importance_summary["target"].unique()
            if target not in TARGET_ORDER
        ]
        for target in extra_targets:
            plot_feature_importance_target(
                target=target,
                importance_summary=importance_summary,
                pdf=pdf,
                top_n=FEATURE_IMPORTANCE_TOP_N,
            )

    print(f"Saved feature-importance PDF report: {FEATURE_IMPORTANCE_REPORT_PDF}")


def make_y_positions(summary_df):
    targets = [
        target for target in TARGET_ORDER
        if target in set(summary_df["target"])
    ]

    extra_targets = [
        target for target in summary_df["target"].unique()
        if target not in targets
    ]
    targets = targets + extra_targets

    y_map = {}
    y_labels = []

    for i, target in enumerate(targets):
        y = len(targets) - 1 - i
        y_map[target] = y
        y_labels.append((y, pretty_target(target)))

    return y_map, y_labels


def build_panel_metric_df(summary_df, metric):
    if metric == "pearson":
        mean_col = "pearson"
        low_col = "pearson_ci_low"
        high_col = "pearson_ci_high"
    elif metric == "r2":
        mean_col = "r2"
        low_col = "r2_ci_low"
        high_col = "r2_ci_high"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return pd.DataFrame({
        "feature_set": summary_df["feature_set"].values,
        "feature_label": summary_df["feature_label"].values,
        "target": summary_df["target"].values,
        "target_label": summary_df["target_label"].values,
        "metric_mean": summary_df[mean_col].values,
        "ci_low": summary_df[low_col].values,
        "ci_high": summary_df[high_col].values,
    })


def plot_lollipop_panel(
    ax,
    metric_df,
    y_map,
    xlabel,
    panel_label,
    xlim,
    draw_zero=True,
    value_format="{:.3f}",
):
    offsets = {
        "freesurfer": 0.16,
        "connectivity": -0.16,
    }

    x_range = xlim[1] - xlim[0]
    text_pad = 0.030 * x_range
    all_label_x = []

    for feature in FEATURE_ORDER:
        sub = metric_df[metric_df["feature_set"] == feature].copy()
        if sub.empty:
            continue

        y = sub["target"].map(y_map).astype(float).values
        y = y + offsets.get(feature, 0.0)

        x = sub["metric_mean"].astype(float).values
        ci_low = sub["ci_low"].astype(float).values
        ci_high = sub["ci_high"].astype(float).values

        lower_err = x - ci_low
        upper_err = ci_high - x

        lower_err = np.where(np.isfinite(lower_err), lower_err, 0.0)
        upper_err = np.where(np.isfinite(upper_err), upper_err, 0.0)
        xerr = np.vstack([lower_err, upper_err])

        color = COLORS[feature]

        ax.errorbar(
            x,
            y,
            xerr=xerr,
            fmt="o",
            markersize=5.8,
            linewidth=1.2,
            elinewidth=1.2,
            capsize=3,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=pretty_feature(feature),
            zorder=3,
        )

        for xi, uerr, yi in zip(x, upper_err, y):
            if not np.isfinite(xi):
                continue

            uerr = 0.0 if not np.isfinite(uerr) else float(uerr)
            label_x = xi + abs(uerr) + text_pad
            all_label_x.append(label_x)

            ax.text(
                label_x,
                yi,
                value_format.format(xi),
                va="center",
                ha="left",
                fontsize=7.2,
                color=color,
                fontweight="bold",
                clip_on=False,
                zorder=5,
            )

    if draw_zero:
        ax.axvline(
            0,
            color=COLORS["zero"],
            linewidth=0.9,
            alpha=0.75,
            zorder=1,
        )

    if all_label_x:
        label_max = np.nanmax(all_label_x)
        right_lim = max(xlim[1], label_max + 0.14 * x_range)
    else:
        right_lim = xlim[1]

    ax.set_xlim(xlim[0], right_lim)
    ax.set_xlabel(xlabel)

    if panel_label:
        ax.set_title(panel_label, loc="left", fontweight="bold")

    ax.grid(axis="x")
    ax.set_axisbelow(True)


def add_regression_background(ax, y_map):
    if not y_map:
        return

    y_values = list(y_map.values())
    ax.axhspan(
        min(y_values) - 0.55,
        max(y_values) + 0.55,
        color=COLORS["regression_bg"],
        alpha=0.75,
        zorder=0,
    )


def get_metric_xlim(metric_df, default=(-0.08, 0.90)):
    means = metric_df["metric_mean"].astype(float).values
    lows = metric_df["ci_low"].astype(float).values
    highs = metric_df["ci_high"].astype(float).values

    finite_parts = [
        values[np.isfinite(values)]
        for values in (means, lows, highs)
    ]
    finite_parts = [values for values in finite_parts if values.size > 0]

    if not finite_parts:
        return default

    finite_vals = np.concatenate(finite_parts)

    return (
        min(default[0], float(np.nanmin(finite_vals)) - 0.03),
        max(default[1], float(np.nanmax(finite_vals)) + 0.08),
    )


def add_figure_legend(fig, ax, bbox_to_anchor=(0.56, 1.005)):
    handles, labels = ax.get_legend_handles_labels()

    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle

    fig.legend(
        list(unique.values()),
        list(unique.keys()),
        loc="upper center",
        bbox_to_anchor=bbox_to_anchor,
        ncol=len(FEATURE_ORDER),
        frameon=False,
        title="Feature set",
    )


def save_single_panel_png(
    metric_df,
    y_map,
    y_labels,
    xlabel,
    xlim,
    output_path,
):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    add_regression_background(ax, y_map)

    plot_lollipop_panel(
        ax=ax,
        metric_df=metric_df,
        y_map=y_map,
        xlabel=xlabel,
        panel_label=None,
        xlim=xlim,
        draw_zero=True,
        value_format="{:.3f}",
    )

    y_ticks = [item[0] for item in y_labels]
    y_ticklabels = [item[1] for item in y_labels]

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels)
    ax.set_ylabel("Normative target")

    add_figure_legend(
        fig,
        ax,
        bbox_to_anchor=(0.56, 1.015),
    )

    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.91])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def make_performance_plot(summary_df):
    set_style()

    y_map, y_labels = make_y_positions(summary_df)

    pearson_df = build_panel_metric_df(summary_df, metric="pearson")
    r2_df = build_panel_metric_df(summary_df, metric="r2")

    y_ticks = [item[0] for item in y_labels]
    y_ticklabels = [item[1] for item in y_labels]

    pearson_xlim = get_metric_xlim(pearson_df, default=(-0.08, 0.90))
    r2_xlim = get_metric_xlim(r2_df, default=(-0.08, 0.90))

    fig = plt.figure(figsize=(8.2, 8.6))

    gs = fig.add_gridspec(
        nrows=2,
        ncols=1,
        height_ratios=[1, 1],
        hspace=0.34,
    )

    ax_top = fig.add_subplot(gs[0, 0])
    ax_bottom = fig.add_subplot(gs[1, 0], sharey=ax_top)

    add_regression_background(ax_top, y_map)
    add_regression_background(ax_bottom, y_map)

    plot_lollipop_panel(
        ax=ax_top,
        metric_df=pearson_df,
        y_map=y_map,
        xlabel="Pearson with 95% bootstrap CI",
        panel_label="A. Pearson",
        xlim=pearson_xlim,
        draw_zero=True,
        value_format="{:.3f}",
    )

    ax_top.set_yticks(y_ticks)
    ax_top.set_yticklabels(y_ticklabels)
    ax_top.set_ylabel("Normative target")

    plot_lollipop_panel(
        ax=ax_bottom,
        metric_df=r2_df,
        y_map=y_map,
        xlabel="R² with 95% bootstrap CI",
        panel_label="B. R²",
        xlim=r2_xlim,
        draw_zero=True,
        value_format="{:.3f}",
    )

    ax_bottom.set_yticks(y_ticks)
    ax_bottom.set_yticklabels(y_ticklabels)
    ax_bottom.set_ylabel("Normative target")

    add_figure_legend(fig, ax_top)

    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.965])
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    save_single_panel_png(
        metric_df=pearson_df,
        y_map=y_map,
        y_labels=y_labels,
        xlabel="Pearson with 95% bootstrap CI",
        xlim=pearson_xlim,
        output_path=OUT_PANEL_A_PNG,
    )

    save_single_panel_png(
        metric_df=r2_df,
        y_map=y_map,
        y_labels=y_labels,
        xlabel="R² with 95% bootstrap CI",
        xlim=r2_xlim,
        output_path=OUT_PANEL_B_PNG,
    )

    print("Saved performance figures:")
    print(OUT_PNG)
    print(OUT_PDF)
    print(OUT_PANEL_A_PNG)
    print(OUT_PANEL_B_PNG)


# =============================================================================
# MAIN
# =============================================================================

def main():
    oof_df, fold_importance_df = compute_all_oof_predictions()

    oof_df.to_csv(OOF_PRED_OUT, index=False)
    print(f"\nSaved OOF predictions:\n{OOF_PRED_OUT}")

    fold_importance_df.to_csv(FEATURE_IMPORTANCE_FOLD_OUT, index=False)
    print(f"Saved fold-wise feature coefficients:\n{FEATURE_IMPORTANCE_FOLD_OUT}")

    performance_summary = summarize_oof_performance(oof_df)
    performance_summary.to_csv(SUMMARY_OUT, index=False)
    print(f"Saved performance summary:\n{SUMMARY_OUT}")

    write_performance_latex_table(
        performance_summary,
        SUPPLEMENTARY_TABLE_TEX,
    )

    importance_summary = aggregate_feature_importance(fold_importance_df)
    importance_summary.to_csv(FEATURE_IMPORTANCE_SUMMARY_OUT, index=False)
    print(f"Saved feature-importance summary:\n{FEATURE_IMPORTANCE_SUMMARY_OUT}")

    top_importance = importance_summary[
        importance_summary["importance_rank"] <= FEATURE_IMPORTANCE_TOP_N
    ].copy()
    top_importance.to_csv(FEATURE_IMPORTANCE_TOP_OUT, index=False)
    print(f"Saved top-feature table:\n{FEATURE_IMPORTANCE_TOP_OUT}")

    print("\nPerformance summary:")
    print(
        performance_summary[
            [
                "feature_set",
                "target",
                "n_oof",
                "pearson",
                "pearson_p",
                "pearson_q_fdr_bh",
                "pearson_ci_low",
                "pearson_ci_high",
                "r2",
                "r2_ci_low",
                "r2_ci_high",
                "alpha",
            ]
        ].to_string(index=False)
    )

    print("\nTop 10 feature importances per target/modality:")
    preview = importance_summary[
        importance_summary["importance_rank"] <= 10
    ][
        [
            "feature_set",
            "target",
            "importance_rank",
            "feature",
            "mean_abs_coefficient",
            "mean_coefficient",
            "sign_consistency",
        ]
    ]
    print(preview.to_string(index=False))

    make_performance_plot(performance_summary)
    make_feature_importance_report(importance_summary)

    print("\nDone.")


if __name__ == "__main__":
    main()

