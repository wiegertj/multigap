#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Missingness sensitivity analysis: normative refitting and downstream associations requiring PHQ-9 and AUDIT-C to be available
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIG
# =============================================================================

SECONDARY_HC = "SECONDARY_HC"
SECONDARY_AUD = "SECONDARY_AUD"
SECONDARY_MDD = "SECONDARY_MDD"
SECONDARY_ANX = "SECONDARY_ANX"

DEFAULT_DIAG_FILE = Path("...")

DEFAULT_MAIN_NORMATIVE_DIR = Path("...")

DEFAULT_SECONDARY_NORMATIVE_DIR = Path("...")

DEFAULT_MAIN_RESULTS_DIR = Path("...")

DEFAULT_SECONDARY_RESULTS_DIR = Path("...")

MAIN_EDGE_TABLE_CANDIDATES = [
    "figure2_adaptive_nodes_predictor_gap_edges.csv",
]


def install_sleep_gap_filter(fig: Any) -> None:
    original_graph_gap_cols = fig.graph_gap_cols

    def graph_gap_cols_without_obsolete_sleep(gap_cols):
        cols = original_graph_gap_cols(gap_cols)
        return [
            c for c in cols
            if not any(target.lower() in str(c).lower() for target in EXCLUDED_SLEEP_TARGETS)
        ]

    fig.graph_gap_cols = graph_gap_cols_without_obsolete_sleep


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Full SECONDARY-definition sensitivity: refit normative models with "
            "SECONDARY_HC and rerun downstream associations."
        )
    )
    p.add_argument("--normative-base-script", type=Path, default=None)
    p.add_argument("--figure-base-script", type=Path, default=None)
    p.add_argument("--diag-file", type=Path, default=DEFAULT_DIAG_FILE)
    p.add_argument(
        "--main-normative-dir",
        type=Path,
        default=DEFAULT_MAIN_NORMATIVE_DIR,
        help="MAIN normative output folder used only to read selected alpha values.",
    )
    p.add_argument(
        "--secondary-normative-dir",
        type=Path,
        default=DEFAULT_SECONDARY_NORMATIVE_DIR,
    )
    p.add_argument(
        "--main-results-dir",
        type=Path,
        default=DEFAULT_MAIN_RESULTS_DIR,
    )
    p.add_argument(
        "--secondary-results-dir",
        type=Path,
        default=DEFAULT_SECONDARY_RESULTS_DIR,
    )
    p.add_argument(
        "--reselect-alpha",
        action="store_true",
    )
    p.add_argument(
        "--skip-normative-refit",
        action="store_true",
    )
    return p.parse_args()


def import_module_from_path(path: Path, module_name: str):
    path = path.expanduser().resolve()
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_script(explicit: Optional[Path], candidates) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        return p

    here = Path(__file__).resolve().parent
    for name in candidates:
        p = here / name
        if p.exists():
            return p.resolve()

    raise FileNotFoundError(
        "Could not locate required companion script. Tried:\n  - "
        + "\n  - ".join(str(here / x) for x in candidates)
    )


# =============================================================================
# MAIN-ALPHA REUSE
# =============================================================================

def load_main_alpha_map(main_normative_dir: Path) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for feature_name in ("freesurfer", "connectivity"):
        path = main_normative_dir / f"model_summary_{feature_name}.json"
        if not path.exists():
            print(f"[WARN] MAIN alpha summary missing for {feature_name}: {path}")
            continue
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        result[feature_name] = {}
        for target_name, info in obj.items():
            alpha = info.get("best_alpha") if isinstance(info, dict) else None
            try:
                alpha = float(alpha)
            except (TypeError, ValueError):
                continue
            if np.isfinite(alpha):
                result[feature_name][target_name] = alpha
    return result


def install_main_alpha_reuse(norm: Any, alpha_map: Dict[str, Dict[str, float]]) -> None:
    original_cv = norm.run_cv_for_target
    original_runner = norm.run_feature_set_from_df
    norm._secondary_current_feature_name = None

    def wrapped_runner(feature_name, df, feature_cols, out_dir, hc_split_map):
        norm._secondary_current_feature_name = feature_name
        return original_runner(
            feature_name=feature_name,
            df=df,
            feature_cols=feature_cols,
            out_dir=out_dir,
            hc_split_map=hc_split_map,
        )

    def fixed_alpha_cv(
        df_train_hc,
        feature_cols,
        target_name,
        spec,
        alphas,
    ):
        feature_name = getattr(norm, "_secondary_current_feature_name", None)
        alpha = alpha_map.get(str(feature_name), {}).get(str(target_name))

        if alpha is None:
            print(
                f"[WARN] No MAIN alpha found for {feature_name}/{target_name}; "
                "re-selecting alpha by CV in SECONDARY train HCs."
            )
            return original_cv(
                df_train_hc=df_train_hc,
                feature_cols=feature_cols,
                target_name=target_name,
                spec=spec,
                alphas=alphas,
            )

        print(
            f"Reusing MAIN-selected alpha only: feature={feature_name}, "
            f"target={target_name}, alpha={alpha:g}"
        )
        audit = pd.DataFrame(
            [{
                "target": target_name,
                "fold": -1,
                "alpha": float(alpha),
                "alpha_source": "MAIN_selected_reused_for_SECONDARY_refit",
            }]
        )
        return float(alpha), audit

    norm.run_feature_set_from_df = wrapped_runner
    norm.run_cv_for_target = fixed_alpha_cv


# =============================================================================
# NORMATIVE REFIT
# =============================================================================

def run_secondary_normative_refit(
    norm: Any,
    diag_file: Path,
    out_dir: Path,
    reuse_main_alpha: bool,
    main_normative_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Core SECONDARY switch.
    norm.HEALTHY_COL = SECONDARY_HC
    norm.DIAG_FILE = str(diag_file)
    norm.OUT_DIR = str(out_dir)
    os.makedirs(norm.OUT_DIR, exist_ok=True)

    if reuse_main_alpha:
        alpha_map = load_main_alpha_map(main_normative_dir)
        install_main_alpha_reuse(norm, alpha_map)

    print("\n" + "=" * 88)
    print("SECONDARY NORMATIVE REFIT")
    print("=" * 88)
    print(f"HEALTHY_COL:          {norm.HEALTHY_COL}")
    print(f"Diagnosis file:       {norm.DIAG_FILE}")
    print(f"Normative output dir: {norm.OUT_DIR}")
    print(
        "Alpha strategy:       "
        + ("reuse MAIN-selected alpha only" if reuse_main_alpha else "reselect by SECONDARY-HC CV")
    )
    print(
        "Refitted parameters:  feature residualizer, scaler, predictive coefficients, "
        "bias correction, gap standardization"
    )

    shared_hc_split_map = norm.make_global_hc_split_map(norm.DIAG_FILE)

    for feature_name, feature_file in norm.FEATURE_FILES.items():
        norm.run_feature_set(
            feature_name=feature_name,
            feature_file=feature_file,
            diag_file=norm.DIAG_FILE,
            out_dir=norm.OUT_DIR,
            hc_split_map=shared_hc_split_map,
        )

def patch_figure_module_for_secondary(
    fig: Any,
    diag_file: Path,
    secondary_normative_dir: Path,
    secondary_results_dir: Path,
) -> None:
    secondary_results_dir.mkdir(parents=True, exist_ok=True)

    fig.CSV_DIAGNOSES = str(diag_file)
    fig.INPUT_FILES = {
        "freesurfer": str(secondary_normative_dir / "normative_scores_freesurfer.csv"),
        "connectivity": str(secondary_normative_dir / "normative_scores_connectivity.csv"),
    }

    fig.HC_COL = SECONDARY_HC
    fig.AUD_COL = SECONDARY_AUD
    fig.MDD_COL = SECONDARY_MDD
    fig.ANX_COL = SECONDARY_ANX
    fig.DIAGNOSIS_COLUMNS_FOR_PLOTTING = [
        fig.HC_COL,
        fig.AUD_COL,
        fig.MDD_COL,
        fig.ANX_COL,
    ]
    fig.GROUP_ORDER_FULL = [fig.AUD_COL, fig.MDD_COL, fig.ANX_COL]

    fig.RESULTS_DIR = str(secondary_results_dir)


def audit_secondary_groups(fig: Any, subject_md: pd.DataFrame) -> None:
    split = subject_md[fig.SPLIT_GROUP_COL].astype(str).str.strip()
    print("\nSECONDARY group audit")
    print("=" * 88)
    print(
        f"heldout SECONDARY_HC: "
        f"{int((split.eq('heldout_hc') & fig.binary_col_positive(subject_md, SECONDARY_HC)).sum()):,}"
    )
    print(f"SECONDARY_AUD:        {int(fig.binary_col_positive(subject_md, SECONDARY_AUD).sum()):,}")
    print(f"SECONDARY_MDD:        {int(fig.binary_col_positive(subject_md, SECONDARY_MDD).sum()):,}")
    print(f"SECONDARY_ANX:        {int(fig.binary_col_positive(subject_md, SECONDARY_ANX).sum()):,}")
    print(
        "MDD+ANX overlap:      "
        f"{int((fig.binary_col_positive(subject_md, SECONDARY_MDD) & fig.binary_col_positive(subject_md, SECONDARY_ANX)).sum()):,}"
    )


def run_secondary_associations(
    fig: Any,
    secondary_results_dir: Path,
    main_results_dir: Path,
) -> pd.DataFrame:
    print("\n" + "=" * 88)
    print("SECONDARY PRIMARY DOWNSTREAM ASSOCIATIONS")
    print("=" * 88)

    first_gap_path = next(iter(fig.INPUT_FILES.values()))
    subject_md = pd.read_csv(first_gap_path, usecols=[fig.ID_COL])
    subject_md[fig.ID_COL] = fig.canonicalize_id_series(subject_md[fig.ID_COL])
    subject_md = subject_md.drop_duplicates(subset=[fig.ID_COL])
    subject_md = fig.add_diagnosis_columns(subject_md)
    subject_md = fig.add_mean_motion_covariate(subject_md)
    subject_md = fig.apply_plotting_diagnosis_labels(subject_md)
    audit_secondary_groups(fig, subject_md)

    gap_df, gap_cols = fig.load_gap_space(subject_md)
    graph_cols = fig.graph_gap_cols(gap_cols)

    diagnosis_all = fig.fit_standard_diagnosis_gap_edges(
        gap_df=gap_df,
        gap_cols=graph_cols,
    )
    score_all = fig.fit_standard_score_gap_edges(
        gap_df=gap_df,
        gap_cols=graph_cols,
    )

    diagnosis_path = secondary_results_dir / "secondary_diagnosis_gap_edges_all.csv"
    score_path = secondary_results_dir / "secondary_score_gap_edges_all.csv"
    diagnosis_all.to_csv(diagnosis_path, index=False)
    score_all.to_csv(score_path, index=False)

    primary_sig = fig.collect_predictor_gap_edges(
        diagnosis_edges=diagnosis_all,
        score_edges=score_all,
    )
    primary_path = secondary_results_dir / "secondary_predictor_gap_edges_primary_fdr.csv"
    primary_sig.to_csv(primary_path, index=False)

    comparison = compare_main_edges_to_secondary(
        main_results_dir=main_results_dir,
        diagnosis_all=diagnosis_all,
        score_all=score_all,
    )
    comparison_path = secondary_results_dir / "secondary_vs_main_primary_edges.csv"
    comparison.to_csv(comparison_path, index=False)

    print("\nSaved SECONDARY sensitivity outputs")
    print("=" * 88)
    for p in [diagnosis_path, score_path, primary_path, comparison_path]:
        print(p)

    return comparison

def find_main_edge_table(main_results_dir: Path) -> Path:
    for name in MAIN_EDGE_TABLE_CANDIDATES:
        p = main_results_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not locate a MAIN significant-edge table in:\n"
        f"{main_results_dir}\nTried: {MAIN_EDGE_TABLE_CANDIDATES}"
    )


def compare_main_edges_to_secondary(
    main_results_dir: Path,
    diagnosis_all: pd.DataFrame,
    score_all: pd.DataFrame,
) -> pd.DataFrame:
    """Match every primary MAIN edge to its corresponding SECONDARY primary result."""
    main_path = find_main_edge_table(main_results_dir)
    main = pd.read_csv(main_path).copy()

    frames = [x for x in [diagnosis_all, score_all] if x is not None and len(x) > 0]
    all_secondary = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    keep_secondary = [
        c for c in [
            "source", "predictor", "gap", "beta_std", "se_hc3", "t_hc3",
            "p_hc3", "q_fdr_bh", "n", "n_case", "n_ref", "score_predictor"
        ]
        if c in all_secondary.columns
    ]
    sec = all_secondary[keep_secondary].copy()
    sec = sec.rename(columns={
        "beta_std": "beta_std_SECONDARY",
        "se_hc3": "se_hc3_SECONDARY",
        "t_hc3": "t_hc3_SECONDARY",
        "p_hc3": "p_hc3_SECONDARY",
        "q_fdr_bh": "q_fdr_bh_SECONDARY",
        "n": "n_SECONDARY",
        "n_case": "n_case_SECONDARY",
        "n_ref": "n_ref_SECONDARY",
    })

    out = main.copy()
    out = out.rename(columns={
        "beta_std": "beta_std_MAIN",
        "p_hc3": "p_hc3_MAIN",
        "q_fdr_bh": "q_fdr_bh_MAIN",
    })
    out = out.merge(sec, on=["source", "predictor", "gap"], how="left")

    out["tested_in_SECONDARY"] = pd.to_numeric(
        out.get("beta_std_SECONDARY"), errors="coerce"
    ).notna()
    out["SECONDARY_primary_fdr05"] = pd.to_numeric(
        out.get("q_fdr_bh_SECONDARY"), errors="coerce"
    ).lt(0.05)
    out["SECONDARY_same_sign_as_MAIN"] = (
        np.sign(pd.to_numeric(out["beta_std_MAIN"], errors="coerce"))
        == np.sign(pd.to_numeric(out["beta_std_SECONDARY"], errors="coerce"))
    )
    out.loc[~out["tested_in_SECONDARY"], "SECONDARY_same_sign_as_MAIN"] = np.nan
    out["SECONDARY_abs_beta_retained_pct"] = (
        pd.to_numeric(out["beta_std_SECONDARY"], errors="coerce").abs()
        / pd.to_numeric(out["beta_std_MAIN"], errors="coerce").abs()
        * 100.0
    )
    out["MAIN_primary_edge"] = True

    preferred = [
        "source", "predictor", "gap",
        "MAIN_primary_edge",
        "beta_std_MAIN", "p_hc3_MAIN", "q_fdr_bh_MAIN",
        "tested_in_SECONDARY",
        "beta_std_SECONDARY", "p_hc3_SECONDARY", "q_fdr_bh_SECONDARY",
        "SECONDARY_primary_fdr05",
        "SECONDARY_same_sign_as_MAIN",
        "SECONDARY_abs_beta_retained_pct",
        "n_SECONDARY", "n_case_SECONDARY", "n_ref_SECONDARY",
    ]
    cols = [c for c in preferred if c in out.columns]
    cols += [c for c in out.columns if c not in cols]
    return out[cols].sort_values(["source", "predictor", "gap"], kind="stable")


def write_text_summary(comparison: pd.DataFrame, output_dir: Path) -> None:
    path = output_dir / "SECONDARY_sensitivity_summary.txt"
    tested = comparison[comparison["tested_in_SECONDARY"].fillna(False)].copy()
    n_main = len(comparison)
    n_tested = len(tested)
    n_fdr = int(tested["SECONDARY_primary_fdr05"].fillna(False).sum())
    n_same = int(tested["SECONDARY_same_sign_as_MAIN"].fillna(False).sum())

    lines = [
        "SECONDARY-definition sensitivity summary",
        "=" * 80,
        f"MAIN primary edges: {n_main}",
        f"MAIN primary edges tested in SECONDARY: {n_tested}",
        f"Same sign in SECONDARY: {n_same}/{n_tested if n_tested else 0}",
        f"SECONDARY primary FDR q<0.05: {n_fdr}/{n_tested if n_tested else 0}",
        "",
        "Definition:",
        "  normative training reference = SECONDARY_HC",
        "  diagnosis contrasts = SECONDARY_AUD/MDD/ANX vs heldout SECONDARY_HC",
        "  MDD/ANX overlap allowed",
        "  no additional all-other-gaps sensitivity is run inside this analysis",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


def main() -> None:
    args = parse_args()

    normative_script = resolve_script(
        args.normative_base_script,
        [
            "1_gap_calculation_final_trail_sleep_binary_framewise_gripmean(4).py",
            "1_gap_calculation_final_trail_sleep_binary_framewise_gripmean.py",
        ],
    )
    figure_script = resolve_script(
        args.figure_base_script,
        [
            "2_EXPEXPEXP_MAIN_groups_INTERNAL_all_other_correction.py",
            "2_EXPEXPEXP_MAIN_groups_INTERNAL_all_other_correction(1).py",
        ],
    )

    diag_file = args.diag_file.expanduser().resolve()
    main_normative_dir = args.main_normative_dir.expanduser().resolve()
    secondary_normative_dir = args.secondary_normative_dir.expanduser().resolve()
    main_results_dir = args.main_results_dir.expanduser().resolve()
    secondary_results_dir = args.secondary_results_dir.expanduser().resolve()

    secondary_normative_dir.mkdir(parents=True, exist_ok=True)
    secondary_results_dir.mkdir(parents=True, exist_ok=True)

    print("3_3 SECONDARY-definition sensitivity analysis")
    print("=" * 88)
    print(f"Normative base:       {normative_script}")
    print(f"Figure/assoc base:    {figure_script}")
    print(f"Diagnosis file:       {diag_file}")
    print(f"SECONDARY norm dir:   {secondary_normative_dir}")
    print(f"SECONDARY result dir: {secondary_results_dir}")
    print(f"Normative HC:         {SECONDARY_HC}")
    print(f"Case groups:          {SECONDARY_AUD}, {SECONDARY_MDD}, {SECONDARY_ANX}")
    print(
        "Alpha:                "
        + ("SECONDARY CV re-selection" if args.reselect_alpha else "reuse MAIN-selected alpha only")
    )

    norm = import_module_from_path(normative_script, "secondary_normative_base")

    if not args.skip_normative_refit:
        run_secondary_normative_refit(
            norm=norm,
            diag_file=diag_file,
            out_dir=secondary_normative_dir,
            reuse_main_alpha=not args.reselect_alpha,
            main_normative_dir=main_normative_dir,
        )
    else:
        print("\n[INFO] Skipping normative refit by explicit request.")

    fig = import_module_from_path(figure_script, "secondary_association_base")
    patch_figure_module_for_secondary(
        fig=fig,
        diag_file=diag_file,
        secondary_normative_dir=secondary_normative_dir,
        secondary_results_dir=secondary_results_dir,
    )

    install_sleep_gap_filter(fig)

    comparison = run_secondary_associations(
        fig=fig,
        secondary_results_dir=secondary_results_dir,
        main_results_dir=main_results_dir,
    )
    write_text_summary(comparison, secondary_results_dir)

    print("\nSECONDARY primary-definition sensitivity complete.")


if __name__ == "__main__":
    main()

