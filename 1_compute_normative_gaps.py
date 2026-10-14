#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Normative modeling pipeline for two UKB brain feature sets:
1) FreeSurfer features
2) Connectivity-network features
"""

import os
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

RANDOM_STATE = 42
N_SPLITS = 5
HC_TRAIN_FRACTION = 0.5

ALPHAS = [
    0.0001,
    0.001,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
    2000.0,
]

ID_COL = "Participant ID"
FEATURE_ID_COL = "Participant_ID"
HEALTHY_COL = "MAIN_HC"

AGE_COL = "age_assessment_center_2"
SEX_COL = "sex"
AGE2_CONFOUND_COL = "age_assessment_center_2_centered_sq"
AGE_SEX_CONFOUND_COL = "age_assessment_center_2_centered_x_sex"
MEAN_MOTION_ID_COL = "eid"
MEAN_MOTION_VALUE_COL = "framewise_dis"
MEAN_MOTION_CONFOUND_COL = "mean_motion"
CSV_MEAN_MOTION = ...
FRAMEWISE_ID_COL = "eid"
FRAMEWISE_VALUE_COL = "framewise_dis"
FRAMEWISE_EXCLUDE_THRESHOLD = 0.5
CSV_FRAMEWISE = ...
HANDGRIP_RIGHT_ID_COL = "eid"
CSV_HANDGRIP_RIGHT = ...
HEIGHT_ID_COL = "eid"
CSV_HEIGHT = ...

FEATURE_FILES = {
    "freesurfer": ...,
    "connectivity": ...,
}

DIAG_FILE = ...

OUT_DIR = ...
os.makedirs(OUT_DIR, exist_ok=True)

BASE_TARGETS = {

"diastolic_bp": {
    "target_col": "diastolic_bp",
    "task": "regression",
    "confounds": ["site", "sex", "age_assessment_center_2", "etiv"],
},
    "brain_age": {
        "target_col": "age_assessment_center_2",
        "task": "regression",
        "confounds": ["site", "sex", "etiv"],
    },
    "Sleep_duration": {
        "target_col": "Sleep_duration",
        "task": "regression",
        "confounds": ["site", "age_assessment_center_2", "sex", "etiv"],
    },
    "Fluid_intelligence_score": {
        "target_col": "Fluid_intelligence_score",
        "task": "regression",
        "confounds": ["site", "age_assessment_center_2", "sex", "etiv"],
    },
    "bmi": {
        "target_col": "bmi",
        "task": "regression",
        "confounds": ["site", "age_assessment_center_2", "sex", "etiv"],
    },
    "Hand_grip_strength_mean": {
        "target_col": "Hand_grip_strength_mean",
        "task": "regression",
        "confounds": ["site", "age_assessment_center_2", "sex", "height", "etiv"],
    },
    "trail_making": {
        "target_col": "duration_trail_time",
        "task": "regression",
        "confounds": ["site", "age_assessment_center_2", "sex", "etiv"]
    }
}


# =============================================================================
# UTILITIES
# =============================================================================

def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def harmonize_id_column(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    df = df.copy()
    df[id_col] = df[id_col].astype(str).str.strip()
    return df


def add_nonlinear_age_terms_to_confounds(confounds: List[str]) -> List[str]:
    """
    Add centered age^2 and centered-age-by-sex terms only when age is already
    part of the covariate set.
    """
    out = list(confounds)

    if AGE_COL not in out:
        return out

    insert_at = out.index(AGE_COL) + 1

    if AGE2_CONFOUND_COL not in out:
        out.insert(insert_at, AGE2_CONFOUND_COL)
        insert_at += 1

    if SEX_COL in out and AGE_SEX_CONFOUND_COL not in out:
        out.insert(insert_at, AGE_SEX_CONFOUND_COL)

    return out


def make_targets_for_feature_set(feature_name: str) -> Dict:
    """
    Add nonlinear age terms where age is already a confound, and add mean_motion as a confound only for
    connectivity models.
    """
    targets = json.loads(json.dumps(BASE_TARGETS))

    for target_name, spec in targets.items():
        if spec.get("add_nonlinear_age_terms", True):
            spec["confounds"] = add_nonlinear_age_terms_to_confounds(spec["confounds"])

    if feature_name == "connectivity":
        for target_name, spec in targets.items():
            if MEAN_MOTION_CONFOUND_COL not in spec["confounds"]:
                spec["confounds"].append(MEAN_MOTION_CONFOUND_COL)

    return targets


def get_base_columns_required_for_confounds(confound_cols: List[str]) -> List[str]:
    required = []

    for col in confound_cols:
        if col == AGE2_CONFOUND_COL:
            required.append(AGE_COL)
        elif col == AGE_SEX_CONFOUND_COL:
            required.extend([AGE_COL, SEX_COL])
        else:
            required.append(col)

    return list(dict.fromkeys(required))


class DerivedConfoundPreprocessor:
    """
    Preprocess target-specific covariates, including centered nonlinear age terms.

    The input dataframe should contain the raw covariates. Derived terms are
    computed internally:
        age_assessment_center_2_centered_sq
            = (age - mean_train_age)^2
        age_assessment_center_2_centered_x_sex
            = (age - mean_train_age) * sex

    """

    def __init__(self, confound_cols: List[str]):
        self.confound_cols = list(confound_cols)
        self.categorical_cols = [c for c in self.confound_cols if c == "site"]
        self.numeric_cols = [c for c in self.confound_cols if c != "site"]
        self.encoder = None
        self.scaler = None
        self.age_center_ = None

    def _check_required_columns(self, df: pd.DataFrame):
        required = get_base_columns_required_for_confounds(self.confound_cols)
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                "Missing required raw columns for confound preprocessing: "
                f"{missing}. Requested confounds: {self.confound_cols}"
            )

    def _needs_centered_age(self) -> bool:
        return (
            AGE2_CONFOUND_COL in self.numeric_cols
            or AGE_SEX_CONFOUND_COL in self.numeric_cols
        )

    def _fit_age_center(self, df: pd.DataFrame):
        if self._needs_centered_age():
            age = safe_numeric(df[AGE_COL]).values.astype(float)
            self.age_center_ = float(np.nanmean(age))
            if not np.isfinite(self.age_center_):
                raise ValueError("Cannot compute centered age terms because age mean is non-finite.")
        else:
            self.age_center_ = None

    def _numeric_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        data = {}

        for col in self.numeric_cols:
            if col == AGE2_CONFOUND_COL:
                age_centered = safe_numeric(df[AGE_COL]).values.astype(float) - self.age_center_
                data[col] = age_centered ** 2
            elif col == AGE_SEX_CONFOUND_COL:
                age_centered = safe_numeric(df[AGE_COL]).values.astype(float) - self.age_center_
                sex = safe_numeric(df[SEX_COL]).values.astype(float)
                data[col] = age_centered * sex
            else:
                data[col] = safe_numeric(df[col]).values.astype(float)

        return pd.DataFrame(data, index=df.index)

    def fit(self, df: pd.DataFrame):
        self._check_required_columns(df)
        self._fit_age_center(df)

        if self.categorical_cols:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.encoder.fit(df[self.categorical_cols])

        if self.numeric_cols:
            self.scaler = StandardScaler()
            self.scaler.fit(self._numeric_dataframe(df))

        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        self._check_required_columns(df)

        chunks = []

        if self.categorical_cols:
            chunks.append(self.encoder.transform(df[self.categorical_cols]))

        if self.numeric_cols:
            chunks.append(self.scaler.transform(self._numeric_dataframe(df)))

        if chunks:
            return np.hstack(chunks)

        return np.empty((len(df), 0), dtype=float)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)


def build_confound_preprocessor(confound_cols: List[str]) -> DerivedConfoundPreprocessor:
    return DerivedConfoundPreprocessor(confound_cols=confound_cols)


def feature_scaler_fit_transform(X_train: np.ndarray, X_test: np.ndarray):
    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)
    X_test_z = scaler.transform(X_test)
    return scaler, X_train_z, X_test_z


def read_feature_table(feature_file: str) -> pd.DataFrame:
    if feature_file.endswith(".csv"):
        feat = pd.read_csv(feature_file)
    elif feature_file.endswith(".tsv") or feature_file.endswith(".txt"):
        feat = pd.read_csv(feature_file, sep="\t")
        if feat.shape[1] == 1:
            feat = pd.read_csv(feature_file, sep=r"\s+", engine="python")
    else:
        feat = pd.read_csv(feature_file)
        if feat.shape[1] == 1:
            feat = pd.read_csv(feature_file, sep=r"\s+", engine="python")

    if FEATURE_ID_COL in feat.columns and ID_COL not in feat.columns:
        feat = feat.rename(columns={FEATURE_ID_COL: ID_COL})

    if ID_COL not in feat.columns:
        raise ValueError(
            f"Could not find ID column '{ID_COL}' or '{FEATURE_ID_COL}' in feature file: {feature_file}\n"
            f"Columns found first 10: {list(feat.columns[:10])}"
        )

    feat = harmonize_id_column(feat, ID_COL)
    return feat


def read_diag_table(diag_file: str) -> pd.DataFrame:
    diag = pd.read_csv(diag_file)

    if ID_COL not in diag.columns:
        raise ValueError(
            f"Could not find ID column '{ID_COL}' in diagnosis file: {diag_file}\n"
            f"Columns found first 10: {list(diag.columns[:10])}"
        )

    if SLEEP_DURATION_COL not in diag.columns:
        raise ValueError(
            f"Could not find continuous sleep-duration column '{SLEEP_DURATION_COL}' "
            f"in diagnosis file: {diag_file}"
        )

    diag = harmonize_id_column(diag, ID_COL)
    return diag


def make_global_hc_split_map(diag_file: str) -> Dict[str, str]:

    diag = read_diag_table(diag_file)

    if HEALTHY_COL not in diag.columns:
        raise ValueError(
            f"Diagnosis file is missing healthy-control column '{HEALTHY_COL}'."
        )

    diag[HEALTHY_COL] = safe_numeric(diag[HEALTHY_COL])

    if diag[ID_COL].duplicated().any():
        n_dup = int(diag[ID_COL].duplicated().sum())
        raise ValueError(
            f"Diagnosis table contains {n_dup:,} duplicated Participant IDs; "
            "cannot build a one-to-one global HC split."
        )

    keep_cols = [ID_COL]
    if SEX_COL in diag.columns:
        keep_cols.append(SEX_COL)

    hc = diag.loc[diag[HEALTHY_COL].eq(1), keep_cols].copy()

    if len(hc) < 2:
        raise ValueError(f"Need at least 2 global HCs to split, got n={len(hc)}")

    stratify = None
    if SEX_COL in hc.columns:
        hc[SEX_COL] = safe_numeric(hc[SEX_COL])
        sex = hc[SEX_COL]
        sex_counts = sex.value_counts(dropna=False)
        sex_ok = sex.notna().all()

        if sex_ok and sex.nunique(dropna=True) > 1 and sex_counts.min() >= 2:
            stratify = sex
        else:
            print(
                "[WARN] Global HC split is not stratified by sex because sex is "
                "missing or at least one stratum has <2 subjects."
            )

    hc_train, hc_holdout = train_test_split(
        hc,
        train_size=HC_TRAIN_FRACTION,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    split_map = {
        **{pid: "train_hc" for pid in hc_train[ID_COL].tolist()},
        **{pid: "heldout_hc" for pid in hc_holdout[ID_COL].tolist()},
    }

    split_table = pd.DataFrame({
        ID_COL: list(split_map.keys()),
        "split_group": list(split_map.values()),
    })
    split_table = split_table.sort_values(ID_COL).reset_index(drop=True)

    split_csv = os.path.join(OUT_DIR, "global_hc_split_assignments.csv")
    split_table.to_csv(split_csv, index=False)

    print("\n" + "=" * 80)
    print("GLOBAL HC SPLIT — SHARED ACROSS FREESURFER AND CONNECTIVITY")
    print("=" * 80)
    print(f"Global HC total:    {len(hc):,}")
    print(f"Global train_hc:    {len(hc_train):,}")
    print(f"Global heldout_hc:  {len(hc_holdout):,}")
    print(f"Saved split map:    {split_csv}")

    return split_map


def read_mean_motion_table(mean_motion_file: str) -> pd.DataFrame:
    mm = pd.read_csv(mean_motion_file)
    mm = mm[[MEAN_MOTION_ID_COL, MEAN_MOTION_VALUE_COL]].copy()
    mm = mm.rename(
        columns={
            MEAN_MOTION_ID_COL: ID_COL,
            MEAN_MOTION_VALUE_COL: MEAN_MOTION_CONFOUND_COL,
        }
    )

    mm = harmonize_id_column(mm, ID_COL)
    mm[MEAN_MOTION_CONFOUND_COL] = safe_numeric(mm[MEAN_MOTION_CONFOUND_COL])

    mm = mm.dropna(subset=[ID_COL, MEAN_MOTION_CONFOUND_COL])
    mm = mm.drop_duplicates(subset=[ID_COL], keep="first")

    return mm


def merge_motion(df: pd.DataFrame, add_mean_motion: bool) -> pd.DataFrame:
    if not add_mean_motion:
        return df

    mm = read_mean_motion_table(CSV_MEAN_MOTION)
    df = df.merge(mm, on=ID_COL, how="inner")

    df[MEAN_MOTION_CONFOUND_COL] = safe_numeric(df[MEAN_MOTION_CONFOUND_COL])
    return df


def read_framewise_table(framewise_file: str) -> pd.DataFrame:
    fw = pd.read_csv(framewise_file)

    fw = fw[[FRAMEWISE_ID_COL, FRAMEWISE_VALUE_COL]].copy()
    fw = fw.rename(columns={FRAMEWISE_ID_COL: ID_COL})
    fw = harmonize_id_column(fw, ID_COL)
    fw[FRAMEWISE_VALUE_COL] = safe_numeric(fw[FRAMEWISE_VALUE_COL])
    fw = fw.dropna(subset=[ID_COL])
    fw = fw.drop_duplicates(subset=[ID_COL], keep="first")
    return fw


def exclude_high_framewise_motion(df: pd.DataFrame, apply_exclusion: bool) -> pd.DataFrame:
    if not apply_exclusion:
        return df

    fw = read_framewise_table(CSV_FRAMEWISE)
    bad_ids = set(
        fw.loc[fw[FRAMEWISE_VALUE_COL] > FRAMEWISE_EXCLUDE_THRESHOLD, ID_COL].tolist()
    )

    before = len(df)
    df = df.loc[~df[ID_COL].isin(bad_ids)].copy()
    print(
        f"[INFO] framewise exclusion > {FRAMEWISE_EXCLUDE_THRESHOLD} mm: "
        f"{before:,} -> {len(df):,} rows; excluded {before - len(df):,}"
    )
    return df


def read_handgrip_right_table(handgrip_file: str) -> pd.DataFrame:
    grip = pd.read_csv(handgrip_file)

    grip = grip[[HANDGRIP_RIGHT_ID_COL, HANDGRIP_RIGHT_VALUE_COL]].copy()
    grip = grip.rename(columns={HANDGRIP_RIGHT_ID_COL: ID_COL})
    grip = harmonize_id_column(grip, ID_COL)
    grip[HANDGRIP_RIGHT_VALUE_COL] = safe_numeric(grip[HANDGRIP_RIGHT_VALUE_COL])
    grip = grip.dropna(subset=[ID_COL])
    grip = grip.drop_duplicates(subset=[ID_COL], keep="first")
    return grip


def merge_and_make_handgrip_mean(df: pd.DataFrame) -> pd.DataFrame:

    grip_right = read_handgrip_right_table(CSV_HANDGRIP_RIGHT)

    df = df.merge(grip_right, on=ID_COL, how="left")
    df[HANDGRIP_LEFT_VALUE_COL] = safe_numeric(df[HANDGRIP_LEFT_VALUE_COL])
    df[HANDGRIP_RIGHT_VALUE_COL] = safe_numeric(df[HANDGRIP_RIGHT_VALUE_COL])
    df[HANDGRIP_MEAN_COL] = (
        df[HANDGRIP_LEFT_VALUE_COL] + df[HANDGRIP_RIGHT_VALUE_COL]
    ) / 2.0

    return df


def read_height_table(height_file: str) -> pd.DataFrame:
    height = pd.read_csv(height_file)

    height = height[[HEIGHT_ID_COL, HEIGHT_VALUE_COL]].copy()
    height = height.rename(
        columns={
            HEIGHT_ID_COL: ID_COL,
            HEIGHT_VALUE_COL: HEIGHT_CONFOUND_COL,
        }
    )

    height = harmonize_id_column(height, ID_COL)
    height[HEIGHT_CONFOUND_COL] = safe_numeric(height[HEIGHT_CONFOUND_COL])

    height = height.dropna(subset=[ID_COL])
    height = height.drop_duplicates(subset=[ID_COL], keep="first")

    return height


def merge_height(df: pd.DataFrame, add_height: bool = True) -> pd.DataFrame:
    height = read_height_table(CSV_HEIGHT)

    if HEIGHT_CONFOUND_COL in df.columns:
        df = df.drop(columns=[HEIGHT_CONFOUND_COL])

    df = df.merge(height, on=ID_COL, how="left")

    df[HEIGHT_CONFOUND_COL] = safe_numeric(df[HEIGHT_CONFOUND_COL])
    return df


# =============================================================================
# FEATURE DECONFOUNDER / BIAS CORRECTION / Z-SCORING
# =============================================================================

@dataclass
class FeatureResidualizer:

    confound_cols: List[str]
    preprocessor: Optional[DerivedConfoundPreprocessor] = None
    model: Optional[LinearRegression] = None

    def fit(self, df: pd.DataFrame, X: np.ndarray):
        self.preprocessor = build_confound_preprocessor(self.confound_cols)
        Xc = self.preprocessor.fit_transform(df)

        self.model = LinearRegression()
        self.model.fit(Xc, X)
        return self

    def predict_confound_part(self, df: pd.DataFrame) -> np.ndarray:
        Xc = self.preprocessor.transform(df)
        return self.model.predict(Xc)

    def residualize(self, df: pd.DataFrame, X: np.ndarray) -> np.ndarray:
        return X - self.predict_confound_part(df)


@dataclass
class BiasCorrector:
    model: Optional[LinearRegression] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model = LinearRegression()
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def correct(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return y - self.predict(X)


@dataclass
class ZScorer:
    mean_: float = 0.0
    std_: float = 1.0

    def fit(self, x: np.ndarray):
        self.mean_ = float(np.nanmean(x))
        self.std_ = float(np.nanstd(x, ddof=0))

        if not np.isfinite(self.std_) or self.std_ == 0:
            self.std_ = 1.0

        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

def get_feature_columns(feature_df: pd.DataFrame) -> List[str]:
    return [c for c in feature_df.columns if c != ID_COL]


def get_required_columns(targets: Dict) -> List[str]:
    required_cols = [ID_COL, HEALTHY_COL]

    for spec in targets.values():
        required_cols.append(spec["target_col"])
        required_cols.extend(get_base_columns_required_for_confounds(spec["confounds"]))

    required_cols = list(dict.fromkeys(required_cols))
    return required_cols


def prepare_dataframe_single(
    feature_file: str,
    diag_file: str,
    targets: Dict,
    add_mean_motion: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    feat = read_feature_table(feature_file)
    diag = read_diag_table(diag_file)

    df = feat.merge(diag, on=ID_COL, how="inner")
    df = merge_motion(df, add_mean_motion=add_mean_motion)
    df = exclude_high_framewise_motion(df, apply_exclusion=add_mean_motion)
    df = merge_and_make_handgrip_mean(df)
    df = merge_height(df, add_height=True)

    feature_cols = get_feature_columns(feat)

    required_cols = get_required_columns(targets)

    missing_required = [c for c in required_cols if c not in df.columns]
    if missing_required:
        raise ValueError(
            "Missing required columns after merge:\n"
            f"{missing_required}\n"
            f"Feature file: {feature_file}\n"
            f"Diagnosis file: {diag_file}\n"
            f"Mean motion added: {add_mean_motion}"
        )

    for col in required_cols:
        if col in df.columns and col not in [ID_COL, "site"]:
            df[col] = safe_numeric(df[col])

    for col in feature_cols:
        df[col] = safe_numeric(df[col])

    before_feature_drop = len(df)
    df = df.dropna(subset=feature_cols)
    print(
        f"[INFO] feature-complete filter: {before_feature_drop:,} -> {len(df):,} "
        f"rows for {os.path.basename(feature_file)}"
    )

    before_health_drop = len(df)
    df = df.dropna(subset=[HEALTHY_COL]).reset_index(drop=True)
    print(
        f"[INFO] healthy-label filter only: {before_health_drop:,} -> {len(df):,} rows"
    )

    return df, feature_cols


def split_hc_cases(
    df: pd.DataFrame,
    hc_split_map: Dict[str, str],
):

    hc = df[df[HEALTHY_COL] == 1].copy()
    cases = df[df[HEALTHY_COL] != 1].copy()

    if len(hc) < 2:
        raise ValueError(f"Need at least 2 healthy controls, got n={len(hc)}")

    hc["_shared_split_group"] = hc[ID_COL].map(hc_split_map)

    missing = hc["_shared_split_group"].isna()
    if missing.any():
        missing_ids = hc.loc[missing, ID_COL].head(10).tolist()
        raise ValueError(
            f"{int(missing.sum()):,} HCs in this imaging dataframe have no global "
            f"split assignment. Example IDs: {missing_ids}"
        )

    hc_train = (
        hc.loc[hc["_shared_split_group"].eq("train_hc")]
        .drop(columns=["_shared_split_group"])
        .reset_index(drop=True)
    )

    hc_holdout = (
        hc.loc[hc["_shared_split_group"].eq("heldout_hc")]
        .drop(columns=["_shared_split_group"])
        .reset_index(drop=True)
    )

    cases = cases.reset_index(drop=True)

    if len(hc_train) == 0 or len(hc_holdout) == 0:
        raise ValueError(
            "Global HC split produced an empty train or heldout subset in this modality."
        )

    return hc_train, hc_holdout, cases


def get_current_target_required_columns(spec: Dict) -> List[str]:
    cols = [spec["target_col"], *get_base_columns_required_for_confounds(spec["confounds"])]
    return list(dict.fromkeys(cols))


def dropna_for_current_target(
    df: pd.DataFrame,
    target_name: str,
    spec: Dict,
    stage_label: str,
) -> pd.DataFrame:

    needed = get_current_target_required_columns(spec)
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Target {target_name!r}, stage {stage_label!r}: missing required columns: "
            f"{missing_cols}"
        )

    before = len(df)
    out = df.dropna(subset=needed).reset_index(drop=True)
    dropped = before - len(out)
    print(
        f"    {stage_label:11s}: {before:8,d} -> {len(out):8,d} "
        f"after current-target dropna; dropped {dropped:8,d}"
    )
    return out


def make_empty_regression_prediction(target_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            ID_COL,
            f"{target_name}_true",
            f"{target_name}_pred_raw",
            f"{target_name}_gap_raw",
            f"{target_name}_pred_bias_corrected",
            f"{target_name}_gap_bias_corrected",
            f"{target_name}_gap_z",
        ]
    )


def make_empty_classification_prediction(target_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            ID_COL,
            f"{target_name}_true",
            f"{target_name}_pred_prob_class1",
            f"{target_name}_pred_prob_sex1",
            f"{target_name}_pred_class",
            f"{target_name}_raw",
            f"{target_name}_bias_corrected",
            f"{target_name}_z",
        ]
    )


def assert_enough_current_target_data(
    target_name: str,
    task: str,
    target_col: str,
    hc_train: pd.DataFrame,
    hc_holdout: pd.DataFrame,
):
    if len(hc_train) < N_SPLITS:
        raise ValueError(
            f"Target {target_name!r}: only {len(hc_train)} complete train-HC rows, "
            f"but N_SPLITS={N_SPLITS}."
        )

    if len(hc_holdout) < 2:
        raise ValueError(
            f"Target {target_name!r}: only {len(hc_holdout)} complete heldout-HC rows. "
            "Need heldout HCs for z-scoring."
        )

    if task == "classification":
        y = hc_train[target_col].astype(int)
        counts = y.value_counts()
        if len(counts) < 2:
            raise ValueError(
                f"Target {target_name!r}: train-HC classification target has only one class. "
                f"Class counts: {counts.to_dict()}"
            )
        min_class_n = int(counts.min())
        if min_class_n < N_SPLITS:
            raise ValueError(
                f"Target {target_name!r}: smallest train-HC class has n={min_class_n}, "
                f"but N_SPLITS={N_SPLITS}. Class counts: {counts.to_dict()}"
            )

def get_cv_iterator(df_train_hc: pd.DataFrame, task: str, target_col: str):
    if task == "classification":
        y = df_train_hc[target_col].astype(int).values
        return StratifiedKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        ).split(df_train_hc, y)

    return KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    ).split(df_train_hc)


# =============================================================================
# CV SCORING
# =============================================================================

def score_regression_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    confound_cols: List[str],
    alpha: float,
) -> Dict:
    y_train = train_df[target_col].values.astype(float)
    y_val = val_df[target_col].values.astype(float)

    X_train = train_df[feature_cols].values.astype(float)
    X_val = val_df[feature_cols].values.astype(float)

    feat_resid = FeatureResidualizer(confound_cols=confound_cols).fit(train_df, X_train)

    X_train_resid = feat_resid.residualize(train_df, X_train)
    X_val_resid = feat_resid.residualize(val_df, X_val)

    _, X_train_z, X_val_z = feature_scaler_fit_transform(X_train_resid, X_val_resid)

    model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
    model.fit(X_train_z, y_train)

    pred_val = model.predict(X_val_z)

    return {
        "alpha": alpha,
        "val_mse": mean_squared_error(y_val, pred_val),
        "val_r2": r2_score(y_val, pred_val),
    }


def score_classification_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    confound_cols: List[str],
    alpha: float,
) -> Dict:
    y_train = train_df[target_col].values.astype(int)
    y_val = val_df[target_col].values.astype(int)

    X_train = train_df[feature_cols].values.astype(float)
    X_val = val_df[feature_cols].values.astype(float)

    feat_resid = FeatureResidualizer(confound_cols=confound_cols).fit(train_df, X_train)

    X_train_resid = feat_resid.residualize(train_df, X_train)
    X_val_resid = feat_resid.residualize(val_df, X_val)

    _, X_train_z, X_val_z = feature_scaler_fit_transform(X_train_resid, X_val_resid)

    C = 1.0 / alpha

    model = LogisticRegression(
        penalty="l2",
        C=C,
        solver="liblinear",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_z, y_train)

    prob_val = model.predict_proba(X_val_z)[:, 1]
    pred_val = (prob_val >= 0.5).astype(int)

    out = {
        "alpha": alpha,
        "val_bal_acc": balanced_accuracy_score(y_val, pred_val),
    }

    if len(np.unique(y_val)) > 1:
        out["val_auc"] = roc_auc_score(y_val, prob_val)
    else:
        out["val_auc"] = np.nan

    return out


def run_cv_for_target(
    df_train_hc: pd.DataFrame,
    feature_cols: List[str],
    target_name: str,
    spec: Dict,
    alphas: List[float],
) -> Tuple[float, pd.DataFrame]:
    task = spec["task"]
    target_col = spec["target_col"]
    confound_cols = spec["confounds"]

    rows = []
    cv_iter = get_cv_iterator(df_train_hc, task, target_col)

    for fold_idx, (tr_idx, va_idx) in enumerate(cv_iter, start=1):
        tr_df = df_train_hc.iloc[tr_idx].reset_index(drop=True)
        va_df = df_train_hc.iloc[va_idx].reset_index(drop=True)

        for alpha in alphas:
            if task == "regression":
                metrics = score_regression_fold(
                    train_df=tr_df,
                    val_df=va_df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    confound_cols=confound_cols,
                    alpha=alpha,
                )
            else:
                metrics = score_classification_fold(
                    train_df=tr_df,
                    val_df=va_df,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    confound_cols=confound_cols,
                    alpha=alpha,
                )

            metrics["target"] = target_name
            metrics["fold"] = fold_idx
            metrics["confounds"] = ",".join(confound_cols)
            rows.append(metrics)

    cv_df = pd.DataFrame(rows)

    if task == "regression":
        best_alpha = (
            cv_df.groupby("alpha", as_index=False)["val_mse"]
            .mean()
            .sort_values("val_mse", ascending=True)
            .iloc[0]["alpha"]
        )
    else:
        best_alpha = (
            cv_df.groupby("alpha", as_index=False)["val_bal_acc"]
            .mean()
            .sort_values("val_bal_acc", ascending=False)
            .iloc[0]["alpha"]
        )

    return float(best_alpha), cv_df


# =============================================================================
# FINAL MODEL FITTING
# =============================================================================

def fit_final_regression(
    df_train_hc: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    confound_cols: List[str],
    alpha: float,
):
    y = df_train_hc[target_col].values.astype(float)
    X = df_train_hc[feature_cols].values.astype(float)

    feat_resid = FeatureResidualizer(confound_cols=confound_cols).fit(df_train_hc, X)
    X_resid = feat_resid.residualize(df_train_hc, X)

    scaler = StandardScaler()
    X_z = scaler.fit_transform(X_resid)

    model = Ridge(alpha=alpha, random_state=RANDOM_STATE)
    model.fit(X_z, y)

    return {
        "feature_residualizer": feat_resid,
        "feature_scaler": scaler,
        "model": model,
    }


def fit_final_classification(
    df_train_hc: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    confound_cols: List[str],
    alpha: float,
):
    y = df_train_hc[target_col].values.astype(int)
    X = df_train_hc[feature_cols].values.astype(float)

    feat_resid = FeatureResidualizer(confound_cols=confound_cols).fit(df_train_hc, X)
    X_resid = feat_resid.residualize(df_train_hc, X)

    scaler = StandardScaler()
    X_z = scaler.fit_transform(X_resid)

    C = 1.0 / alpha

    model = LogisticRegression(
        penalty="l2",
        C=C,
        solver="liblinear",
        max_iter=5000,
        random_state=RANDOM_STATE,
    )
    model.fit(X_z, y)

    return {
        "feature_residualizer": feat_resid,
        "feature_scaler": scaler,
        "model": model,
    }


# =============================================================================
# APPLY MODELS
# =============================================================================

def apply_regression_model(
    fitted: Dict,
    df_apply: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    target_name: str,
) -> pd.DataFrame:
    y_true = df_apply[target_col].values.astype(float)
    X = df_apply[feature_cols].values.astype(float)

    X_resid = fitted["feature_residualizer"].residualize(df_apply, X)
    X_z = fitted["feature_scaler"].transform(X_resid)

    pred_full = fitted["model"].predict(X_z)
    gap_raw = pred_full - y_true

    return pd.DataFrame(
        {
            ID_COL: df_apply[ID_COL].values,
            f"{target_name}_true": y_true,
            f"{target_name}_pred_raw": pred_full,
            f"{target_name}_gap_raw": gap_raw,
        }
    )


def apply_classification_model(
    fitted: Dict,
    df_apply: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    target_name: str,
) -> pd.DataFrame:
    y_true = df_apply[target_col].values.astype(int)
    X = df_apply[feature_cols].values.astype(float)

    X_resid = fitted["feature_residualizer"].residualize(df_apply, X)
    X_z = fitted["feature_scaler"].transform(X_resid)

    prob_class1 = fitted["model"].predict_proba(X_z)[:, 1]
    pred_class = (prob_class1 >= 0.5).astype(int)

    atypicality_raw = np.where(y_true == 1, 1.0 - prob_class1, prob_class1)

    return pd.DataFrame(
        {
            ID_COL: df_apply[ID_COL].values,
            f"{target_name}_true": y_true,
            f"{target_name}_pred_prob_class1": prob_class1,
            f"{target_name}_pred_prob_sex1": prob_class1,
            f"{target_name}_pred_class": pred_class,
            f"{target_name}_raw": atypicality_raw,
        }
    )


# =============================================================================
# BIAS CORRECTION AND Z-SCORING
# =============================================================================

def fit_bias_correction_regression(
    pred_train: pd.DataFrame,
    target_name: str,
):
    y_true = pred_train[f"{target_name}_true"].values.astype(float).reshape(-1, 1)
    gap_raw = pred_train[f"{target_name}_gap_raw"].values.astype(float)

    bias = BiasCorrector().fit(y_true, gap_raw)
    return bias


def apply_bias_correction_regression(
    pred_df: pd.DataFrame,
    target_name: str,
    bias: BiasCorrector,
) -> pd.DataFrame:
    y_true = pred_df[f"{target_name}_true"].values.astype(float)
    gap_raw = pred_df[f"{target_name}_gap_raw"].values.astype(float)

    X_bias = y_true.reshape(-1, 1)
    gap_bc = bias.correct(X_bias, gap_raw)
    pred_bc = y_true + gap_bc

    pred_df[f"{target_name}_pred_bias_corrected"] = pred_bc
    pred_df[f"{target_name}_gap_bias_corrected"] = gap_bc

    return pred_df


def fit_zscorer_regression_on_holdout(
    pred_holdout: pd.DataFrame,
    target_name: str,
) -> ZScorer:
    z = ZScorer().fit(
        pred_holdout[f"{target_name}_gap_bias_corrected"].values.astype(float)
    )
    return z


def apply_zscoring_regression(
    pred_df: pd.DataFrame,
    target_name: str,
    z: ZScorer,
) -> pd.DataFrame:
    gap_bc = pred_df[f"{target_name}_gap_bias_corrected"].values.astype(float)
    pred_df[f"{target_name}_gap_z"] = z.transform(gap_bc)

    return pred_df


def fit_bias_correction_classification(
    pred_train: pd.DataFrame,
    df_train_hc: pd.DataFrame,
    confound_cols: List[str],
    target_name: str,
):
    bias_prep = build_confound_preprocessor(confound_cols)
    X_bias = bias_prep.fit_transform(df_train_hc)

    raw = pred_train[f"{target_name}_raw"].values.astype(float)

    bias = BiasCorrector().fit(X_bias, raw)

    return bias, bias_prep


def apply_bias_correction_classification(
    pred_df: pd.DataFrame,
    df_apply: pd.DataFrame,
    confound_cols: List[str],
    target_name: str,
    bias: BiasCorrector,
    bias_prep: DerivedConfoundPreprocessor,
) -> pd.DataFrame:
    X_bias = bias_prep.transform(df_apply)
    raw = pred_df[f"{target_name}_raw"].values.astype(float)

    raw_bc = bias.correct(X_bias, raw)

    pred_df[f"{target_name}_bias_corrected"] = raw_bc

    return pred_df


def fit_zscorer_classification_on_holdout(
    pred_holdout: pd.DataFrame,
    target_name: str,
) -> ZScorer:
    z = ZScorer().fit(pred_holdout[f"{target_name}_bias_corrected"].values.astype(float))
    return z


def apply_zscoring_classification(
    pred_df: pd.DataFrame,
    target_name: str,
    z: ZScorer,
) -> pd.DataFrame:
    raw_bc = pred_df[f"{target_name}_bias_corrected"].values.astype(float)
    pred_df[f"{target_name}_z"] = z.transform(raw_bc)

    return pred_df


# =============================================================================
# OUTPUT BUILDING
# =============================================================================

def build_base_output(
    df_stage: pd.DataFrame,
    train_ids: set,
    holdout_ids: set,
    case_ids: set,
) -> pd.DataFrame:
    out = pd.DataFrame({ID_COL: df_stage[ID_COL].values})

    split_group = []
    for pid in df_stage[ID_COL].values:
        if pid in train_ids:
            split_group.append("train_hc")
        elif pid in holdout_ids:
            split_group.append("heldout_hc")
        elif pid in case_ids:
            split_group.append("case")
        else:
            split_group.append("unknown")

    out["split_group"] = split_group
    out[HEALTHY_COL] = df_stage[HEALTHY_COL].values

    meta_cols = [
        "sex",
        "age_assessment_center_2",
        "site",
        "etiv",
        MEAN_MOTION_CONFOUND_COL,
        "bmi",
        "cognition_score",
        "Body fat percentage",
        "Sleep duration",
        SLEEP_DURATION_COL,
        "Systolic blood pressure",
        "duration_trail_time",
        "MDD",
        "ANX",
        "AUD",
        "SCZ",
        "BIPOLAR",
        "ADHD",
        "BPD",
        "ALZ",
        "n_dx",
    ]

    for col in meta_cols:
        if col in df_stage.columns:
            out[col] = df_stage[col].values

    return out

def run_feature_set_from_df(
    feature_name: str,
    df: pd.DataFrame,
    feature_cols: List[str],
    out_dir: str,
    hc_split_map: Dict[str, str],
):
    print("\n" + "=" * 80)
    print(f"Running feature set: {feature_name}")
    print("=" * 80)

    targets = make_targets_for_feature_set(feature_name)

    hc_train_all, hc_holdout_all, cases_all = split_hc_cases(
        df,
        hc_split_map=hc_split_map,
    )

    print(f"Merged rows after feature/health filters: {len(df)}")
    print(f"Feature columns:                    {len(feature_cols)}")
    print(f"Stable train HC before target NA:   {len(hc_train_all)}")
    print(f"Stable heldout HC before target NA: {len(hc_holdout_all)}")
    print(f"Stable cases before target NA:      {len(cases_all)}")

    if MEAN_MOTION_CONFOUND_COL in df.columns:
        print(f"Mean motion rows: {df[MEAN_MOTION_CONFOUND_COL].notna().sum()}")
        print(
            f"Mean motion range: "
            f"{df[MEAN_MOTION_CONFOUND_COL].min():.6f} - "
            f"{df[MEAN_MOTION_CONFOUND_COL].max():.6f}"
        )

    if HEIGHT_CONFOUND_COL in df.columns:
        print(f"Height rows: {df[HEIGHT_CONFOUND_COL].notna().sum()}")
        if df[HEIGHT_CONFOUND_COL].notna().any():
            print(
                f"Height range: "
                f"{df[HEIGHT_CONFOUND_COL].min():.6f} - "
                f"{df[HEIGHT_CONFOUND_COL].max():.6f}"
            )

    train_ids = set(hc_train_all[ID_COL].tolist())
    holdout_ids = set(hc_holdout_all[ID_COL].tolist())
    case_ids = set(cases_all[ID_COL].tolist())

    df_stage = pd.concat([hc_train_all, hc_holdout_all, cases_all], axis=0, ignore_index=True)
    output_df = build_base_output(df_stage, train_ids, holdout_ids, case_ids)

    all_cv = []
    model_summary = {}
    predictions = {}
    target_counts = []

    for target_name, spec in targets.items():
        print(f"\n--- Target: {target_name} ---")
        print(f"Target column: {spec['target_col']}")
        print(f"Confounds: {spec['confounds']}")

        task = spec["task"]
        target_col = spec["target_col"]
        confound_cols = spec["confounds"]

        hc_train = dropna_for_current_target(
            hc_train_all,
            target_name=target_name,
            spec=spec,
            stage_label="train_hc",
        )
        hc_holdout = dropna_for_current_target(
            hc_holdout_all,
            target_name=target_name,
            spec=spec,
            stage_label="heldout_hc",
        )
        cases = dropna_for_current_target(
            cases_all,
            target_name=target_name,
            spec=spec,
            stage_label="case",
        )

        assert_enough_current_target_data(
            target_name=target_name,
            task=task,
            target_col=target_col,
            hc_train=hc_train,
            hc_holdout=hc_holdout,
        )

        target_counts.append({
            "feature_set": feature_name,
            "target": target_name,
            "task": task,
            "target_col": target_col,
            "confounds": ",".join(confound_cols),
            "n_train_hc_complete_current_target": int(len(hc_train)),
            "n_heldout_hc_complete_current_target": int(len(hc_holdout)),
            "n_cases_complete_current_target": int(len(cases)),
            "n_total_complete_current_target": int(len(hc_train) + len(hc_holdout) + len(cases)),
        })

        best_alpha, cv_df = run_cv_for_target(
            df_train_hc=hc_train,
            feature_cols=feature_cols,
            target_name=target_name,
            spec=spec,
            alphas=ALPHAS,
        )

        print(f"Best alpha: {best_alpha}")

        all_cv.append(cv_df)

        if task == "regression":
            fitted = fit_final_regression(
                df_train_hc=hc_train,
                feature_cols=feature_cols,
                target_col=target_col,
                confound_cols=confound_cols,
                alpha=best_alpha,
            )

            pred_train = apply_regression_model(
                fitted,
                hc_train,
                feature_cols,
                target_col,
                target_name,
            )
            pred_holdout = apply_regression_model(
                fitted,
                hc_holdout,
                feature_cols,
                target_col,
                target_name,
            )
            pred_cases = (
                apply_regression_model(
                    fitted,
                    cases,
                    feature_cols,
                    target_col,
                    target_name,
                )
                if len(cases) > 0
                else make_empty_regression_prediction(target_name)
            )

            bias = fit_bias_correction_regression(pred_train, target_name)

            pred_train = apply_bias_correction_regression(pred_train, target_name, bias)
            pred_holdout = apply_bias_correction_regression(pred_holdout, target_name, bias)
            if len(pred_cases) > 0:
                pred_cases = apply_bias_correction_regression(pred_cases, target_name, bias)

            z = fit_zscorer_regression_on_holdout(pred_holdout, target_name)

            pred_train = apply_zscoring_regression(pred_train, target_name, z)
            pred_holdout = apply_zscoring_regression(pred_holdout, target_name, z)
            if len(pred_cases) > 0:
                pred_cases = apply_zscoring_regression(pred_cases, target_name, z)

        else:
            fitted = fit_final_classification(
                df_train_hc=hc_train,
                feature_cols=feature_cols,
                target_col=target_col,
                confound_cols=confound_cols,
                alpha=best_alpha,
            )

            pred_train = apply_classification_model(
                fitted,
                hc_train,
                feature_cols,
                target_col,
                target_name,
            )
            pred_holdout = apply_classification_model(
                fitted,
                hc_holdout,
                feature_cols,
                target_col,
                target_name,
            )
            pred_cases = (
                apply_classification_model(
                    fitted,
                    cases,
                    feature_cols,
                    target_col,
                    target_name,
                )
                if len(cases) > 0
                else make_empty_classification_prediction(target_name)
            )

            bias, bias_prep = fit_bias_correction_classification(
                pred_train=pred_train,
                df_train_hc=hc_train,
                confound_cols=confound_cols,
                target_name=target_name,
            )

            pred_train = apply_bias_correction_classification(
                pred_train,
                hc_train,
                confound_cols,
                target_name,
                bias,
                bias_prep,
            )
            pred_holdout = apply_bias_correction_classification(
                pred_holdout,
                hc_holdout,
                confound_cols,
                target_name,
                bias,
                bias_prep,
            )
            if len(pred_cases) > 0:
                pred_cases = apply_bias_correction_classification(
                    pred_cases,
                    cases,
                    confound_cols,
                    target_name,
                    bias,
                    bias_prep,
                )

            z = fit_zscorer_classification_on_holdout(pred_holdout, target_name)

            pred_train = apply_zscoring_classification(pred_train, target_name, z)
            pred_holdout = apply_zscoring_classification(pred_holdout, target_name, z)
            if len(pred_cases) > 0:
                pred_cases = apply_zscoring_classification(pred_cases, target_name, z)

        pred_all = pd.concat(
            [pred_train, pred_holdout, pred_cases],
            axis=0,
            ignore_index=True,
        )

        predictions[target_name] = pred_all

        model_summary[target_name] = {
            "task": task,
            "target_col": target_col,
            "feature_confounds_removed": confound_cols,
            "best_alpha": best_alpha,
            "bias_correction_fit_on": "train_hc complete for current target only",
            "z_scoring_fit_on": "heldout_hc complete for current target only",
            "deconfounding_strategy": "features_residualized_not_target",
            "missingness_strategy": "target_wise_dropna_only_current_target_and_confounds",
            "feature_set": feature_name,
            "hc_split_strategy": "single_global_split_from_final_diagnosis_table_shared_across_modalities",
            "n_features": len(feature_cols),
            "n_train_hc_complete_current_target": int(len(hc_train)),
            "n_heldout_hc_complete_current_target": int(len(hc_holdout)),
            "n_cases_complete_current_target": int(len(cases)),
            "n_total_complete_current_target": int(len(pred_all)),
            "mean_motion_added_as_confound": MEAN_MOTION_CONFOUND_COL in confound_cols,
            "mean_motion_column": (
                MEAN_MOTION_CONFOUND_COL
                if MEAN_MOTION_CONFOUND_COL in confound_cols
                else None
            ),
            "mean_motion_source_file": (
                CSV_MEAN_MOTION
                if MEAN_MOTION_CONFOUND_COL in confound_cols
                else None
            ),
            "mean_motion_source_id_col": (
                MEAN_MOTION_ID_COL
                if MEAN_MOTION_CONFOUND_COL in confound_cols
                else None
            ),
            "mean_motion_source_value_col": (
                MEAN_MOTION_VALUE_COL
                if MEAN_MOTION_CONFOUND_COL in confound_cols
                else None
            ),
            "height_added_as_confound": HEIGHT_CONFOUND_COL in confound_cols,
            "height_column": (
                HEIGHT_CONFOUND_COL
                if HEIGHT_CONFOUND_COL in confound_cols
                else None
            ),
            "height_source_file": (
                CSV_HEIGHT
                if HEIGHT_CONFOUND_COL in confound_cols
                else None
            ),
            "height_source_id_col": (
                HEIGHT_ID_COL
                if HEIGHT_CONFOUND_COL in confound_cols
                else None
            ),
            "height_source_value_col": (
                HEIGHT_VALUE_COL
                if HEIGHT_CONFOUND_COL in confound_cols
                else None
            ),
        }

    for target_name, pred_df in predictions.items():
        output_df = output_df.merge(pred_df, on=ID_COL, how="left")

    cv_all_df = pd.concat(all_cv, axis=0, ignore_index=True)
    target_counts_df = pd.DataFrame(target_counts)

    out_csv = os.path.join(out_dir, f"normative_scores_{feature_name}.csv")
    cv_csv = os.path.join(out_dir, f"cv_results_{feature_name}.csv")
    target_counts_csv = os.path.join(out_dir, f"targetwise_sample_counts_{feature_name}.csv")
    summary_json = os.path.join(out_dir, f"model_summary_{feature_name}.json")

    output_df.to_csv(out_csv, index=False)
    cv_all_df.to_csv(cv_csv, index=False)
    target_counts_df.to_csv(target_counts_csv, index=False)

    with open(summary_json, "w") as f:
        json.dump(model_summary, f, indent=2)

    print("\nSaved:")
    print(out_csv)
    print(cv_csv)
    print(target_counts_csv)
    print(summary_json)

def run_feature_set(
    feature_name: str,
    feature_file: str,
    diag_file: str,
    out_dir: str,
    hc_split_map: Dict[str, str],
):
    targets = make_targets_for_feature_set(feature_name)
    add_mean_motion = feature_name == "connectivity"

    df, feature_cols = prepare_dataframe_single(
        feature_file=feature_file,
        diag_file=diag_file,
        targets=targets,
        add_mean_motion=add_mean_motion,
    )

    run_feature_set_from_df(
        feature_name=feature_name,
        df=df,
        feature_cols=feature_cols,
        out_dir=out_dir,
        hc_split_map=hc_split_map,
    )


def audit_cross_modality_split_consistency(out_dir: str) -> None:
    """Verify that common HCs have identical split labels across modalities."""
    fs_path = os.path.join(out_dir, "normative_scores_freesurfer.csv")
    con_path = os.path.join(out_dir, "normative_scores_connectivity.csv")

    fs = pd.read_csv(fs_path, usecols=[ID_COL, "split_group", HEALTHY_COL])
    con = pd.read_csv(con_path, usecols=[ID_COL, "split_group", HEALTHY_COL])

    fs = harmonize_id_column(fs, ID_COL)
    con = harmonize_id_column(con, ID_COL)

    fs[HEALTHY_COL] = safe_numeric(fs[HEALTHY_COL])
    con[HEALTHY_COL] = safe_numeric(con[HEALTHY_COL])

    common = fs.merge(
        con,
        on=ID_COL,
        how="inner",
        suffixes=("_freesurfer", "_connectivity"),
        validate="one_to_one",
    )

    common_hc = common.loc[
        common[f"{HEALTHY_COL}_freesurfer"].eq(1)
        & common[f"{HEALTHY_COL}_connectivity"].eq(1)
    ].copy()

    mismatch = common_hc[
        "split_group_freesurfer"
    ].ne(common_hc["split_group_connectivity"])

    print("\n" + "=" * 80)
    print("CROSS-MODALITY HC SPLIT CONSISTENCY AUDIT")
    print("=" * 80)
    print(f"Common HCs across modalities: {len(common_hc):,}")
    print(f"Split-label mismatches:        {int(mismatch.sum()):,}")

    if mismatch.any():
        examples = common_hc.loc[
            mismatch,
            [ID_COL, "split_group_freesurfer", "split_group_connectivity"],
        ].head(20)
        raise AssertionError(
            "Common HCs have inconsistent FreeSurfer/connectivity split labels.\n"
            + examples.to_string(index=False)
        )

    print("PASS: every common HC has the same split_group in both modalities.")

if __name__ == "__main__":
    shared_hc_split_map = make_global_hc_split_map(DIAG_FILE)

    for feature_name, feature_file in FEATURE_FILES.items():
        run_feature_set(
            feature_name=feature_name,
            feature_file=feature_file,
            diag_file=DIAG_FILE,
            out_dir=OUT_DIR,
            hc_split_map=shared_hc_split_map,
        )

    audit_cross_modality_split_consistency(OUT_DIR)
