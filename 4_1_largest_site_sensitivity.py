#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strict largest-site holdout sensitivity pipeline.
"""

import os
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

RANDOM_STATE = 42
N_SPLITS = 5

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
SLEEP_DURATION_COL = "Sleep_duration"
MEAN_MOTION_ID_COL = "eid"
MEAN_MOTION_VALUE_COL = "framewise_dis"
MEAN_MOTION_CONFOUND_COL = "mean_motion"
CSV_MEAN_MOTION = ...
FRAMEWISE_ID_COL = "eid"
FRAMEWISE_VALUE_COL = "framewise_dis"
FRAMEWISE_EXCLUDE_THRESHOLD = 0.5
CSV_FRAMEWISE = ...

HANDGRIP_RIGHT_ID_COL = "eid"
HANDGRIP_RIGHT_VALUE_COL = "handgrip_right"
HANDGRIP_LEFT_VALUE_COL = "Hand_grip_strength_left"
HANDGRIP_MEAN_COL = "Hand_grip_strength_mean"
CSV_HANDGRIP_RIGHT = ...

HEIGHT_ID_COL = "eid"
HEIGHT_VALUE_COL = "height"
HEIGHT_CONFOUND_COL = "height"
CSV_HEIGHT = ...

FEATURE_FILES = {
    "freesurfer": ...,
    "connectivity": ...
}

DIAG_FILE = ...
OUT_DIR = ...
os.makedirs(OUT_DIR, exist_ok=True)

MAIN_MODEL_SUMMARY_DIR = (
    ...
)
REUSE_MAIN_SELECTED_ALPHA = True
GLOBAL_HELDOUT_SITE = None


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
        "target_col": SLEEP_DURATION_COL,
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
    "Systolic_blood_pressure": {
        "target_col": "Systolic_blood_pressure",
        "task": "regression",
        "confounds": ["site", "age_assessment_center_2", "sex", "etiv"],
    },
    "Hand_grip_strength_mean": {
        "target_col": HANDGRIP_MEAN_COL,
        "task": "regression",
        "confounds": ["site", "age_assessment_center_2", "sex", HEIGHT_CONFOUND_COL, "etiv"],
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

    targets = json.loads(json.dumps(BASE_TARGETS))

    if feature_name == "freesurfer":
        targets.pop("Sleep_duration", None)

    for target_name, spec in targets.items():
        if spec.get("add_nonlinear_age_terms", True):
            spec["confounds"] = add_nonlinear_age_terms_to_confounds(spec["confounds"])

    if feature_name in {"connectivity", "joint"}:
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

    def _needs_centered_age(self) -> bool:
        return (
            AGE2_CONFOUND_COL in self.numeric_cols
            or AGE_SEX_CONFOUND_COL in self.numeric_cols
        )

    def _fit_age_center(self, df: pd.DataFrame):
        if self._needs_centered_age():
            age = safe_numeric(df[AGE_COL]).values.astype(float)
            self.age_center_ = float(np.nanmean(age))
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


    feat = harmonize_id_column(feat, ID_COL)
    return feat


def read_diag_table(diag_file: str) -> pd.DataFrame:
    diag = pd.read_csv(diag_file)


    diag = harmonize_id_column(diag, ID_COL)
    return diag


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


def maybe_merge_mean_motion(df: pd.DataFrame, add_mean_motion: bool) -> pd.DataFrame:
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

    if HANDGRIP_RIGHT_VALUE_COL in df.columns:
        df = df.drop(columns=[HANDGRIP_RIGHT_VALUE_COL])

    df = df.merge(grip_right, on=ID_COL, how="left")
    df[HANDGRIP_LEFT_VALUE_COL] = safe_numeric(df[HANDGRIP_LEFT_VALUE_COL])
    df[HANDGRIP_RIGHT_VALUE_COL] = safe_numeric(df[HANDGRIP_RIGHT_VALUE_COL])

    # Strict bilateral mean: if either hand is missing, the mean remains missing.
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


def maybe_merge_height(df: pd.DataFrame, add_height: bool = True) -> pd.DataFrame:
    if not add_height:
        return df

    height = read_height_table(CSV_HEIGHT)

    # Prefer the explicitly generated height CSV if a stale/empty height column
    # already exists in the diagnosis table.
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
    df = maybe_merge_mean_motion(df, add_mean_motion=add_mean_motion)
    df = exclude_high_framewise_motion(df, apply_exclusion=add_mean_motion)
    df = merge_and_make_handgrip_mean(df)
    df = maybe_merge_height(df, add_height=True)

    feature_cols = get_feature_columns(feat)

    # Keep this as a schema check only. Do not drop rows based on all targets.
    required_cols = get_required_columns(targets)


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


def split_hc_cases(df: pd.DataFrame):

    global GLOBAL_HELDOUT_SITE


    site = df["site"].astype(str).str.strip()
    held = str(GLOBAL_HELDOUT_SITE).strip()
    is_hc = pd.to_numeric(df[HEALTHY_COL], errors="coerce").eq(1)

    hc_train = df.loc[is_hc & ~site.eq(held)].copy()
    hc_holdout = df.loc[is_hc & site.eq(held)].copy()
    cases = df.loc[~is_hc & site.eq(held)].copy()


    print(f"[STRICT SITE HOLDOUT] held-out site = {held}")
    print(f"[STRICT SITE HOLDOUT] train MAIN_HC outside site: {len(hc_train):,}")
    print(f"[STRICT SITE HOLDOUT] held-out MAIN_HC at site:  {len(hc_holdout):,}")
    print(f"[STRICT SITE HOLDOUT] held-out non-HC at site:  {len(cases):,}")

    return (
        hc_train.reset_index(drop=True),
        hc_holdout.reset_index(drop=True),
        cases.reset_index(drop=True),
    )


def infer_common_largest_site_from_both_modalities() -> str:
    diag = read_diag_table(DIAG_FILE)
    need = [ID_COL, HEALTHY_COL, "site"]
    missing = [c for c in need if c not in diag.columns]

    fs = read_feature_table(FEATURE_FILES["freesurfer"])[[ID_COL]].drop_duplicates()
    con = read_feature_table(FEATURE_FILES["connectivity"])[[ID_COL]].drop_duplicates()
    both = fs.merge(con, on=ID_COL, how="inner")
    x = both.merge(diag[need], on=ID_COL, how="inner")
    x[HEALTHY_COL] = pd.to_numeric(x[HEALTHY_COL], errors="coerce")
    x = x.loc[x[HEALTHY_COL].eq(1) & x["site"].notna()].copy()

    counts = x["site"].astype(str).str.strip().value_counts()
    held = str(counts.index[0])
    print("\nCommon held-out site selection")
    print("=" * 80)
    print(counts.to_string())
    print(f"Selected largest common MAIN_HC site: {held} (n={int(counts.iloc[0]):,})")
    return held


def get_reused_main_alpha(feature_name: str, target_name: str):
    path = os.path.join(MAIN_MODEL_SUMMARY_DIR, f"model_summary_{feature_name}.json")
    if not REUSE_MAIN_SELECTED_ALPHA or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        summary = json.load(f)
    if target_name not in summary:
        return None
    value = summary[target_name].get("best_alpha", None)
    if value is None:
        return None
    return float(value)


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
):
    print("\n" + "=" * 80)
    print(f"Running feature set: {feature_name}")
    print("=" * 80)

    targets = make_targets_for_feature_set(feature_name)

    hc_train_all, hc_holdout_all, cases_all = split_hc_cases(df)

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

        reused_alpha = get_reused_main_alpha(feature_name, target_name)
        if reused_alpha is not None:
            best_alpha = float(reused_alpha)
            cv_df = pd.DataFrame([{
                "target": target_name,
                "alpha": best_alpha,
                "alpha_source": "reused_from_MAIN_selection_only",
                "fold": np.nan,
            }])
            print(f"Alpha reused from MAIN selection: {best_alpha}")
        else:
            best_alpha, cv_df = run_cv_for_target(
                df_train_hc=hc_train,
                feature_cols=feature_cols,
                target_name=target_name,
                spec=spec,
                alphas=ALPHAS,
            )
            cv_df = cv_df.copy()
            cv_df["alpha_source"] = "reselected_with_nonheldout_MAIN_HC_CV"
            print(f"Alpha reselected outside held-out site: {best_alpha}")

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
            "alpha_source": ("reused_from_MAIN_selection_only" if reused_alpha is not None else "reselected_with_nonheldout_MAIN_HC_CV"),
            "strict_heldout_site": str(GLOBAL_HELDOUT_SITE),
            "bias_correction_fit_on": "MAIN_HC outside held-out site; current-target complete cases only",
            "z_scoring_fit_on": "MAIN_HC inside held-out site; current-target complete cases only",
            "deconfounding_strategy": "features_residualized_not_target",
            "missingness_strategy": "target_wise_dropna_only_current_target_and_confounds",
            "feature_set": feature_name,
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
    )




if __name__ == "__main__":
    GLOBAL_HELDOUT_SITE = infer_common_largest_site_from_both_modalities()

    with open(os.path.join(OUT_DIR, "strict_holdout_site.json"), "w") as f:
        json.dump({
            "heldout_site": str(GLOBAL_HELDOUT_SITE),
            "healthy_definition": HEALTHY_COL,
            "site_selection": "largest site among MAIN_HC with both FreeSurfer and connectivity IDs",
            "alpha_policy": (
                "reuse MAIN-selected alpha only; refit every model/preprocessing parameter outside site"
                if REUSE_MAIN_SELECTED_ALPHA
                else "reselect alpha using only MAIN_HC outside held-out site"
            ),
        }, f, indent=2)

    for feature_name, feature_file in FEATURE_FILES.items():
        run_feature_set(
            feature_name=feature_name,
            feature_file=feature_file,
            diag_file=DIAG_FILE,
            out_dir=OUT_DIR,
        )

    print("\nStrict normative refit complete. Starting held-out-site association analysis...\n")



"""
Largest-site-holdout association robustness test for Figure 2.

Purpose
-------
This script is the downstream association-test companion for the
largest-site-holdout normative gap pipeline.

It uses the largest-site-holdout normative gap CSVs, reruns the standard
Figure-2 predictor -> gap association models, and compares the resulting effects
against the updated main Figure-2 predictor-gap edge table.
"""

import os
import warnings
from itertools import combinations
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

ID_COL = "Participant ID"
SPLIT_GROUP_COL = "split_group"
TRAIN_HC_SPLIT_VALUE = "train_hc"
HELDOUT_HC_SPLIT_VALUE = "heldout_hc"

LARGEST_SITE_HOLDOUT_GAP_DIR = (
    "/zi/flstorage/HITKIP/share/Julius/brain_age_gap/"
    "normative_outputs_MAIN_strict_largest_site_holdout"
)

INPUT_FILES = {
    "freesurfer": os.path.join(
        LARGEST_SITE_HOLDOUT_GAP_DIR,
        "normative_scores_freesurfer.csv",
    ),
    "connectivity": os.path.join(
        LARGEST_SITE_HOLDOUT_GAP_DIR,
        "normative_scores_connectivity.csv",
    ),
}



# Significant predictor-gap edge table from the updated main Figure-2 script.
ORIGINAL_PREDICTOR_EDGE_TABLE = os.path.join(
    ...
)

# Significant gap-gap edge table from the updated main Figure-2 script.
ORIGINAL_GAP_GAP_EDGE_TABLE = os.path.join(
    ...
)

# Keep the site-holdout robustness outputs in their existing top-level location.
OUT_DIR = os.path.join(
    ...
)
os.makedirs(OUT_DIR, exist_ok=True)

CSV_DIAGNOSES = (
    ...
)

CSV_MEAN_MOTION = ...
MEAN_MOTION_ID_COL = "eid"
MEAN_MOTION_VALUE_COL = "framewise_dis"
MEAN_MOTION_COVARIATE_COL = "mean_motion"

CSV_FRAMEWISE = ...
FRAMEWISE_ID_COL = "eid"
FRAMEWISE_VALUE_COL = "framewise_dis"
FRAMEWISE_EXCLUDE_THRESHOLD = 0.5

SITE_COL = "site"

REGRESSION_TARGETS = [
    "brain_age",
    "Fluid_intelligence_score",
    "body_fat_percentage",
    "Sleep_duration",
    "trail_making",
    "Hand_grip_strength_mean",
    "bmi",
    "diastolic_bp",
]




FEATURE_TARGET_EXCLUSIONS = {

    "freesurfer": {
        "Sleep_duration",
    },
    "connectivity": {
    },
}


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
MIN_N_MODEL = 30
MIN_CASE_N = 10
MIN_REF_N = 10

GAP_GRAPH_MIN_ABS_BETA_STD = 0.05


def canonicalize_id_series(x: pd.Series) -> pd.Series:
    return (
        x.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def zscore_series(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").astype(float)
    sd = x.std(ddof=0)

    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=x.index, dtype=float)

    return (x - x.mean()) / sd


def ranknorm_series(x: pd.Series) -> pd.Series:
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


def safe_numeric_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()

    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def binary_col_positive(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").fillna(0).eq(1)


def is_valid_graph_label(x) -> bool:
    if x is None or pd.isna(x):
        return False

    s = str(x).strip()

    if s == "":
        return False
    if s.lower() in {"none", "nan", "null"}:
        return False

    return True


def gap_to_feature_target(gap: str) -> Tuple[str, str]:
    if "__" in str(gap):
        fs, target = str(gap).split("__", 1)
    else:
        fs, target = "", str(gap)

    return fs, target


def is_graph_excluded_gap(gap: str) -> bool:
    if gap is None or pd.isna(gap):
        return False

    _, target = gap_to_feature_target(str(gap).strip())
    return target in GRAPH_EXCLUDE_TARGETS


def graph_gap_cols(gap_cols: List[str]) -> List[str]:
    return [
        g for g in gap_cols
        if is_valid_graph_label(g) and not is_graph_excluded_gap(g)
    ]


def short_gap_name(name: str) -> str:
    if "__" in name:
        _, target = name.split("__", 1)
    else:
        target = name

    replacements = {
        "brain_age": "Age",
        "Fluid_intelligence_score": "Intel.",
        "body_fat_percentage": "Body fat",
        "trail_making": "Trail",
        "Hand_grip_strength_mean": "Grip",
        "bmi": "BMI",
        "diastolic_bp": "DBP",
    }

    return replacements.get(target, target)


def available_covariates(df: pd.DataFrame) -> List[str]:
    return [c for c in COVARIATE_CANDIDATES if c in df.columns]


# =============================================================================
# DATA LOADING
# =============================================================================

def read_framewise_table(framewise_file: str) -> pd.DataFrame:
    fw = pd.read_csv(framewise_file)



    fw = fw[[FRAMEWISE_ID_COL, FRAMEWISE_VALUE_COL]].copy()
    fw = fw.rename(columns={FRAMEWISE_ID_COL: ID_COL})
    fw[ID_COL] = canonicalize_id_series(fw[ID_COL])
    fw[FRAMEWISE_VALUE_COL] = pd.to_numeric(fw[FRAMEWISE_VALUE_COL], errors="coerce")
    fw = fw.dropna(subset=[ID_COL])
    fw = fw.drop_duplicates(subset=[ID_COL], keep="first")
    return fw


def exclude_high_framewise_from_connectivity_gaps(
    gap_df: pd.DataFrame,
    gap_cols: List[str],
) -> pd.DataFrame:

    fw = read_framewise_table(CSV_FRAMEWISE)
    bad_ids = set(
        fw.loc[
            fw[FRAMEWISE_VALUE_COL] > FRAMEWISE_EXCLUDE_THRESHOLD,
            ID_COL,
        ].tolist()
    )

    connectivity_gap_cols = [
        c for c in gap_cols
        if str(c).startswith("connectivity__") and c in gap_df.columns
    ]

    out = gap_df.copy()
    bad_mask = out[ID_COL].isin(bad_ids)
    n_bad = int(bad_mask.sum())

    if connectivity_gap_cols:
        out.loc[bad_mask, connectivity_gap_cols] = np.nan

    print(
        f"[INFO] framewise exclusion > {FRAMEWISE_EXCLUDE_THRESHOLD} mm for "
        f"connectivity gaps: flagged {n_bad:,} heldout-site subjects; "
        f"set {len(connectivity_gap_cols)} connectivity gap column(s) to NaN "
        f"for those subjects. FreeSurfer gaps unchanged."
    )

    return out


def get_gap_specs(df: pd.DataFrame, feature_set: str) -> List[Tuple[str, str, str]]:
    specs = []
    excluded_targets = FEATURE_TARGET_EXCLUSIONS.get(feature_set, set())

    for t in REGRESSION_TARGETS:
        if t in excluded_targets:
            print(f"[INFO] Excluding {feature_set} target: {t}")
            continue

        c = f"{t}_gap_z"
        if c in df.columns:
            specs.append((feature_set, t, c))
        else:
            print(
                f"[WARN] Expected regression gap column not found for "
                f"{feature_set}/{t}: {c}"
            )

    return specs


def load_largest_site_holdout_gap_space() -> Tuple[pd.DataFrame, List[str]]:
    frames = []
    specs_all = []

    for feature_set, path in INPUT_FILES.items():

        dfi = pd.read_csv(path)
        dfi[ID_COL] = canonicalize_id_series(dfi[ID_COL])

        specs = get_gap_specs(dfi, feature_set)


        keep = [ID_COL]

        for c in [SPLIT_GROUP_COL, HC_COL, SITE_COL, "sex", "age_assessment_center_2", "etiv"]:
            if c in dfi.columns and c not in keep:
                keep.append(c)

        keep += [old for _, _, old in specs]
        keep = list(dict.fromkeys(keep))
        dfi = dfi[keep].copy()

        rename = {}

        for fs_name, target, old in specs:
            new = f"{fs_name}__{target}"
            rename[old] = new
            specs_all.append((fs_name, target, new))

        dfi = dfi.rename(columns=rename)

        if frames:
            # Avoid duplicate metadata columns from the second feature set.
            keep_second = [ID_COL]
            if SPLIT_GROUP_COL in dfi.columns:
                keep_second.append(SPLIT_GROUP_COL)
            keep_second += list(rename.values())
            dfi = dfi[keep_second].copy()

        frames.append(dfi)

    gap_df = frames[0].copy()

    for f in frames[1:]:
        merge_cols = [ID_COL]

        if SPLIT_GROUP_COL in gap_df.columns and SPLIT_GROUP_COL in f.columns:
            merge_cols.append(SPLIT_GROUP_COL)

        gap_df = gap_df.merge(
            f,
            on=merge_cols,
            how="inner",
            validate="one_to_one",
        )

    gap_cols = [s[2] for s in specs_all]
    gap_cols = list(dict.fromkeys(gap_cols))

    print("[STRICT SITE HOLDOUT] verified final target space, including connectivity Sleep_duration.")

    gap_df = safe_numeric_df(gap_df, gap_cols)


    print("\nSplit counts before train-HC removal")
    print("=" * 80)
    print(gap_df[SPLIT_GROUP_COL].value_counts(dropna=False).to_string())

    # Remove train HCs before downstream association testing.
    before = len(gap_df)
    gap_df = gap_df.loc[
        ~gap_df[SPLIT_GROUP_COL].astype(str).str.strip().eq(TRAIN_HC_SPLIT_VALUE)
    ].copy()
    print(f"\nRemoved train HCs: {before - len(gap_df):,}; remaining n={len(gap_df):,}")

    # Infer heldout site from heldout HCs.
    if SITE_COL in gap_df.columns:
        heldout_sites = (
            gap_df.loc[
                gap_df[SPLIT_GROUP_COL].astype(str).str.strip().eq(HELDOUT_HC_SPLIT_VALUE),
                SITE_COL,
            ]
            .astype(str)
            .str.strip()
            .dropna()
            .unique()
            .tolist()
        )

        if len(heldout_sites) == 1:
            heldout_site = heldout_sites[0]
            before_site = len(gap_df)
            gap_df = gap_df.loc[gap_df[SITE_COL].astype(str).str.strip().eq(heldout_site)].copy()
            print(f"Heldout site inferred from heldout HCs: {heldout_site}")
            print(f"Filtered to heldout site: {before_site:,} -> {len(gap_df):,}")
        elif len(heldout_sites) > 1:
            print(
                "[WARN] Multiple heldout-HC sites found; not filtering further: "
                + ", ".join(map(str, heldout_sites))
            )
        else:
            print("[WARN] No heldout-HC site could be inferred.")

    print("\nSplit counts after site filtering")
    print("=" * 80)
    print(gap_df[SPLIT_GROUP_COL].value_counts(dropna=False).to_string())

    return gap_df.reset_index(drop=True), gap_cols


def resolve_covariate_source_columns(header: List[str], existing_columns: List[str]) -> Tuple[List[str], Dict[str, str]]:
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


def merge_diagnosis_scores_and_mean_motion(gap_df: pd.DataFrame) -> pd.DataFrame:
    header = pd.read_csv(CSV_DIAGNOSES, nrows=0).columns.tolist()

    optional_covariates, rename_map = resolve_covariate_source_columns(
        header=header,
        existing_columns=gap_df.columns.tolist(),
    )

    optional_scores = [
        c for c in SCORE_PREDICTOR_COLS
        if c in header and c not in gap_df.columns
    ]

    required = [
        ID_COL,
        *DIAGNOSIS_COLUMNS_FOR_PLOTTING,
        *optional_covariates,
        *optional_scores,
    ]
    required = list(dict.fromkeys([c for c in required if c in header or c == ID_COL]))

    diag = pd.read_csv(CSV_DIAGNOSES, usecols=required)
    diag[ID_COL] = canonicalize_id_series(diag[ID_COL])

    if rename_map:
        diag = diag.rename(columns=rename_map)

    if diag[ID_COL].duplicated().any():
        diag = diag.drop_duplicates(subset=[ID_COL], keep="first")

    out = gap_df.copy()
    out[ID_COL] = canonicalize_id_series(out[ID_COL])

    # Drop stale diagnosis / score columns before merge.
    stale = [
        c for c in DIAGNOSIS_COLUMNS_FOR_PLOTTING + SCORE_PREDICTOR_COLS
        if c in out.columns
    ]
    if stale:
        out = out.drop(columns=stale)

    merge_cols = [
        c for c in diag.columns
        if c == ID_COL or c not in out.columns
    ]
    diag = diag[merge_cols].copy()

    out = out.merge(
        diag,
        on=ID_COL,
        how="left",
        validate="many_to_one",
    )

    if os.path.exists(CSV_MEAN_MOTION):
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
        mm = mm.dropna(subset=[MEAN_MOTION_COVARIATE_COL])
        mm = mm.drop_duplicates(subset=[ID_COL], keep="first")

        out = out.drop(columns=[MEAN_MOTION_COVARIATE_COL], errors="ignore")
        out = out.merge(mm, on=ID_COL, how="left", validate="many_to_one")
    else:
        print(f"[WARN] Mean-motion file not found; skipping: {CSV_MEAN_MOTION}")

    out = apply_plotting_diagnosis_labels(out)

    print("\nCovariates available")
    print("=" * 80)
    print(", ".join(available_covariates(out)))

    return out


def apply_plotting_diagnosis_labels(df: pd.DataFrame) -> pd.DataFrame:

    missing = [c for c in DIAGNOSIS_COLUMNS_FOR_PLOTTING if c not in df.columns]

    out = df.copy()
    split = out[SPLIT_GROUP_COL].astype(str).str.strip()

    is_hc = split.eq(HELDOUT_HC_SPLIT_VALUE) & binary_col_positive(out, HC_COL)
    is_aud = binary_col_positive(out, AUD_COL)
    is_mdd = binary_col_positive(out, MDD_COL)
    is_anx = binary_col_positive(out, ANX_COL)

    # Display label only. Independent MAIN flags below are used for statistics.
    out["analysis_label"] = np.nan
    out.loc[is_hc, "analysis_label"] = "heldout_hc"
    out.loc[is_aud, "analysis_label"] = AUD_COL
    out.loc[is_anx, "analysis_label"] = ANX_COL
    out.loc[is_mdd, "analysis_label"] = MDD_COL

    print("\nLargest-site-holdout MAIN group counts")
    print("=" * 80)
    print(f"{'heldout_hc & MAIN_HC':30s}: n={int(is_hc.sum()):8,d}")
    print(f"{AUD_COL:30s}: n={int(is_aud.sum()):8,d}")
    print(f"{MDD_COL:30s}: n={int(is_mdd.sum()):8,d}")
    print(f"{ANX_COL:30s}: n={int(is_anx.sum()):8,d}")
    print(
        f"{'MAIN_MDD & MAIN_ANX':30s}: "
        f"n={int((is_mdd & is_anx).sum()):8,d}"
    )

    return out


# =============================================================================
# HC3 OLS MODEL HELPERS
# =============================================================================

def prepare_covariate_design(df: pd.DataFrame, covariates: List[str]) -> pd.DataFrame:
    if not covariates:
        return pd.DataFrame(index=df.index)

    pieces = []

    for c in covariates:
        if c not in df.columns:
            continue

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


def drop_constant_predictors(X: pd.DataFrame) -> pd.DataFrame:
    keep = []

    for c in X.columns:
        vals = pd.to_numeric(X[c], errors="coerce")
        if vals.notna().sum() >= 10 and vals.nunique(dropna=True) > 1:
            keep.append(c)

    return X[keep].astype(float)


def standardize_model_dataframe(
    model_df: pd.DataFrame,
    outcome_col: str,
    predictor_of_interest: str,
) -> pd.DataFrame:
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
    df: pd.DataFrame,
    outcome_col: str,
    predictor_col: str,
    covariates: List[str],
    extra_predictors: List[str] = None,
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

    if len(tmp) < MIN_N_MODEL:
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

    if len(model_df) < MIN_N_MODEL:
        return None

    model_df_std = standardize_model_dataframe(
        model_df=model_df,
        outcome_col=outcome_col,
        predictor_of_interest=predictor_col,
    )

    if len(model_df_std) < MIN_N_MODEL:
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
            "n_covariates_model": int(X.shape[1] - 2),
        }
    except Exception as e:
        print(f"[WARN] OLS failed for {predictor_col} -> {outcome_col}: {e}")
        return None

def add_binary_diagnosis_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for c in DIAGNOSIS_COLUMNS_FOR_PLOTTING:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    split = out[SPLIT_GROUP_COL].astype(str).str.strip()

    out["is_hc"] = split.eq(HELDOUT_HC_SPLIT_VALUE) & out[HC_COL].eq(1)
    out["is_aud"] = out[AUD_COL].eq(1)
    out["is_mdd"] = out[MDD_COL].eq(1)
    out["is_anx"] = out[ANX_COL].eq(1)

    return out


def make_contrast_subset(df: pd.DataFrame, contrast_name: str) -> pd.DataFrame:

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

    if n_case < MIN_CASE_N or n_ref < MIN_REF_N:
        print(
            f"[WARN] Too few subjects for {contrast_name}: "
            f"case={n_case}, ref={n_ref}"
        )

    return work


def fit_standard_diagnosis_gap_edges(gap_df: pd.DataFrame, gap_cols: List[str]) -> pd.DataFrame:
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

        n_case = int(sub["diagnosis_indicator"].eq(1).sum()) if len(sub) else 0
        n_ref = int(sub["diagnosis_indicator"].eq(0).sum()) if len(sub) else 0

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
                "n_case": n_case,
                "n_ref": n_ref,
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
                alpha=FDR_ALPHA,
                method="fdr_bh",
            )[1]

    out["q_sig"] = out["q_fdr_bh"].lt(FDR_ALPHA)
    out["p_sig"] = out["p_hc3"].lt(0.05)

    return out


def fit_standard_score_gap_edges(gap_df: pd.DataFrame, gap_cols: List[str]) -> pd.DataFrame:
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
                alpha=FDR_ALPHA,
                method="fdr_bh",
            )[1]

    out["q_sig"] = out["q_fdr_bh"].lt(FDR_ALPHA)
    out["p_sig"] = out["p_hc3"].lt(0.05)

    return out


def combine_predictor_gap_edges(
    diagnosis_edges: pd.DataFrame,
    score_edges: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = []

    for df in [diagnosis_edges, score_edges]:
        if df is not None and len(df) > 0:
            frames.append(df.copy())

    if frames:
        all_edges = pd.concat(frames, axis=0, ignore_index=True)
    else:
        all_edges = pd.DataFrame()

    if len(all_edges) == 0:
        return all_edges, all_edges

    fdr_edges = all_edges[
        all_edges["q_fdr_bh"].lt(FDR_ALPHA)
        & all_edges["beta_std"].notna()
        & all_edges["predictor"].map(is_valid_graph_label)
        & all_edges["gap"].map(is_valid_graph_label)
        & ~all_edges["gap"].map(is_graph_excluded_gap)
    ].copy()

    fdr_edges = fdr_edges.sort_values(
        ["source", "predictor", "q_fdr_bh", "p_hc3"],
        ascending=[True, True, True, True],
    )

    return all_edges, fdr_edges


# =============================================================================
# GAP-GAP ASSOCIATIONS
# =============================================================================

def compute_gap_adjusted_ols_hc3_edges(gap_df: pd.DataFrame, gap_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
            "beta_ab": float(beta_ab),
            "beta_ba": float(beta_ba),
            "p_pair": float(p_pair),
            "n_complete": int(n_pair),
        })

    out = pd.DataFrame(rows)

    if len(out) == 0:
        return out, out

    out["q_fdr_bh_unique_pairs"] = multipletests(
        out["p_pair"],
        alpha=FDR_ALPHA,
        method="fdr_bh",
    )[1]

    fdr_edges = out[
        out["q_fdr_bh_unique_pairs"].lt(FDR_ALPHA)
        & out["beta_std_symmetric"].abs().ge(GAP_GRAPH_MIN_ABS_BETA_STD)
    ].copy()

    fdr_edges = fdr_edges.sort_values(
        "beta_std_symmetric",
        key=lambda x: x.abs(),
        ascending=False,
    )

    return out, fdr_edges


# =============================================================================
# ORIGINAL EDGE COMPARISON
# =============================================================================

def load_original_predictor_edges() -> pd.DataFrame:
    if not os.path.exists(ORIGINAL_PREDICTOR_EDGE_TABLE):
        print(f"[WARN] Original predictor edge table not found:\n{ORIGINAL_PREDICTOR_EDGE_TABLE}")
        return pd.DataFrame()

    orig = pd.read_csv(ORIGINAL_PREDICTOR_EDGE_TABLE)

    required = ["predictor", "gap", "source", "beta_std"]
    missing = [c for c in required if c not in orig.columns]
    if missing:
        print(
            "[WARN] Original predictor edge table lacks required columns: "
            + ", ".join(missing)
        )
        return pd.DataFrame()

    orig = orig.copy()

    excluded_mask = orig["gap"].astype(str).str.strip().isin(EXCLUDED_GAPS)
    if excluded_mask.any():
        print(
            "[INFO] Original predictor-gap table: removed "
            f"{int(excluded_mask.sum()):,} excluded edge rows."
        )
        orig = orig.loc[~excluded_mask].copy()

    orig["beta_std_original"] = pd.to_numeric(orig["beta_std"], errors="coerce")

    if "q_fdr_bh" in orig.columns:
        orig["q_fdr_bh_original"] = pd.to_numeric(orig["q_fdr_bh"], errors="coerce")
    else:
        orig["q_fdr_bh_original"] = np.nan

    if "p_hc3" in orig.columns:
        orig["p_hc3_original"] = pd.to_numeric(orig["p_hc3"], errors="coerce")
    else:
        orig["p_hc3_original"] = np.nan

    keep = [
        "source",
        "predictor",
        "gap",
        "beta_std_original",
        "p_hc3_original",
        "q_fdr_bh_original",
    ]

    if "score_predictor" in orig.columns:
        keep.append("score_predictor")

    return orig[keep].copy()


def compare_predictor_edges_to_original(
    holdout_all_edges: pd.DataFrame,
    original_edges: pd.DataFrame,
) -> pd.DataFrame:
    if original_edges.empty or holdout_all_edges.empty:
        return pd.DataFrame()

    hold = holdout_all_edges.copy()
    hold = hold.rename(
        columns={
            "beta_std": "beta_std_holdout",
            "se_hc3": "se_hc3_holdout",
            "t_hc3": "t_hc3_holdout",
            "p_hc3": "p_hc3_holdout",
            "q_fdr_bh": "q_fdr_bh_holdout",
            "n": "n_holdout",
        }
    )

    keep_hold = [
        "source",
        "predictor",
        "gap",
        "beta_std_holdout",
        "se_hc3_holdout",
        "t_hc3_holdout",
        "p_hc3_holdout",
        "q_fdr_bh_holdout",
        "n_holdout",
    ]

    for c in ["n_case", "n_ref", "score_predictor"]:
        if c in hold.columns and c not in keep_hold:
            keep_hold.append(c)

    hold = hold[keep_hold].copy()

    comp = original_edges.merge(
        hold,
        on=["source", "predictor", "gap"],
        how="left",
        suffixes=("_original", "_holdout"),
    )

    comp["tested_in_holdout"] = comp["beta_std_holdout"].notna()

    comp["same_sign"] = (
        np.sign(comp["beta_std_original"]) == np.sign(comp["beta_std_holdout"])
    )
    comp.loc[~comp["tested_in_holdout"], "same_sign"] = np.nan

    comp["beta_delta_holdout_minus_original"] = (
        comp["beta_std_holdout"] - comp["beta_std_original"]
    )

    comp["abs_beta_retained_pct"] = (
        comp["beta_std_holdout"].abs()
        / comp["beta_std_original"].abs()
        * 100.0
    )

    comp["p05_holdout"] = comp["p_hc3_holdout"].lt(0.05)
    comp["fdr05_holdout"] = comp["q_fdr_bh_holdout"].lt(FDR_ALPHA)

    return comp.sort_values(
        ["tested_in_holdout", "same_sign", "abs_beta_retained_pct"],
        ascending=[True, True, True],
    )


def summarize_predictor_edge_robustness(comp: pd.DataFrame) -> pd.DataFrame:
    if comp.empty:
        return pd.DataFrame()

    rows = []

    groupings = {
        "overall": [],
        "by_source": ["source"],
        "by_predictor": ["source", "predictor"],
    }

    for summary_level, group_cols in groupings.items():
        if group_cols:
            iterator = comp.groupby(group_cols, dropna=False)
        else:
            iterator = [(("all",), comp)]

        for key, g in iterator:
            if not isinstance(key, tuple):
                key = (key,)

            tested = g[g["tested_in_holdout"].fillna(False)].copy()

            row = {
                "summary_level": summary_level,
                "n_original_edges": int(len(g)),
                "n_tested_in_holdout": int(len(tested)),
                "n_same_sign": int(tested["same_sign"].fillna(False).sum()),
                "pct_same_sign": (
                    float(100.0 * tested["same_sign"].mean())
                    if len(tested) else np.nan
                ),
                "n_p05_holdout": int(tested["p05_holdout"].fillna(False).sum()),
                "pct_p05_holdout": (
                    float(100.0 * tested["p05_holdout"].mean())
                    if len(tested) else np.nan
                ),
                "n_fdr05_holdout": int(tested["fdr05_holdout"].fillna(False).sum()),
                "pct_fdr05_holdout": (
                    float(100.0 * tested["fdr05_holdout"].mean())
                    if len(tested) else np.nan
                ),
                "median_abs_beta_retained_pct": (
                    float(tested["abs_beta_retained_pct"].median())
                    if len(tested) else np.nan
                ),
                "min_abs_beta_retained_pct": (
                    float(tested["abs_beta_retained_pct"].min())
                    if len(tested) else np.nan
                ),
                "median_abs_beta_delta": (
                    float(tested["beta_delta_holdout_minus_original"].abs().median())
                    if len(tested) else np.nan
                ),
            }

            for col, value in zip(group_cols, key):
                row[col] = value

            rows.append(row)

    return pd.DataFrame(rows)


def load_original_gap_gap_edges() -> pd.DataFrame:
    if not os.path.exists(ORIGINAL_GAP_GAP_EDGE_TABLE):
        print(f"[WARN] Original gap-gap edge table not found:\n{ORIGINAL_GAP_GAP_EDGE_TABLE}")
        return pd.DataFrame()

    orig = pd.read_csv(ORIGINAL_GAP_GAP_EDGE_TABLE)

    required = ["gap_a", "gap_b", "beta_std_symmetric"]
    missing = [c for c in required if c not in orig.columns]
    if missing:
        print(
            "[WARN] Original gap-gap edge table lacks required columns: "
            + ", ".join(missing)
        )
        return pd.DataFrame()

    orig = orig.copy()

    excluded_mask = (
        orig["gap_a"].astype(str).str.strip().isin(EXCLUDED_GAPS)
        | orig["gap_b"].astype(str).str.strip().isin(EXCLUDED_GAPS)
    )
    if excluded_mask.any():
        print(
            "[INFO] Original gap-gap table: removed "
            f"{int(excluded_mask.sum()):,} excluded edge rows."
        )
        orig = orig.loc[~excluded_mask].copy()

    orig["gap_pair_key"] = orig.apply(
        lambda r: " || ".join(sorted([str(r["gap_a"]), str(r["gap_b"])])),
        axis=1,
    )
    orig["beta_std_symmetric_original"] = pd.to_numeric(
        orig["beta_std_symmetric"],
        errors="coerce",
    )

    if "q_fdr_bh_unique_pairs" in orig.columns:
        orig["q_fdr_bh_unique_pairs_original"] = pd.to_numeric(
            orig["q_fdr_bh_unique_pairs"],
            errors="coerce",
        )
    else:
        orig["q_fdr_bh_unique_pairs_original"] = np.nan

    keep = [
        "gap_pair_key",
        "gap_a",
        "gap_b",
        "beta_std_symmetric_original",
        "q_fdr_bh_unique_pairs_original",
    ]

    return orig[keep].copy()


def compare_gap_gap_edges_to_original(
    holdout_gap_gap_all: pd.DataFrame,
    original_gap_gap: pd.DataFrame,
) -> pd.DataFrame:
    if original_gap_gap.empty or holdout_gap_gap_all.empty:
        return pd.DataFrame()

    hold = holdout_gap_gap_all.copy()
    hold["gap_pair_key"] = hold.apply(
        lambda r: " || ".join(sorted([str(r["gap_a"]), str(r["gap_b"])])),
        axis=1,
    )

    hold = hold.rename(
        columns={
            "beta_std_symmetric": "beta_std_symmetric_holdout",
            "p_pair": "p_pair_holdout",
            "q_fdr_bh_unique_pairs": "q_fdr_bh_unique_pairs_holdout",
            "n_complete": "n_complete_holdout",
        }
    )

    keep_hold = [
        "gap_pair_key",
        "beta_std_symmetric_holdout",
        "p_pair_holdout",
        "q_fdr_bh_unique_pairs_holdout",
        "n_complete_holdout",
    ]

    comp = original_gap_gap.merge(
        hold[keep_hold],
        on="gap_pair_key",
        how="left",
    )

    comp["tested_in_holdout"] = comp["beta_std_symmetric_holdout"].notna()
    comp["same_sign"] = (
        np.sign(comp["beta_std_symmetric_original"])
        == np.sign(comp["beta_std_symmetric_holdout"])
    )
    comp.loc[~comp["tested_in_holdout"], "same_sign"] = np.nan

    comp["abs_beta_retained_pct"] = (
        comp["beta_std_symmetric_holdout"].abs()
        / comp["beta_std_symmetric_original"].abs()
        * 100.0
    )

    comp["p05_holdout"] = comp["p_pair_holdout"].lt(0.05)
    comp["fdr05_holdout"] = comp["q_fdr_bh_unique_pairs_holdout"].lt(FDR_ALPHA)

    return comp.sort_values(
        ["tested_in_holdout", "same_sign", "abs_beta_retained_pct"],
        ascending=[True, True, True],
    )


# =============================================================================
# REPORT
# =============================================================================

def format_pct(x) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):.1f}%"


def make_text_report(
    gap_df: pd.DataFrame,
    predictor_comp: pd.DataFrame,
    predictor_summary: pd.DataFrame,
    holdout_fdr_edges: pd.DataFrame,
    gap_gap_comp: pd.DataFrame,
) -> str:
    lines = []

    lines.append("Largest-site-holdout association robustness report")
    lines.append("=" * 80)
    lines.append("")

    if SITE_COL in gap_df.columns:
        sites = sorted(gap_df[SITE_COL].astype(str).dropna().unique().tolist())
        lines.append(f"Heldout site(s) included: {', '.join(sites)}")

    lines.append(f"Total subjects in heldout-site association set: n={len(gap_df):,}")
    lines.append("")
    lines.append("Split counts:")
    lines.append(gap_df[SPLIT_GROUP_COL].value_counts(dropna=False).to_string())
    lines.append("")
    lines.append("Analysis-label counts:")
    lines.append(gap_df["analysis_label"].value_counts(dropna=False).to_string())
    lines.append("")

    lines.append("Heldout-site FDR-significant predictor-gap edges")
    lines.append("-" * 80)
    lines.append(f"n={len(holdout_fdr_edges):,}")
    lines.append("")

    if predictor_summary is not None and len(predictor_summary) > 0:
        overall = predictor_summary[predictor_summary["summary_level"].eq("overall")]
        if len(overall) > 0:
            r = overall.iloc[0]
            lines.append("Comparison to original significant predictor-gap edges")
            lines.append("-" * 80)
            lines.append(
                f"Original significant edges tested in heldout site: "
                f"{int(r['n_tested_in_holdout'])}/{int(r['n_original_edges'])}"
            )
            lines.append(
                f"Same sign: {int(r['n_same_sign'])}/{int(r['n_tested_in_holdout'])} "
                f"({format_pct(r['pct_same_sign'])})"
            )
            lines.append(
                f"Nominal p<0.05 in heldout site: "
                f"{int(r['n_p05_holdout'])}/{int(r['n_tested_in_holdout'])} "
                f"({format_pct(r['pct_p05_holdout'])})"
            )
            lines.append(
                f"FDR q<0.05 in heldout site: "
                f"{int(r['n_fdr05_holdout'])}/{int(r['n_tested_in_holdout'])} "
                f"({format_pct(r['pct_fdr05_holdout'])})"
            )
            lines.append(
                f"Median retained absolute beta: "
                f"{format_pct(r['median_abs_beta_retained_pct'])}"
            )
            lines.append(
                f"Minimum retained absolute beta: "
                f"{format_pct(r['min_abs_beta_retained_pct'])}"
            )
            lines.append("")

    if predictor_comp is not None and len(predictor_comp) > 0:
        flips = predictor_comp[
            predictor_comp["tested_in_holdout"].fillna(False)
            & predictor_comp["same_sign"].eq(False)
        ].copy()

        lines.append("Potential sign-flip predictor-gap edges")
        lines.append("-" * 80)
        if len(flips) == 0:
            lines.append("None.")
        else:
            cols = [
                "source",
                "predictor",
                "gap",
                "beta_std_original",
                "beta_std_holdout",
                "p_hc3_holdout",
                "q_fdr_bh_holdout",
            ]
            lines.append(flips[cols].to_string(index=False))
        lines.append("")

    if gap_gap_comp is not None and len(gap_gap_comp) > 0:
        tested = gap_gap_comp[gap_gap_comp["tested_in_holdout"].fillna(False)].copy()
        if len(tested) > 0:
            same_sign = tested["same_sign"].mean() * 100.0
            n_same = int(tested["same_sign"].fillna(False).sum())
            n_fdr = int(tested["fdr05_holdout"].fillna(False).sum())
            lines.append("Gap-gap edge comparison")
            lines.append("-" * 80)
            lines.append(
                f"Original significant gap-gap edges tested in heldout site: "
                f"{len(tested)}/{len(gap_gap_comp)}"
            )
            lines.append(
                f"Same sign: {n_same}/{len(tested)} ({same_sign:.1f}%)"
            )
            lines.append(
                f"FDR q<0.05 in heldout site: {n_fdr}/{len(tested)}"
            )
            lines.append(
                f"Median retained absolute beta: "
                f"{tested['abs_beta_retained_pct'].median():.1f}%"
            )
            lines.append("")

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 100)
    print("Largest-site-holdout Figure-2 association robustness")
    print("=" * 100)

    gap_df, gap_cols = load_largest_site_holdout_gap_space()
    gap_df = exclude_high_framewise_from_connectivity_gaps(gap_df, gap_cols)
    gap_df = merge_diagnosis_scores_and_mean_motion(gap_df)

    graph_cols = graph_gap_cols(gap_cols)

    print("\nGap columns used for predictor-gap tests")
    print("=" * 80)
    for g in graph_cols:
        print(g)

    print("\nFitting standard diagnosis -> gap HC3 models")
    print("=" * 80)
    diagnosis_edges = fit_standard_diagnosis_gap_edges(
        gap_df=gap_df,
        gap_cols=graph_cols,
    )

    print("\nFitting standard clinical-score -> gap HC3 models")
    print("=" * 80)
    score_edges = fit_standard_score_gap_edges(
        gap_df=gap_df,
        gap_cols=graph_cols,
    )

    all_predictor_edges, fdr_predictor_edges = combine_predictor_gap_edges(
        diagnosis_edges=diagnosis_edges,
        score_edges=score_edges,
    )

    print("\nFitting gap-gap HC3 models")
    print("=" * 80)
    gap_gap_all, gap_gap_fdr = compute_gap_adjusted_ols_hc3_edges(
        gap_df=gap_df,
        gap_cols=gap_cols,
    )

    # Save heldout results.
    diagnosis_path = os.path.join(OUT_DIR, "largest_site_holdout_diagnosis_gap_edges_all.csv")
    score_path = os.path.join(OUT_DIR, "largest_site_holdout_score_gap_edges_all.csv")
    all_pred_path = os.path.join(OUT_DIR, "largest_site_holdout_predictor_gap_edges_all.csv")
    fdr_pred_path = os.path.join(OUT_DIR, "largest_site_holdout_predictor_gap_edges_fdr_significant.csv")
    gapgap_all_path = os.path.join(OUT_DIR, "largest_site_holdout_gap_gap_edges_all.csv")
    gapgap_fdr_path = os.path.join(OUT_DIR, "largest_site_holdout_gap_gap_edges_fdr_significant.csv")

    diagnosis_edges.to_csv(diagnosis_path, index=False)
    score_edges.to_csv(score_path, index=False)
    all_predictor_edges.to_csv(all_pred_path, index=False)
    fdr_predictor_edges.to_csv(fdr_pred_path, index=False)
    gap_gap_all.to_csv(gapgap_all_path, index=False)
    gap_gap_fdr.to_csv(gapgap_fdr_path, index=False)

    print("\nSaved heldout-site association results")
    print("=" * 80)
    print(diagnosis_path)
    print(score_path)
    print(all_pred_path)
    print(fdr_pred_path)
    print(gapgap_all_path)
    print(gapgap_fdr_path)

    # Compare predictor-gap edges to original main Figure-2 significant edges.
    original_predictor_edges = load_original_predictor_edges()
    predictor_comp = compare_predictor_edges_to_original(
        holdout_all_edges=all_predictor_edges,
        original_edges=original_predictor_edges,
    )

    predictor_summary = summarize_predictor_edge_robustness(predictor_comp)

    comp_path = os.path.join(OUT_DIR, "largest_site_holdout_predictor_gap_edges_vs_original.csv")
    summary_path = os.path.join(OUT_DIR, "largest_site_holdout_predictor_gap_edges_robustness_summary.csv")

    if len(predictor_comp) > 0:
        predictor_comp.to_csv(comp_path, index=False)
        print("\nSaved predictor-gap comparison to original significant edges:")
        print(comp_path)

    if len(predictor_summary) > 0:
        predictor_summary.to_csv(summary_path, index=False)
        print("\nSaved predictor-gap robustness summary:")
        print(summary_path)

        print("\nRobustness summary")
        print("=" * 80)
        print(predictor_summary.to_string(index=False))

    # Compare gap-gap edges to original significant gap-gap table.
    original_gap_gap = load_original_gap_gap_edges()
    gap_gap_comp = compare_gap_gap_edges_to_original(
        holdout_gap_gap_all=gap_gap_all,
        original_gap_gap=original_gap_gap,
    )

    gapgap_comp_path = os.path.join(OUT_DIR, "largest_site_holdout_gap_gap_edges_vs_original.csv")
    if len(gap_gap_comp) > 0:
        gap_gap_comp.to_csv(gapgap_comp_path, index=False)
        print("\nSaved gap-gap comparison to original significant edges:")
        print(gapgap_comp_path)

    report = make_text_report(
        gap_df=gap_df,
        predictor_comp=predictor_comp,
        predictor_summary=predictor_summary,
        holdout_fdr_edges=fdr_predictor_edges,
        gap_gap_comp=gap_gap_comp,
    )

    report_path = os.path.join(OUT_DIR, "largest_site_holdout_association_robustness_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\nSaved text report:")
    print(report_path)

    print("\n" + "=" * 100)
    print("Report preview")
    print("=" * 100)
    print(report)

    print("\nDone.")


if __name__ == "__main__":
    main()

