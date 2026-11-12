from __future__ import annotations

import hashlib
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shap
from catboost import Pool
from rasterio.transform import xy
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


warnings.filterwarnings(
    "ignore",
    message="X has feature names, but StandardScaler was fitted without feature names",
)
warnings.filterwarnings(
    "ignore",
    message="The NumPy global RNG was seeded by calling `np.random.seed`",
    category=FutureWarning,
)


ROOT = Path(r".")
DATA_PATH = ROOT / "Data" / "tabular_data_14" / "train" / "v4" / "rswq_model_input_relaxed_s2_context_2562_2568.csv"
ARTIFACT_DIR = ROOT / "Models" / "mapping_v1" / "artifacts"
MOVING_WINDOW_DIR = ROOT / "Data" / "tabular_data_14" / "moving_window"
PREDICTION_DIR = ROOT / "Data" / "tabular_data_14" / "prediction_maps" / "v1" / "geotiff"
OUTPUT_DIR = ROOT / "Results" / "XAI" / "v1"

TRAIN_YEAR_MIN = 2562
TRAIN_YEAR_MAX = 2566
EVAL_YEAR_MIN = 2567
EVAL_YEAR_MAX = 2568
MAX_MATCH_DAYS = 1
SS_MAX_MG_L = 130.0
RANDOM_STATE = 20260804
N_BOOTSTRAP = 1000
PERMUTATION_ROUNDS = 5
SHAP_BACKGROUND_SIZE = 25
OOD_MAX = 0.25
MAX_STATION_SNAP_M = 500.0

MAPS = {
    "WB02": {"name": "Thale Luang", "thai": "ทะเลหลวง", "date": "2024-07-19"},
    "WB07": {"name": "Lower Lang Suan River", "thai": "แม่น้ำหลังสวนตอนล่าง", "date": "2025-01-20"},
    "WB05": {"name": "Pak Phanang River", "thai": "แม่น้ำปากพนัง", "date": "2025-07-04"},
}

# These station locations are selected before SHAP is inspected. The two river-mouth
# labels are station-location proxies and must be checked by a domain expert before
# the word "river mouth" is used in the manuscript.
PREDEFINED_LOCATION_ROLES = {
    ("WB02", "SK07"): "representative_field_station",
    ("WB05", "PN02"): "river_mouth_station_proxy_requires_confirmation",
    ("WB07", "LS01"): "river_mouth_station_proxy_requires_confirmation",
}

TARGET_CONFIG = {
    "Turbidity": {
        "artifact": "turbidity_s2_context_linearsvr.joblib",
        "target_col": "turbidity_NTU_clean",
        "unit": "NTU",
        "global_scale": "log1p(NTU)",
        "prediction_band": "turbidity_s2_context",
        "ood_band": "turbidity_s2_context_ood_fraction",
    },
    "SS": {
        "artifact": "ss_s2_context_two_stage_bundle.joblib",
        "target_col": "SS_mg_l_clean",
        "unit": "mg/L",
        "global_scale": "mg/L",
        "prediction_band": "ss_s2_context",
        "ood_band": "ss_s2_context_ood_fraction",
    },
    "DO proxy": {
        "artifact": "do_s2_context_catboost.joblib",
        "target_col": "DO_mg_l_clean",
        "unit": "mg/L",
        "global_scale": "mg/L",
        "prediction_band": "do_s2_context",
        "ood_band": "do_s2_context_ood_fraction",
    },
}


@dataclass
class ShapResult:
    target: str
    feature_names: list[str]
    feature_groups: list[str]
    output_scale: str
    eval_values: np.ndarray
    eval_base_values: np.ndarray
    eval_raw: pd.DataFrame
    eval_frame: pd.DataFrame
    eval_pred_output: np.ndarray
    eval_pred_original: np.ndarray
    train_values: np.ndarray
    train_base_values: np.ndarray
    train_raw: pd.DataFrame
    train_frame: pd.DataFrame
    train_pred_output: np.ndarray
    train_pred_original: np.ndarray
    explainer: object | None = None


def ensure_dirs() -> dict[str, Path]:
    dirs = {
        "root": OUTPUT_DIR,
        "tables": OUTPUT_DIR / "tables",
        "figures": OUTPUT_DIR / "figures",
        "local": OUTPUT_DIR / "figures" / "local_waterfalls",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_ratio(numerator: pd.Series, denominator: pd.Series, eps: float = 1e-12) -> np.ndarray:
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    out = np.where(np.abs(den) > eps, num / den, np.nan)
    return np.where(np.isfinite(out), out, np.nan)


def normalized_difference(a: pd.Series, b: pd.Series, eps: float = 1e-12) -> np.ndarray:
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    denominator = a + b
    out = np.where(np.abs(denominator) > eps, (a - b) / denominator, np.nan)
    return np.where(np.isfinite(out), out, np.nan)


def add_custom_s2_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    blue = out["B2_median"]
    green = out["B3_median"]
    red = out["B4_median"]
    nir = out["B8_median"]
    swir_1 = out["B11_median"]
    swir_2 = out["B12_median"]
    out["NDWI1"] = normalized_difference(green, nir)
    out["NDWI2"] = normalized_difference(nir, swir_1)
    out["NDWI3"] = normalized_difference(nir, swir_2)
    out["B4_B3_ratio"] = safe_ratio(red, green)
    out["B8_B4_ratio"] = safe_ratio(nir, red)
    out["NIR_Blue_ratio"] = safe_ratio(nir, blue)
    out["NIR_Green_ratio"] = safe_ratio(nir, green)
    out["NIR_Red_ratio"] = safe_ratio(nir, red)
    out["Red_NIR_ND"] = normalized_difference(red, nir)
    return out


def target_feature_group(target: str, feature: str, s2_features: set[str]) -> str:
    if feature in s2_features:
        return "Sentinel-2 spectral"
    if target == "Turbidity":
        if feature.startswith("rain_") or feature.startswith("rainy_"):
            return "Rainfall"
        if "runoff" in feature:
            return "Runoff"
        if feature.startswith("wc_water_"):
            return "Surrounding-water fraction"
    elif target == "SS":
        if feature in {"local_cloud_pct", "scene_cloud_pct"}:
            return "Cloud QA"
        if feature.startswith("jrc_"):
            return "Historical-water occurrence"
        if feature.startswith("wc_water_"):
            return "Surrounding-water fraction"
        if feature in {
            "valid_water_pixel_count",
            "water_pixel_fraction",
            "candidate_water_count",
            "water_pixel_pct",
        }:
            return "Water-pixel QA"
    elif target == "DO proxy":
        if feature in {"month", "day_of_year", "doy_sin", "doy_cos"}:
            return "Seasonality"
        if feature.startswith("rain_") or feature.startswith("rainy_"):
            return "Rainfall"
        if "wind" in feature:
            return "Wind"
        if "solar" in feature:
            return "Solar radiation"
        if "air_temp" in feature:
            return "Air temperature"
        if feature.startswith("wc_"):
            return "Land cover"
        if feature.startswith("jrc_"):
            return "Historical water"
    return "Other context"


def s2_feature_set(target: str, features: list[str]) -> set[str]:
    if target == "SS":
        return {
            feature for feature in features
            if feature.startswith(("B", "ND", "NIR_", "Red_"))
        }
    return {
        feature for feature in features
        if feature.startswith("B") or feature.startswith("ND")
    }


def broadcast_base_values(base_values: np.ndarray | float, n_rows: int) -> np.ndarray:
    values = np.asarray(base_values, dtype=float)
    if values.ndim == 0:
        return np.full(n_rows, float(values))
    values = values.reshape(-1)
    if len(values) == 1:
        return np.full(n_rows, float(values[0]))
    if len(values) != n_rows:
        raise ValueError(f"Unexpected SHAP base-value shape: {values.shape}; expected {n_rows} rows")
    return values


def exact_prediction(bundle: dict, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    strategy = bundle["prediction_contract"]["strategy"]
    features = bundle["feature_names"]
    X = X[features]
    if strategy == "log1p_regression":
        pred_output = np.asarray(bundle["estimator"].predict(X), dtype=float).reshape(-1)
        pred_original = np.expm1(np.clip(pred_output, 0, None))
    elif strategy == "two_stage_soft_gate":
        p_high = bundle["classifier"].predict_proba(X)[:, 1]
        high_log = np.asarray(bundle["regressor"].predict(X), dtype=float).reshape(-1)
        high_original = np.expm1(np.clip(high_log, 0, None))
        pred_original = p_high * high_original + (1.0 - p_high) * float(bundle["low_value"])
        pred_output = pred_original.copy()
    elif strategy == "raw_regression":
        pred_original = np.asarray(bundle["estimator"].predict(X), dtype=float).reshape(-1)
        pred_output = pred_original.copy()
    else:
        raise ValueError(f"Unsupported prediction strategy: {strategy}")
    return pred_output, pred_original


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman_r": float(spearmanr(y_true, y_pred).statistic),
        "pearson_r": float(pearsonr(y_true, y_pred).statistic),
        "bias": float(np.mean(y_pred - y_true)),
    }


def load_inputs() -> tuple[pd.DataFrame, dict[str, dict], dict[str, dict[str, pd.DataFrame]], pd.DataFrame]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)
    data = add_custom_s2_features(pd.read_csv(DATA_PATH, low_memory=False))
    data["year_be"] = pd.to_numeric(data["year_be"], errors="coerce")
    data["abs_date_diff_days"] = pd.to_numeric(data["abs_date_diff_days"], errors="coerce")

    bundles: dict[str, dict] = {}
    splits: dict[str, dict[str, pd.DataFrame]] = {}
    verification_rows: list[dict] = []

    for target, config in TARGET_CONFIG.items():
        artifact_path = ARTIFACT_DIR / config["artifact"]
        if not artifact_path.exists():
            raise FileNotFoundError(artifact_path)
        bundle = joblib.load(artifact_path)
        features = list(bundle["feature_names"])
        missing = [feature for feature in features if feature not in data.columns]
        if missing:
            raise KeyError(f"{target}: missing dataset features: {missing}")

        target_values = pd.to_numeric(data[config["target_col"]], errors="coerce")
        base_mask = target_values.notna() & data["abs_date_diff_days"].le(MAX_MATCH_DAYS)
        if target == "SS":
            base_mask &= target_values.le(SS_MAX_MG_L)
        train = data.loc[
            base_mask & data["year_be"].between(TRAIN_YEAR_MIN, TRAIN_YEAR_MAX)
        ].copy()
        evaluation = data.loc[
            base_mask & data["year_be"].between(EVAL_YEAR_MIN, EVAL_YEAR_MAX)
        ].copy()

        if len(train) != int(bundle["n_train"]) or len(evaluation) != int(bundle["n_test"]):
            raise AssertionError(
                f"{target}: split differs from artifact: got {len(train)}/{len(evaluation)}, "
                f"expected {bundle['n_train']}/{bundle['n_test']}"
            )

        pred_output, pred_original = exact_prediction(bundle, evaluation[features])
        metrics = regression_metrics(evaluation[config["target_col"]].to_numpy(float), pred_original)
        stored_r2 = bundle["validation_metrics"].get("original_r2", bundle["validation_metrics"].get("r2"))
        stored_rmse = bundle["validation_metrics"].get("original_rmse", bundle["validation_metrics"].get("rmse"))
        if not np.isclose(metrics["r2"], stored_r2, atol=1e-10):
            raise AssertionError(f"{target}: recomputed R2 {metrics['r2']} != stored {stored_r2}")
        if not np.isclose(metrics["rmse"], stored_rmse, atol=1e-10):
            raise AssertionError(f"{target}: recomputed RMSE {metrics['rmse']} != stored {stored_rmse}")

        bundles[target] = bundle
        splits[target] = {"train": train, "evaluation": evaluation}
        verification_rows.append({
            "target": target,
            "artifact_id": bundle["artifact_id"],
            "feature_set": bundle["feature_set"],
            "model": (
                "LogisticRegression + LinearSVR (soft gate)"
                if bundle["prediction_contract"]["strategy"] == "two_stage_soft_gate"
                else type(bundle["estimator"].named_steps["model"]).__name__
            ),
            "n_features": len(features),
            "n_train": len(train),
            "n_evaluation": len(evaluation),
            "evaluation_years": "2567-2568",
            "r2": metrics["r2"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "spearman_r": metrics["spearman_r"],
            "bias": metrics["bias"],
            "metric_match_artifact": True,
            "artifact_path": str(artifact_path),
        })
    return data, bundles, splits, pd.DataFrame(verification_rows)


def linear_shap_result(
    target: str,
    bundle: dict,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    target_col: str,
) -> ShapResult:
    features = list(bundle["feature_names"])
    pipeline: Pipeline = bundle["estimator"]
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    X_train_raw = train[features].copy()
    X_eval_raw = evaluation[features].copy()
    X_train_imputed = pd.DataFrame(imputer.transform(X_train_raw), columns=features, index=train.index)
    X_eval_imputed = pd.DataFrame(imputer.transform(X_eval_raw), columns=features, index=evaluation.index)
    X_train_model = pd.DataFrame(scaler.transform(X_train_imputed), columns=features, index=train.index)
    X_eval_model = pd.DataFrame(scaler.transform(X_eval_imputed), columns=features, index=evaluation.index)

    explainer = shap.LinearExplainer(model, X_train_model)
    train_exp = explainer(X_train_model)
    eval_exp = explainer(X_eval_model)
    train_output, train_original = exact_prediction(bundle, X_train_raw)
    eval_output, eval_original = exact_prediction(bundle, X_eval_raw)
    s2_features = s2_feature_set(target, features)

    return ShapResult(
        target=target,
        feature_names=features,
        feature_groups=[target_feature_group(target, feature, s2_features) for feature in features],
        output_scale=TARGET_CONFIG[target]["global_scale"],
        eval_values=np.asarray(eval_exp.values, dtype=float),
        eval_base_values=broadcast_base_values(eval_exp.base_values, len(evaluation)),
        eval_raw=X_eval_imputed,
        eval_frame=evaluation,
        eval_pred_output=eval_output,
        eval_pred_original=eval_original,
        train_values=np.asarray(train_exp.values, dtype=float),
        train_base_values=broadcast_base_values(train_exp.base_values, len(train)),
        train_raw=X_train_imputed,
        train_frame=train,
        train_pred_output=train_output,
        train_pred_original=train_original,
        explainer=explainer,
    )


def permutation_shap_result(
    target: str,
    bundle: dict,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    random_state: int,
) -> ShapResult:
    features = list(bundle["feature_names"])
    X_train = train[features].copy()
    X_eval = evaluation[features].copy()
    background = X_train.sample(n=min(SHAP_BACKGROUND_SIZE, len(X_train)), random_state=random_state)

    def predict_fn(values: np.ndarray | pd.DataFrame) -> np.ndarray:
        frame = pd.DataFrame(values, columns=features)
        return exact_prediction(bundle, frame)[1]

    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.Explainer(
        predict_fn,
        masker,
        algorithm="permutation",
        feature_names=features,
        seed=random_state,
    )
    max_evals = PERMUTATION_ROUNDS * (2 * len(features) + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_exp = explainer(X_train, max_evals=max_evals, silent=True)
        eval_exp = explainer(X_eval, max_evals=max_evals, silent=True)

    train_output, train_original = exact_prediction(bundle, X_train)
    eval_output, eval_original = exact_prediction(bundle, X_eval)
    s2_features = s2_feature_set(target, features)
    return ShapResult(
        target=target,
        feature_names=features,
        feature_groups=[target_feature_group(target, feature, s2_features) for feature in features],
        output_scale=TARGET_CONFIG[target]["global_scale"],
        eval_values=np.asarray(eval_exp.values, dtype=float),
        eval_base_values=broadcast_base_values(eval_exp.base_values, len(evaluation)),
        eval_raw=X_eval,
        eval_frame=evaluation,
        eval_pred_output=eval_output,
        eval_pred_original=eval_original,
        train_values=np.asarray(train_exp.values, dtype=float),
        train_base_values=broadcast_base_values(train_exp.base_values, len(train)),
        train_raw=X_train,
        train_frame=train,
        train_pred_output=train_output,
        train_pred_original=train_original,
        explainer=explainer,
    )


def catboost_shap_result(
    target: str,
    bundle: dict,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> ShapResult:
    features = list(bundle["feature_names"])
    pipeline: Pipeline = bundle["estimator"]
    imputer = pipeline.named_steps["imputer"]
    model = pipeline.named_steps["model"]
    X_train_raw = train[features].copy()
    X_eval_raw = evaluation[features].copy()
    X_train = pd.DataFrame(imputer.transform(X_train_raw), columns=features, index=train.index)
    X_eval = pd.DataFrame(imputer.transform(X_eval_raw), columns=features, index=evaluation.index)

    train_native = np.asarray(model.get_feature_importance(Pool(X_train, feature_names=features), type="ShapValues"))
    eval_native = np.asarray(model.get_feature_importance(Pool(X_eval, feature_names=features), type="ShapValues"))
    train_output, train_original = exact_prediction(bundle, X_train_raw)
    eval_output, eval_original = exact_prediction(bundle, X_eval_raw)
    s2_features = s2_feature_set(target, features)
    return ShapResult(
        target=target,
        feature_names=features,
        feature_groups=[target_feature_group(target, feature, s2_features) for feature in features],
        output_scale=TARGET_CONFIG[target]["global_scale"],
        eval_values=train_native[:0] if False else eval_native[:, :-1],
        eval_base_values=eval_native[:, -1],
        eval_raw=X_eval,
        eval_frame=evaluation,
        eval_pred_output=eval_output,
        eval_pred_original=eval_original,
        train_values=train_native[:, :-1],
        train_base_values=train_native[:, -1],
        train_raw=X_train,
        train_frame=train,
        train_pred_output=train_output,
        train_pred_original=train_original,
        explainer=None,
    )


def compute_stage_specific_ss_shap(
    bundle: dict,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    dirs: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = list(bundle["feature_names"])
    target_col = TARGET_CONFIG["SS"]["target_col"]

    classifier: Pipeline = bundle["classifier"]
    cls_imputer = classifier.named_steps["imputer"]
    cls_scaler = classifier.named_steps["scaler"]
    cls_model = classifier.named_steps["model"]
    X_train_imp = pd.DataFrame(cls_imputer.transform(train[features]), columns=features)
    X_eval_imp = pd.DataFrame(cls_imputer.transform(evaluation[features]), columns=features)
    X_train_scaled = pd.DataFrame(cls_scaler.transform(X_train_imp), columns=features)
    X_eval_scaled = pd.DataFrame(cls_scaler.transform(X_eval_imp), columns=features)
    cls_explainer = shap.LinearExplainer(cls_model, X_train_scaled)
    cls_exp = cls_explainer(X_eval_scaled)
    cls_values = np.asarray(cls_exp.values, dtype=float)

    regressor: Pipeline = bundle["regressor"]
    high_train = train[target_col].gt(float(bundle["low_threshold"]))
    high_eval = evaluation[target_col].gt(float(bundle["low_threshold"]))
    reg_imputer = regressor.named_steps["imputer"]
    reg_scaler = regressor.named_steps["scaler"]
    reg_model = regressor.named_steps["model"]
    X_reg_train_imp = pd.DataFrame(reg_imputer.transform(train.loc[high_train, features]), columns=features)
    X_reg_eval_imp = pd.DataFrame(reg_imputer.transform(evaluation.loc[high_eval, features]), columns=features)
    X_reg_train_scaled = pd.DataFrame(reg_scaler.transform(X_reg_train_imp), columns=features)
    X_reg_eval_scaled = pd.DataFrame(reg_scaler.transform(X_reg_eval_imp), columns=features)
    reg_explainer = shap.LinearExplainer(reg_model, X_reg_train_scaled)
    reg_exp = reg_explainer(X_reg_eval_scaled)
    reg_values = np.asarray(reg_exp.values, dtype=float)

    s2_features = s2_feature_set("SS", features)
    groups = [target_feature_group("SS", feature, s2_features) for feature in features]
    classifier_table = pd.DataFrame({
        "stage": "Stage 1 LogisticRegression",
        "output_scale": "log-odds of high SS regime",
        "feature": features,
        "feature_group": groups,
        "mean_abs_shap": np.mean(np.abs(cls_values), axis=0),
        "mean_shap": np.mean(cls_values, axis=0),
        "n_explained": len(evaluation),
    }).sort_values("mean_abs_shap", ascending=False)
    regressor_table = pd.DataFrame({
        "stage": "Stage 2 LinearSVR",
        "output_scale": "log1p(SS) among observed high-regime rows",
        "feature": features,
        "feature_group": groups,
        "mean_abs_shap": np.mean(np.abs(reg_values), axis=0),
        "mean_shap": np.mean(reg_values, axis=0),
        "n_explained": int(high_eval.sum()),
    }).sort_values("mean_abs_shap", ascending=False)
    classifier_table.to_csv(dirs["tables"] / "ss_stage1_classifier_shap.csv", index=False)
    regressor_table.to_csv(dirs["tables"] / "ss_stage2_regressor_shap.csv", index=False)

    for table, values, raw, title, filename, xlabel in [
        (
            classifier_table,
            cls_values,
            X_eval_imp,
            "SS Stage 1: low/high-regime classifier",
            "figure_ss_stage1_classifier_shap.png",
            "Mean |SHAP| (classifier log-odds)",
        ),
        (
            regressor_table,
            reg_values,
            X_reg_eval_imp,
            "SS Stage 2: high-regime LinearSVR",
            "figure_ss_stage2_regressor_shap.png",
            "Mean |SHAP| on log1p(SS)",
        ),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
        top = table.head(12).sort_values("mean_abs_shap")
        axes[0].barh(top["feature"], top["mean_abs_shap"], color="#4C78A8")
        axes[0].set_xlabel(xlabel)
        axes[0].set_title("Global importance")
        axes[0].grid(axis="x", alpha=0.25)
        plt.sca(axes[1])
        shap.summary_plot(values, raw, feature_names=features, max_display=12, show=False)
        axes[1].set_title("Direction and distribution")
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(dirs["figures"] / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return classifier_table, regressor_table


def result_to_long(result: ShapResult, split: str) -> pd.DataFrame:
    if split == "evaluation":
        values = result.eval_values
        base_values = result.eval_base_values
        raw = result.eval_raw
        frame = result.eval_frame
        pred_output = result.eval_pred_output
        pred_original = result.eval_pred_original
    else:
        values = result.train_values
        base_values = result.train_base_values
        raw = result.train_raw
        frame = result.train_frame
        pred_output = result.train_pred_output
        pred_original = result.train_pred_original

    target_col = TARGET_CONFIG[result.target]["target_col"]
    metadata = pd.DataFrame({
        "row_position": np.arange(len(frame)),
        "observation_id": frame.get("sample_id", pd.Series(frame.index, index=frame.index)).astype(str).to_numpy(),
        "station": frame.get("station_canonical", pd.Series("unknown", index=frame.index)).astype(str).to_numpy(),
        "field_date": frame.get("field_date", pd.Series(pd.NA, index=frame.index)).astype(str).to_numpy(),
        "sat_date": frame.get("sat_date", pd.Series(pd.NA, index=frame.index)).astype(str).to_numpy(),
        "year_be": frame["year_be"].to_numpy(),
        "observed_original": frame[target_col].to_numpy(float),
        "predicted_original": pred_original,
        "predicted_output_scale": pred_output,
        "base_value": base_values,
    })
    pieces = []
    for feature_idx, feature in enumerate(result.feature_names):
        part = metadata.copy()
        part["target"] = result.target
        part["split"] = split
        part["output_scale"] = result.output_scale
        part["feature"] = feature
        part["feature_group"] = result.feature_groups[feature_idx]
        part["raw_feature_value"] = pd.to_numeric(raw.iloc[:, feature_idx], errors="coerce").to_numpy()
        part["shap_value"] = values[:, feature_idx]
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def make_global_tables(results: dict[str, ShapResult], dirs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    long_table = pd.concat(
        [result_to_long(result, split) for result in results.values() for split in ["train", "evaluation"]],
        ignore_index=True,
    )
    long_table.to_csv(dirs["tables"] / "global_shap_long.csv", index=False)

    importance_rows = []
    group_rows = []
    additivity_rows = []
    for target, result in results.items():
        mean_abs = np.mean(np.abs(result.eval_values), axis=0)
        mean_signed = np.mean(result.eval_values, axis=0)
        total = float(mean_abs.sum())
        for rank, feature_index in enumerate(np.argsort(-mean_abs), start=1):
            importance_rows.append({
                "target": target,
                "rank": rank,
                "feature": result.feature_names[feature_index],
                "feature_group": result.feature_groups[feature_index],
                "mean_abs_shap": mean_abs[feature_index],
                "mean_shap": mean_signed[feature_index],
                "share_of_absolute_shap": mean_abs[feature_index] / total if total else np.nan,
                "output_scale": result.output_scale,
                "n_evaluation": len(result.eval_frame),
            })
        target_importance = pd.DataFrame([row for row in importance_rows if row["target"] == target])
        for group, group_df in target_importance.groupby("feature_group"):
            group_total = float(group_df["mean_abs_shap"].sum())
            group_rows.append({
                "target": target,
                "feature_group": group,
                "n_features": len(group_df),
                "total_mean_abs_shap": group_total,
                "mean_abs_shap_per_feature": float(group_df["mean_abs_shap"].mean()),
                "group_attribution_share": group_total / total if total else np.nan,
                "output_scale": result.output_scale,
            })
        reconstructed_eval = result.eval_base_values + result.eval_values.sum(axis=1)
        reconstructed_train = result.train_base_values + result.train_values.sum(axis=1)
        additivity_rows.extend([
            {
                "target": target,
                "split": "evaluation",
                "output_scale": result.output_scale,
                "max_abs_additivity_error": float(np.max(np.abs(reconstructed_eval - result.eval_pred_output))),
                "mean_abs_additivity_error": float(np.mean(np.abs(reconstructed_eval - result.eval_pred_output))),
            },
            {
                "target": target,
                "split": "train",
                "output_scale": result.output_scale,
                "max_abs_additivity_error": float(np.max(np.abs(reconstructed_train - result.train_pred_output))),
                "mean_abs_additivity_error": float(np.mean(np.abs(reconstructed_train - result.train_pred_output))),
            },
        ])

    importance = pd.DataFrame(importance_rows)
    groups = pd.DataFrame(group_rows)
    top_five = importance.loc[importance["rank"].le(5)].copy()
    additivity = pd.DataFrame(additivity_rows)
    importance.to_csv(dirs["tables"] / "global_feature_importance.csv", index=False)
    top_five.to_csv(dirs["tables"] / "top_five_features.csv", index=False)
    groups.to_csv(dirs["tables"] / "feature_group_attribution.csv", index=False)
    additivity.to_csv(dirs["tables"] / "shap_additivity_audit.csv", index=False)
    return {
        "long": long_table,
        "importance": importance,
        "top_five": top_five,
        "groups": groups,
        "additivity": additivity,
    }


def make_global_figures(results: dict[str, ShapResult], tables: dict[str, pd.DataFrame], dirs: dict[str, Path]) -> pd.DataFrame:
    group_palette = {
        "Sentinel-2 spectral": "#2F6B9A",
        "Rainfall": "#3A9D5D",
        "Runoff": "#1B998B",
        "Surrounding-water fraction": "#6BAED6",
        "Water-pixel QA": "#7B6FD0",
        "Cloud QA": "#A76DCA",
        "Historical-water occurrence": "#4C956C",
        "Seasonality": "#E69F00",
        "Wind": "#56B4E9",
        "Solar radiation": "#F0C808",
        "Air temperature": "#D55E00",
        "Land cover": "#8C6D31",
        "Historical water": "#4C956C",
        "Other context": "#8C8C8C",
    }

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2))
    for ax, target in zip(axes, TARGET_CONFIG):
        target_df = tables["importance"].loc[tables["importance"]["target"].eq(target)].head(12).sort_values("mean_abs_shap")
        ax.barh(
            target_df["feature"],
            target_df["mean_abs_shap"],
            color=target_df["feature_group"].map(group_palette).fillna("#8C8C8C"),
        )
        ax.set_title(target)
        ax.set_xlabel(f"Mean |SHAP| [{TARGET_CONFIG[target]['global_scale']}]")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Global SHAP importance of the locked S2 + context models", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "figure_global_shap_bar.png", dpi=300, bbox_inches="tight")
    fig.savefig(dirs["figures"] / "figure_global_shap_bar.pdf", bbox_inches="tight")
    plt.close(fig)

    for target, result in results.items():
        explanation = shap.Explanation(
            values=result.eval_values,
            base_values=result.eval_base_values,
            data=result.eval_raw.to_numpy(float),
            feature_names=result.feature_names,
        )
        shap.plots.beeswarm(explanation, max_display=15, show=False)
        plt.title(f"{target}: SHAP distribution on temporal evaluation data")
        plt.tight_layout()
        safe_target = target.lower().replace(" ", "_")
        plt.savefig(dirs["figures"] / f"figure_beeswarm_{safe_target}.png", dpi=300, bbox_inches="tight")
        plt.savefig(dirs["figures"] / f"figure_beeswarm_{safe_target}.pdf", bbox_inches="tight")
        plt.close()

    group_table = tables["groups"].pivot(index="target", columns="feature_group", values="group_attribution_share").fillna(0)
    group_table = group_table.reindex(list(TARGET_CONFIG))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    left = np.zeros(len(group_table))
    for group in group_table.columns:
        values = group_table[group].to_numpy(float)
        ax.barh(group_table.index, values, left=left, label=group, color=group_palette.get(group, "#8C8C8C"))
        left += values
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of absolute SHAP attribution")
    ax.set_title("Feature-group attribution of the models")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "figure_feature_group_attribution.png", dpi=300, bbox_inches="tight")
    fig.savefig(dirs["figures"] / "figure_feature_group_attribution.pdf", bbox_inches="tight")
    plt.close(fig)

    dependence_rows = []
    for target, result in results.items():
        top_features = (
            tables["importance"].loc[tables["importance"]["target"].eq(target)]
            .sort_values("rank")
            .head(3)["feature"]
            .tolist()
        )
        fig, axes = plt.subplots(1, len(top_features), figsize=(15, 4.6), squeeze=False)
        for ax, feature in zip(axes[0], top_features):
            feature_index = result.feature_names.index(feature)
            x_values = pd.to_numeric(result.eval_raw[feature], errors="coerce").to_numpy(float)
            y_values = result.eval_values[:, feature_index]
            valid = np.isfinite(x_values) & np.isfinite(y_values)
            rho = float(spearmanr(x_values[valid], y_values[valid]).statistic) if valid.sum() >= 3 else np.nan
            interaction_feature = None
            if target in {"SS", "DO proxy"}:
                try:
                    interactions = shap.approximate_interactions(
                        feature,
                        result.eval_values,
                        result.eval_raw,
                    )
                    if len(interactions):
                        interaction_feature = result.feature_names[int(interactions[0])]
                except Exception:
                    interaction_feature = None
            ax.scatter(x_values, y_values, color="#2F6B9A", s=35, alpha=0.85, edgecolor="none")
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
            ax.set_xlabel(feature)
            ax.set_ylabel(f"SHAP [{result.output_scale}]")
            ax.set_title(f"Spearman rho={rho:.2f}")
            ax.grid(alpha=0.2)
            dependence_rows.append({
                "target": target,
                "feature": feature,
                "feature_group": result.feature_groups[feature_index],
                "spearman_raw_value_vs_shap": rho,
                "suggested_interaction_feature": interaction_feature,
                "output_scale": result.output_scale,
                "n_evaluation": int(valid.sum()),
            })
        fig.suptitle(f"{target}: dependence of the top SHAP predictors", fontsize=13, fontweight="bold")
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.16, top=0.82, wspace=0.30)
        safe_target = target.lower().replace(" ", "_")
        fig.savefig(dirs["figures"] / f"figure_dependence_{safe_target}.png", dpi=300, bbox_inches="tight")
        fig.savefig(dirs["figures"] / f"figure_dependence_{safe_target}.pdf", bbox_inches="tight")
        plt.close(fig)

    dependence = pd.DataFrame(dependence_rows)
    dependence.to_csv(dirs["tables"] / "dependence_plot_summary.csv", index=False)
    return dependence


def cluster_bootstrap_stability(
    results: dict[str, ShapResult],
    tables: dict[str, pd.DataFrame],
    dirs: dict[str, Path],
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    feature_draws = []
    group_draws = []
    rank_rows = []

    for target, result in results.items():
        stations = result.eval_frame.get(
            "station_canonical", pd.Series(np.arange(len(result.eval_frame)), index=result.eval_frame.index)
        ).astype(str).to_numpy()
        unique_stations = np.unique(stations)
        station_rows = {station: np.flatnonzero(stations == station) for station in unique_stations}
        values = result.eval_values

        for draw in range(N_BOOTSTRAP):
            sampled_stations = rng.choice(unique_stations, size=len(unique_stations), replace=True)
            sampled_rows = np.concatenate([station_rows[station] for station in sampled_stations])
            means = np.mean(np.abs(values[sampled_rows]), axis=0)
            total = float(means.sum())
            top_five = set(np.argsort(-means)[:5])
            for feature_index, feature in enumerate(result.feature_names):
                feature_draws.append({
                    "target": target,
                    "bootstrap": draw,
                    "feature": feature,
                    "feature_group": result.feature_groups[feature_index],
                    "mean_abs_shap": means[feature_index],
                    "top_five": int(feature_index in top_five),
                })
            for group in sorted(set(result.feature_groups)):
                group_total = float(sum(means[i] for i, value in enumerate(result.feature_groups) if value == group))
                group_draws.append({
                    "target": target,
                    "bootstrap": draw,
                    "feature_group": group,
                    "group_attribution_share": group_total / total if total else np.nan,
                })

        train_mean = np.mean(np.abs(result.train_values), axis=0)
        eval_mean = np.mean(np.abs(result.eval_values), axis=0)
        train_rank = pd.Series(train_mean).rank(ascending=False, method="average").to_numpy()
        eval_rank = pd.Series(eval_mean).rank(ascending=False, method="average").to_numpy()
        rank_rho = float(spearmanr(train_rank, eval_rank).statistic)
        train_top = {result.feature_names[index] for index in np.argsort(-train_mean)[:5]}
        eval_top = {result.feature_names[index] for index in np.argsort(-eval_mean)[:5]}
        rank_rows.append({
            "target": target,
            "n_train": len(result.train_frame),
            "n_evaluation": len(result.eval_frame),
            "n_evaluation_stations": len(unique_stations),
            "train_evaluation_rank_spearman": rank_rho,
            "top_five_overlap_count": len(train_top & eval_top),
            "train_top_five": "; ".join(sorted(train_top)),
            "evaluation_top_five": "; ".join(sorted(eval_top)),
            "interpretation": "Conditional attribution stability; model was not refitted inside bootstrap.",
        })

    feature_draws_df = pd.DataFrame(feature_draws)
    group_draws_df = pd.DataFrame(group_draws)
    feature_ci = (
        feature_draws_df.groupby(["target", "feature", "feature_group"], as_index=False)
        .agg(
            bootstrap_mean_abs_shap=("mean_abs_shap", "mean"),
            ci95_low=("mean_abs_shap", lambda x: np.quantile(x, 0.025)),
            ci95_high=("mean_abs_shap", lambda x: np.quantile(x, 0.975)),
            top_five_frequency=("top_five", "mean"),
        )
    )
    group_ci = (
        group_draws_df.groupby(["target", "feature_group"], as_index=False)
        .agg(
            bootstrap_mean_share=("group_attribution_share", "mean"),
            ci95_low=("group_attribution_share", lambda x: np.quantile(x, 0.025)),
            ci95_high=("group_attribution_share", lambda x: np.quantile(x, 0.975)),
        )
    )
    rank_stability = pd.DataFrame(rank_rows)
    feature_ci.to_csv(dirs["tables"] / "shap_bootstrap_feature_ci.csv", index=False)
    group_ci.to_csv(dirs["tables"] / "shap_bootstrap_group_ci.csv", index=False)
    rank_stability.to_csv(dirs["tables"] / "train_evaluation_rank_stability.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2))
    for ax, target in zip(axes, TARGET_CONFIG):
        current = feature_ci.loc[feature_ci["target"].eq(target)].sort_values("bootstrap_mean_abs_shap", ascending=False).head(10)
        current = current.sort_values("bootstrap_mean_abs_shap")
        errors = np.vstack([
            current["bootstrap_mean_abs_shap"] - current["ci95_low"],
            current["ci95_high"] - current["bootstrap_mean_abs_shap"],
        ])
        ax.errorbar(current["bootstrap_mean_abs_shap"], current["feature"], xerr=errors, fmt="o", color="#2F6B9A", ecolor="#9ECAE1", capsize=3)
        ax.set_title(target)
        ax.set_xlabel(f"Bootstrap mean |SHAP| [{TARGET_CONFIG[target]['global_scale']}]")
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Station-cluster bootstrap intervals for global SHAP importance", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "figure_shap_bootstrap_ci.png", dpi=300, bbox_inches="tight")
    fig.savefig(dirs["figures"] / "figure_shap_bootstrap_ci.pdf", bbox_inches="tight")
    plt.close(fig)
    return {"feature_ci": feature_ci, "group_ci": group_ci, "rank": rank_stability}


def find_raster(map_id: str, suffix: str) -> Path:
    matches = sorted(MOVING_WINDOW_DIR.rglob(f"{map_id}_*_{suffix}"))
    if not matches:
        raise FileNotFoundError(f"No raster for {map_id} and suffix {suffix}")
    if len(matches) > 1:
        non_tiles = [path for path in matches if "-000000" not in path.name]
        if len(non_tiles) == 1:
            return non_tiles[0]
    return matches[0]


def raster_band_map(src: rasterio.io.DatasetReader) -> dict[str, int]:
    return {name: index for index, name in enumerate(src.descriptions, start=1) if name}


def read_band(src: rasterio.io.DatasetReader, band_name: str) -> np.ndarray:
    bands = raster_band_map(src)
    if band_name not in bands:
        raise KeyError(f"{src.name}: missing band {band_name}")
    array = src.read(bands[band_name]).astype(float)
    if src.nodata is not None:
        array[array == src.nodata] = np.nan
    return array


def nearest_valid_cell(valid_mask: np.ndarray, row: int, col: int) -> tuple[int, int, float]:
    valid_rows, valid_cols = np.where(valid_mask)
    if not len(valid_rows):
        raise ValueError("No valid map pixels")
    tree = cKDTree(np.column_stack([valid_rows, valid_cols]))
    distance, position = tree.query([row, col], k=1)
    return int(valid_rows[position]), int(valid_cols[position]), float(distance)


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def select_spatial_points(data: pd.DataFrame, bundles: dict[str, dict], dirs: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    point_rows = []
    station_exclusion_rows = []
    feature_frames: dict[str, list[dict]] = {target: [] for target in TARGET_CONFIG}

    for map_id, map_info in MAPS.items():
        feature_path = find_raster(map_id, "moving_window_features.tif")
        prediction_path = PREDICTION_DIR / f"{map_id}_{map_info['date'].replace('-', '')}_water_quality_predictions.tif"
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)

        with rasterio.open(feature_path) as feature_src, rasterio.open(prediction_path) as pred_src:
            if (feature_src.width, feature_src.height, feature_src.transform) != (pred_src.width, pred_src.height, pred_src.transform):
                raise AssertionError(f"{map_id}: feature and prediction rasters are not aligned")
            feature_band_names = raster_band_map(feature_src)
            support = read_band(pred_src, "prediction_support")
            station_subset = (
                data.loc[data["coordinate_waterbody"].astype(str).eq(map_info["thai"])]
                .groupby("station_canonical", as_index=False)
                .agg(latitude=("lat", "median"), longitude=("lon", "median"))
            )

            prediction_arrays = {
                target: read_band(pred_src, config["prediction_band"])
                for target, config in TARGET_CONFIG.items()
            }
            ood_arrays = {
                target: read_band(pred_src, config["ood_band"])
                for target, config in TARGET_CONFIG.items()
            }
            common_valid = np.isfinite(support) & (support > 0)
            for target in TARGET_CONFIG:
                common_valid &= np.isfinite(prediction_arrays[target])
                common_valid &= np.isfinite(ood_arrays[target]) & (ood_arrays[target] <= OOD_MAX)

            common_station_cells = []
            for station_record in station_subset.itertuples(index=False):
                station_row, station_col = feature_src.index(float(station_record.longitude), float(station_record.latitude))
                nearest_row, nearest_col, _ = nearest_valid_cell(common_valid, station_row, station_col)
                nearest_lon, nearest_lat = xy(feature_src.transform, nearest_row, nearest_col, offset="center")
                snap_distance_m = haversine_m(
                    float(station_record.longitude),
                    float(station_record.latitude),
                    float(nearest_lon),
                    float(nearest_lat),
                )
                role = PREDEFINED_LOCATION_ROLES.get((map_id, str(station_record.station_canonical)), "field_station_location")
                requires_confirmation = "requires_confirmation" in role
                if snap_distance_m > MAX_STATION_SNAP_M:
                    station_exclusion_rows.append({
                        "map_id": map_id,
                        "waterbody": map_info["name"],
                        "target": "all_targets_common_validity",
                        "station": str(station_record.station_canonical),
                        "station_longitude": float(station_record.longitude),
                        "station_latitude": float(station_record.latitude),
                        "nearest_valid_distance_m": snap_distance_m,
                        "maximum_allowed_distance_m": MAX_STATION_SNAP_M,
                        "reason": "Nearest pixel valid for all three low-OOD model maps is too far from station.",
                    })
                    continue
                common_station_cells.append((
                    role,
                    str(station_record.station_canonical),
                    nearest_row,
                    nearest_col,
                    float(station_record.longitude),
                    float(station_record.latitude),
                    requires_confirmation,
                ))

            for target, config in TARGET_CONFIG.items():
                bundle = bundles[target]
                features = list(bundle["feature_names"])
                missing = [feature for feature in features if feature not in feature_band_names]
                if missing:
                    raise KeyError(f"{map_id} {target}: missing feature raster bands: {missing}")
                prediction = prediction_arrays[target]
                ood = ood_arrays[target]
                valid = np.isfinite(prediction) & np.isfinite(support) & (support > 0)
                valid &= np.isfinite(ood) & (ood <= OOD_MAX)
                valid_rows, valid_cols = np.where(valid)
                valid_predictions = prediction[valid]
                if len(valid_predictions) < 3:
                    raise ValueError(f"{map_id} {target}: fewer than three valid low-OOD pixels")

                selected_cells: list[tuple[str, str, int, int, float | None, float | None, bool]] = []
                order = np.argsort(valid_predictions)
                for quantile, label in [(0.05, "P05"), (0.50, "P50"), (0.95, "P95")]:
                    position = int(round(quantile * (len(order) - 1)))
                    selected = order[position]
                    selected_cells.append(("prediction_quantile", label, int(valid_rows[selected]), int(valid_cols[selected]), None, None, False))

                selected_cells.extend(common_station_cells)

                for location_type, location_label, row, col, source_lon, source_lat, requires_confirmation in selected_cells:
                    pixel_lon, pixel_lat = xy(feature_src.transform, row, col, offset="center")
                    raw_values = {}
                    for feature in features:
                        value = feature_src.read(feature_band_names[feature], window=((row, row + 1), (col, col + 1))).astype(float)[0, 0]
                        raw_values[feature] = np.nan if feature_src.nodata is not None and value == feature_src.nodata else float(value)
                    record_key = f"{map_id}|{target}|{location_type}|{location_label}"
                    feature_frames[target].append({"record_key": record_key, **raw_values})
                    source_distance = (
                        haversine_m(source_lon, source_lat, float(pixel_lon), float(pixel_lat))
                        if source_lon is not None and source_lat is not None else np.nan
                    )
                    point_rows.append({
                        "record_key": record_key,
                        "map_id": map_id,
                        "waterbody": map_info["name"],
                        "sat_date": map_info["date"],
                        "target": target,
                        "location_type": location_type,
                        "location_label": location_label,
                        "requires_domain_confirmation": requires_confirmation,
                        "row": row,
                        "col": col,
                        "longitude": float(pixel_lon),
                        "latitude": float(pixel_lat),
                        "source_longitude": source_lon,
                        "source_latitude": source_lat,
                        "distance_from_source_location_m": source_distance,
                        "prediction_raster": float(prediction[row, col]),
                        "ood_fraction": float(ood[row, col]),
                        "prediction_support": float(support[row, col]),
                        "feature_raster_path": str(feature_path),
                        "prediction_raster_path": str(prediction_path),
                    })

    point_manifest = pd.DataFrame(point_rows)
    output_frames = {
        target: pd.DataFrame(rows).drop_duplicates("record_key").set_index("record_key")
        for target, rows in feature_frames.items()
    }
    point_manifest.to_csv(dirs["tables"] / "local_spatial_point_manifest_pre_shap.csv", index=False)
    pd.DataFrame(station_exclusion_rows).to_csv(
        dirs["tables"] / "local_spatial_station_exclusions.csv",
        index=False,
    )
    return point_manifest, output_frames


def explain_spatial_points(
    point_manifest: pd.DataFrame,
    feature_frames: dict[str, pd.DataFrame],
    bundles: dict[str, dict],
    splits: dict[str, dict[str, pd.DataFrame]],
    results: dict[str, ShapResult],
    dirs: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    local_long_parts = []
    completed_manifest = point_manifest.copy()

    for target, config in TARGET_CONFIG.items():
        bundle = bundles[target]
        features = list(bundle["feature_names"])
        X_local = feature_frames[target][features].copy()
        keys = X_local.index.to_list()
        if target == "Turbidity":
            X_train = splits[target]["train"][features].copy()
            background = X_train.sample(n=min(SHAP_BACKGROUND_SIZE, len(X_train)), random_state=RANDOM_STATE)

            def predict_original(values: np.ndarray | pd.DataFrame) -> np.ndarray:
                return exact_prediction(bundle, pd.DataFrame(values, columns=features))[1]

            explainer = shap.Explainer(
                predict_original,
                shap.maskers.Independent(background, max_samples=len(background)),
                algorithm="permutation",
                feature_names=features,
                seed=RANDOM_STATE,
            )
            local_exp = explainer(
                X_local,
                max_evals=PERMUTATION_ROUNDS * (2 * len(features) + 1),
                silent=True,
            )
            local_values = np.asarray(local_exp.values, dtype=float)
            local_base = broadcast_base_values(local_exp.base_values, len(X_local))
            output_scale = "NTU"
        elif target == "SS":
            explainer = results[target].explainer
            local_exp = explainer(
                X_local,
                max_evals=PERMUTATION_ROUNDS * (2 * len(features) + 1),
                silent=True,
            )
            local_values = np.asarray(local_exp.values, dtype=float)
            local_base = broadcast_base_values(local_exp.base_values, len(X_local))
            output_scale = "mg/L"
        else:
            pipeline: Pipeline = bundle["estimator"]
            imputer = pipeline.named_steps["imputer"]
            model = pipeline.named_steps["model"]
            X_local_imputed = pd.DataFrame(imputer.transform(X_local), columns=features, index=X_local.index)
            native = np.asarray(model.get_feature_importance(Pool(X_local_imputed, feature_names=features), type="ShapValues"))
            local_values = native[:, :-1]
            local_base = native[:, -1]
            X_local = X_local_imputed
            output_scale = "mg/L"

        _, local_predictions = exact_prediction(bundle, X_local)
        target_rows = completed_manifest["target"].eq(target)
        prediction_by_key = dict(zip(keys, local_predictions))
        completed_manifest.loc[target_rows, "prediction_recomputed"] = completed_manifest.loc[target_rows, "record_key"].map(prediction_by_key)
        completed_manifest.loc[target_rows, "prediction_difference_recomputed_minus_raster"] = (
            completed_manifest.loc[target_rows, "prediction_recomputed"]
            - completed_manifest.loc[target_rows, "prediction_raster"]
        )

        s2_features = s2_feature_set(target, features)
        groups = [target_feature_group(target, feature, s2_features) for feature in features]
        target_meta = completed_manifest.loc[target_rows].set_index("record_key").loc[keys].reset_index()
        for row_position, record_key in enumerate(keys):
            for feature_index, feature in enumerate(features):
                local_long_parts.append({
                    "record_key": record_key,
                    "map_id": target_meta.loc[row_position, "map_id"],
                    "waterbody": target_meta.loc[row_position, "waterbody"],
                    "sat_date": target_meta.loc[row_position, "sat_date"],
                    "target": target,
                    "location_type": target_meta.loc[row_position, "location_type"],
                    "location_label": target_meta.loc[row_position, "location_label"],
                    "longitude": target_meta.loc[row_position, "longitude"],
                    "latitude": target_meta.loc[row_position, "latitude"],
                    "feature": feature,
                    "feature_group": groups[feature_index],
                    "raw_feature_value": pd.to_numeric(X_local.iloc[row_position, feature_index], errors="coerce"),
                    "shap_value": local_values[row_position, feature_index],
                    "base_value": local_base[row_position],
                    "predicted_value": local_predictions[row_position],
                    "output_scale": output_scale,
                    "ood_fraction": target_meta.loc[row_position, "ood_fraction"],
                })

        p95_positions = target_meta.index[
            target_meta["location_type"].eq("prediction_quantile") & target_meta["location_label"].eq("P95")
        ].tolist()
        for row_position in p95_positions:
            meta = target_meta.loc[row_position]
            explanation = shap.Explanation(
                values=local_values[row_position],
                base_values=local_base[row_position],
                data=X_local.iloc[row_position].to_numpy(float),
                feature_names=features,
            )
            shap.plots.waterfall(explanation, max_display=12, show=False)
            plt.title(f"{meta['waterbody']} | {target} | P95 mapped prediction")
            plt.tight_layout()
            safe_target = target.lower().replace(" ", "_")
            filename = f"waterfall_{meta['map_id']}_{safe_target}_p95.png"
            plt.savefig(dirs["local"] / filename, dpi=300, bbox_inches="tight")
            plt.close()

    local_long = pd.DataFrame(local_long_parts)
    completed_manifest.to_csv(dirs["tables"] / "local_spatial_point_manifest.csv", index=False)
    local_long.to_csv(dirs["tables"] / "local_spatial_shap_long.csv", index=False)

    top_local = (
        local_long.assign(abs_shap=lambda frame: frame["shap_value"].abs())
        .sort_values(["record_key", "abs_shap"], ascending=[True, False])
        .groupby("record_key", as_index=False)
        .head(10)
    )
    top_local.to_csv(dirs["tables"] / "local_spatial_top_contributions.csv", index=False)

    p95 = local_long.loc[
        local_long["location_type"].eq("prediction_quantile")
        & local_long["location_label"].eq("P95")
    ].copy()
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    for row_index, map_id in enumerate(MAPS):
        for col_index, target in enumerate(TARGET_CONFIG):
            ax = axes[row_index, col_index]
            current = p95.loc[p95["map_id"].eq(map_id) & p95["target"].eq(target)].copy()
            current["abs_shap"] = current["shap_value"].abs()
            current = current.nlargest(8, "abs_shap").sort_values("shap_value")
            colors = np.where(current["shap_value"].ge(0), "#D55E00", "#0072B2")
            ax.barh(current["feature"], current["shap_value"], color=colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(f"{MAPS[map_id]['name']} | {target}")
            ax.set_xlabel(f"Local SHAP [{current['output_scale'].iloc[0]}]")
            ax.grid(axis="x", alpha=0.2)
    fig.suptitle("Local explanations of P95 mapped predictions", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "figure_local_p95_contributions.png", dpi=300, bbox_inches="tight")
    fig.savefig(dirs["figures"] / "figure_local_p95_contributions.pdf", bbox_inches="tight")
    plt.close(fig)
    return completed_manifest, local_long


def catboost_interaction_table(bundle: dict, dirs: dict[str, Path]) -> pd.DataFrame:
    features = list(bundle["feature_names"])
    model = bundle["estimator"].named_steps["model"]
    interactions = np.asarray(model.get_feature_importance(type="Interaction"), dtype=float)
    rows = []
    for first, second, score in interactions:
        first_index = int(first)
        second_index = int(second)
        rows.append({
            "target": "DO proxy",
            "feature_1": features[first_index],
            "feature_2": features[second_index],
            "interaction_strength": float(score),
        })
    table = pd.DataFrame(rows).sort_values("interaction_strength", ascending=False)
    table.to_csv(dirs["tables"] / "do_catboost_interactions.csv", index=False)
    return table


def dataframe_to_markdown(data: pd.DataFrame) -> str:
    formatted = data.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{value:.5g}")
        else:
            formatted[column] = formatted[column].fillna("").astype(str)
    headers = [str(column).replace("|", "\\|") for column in formatted.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in formatted.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_reporting_summary(
    dirs: dict[str, Path],
    verification: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    stability: dict[str, pd.DataFrame],
    spatial_manifest: pd.DataFrame,
) -> Path:
    performance = verification[[
        "target", "model", "n_features", "n_train", "n_evaluation", "r2", "rmse", "mae", "spearman_r", "bias"
    ]].copy()
    top_features = tables["top_five"][[
        "target", "rank", "feature", "feature_group", "mean_abs_shap", "share_of_absolute_shap", "output_scale"
    ]].copy()
    groups = (
        tables["groups"][[
            "target", "feature_group", "n_features", "group_attribution_share", "mean_abs_shap_per_feature", "output_scale"
        ]]
        .sort_values(["target", "group_attribution_share"], ascending=[True, False])
    )
    rank = stability["rank"][[
        "target", "n_evaluation_stations", "train_evaluation_rank_spearman", "top_five_overlap_count"
    ]].copy()
    quantile_points = spatial_manifest.loc[spatial_manifest["location_type"].eq("prediction_quantile")]

    lines = [
        "# Final-model SHAP reporting summary",
        "",
        "This report is generated from the locked S2 + context artifacts. The 2024-2025 data are called the temporal evaluation set because they were previously used during model screening.",
        "",
        "## Locked model verification",
        "",
        dataframe_to_markdown(performance.round(4)),
        "",
        "## Top five global SHAP features",
        "",
        dataframe_to_markdown(top_features.round(5)),
        "",
        "## Feature-group attribution",
        "",
        dataframe_to_markdown(groups.round(5)),
        "",
        "`group_attribution_share` is the sum of mean absolute SHAP within a group divided by the sum across all model features. It is not a causal contribution.",
        "",
        "## Conditional attribution stability",
        "",
        dataframe_to_markdown(rank.round(4)),
        "",
        "The bootstrap intervals condition on the locked fitted models and quantify sampling variability across evaluation stations; the models were not refitted within bootstrap samples.",
        "",
        "## Spatial explanation inventory",
        "",
        f"- Quantile locations explained: {len(quantile_points)} ({len(quantile_points) // 3} map-target combinations x P05/P50/P95).",
        f"- All retained local locations: {len(spatial_manifest)}.",
        f"- Map prediction reproduction maximum absolute difference: {spatial_manifest['prediction_difference_recomputed_minus_raster'].abs().max():.3e}.",
        "- Station locations are snapped only to a common pixel valid for all three model maps within 500 m.",
        "- PN02 and LS01 are provisional river-mouth station proxies and require domain confirmation.",
        "",
        "## Interpretation guardrails",
        "",
        "- Turbidity global SHAP is on the log1p(NTU) model-output scale; local mapped SHAP is computed end-to-end in NTU.",
        "- SS main SHAP explains the final two-stage soft-gated output in mg/L. Stage 1 log-odds and Stage 2 log1p(SS) explanations are supplementary.",
        "- DO is an indirect proxy model. Its context features are coarse and do not independently create 20 m spatial detail.",
        "- B1 and B9 have 60 m native resolution. Their high Turbidity attribution must not be interpreted as independent 20 m spatial evidence after resampling.",
        "- The DO model uses raw day-of-year and month predictors; seasonal attribution may include calendar encoding and site-year structure.",
        "- Correlated bands, indices, QA variables, and land-cover fractions can redistribute SHAP attribution among themselves.",
        "- Absolute SHAP magnitudes are on different target-output scales and must not be compared numerically across targets; normalized within-target shares may be compared descriptively.",
        "- SHAP attribution is model-specific association and must not be described as a causal effect.",
    ]
    path = dirs["root"] / "xai_reporting_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_manifest(
    dirs: dict[str, Path],
    bundles: dict[str, dict],
    verification: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    stability: dict[str, pd.DataFrame],
    spatial_manifest: pd.DataFrame,
) -> Path:
    manifest = {
        "workflow_version": "1.0.0",
        "data_path": str(DATA_PATH),
        "data_sha256": file_sha256(DATA_PATH),
        "temporal_contract": {
            "train_year_be": [TRAIN_YEAR_MIN, TRAIN_YEAR_MAX],
            "evaluation_year_be": [EVAL_YEAR_MIN, EVAL_YEAR_MAX],
            "match_window_days": MAX_MATCH_DAYS,
            "terminology": "temporal evaluation set; not described as an untouched test set",
        },
        "artifacts": {
            target: {
                "artifact_id": bundle["artifact_id"],
                "artifact_path": str(ARTIFACT_DIR / TARGET_CONFIG[target]["artifact"]),
                "artifact_sha256": file_sha256(ARTIFACT_DIR / TARGET_CONFIG[target]["artifact"]),
                "feature_count": len(bundle["feature_names"]),
                "prediction_contract": bundle["prediction_contract"],
            }
            for target, bundle in bundles.items()
        },
        "shap_methods": {
            "Turbidity_global": "LinearSHAP on standardized LinearSVR inputs; output is log1p(NTU)",
            "Turbidity_local": "Permutation SHAP on end-to-end inverse-transformed predictor; output is NTU",
            "SS_global_and_local": "Permutation SHAP on end-to-end two-stage soft-gated predictor; output is mg/L",
            "SS_stage_1": "LinearSHAP for LogisticRegression; output is classifier log-odds",
            "SS_stage_2": "LinearSHAP for high-regime LinearSVR; output is log1p(SS)",
            "DO_global_and_local": "CatBoost native TreeSHAP; output is mg/L",
        },
        "bootstrap": {
            "method": "nonparametric cluster bootstrap of temporal-evaluation rows by station",
            "n_bootstrap": N_BOOTSTRAP,
            "limitation": "The fitted model is held fixed; intervals describe attribution-sampling uncertainty, not refit uncertainty.",
        },
        "spatial": {
            "map_ids": list(MAPS),
            "ood_max": OOD_MAX,
            "maximum_station_snap_distance_m": MAX_STATION_SNAP_M,
            "quantiles": [0.05, 0.50, 0.95],
            "moving_window_note": "Local explanations refer to mapped moving-window feature vectors, not unsmoothed native pixels.",
            "station_note": "Station-location points are nearest valid mapped pixels; they do not imply same-date in-situ validation.",
            "river_mouth_note": "PN02 and LS01 are provisional station proxies requiring domain confirmation before manuscript labeling.",
        },
        "counts": {
            "global_shap_rows": len(tables["long"]),
            "feature_importance_rows": len(tables["importance"]),
            "bootstrap_feature_rows": len(stability["feature_ci"]),
            "spatial_points": len(spatial_manifest),
        },
        "verification": verification.to_dict(orient="records"),
        "interpretation_guardrails": [
            "SHAP attribution is associational and model-specific, not causal contribution.",
            "Correlated and derived predictors can share or redistribute attribution.",
            "Coarse context features can change scene-level predictions but do not create independent 20 m spatial detail.",
            "B1 and B9 are native 60 m bands; their attribution does not imply independent 20 m information after resampling.",
            "Raw day-of-year and month can encode calendar and site-year structure in the DO proxy model.",
            "Absolute SHAP magnitudes must not be compared across targets because their output scales differ.",
            "The 2024-2025 period is called temporal evaluation because model screening used this period.",
        ],
    }
    path = dirs["root"] / "xai_run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_workflow() -> dict[str, object]:
    dirs = ensure_dirs()
    data, bundles, splits, verification = load_inputs()
    verification.to_csv(dirs["tables"] / "model_artifact_verification.csv", index=False)

    results = {
        "Turbidity": linear_shap_result(
            "Turbidity",
            bundles["Turbidity"],
            splits["Turbidity"]["train"],
            splits["Turbidity"]["evaluation"],
            TARGET_CONFIG["Turbidity"]["target_col"],
        ),
        "SS": permutation_shap_result(
            "SS",
            bundles["SS"],
            splits["SS"]["train"],
            splits["SS"]["evaluation"],
            RANDOM_STATE,
        ),
        "DO proxy": catboost_shap_result(
            "DO proxy",
            bundles["DO proxy"],
            splits["DO proxy"]["train"],
            splits["DO proxy"]["evaluation"],
        ),
    }

    ss_stage1, ss_stage2 = compute_stage_specific_ss_shap(
        bundles["SS"],
        splits["SS"]["train"],
        splits["SS"]["evaluation"],
        dirs,
    )
    tables = make_global_tables(results, dirs)
    dependence = make_global_figures(results, tables, dirs)
    stability = cluster_bootstrap_stability(results, tables, dirs)
    do_interactions = catboost_interaction_table(bundles["DO proxy"], dirs)
    point_manifest, feature_frames = select_spatial_points(data, bundles, dirs)
    spatial_manifest, spatial_long = explain_spatial_points(
        point_manifest,
        feature_frames,
        bundles,
        splits,
        results,
        dirs,
    )
    reporting_summary_path = write_reporting_summary(
        dirs,
        verification,
        tables,
        stability,
        spatial_manifest,
    )
    manifest_path = write_manifest(dirs, bundles, verification, tables, stability, spatial_manifest)
    return {
        "output_dir": dirs["root"],
        "verification": verification,
        "global_importance": tables["importance"],
        "top_five": tables["top_five"],
        "group_attribution": tables["groups"],
        "additivity": tables["additivity"],
        "dependence": dependence,
        "bootstrap_feature_ci": stability["feature_ci"],
        "bootstrap_group_ci": stability["group_ci"],
        "rank_stability": stability["rank"],
        "ss_stage1": ss_stage1,
        "ss_stage2": ss_stage2,
        "do_interactions": do_interactions,
        "spatial_manifest": spatial_manifest,
        "spatial_shap": spatial_long,
        "reporting_summary_path": reporting_summary_path,
        "manifest_path": manifest_path,
    }


if __name__ == "__main__":
    outputs = run_workflow()
    print(outputs["verification"].to_string(index=False))
    print(f"\nOutputs: {outputs['output_dir']}")
