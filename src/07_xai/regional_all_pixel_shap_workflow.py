from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import rasterio
import shap
from catboost import Pool


ROOT = Path(r".")
BASE_WORKFLOW_PATH = ROOT / "Resources" / "notebooks" / "XAI" / "final_model_shap_workflow.py"
OUTPUT_DIR = ROOT / "Results" / "XAI" / "regional_all_pixel_v1"
BATCH_DIR = OUTPUT_DIR / "pixel_shap_batches"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

BATCH_SIZE = 2_000
DISPLAY_MAX_PER_PANEL = 2_500
TOP_FEATURES_PER_TARGET = 8
RANDOM_STATE = 20260807

MAP_ORDER = ["WB02", "WB07", "WB05"]
TARGET_ORDER = ["Turbidity", "SS", "DO proxy"]
TARGET_LABELS = {
    "Turbidity": "Turbidity (NTU)",
    "SS": "Suspended solids (mg/L)",
    "DO proxy": "DO proxy (mg/L)",
}
SITE_LABELS = {
    "WB02": "Thale Luang",
    "WB07": "Lower Lang Suan River",
    "WB05": "Pak Phanang River",
}

GROUP_ORDER = [
    "Sentinel-2 spectral",
    "Rainfall",
    "Runoff",
    "Wind",
    "Air temperature",
    "Solar radiation",
    "Seasonality",
    "Water-pixel QA",
    "Cloud QA",
    "Historical-water occurrence",
    "Historical water",
    "Surrounding-water fraction",
    "Land cover",
    "Terrain",
    "Other context",
]
GROUP_COLORS = {
    "Sentinel-2 spectral": "#2166AC",
    "Rainfall": "#2CA25F",
    "Runoff": "#006D2C",
    "Wind": "#67A9CF",
    "Air temperature": "#D73027",
    "Solar radiation": "#FDAE61",
    "Seasonality": "#984EA3",
    "Water-pixel QA": "#756BB1",
    "Cloud QA": "#BDBDBD",
    "Historical-water occurrence": "#1B9E77",
    "Historical water": "#238B45",
    "Surrounding-water fraction": "#80CDC1",
    "Land cover": "#D95F02",
    "Terrain": "#8C510A",
    "Other context": "#636363",
}


def load_base_workflow():
    module_name = "rswq_final_model_shap_workflow"
    spec = importlib.util.spec_from_file_location(module_name, BASE_WORKFLOW_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load workflow: {BASE_WORKFLOW_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_dirs() -> None:
    for path in [OUTPUT_DIR, BATCH_DIR, TABLE_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def feature_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    return [column for column in frame.columns if column.startswith(prefix)]


def safe_slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without the optional tabulate package."""
    display = frame.copy()
    for column in display.columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else str(value).replace("|", "\\|")
        )
    header = "| " + " | ".join(display.columns.astype(str)) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(row) + " |"
        for row in display.astype(str).itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def common_valid_pixels(wf, prediction_src: rasterio.io.DatasetReader):
    support = wf.read_band(prediction_src, "prediction_support")
    common = np.isfinite(support) & (support > 0)
    predictions = {}
    ood = {}
    for target, config in wf.TARGET_CONFIG.items():
        predictions[target] = wf.read_band(prediction_src, config["prediction_band"])
        ood[target] = wf.read_band(prediction_src, config["ood_band"])
        common &= np.isfinite(predictions[target])
        common &= np.isfinite(ood[target]) & (ood[target] <= wf.OOD_MAX)
    rows, cols = np.where(common)
    return rows, cols, support, predictions, ood


def read_feature_frame(
    wf,
    feature_src: rasterio.io.DatasetReader,
    rows: np.ndarray,
    cols: np.ndarray,
    features: list[str],
) -> pd.DataFrame:
    bands = wf.raster_band_map(feature_src)
    missing = [feature for feature in features if feature not in bands]
    if missing:
        raise KeyError(f"{feature_src.name}: missing model features {missing}")
    values = {}
    for feature in features:
        band = feature_src.read(bands[feature], masked=True)
        selected = np.ma.asarray(band[rows, cols]).filled(np.nan).astype(float)
        values[feature] = selected
    return pd.DataFrame(values)


def permutation_explanation(wf, target, bundle, train, X, batch_seed):
    features = list(bundle["feature_names"])
    background = train[features].sample(
        n=min(wf.SHAP_BACKGROUND_SIZE, len(train)),
        random_state=wf.RANDOM_STATE,
    )

    def predict_original(values):
        frame = pd.DataFrame(values, columns=features)
        return wf.exact_prediction(bundle, frame)[1]

    explainer = shap.Explainer(
        predict_original,
        shap.maskers.Independent(background, max_samples=len(background)),
        algorithm="permutation",
        feature_names=features,
        seed=batch_seed,
    )
    explanation = explainer(
        X,
        max_evals=wf.PERMUTATION_ROUNDS * (2 * len(features) + 1),
        silent=True,
    )
    return (
        np.asarray(explanation.values, dtype=float),
        wf.broadcast_base_values(explanation.base_values, len(X)),
    )


def catboost_explanation(bundle, X):
    features = list(bundle["feature_names"])
    pipeline = bundle["estimator"]
    imputer = pipeline.named_steps["imputer"]
    model = pipeline.named_steps["model"]
    X_imputed = pd.DataFrame(imputer.transform(X), columns=features)
    native = np.asarray(
        model.get_feature_importance(
            Pool(X_imputed, feature_names=features),
            type="ShapValues",
        ),
        dtype=float,
    )
    return native[:, :-1], native[:, -1]


def explain_all_pixels(wf, bundles, splits) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_rows = []
    batch_rows = []
    total_started = time.perf_counter()

    for map_id in MAP_ORDER:
        map_info = wf.MAPS[map_id]
        feature_path = wf.find_raster(map_id, "moving_window_features.tif")
        prediction_path = wf.PREDICTION_DIR / (
            f"{map_id}_{map_info['date'].replace('-', '')}_water_quality_predictions.tif"
        )
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)

        with rasterio.open(feature_path) as feature_src, rasterio.open(prediction_path) as prediction_src:
            if (
                feature_src.width,
                feature_src.height,
                feature_src.transform,
            ) != (
                prediction_src.width,
                prediction_src.height,
                prediction_src.transform,
            ):
                raise AssertionError(f"{map_id}: feature and prediction rasters are not aligned")

            rows, cols, support, predictions, ood = common_valid_pixels(wf, prediction_src)
            if not len(rows):
                raise ValueError(f"{map_id}: no common valid low-OOD water pixels")

            longitudes, latitudes = rasterio.transform.xy(
                feature_src.transform,
                rows,
                cols,
                offset="center",
            )
            longitudes = np.asarray(longitudes, dtype=float)
            latitudes = np.asarray(latitudes, dtype=float)

            audit_rows.append({
                "map_id": map_id,
                "waterbody": SITE_LABELS[map_id],
                "sat_date": map_info["date"],
                "raster_height": feature_src.height,
                "raster_width": feature_src.width,
                "common_valid_low_ood_pixels": len(rows),
                "ood_threshold": wf.OOD_MAX,
                "feature_raster": str(feature_path),
                "prediction_raster": str(prediction_path),
            })

            for target in TARGET_ORDER:
                bundle = bundles[target]
                features = list(bundle["feature_names"])
                target_dir = BATCH_DIR / map_id / safe_slug(target)
                target_dir.mkdir(parents=True, exist_ok=True)
                n_batches = math.ceil(len(rows) / BATCH_SIZE)
                print(
                    f"{map_id} | {target} | {len(rows):,} pixels | "
                    f"{len(features)} features | {n_batches} batches",
                    flush=True,
                )

                for batch_index, start in enumerate(range(0, len(rows), BATCH_SIZE)):
                    stop = min(start + BATCH_SIZE, len(rows))
                    batch_path = target_dir / f"batch_{batch_index:04d}.parquet"
                    if batch_path.exists():
                        existing = pd.read_parquet(batch_path, columns=["pixel_index"])
                        expected = stop - start
                        if len(existing) == expected:
                            print(
                                f"  batch {batch_index + 1:02d}/{n_batches}: cached ({expected:,})",
                                flush=True,
                            )
                            batch_rows.append({
                                "map_id": map_id,
                                "target": target,
                                "batch": batch_index,
                                "start": start,
                                "stop": stop,
                                "n_pixels": expected,
                                "elapsed_seconds": 0.0,
                                "status": "cached",
                                "path": str(batch_path),
                            })
                            continue
                        batch_path.unlink()

                    batch_started = time.perf_counter()
                    batch_rows_idx = rows[start:stop]
                    batch_cols_idx = cols[start:stop]
                    X = read_feature_frame(
                        wf,
                        feature_src,
                        batch_rows_idx,
                        batch_cols_idx,
                        features,
                    )
                    if target in {"Turbidity", "SS"}:
                        shap_values, base_values = permutation_explanation(
                            wf,
                            target,
                            bundle,
                            splits[target]["train"],
                            X,
                            RANDOM_STATE + batch_index,
                        )
                    else:
                        shap_values, base_values = catboost_explanation(bundle, X)

                    _, exact_prediction = wf.exact_prediction(bundle, X)
                    reconstructed = base_values + shap_values.sum(axis=1)
                    additivity_error = reconstructed - exact_prediction

                    result = pd.DataFrame({
                        "pixel_index": np.arange(start, stop, dtype=np.int64),
                        "map_id": map_id,
                        "waterbody": SITE_LABELS[map_id],
                        "sat_date": map_info["date"],
                        "target": target,
                        "row": batch_rows_idx.astype(np.int32),
                        "col": batch_cols_idx.astype(np.int32),
                        "longitude": longitudes[start:stop],
                        "latitude": latitudes[start:stop],
                        "prediction_raster": predictions[target][batch_rows_idx, batch_cols_idx],
                        "prediction_model": exact_prediction,
                        "ood_fraction": ood[target][batch_rows_idx, batch_cols_idx],
                        "prediction_support": support[batch_rows_idx, batch_cols_idx],
                        "base_value": base_values,
                        "reconstructed_prediction": reconstructed,
                        "additivity_error": additivity_error,
                    })
                    for feature_index, feature in enumerate(features):
                        result[f"value__{feature}"] = X[feature].to_numpy(float)
                        result[f"shap__{feature}"] = shap_values[:, feature_index]

                    result.to_parquet(batch_path, index=False, compression="zstd")
                    elapsed = time.perf_counter() - batch_started
                    batch_rows.append({
                        "map_id": map_id,
                        "target": target,
                        "batch": batch_index,
                        "start": start,
                        "stop": stop,
                        "n_pixels": len(result),
                        "elapsed_seconds": elapsed,
                        "status": "completed",
                        "path": str(batch_path),
                    })
                    print(
                        f"  batch {batch_index + 1:02d}/{n_batches}: "
                        f"{len(result):,} pixels in {elapsed:.1f} s",
                        flush=True,
                    )

    pixel_audit = pd.DataFrame(audit_rows)
    batch_manifest = pd.DataFrame(batch_rows)
    pixel_audit.to_csv(TABLE_DIR / "regional_pixel_frame_audit.csv", index=False)
    batch_manifest.to_csv(TABLE_DIR / "regional_shap_batch_manifest.csv", index=False)
    print(f"All SHAP batches finished in {(time.perf_counter() - total_started) / 60:.1f} min")
    return pixel_audit, batch_manifest


def batch_paths(map_id: str, target: str) -> list[Path]:
    paths = sorted((BATCH_DIR / map_id / safe_slug(target)).glob("batch_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No SHAP batches for {map_id} {target}")
    return paths


def load_map_target(map_id: str, target: str) -> pd.DataFrame:
    return pd.concat(
        [pd.read_parquet(path) for path in batch_paths(map_id, target)],
        ignore_index=True,
    ).sort_values("pixel_index").reset_index(drop=True)


def summarize_all_pixels(wf, bundles):
    feature_rows = []
    group_rows = []
    dominant_rows = []
    additivity_rows = []
    missingness_rows = []

    for map_id in MAP_ORDER:
        for target in TARGET_ORDER:
            frame = load_map_target(map_id, target)
            features = list(bundles[target]["feature_names"])
            s2_features = wf.s2_feature_set(target, features)
            groups = {
                feature: wf.target_feature_group(target, feature, s2_features)
                for feature in features
            }
            shap_matrix = frame[[f"shap__{feature}" for feature in features]].to_numpy(float)
            mean_abs = np.mean(np.abs(shap_matrix), axis=0)
            total_mean_abs = float(mean_abs.sum())

            for feature_index, feature in enumerate(features):
                values = shap_matrix[:, feature_index]
                raw_values = frame[f"value__{feature}"].to_numpy(float)
                finite_raw = raw_values[np.isfinite(raw_values)]
                n_unique_nonmissing = int(pd.Series(finite_raw).nunique())
                feature_rows.append({
                    "map_id": map_id,
                    "waterbody": SITE_LABELS[map_id],
                    "target": target,
                    "feature": feature,
                    "feature_group": groups[feature],
                    "n_pixels": len(frame),
                    "mean_abs_shap": mean_abs[feature_index],
                    "attribution_share": (
                        mean_abs[feature_index] / total_mean_abs if total_mean_abs else np.nan
                    ),
                    "mean_signed_shap": np.mean(values),
                    "median_shap": np.median(values),
                    "shap_sd": np.std(values, ddof=1),
                    "shap_q05": np.quantile(values, 0.05),
                    "shap_q95": np.quantile(values, 0.95),
                    "positive_fraction": np.mean(values > 0),
                })
                missingness_rows.append({
                    "map_id": map_id,
                    "waterbody": SITE_LABELS[map_id],
                    "target": target,
                    "feature": feature,
                    "feature_group": groups[feature],
                    "n_pixels": len(frame),
                    "missing_count": int(np.sum(~np.isfinite(raw_values))),
                    "missing_fraction": float(np.mean(~np.isfinite(raw_values))),
                    "n_unique_nonmissing": n_unique_nonmissing,
                    "all_missing": len(finite_raw) == 0,
                    "constant_within_scene": n_unique_nonmissing <= 1,
                })

            unique_groups = [group for group in GROUP_ORDER if group in set(groups.values())]
            group_abs_matrix = []
            for group in unique_groups:
                indices = [index for index, feature in enumerate(features) if groups[feature] == group]
                group_abs = np.sum(np.abs(shap_matrix[:, indices]), axis=1)
                group_signed = np.sum(shap_matrix[:, indices], axis=1)
                centered_group_signed = group_signed - np.mean(group_signed)
                group_abs_matrix.append(group_abs)
                group_rows.append({
                    "map_id": map_id,
                    "waterbody": SITE_LABELS[map_id],
                    "target": target,
                    "feature_group": group,
                    "n_features": len(indices),
                    "n_pixels": len(frame),
                    "mean_abs_shap_sum": np.mean(group_abs),
                    "attribution_share": np.mean(group_abs) / total_mean_abs if total_mean_abs else np.nan,
                    "mean_signed_shap_sum": np.mean(group_signed),
                    "spatial_sd_signed_shap": np.std(group_signed, ddof=1),
                    "mean_abs_centered_shap": np.mean(np.abs(centered_group_signed)),
                })

            group_abs_matrix = np.column_stack(group_abs_matrix)
            dominant_indices = np.argmax(group_abs_matrix, axis=1)
            for group_index, group in enumerate(unique_groups):
                count = int(np.sum(dominant_indices == group_index))
                dominant_rows.append({
                    "map_id": map_id,
                    "waterbody": SITE_LABELS[map_id],
                    "target": target,
                    "feature_group": group,
                    "dominant_pixel_count": count,
                    "dominant_area_share": count / len(frame),
                })

            errors = frame["additivity_error"].to_numpy(float)
            raster_delta = frame["prediction_model"] - frame["prediction_raster"]
            additivity_rows.append({
                "map_id": map_id,
                "waterbody": SITE_LABELS[map_id],
                "target": target,
                "n_pixels": len(frame),
                "max_abs_additivity_error": np.max(np.abs(errors)),
                "mean_abs_additivity_error": np.mean(np.abs(errors)),
                "max_abs_model_raster_difference": np.max(np.abs(raster_delta)),
                "mean_abs_model_raster_difference": np.mean(np.abs(raster_delta)),
            })

    feature_table = pd.DataFrame(feature_rows)
    group_table = pd.DataFrame(group_rows)
    dominant_table = pd.DataFrame(dominant_rows)
    additivity_table = pd.DataFrame(additivity_rows)
    missingness_table = pd.DataFrame(missingness_rows)

    variation_total = group_table.groupby(["map_id", "target"])[
        "mean_abs_centered_shap"
    ].transform("sum")
    group_table["spatial_variation_share"] = np.where(
        variation_total > 0,
        group_table["mean_abs_centered_shap"] / variation_total,
        np.nan,
    )

    feature_table.to_csv(TABLE_DIR / "regional_feature_attribution.csv", index=False)
    group_table.to_csv(TABLE_DIR / "regional_feature_group_attribution.csv", index=False)
    dominant_table.to_csv(TABLE_DIR / "regional_dominant_group_area_share.csv", index=False)
    additivity_table.to_csv(TABLE_DIR / "regional_shap_additivity_audit.csv", index=False)
    missingness_table.to_csv(TABLE_DIR / "regional_feature_missingness_audit.csv", index=False)
    return feature_table, group_table, dominant_table, additivity_table, missingness_table


def plot_group_attribution(group_table: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.6), sharey=True)
    for target_index, target in enumerate(TARGET_ORDER):
        ax = axes[target_index]
        subset = group_table[group_table["target"].eq(target)].copy()
        pivot = subset.pivot(index="map_id", columns="feature_group", values="attribution_share")
        pivot = pivot.reindex(MAP_ORDER).fillna(0)
        left = np.zeros(len(MAP_ORDER))
        for group in GROUP_ORDER:
            if group not in pivot.columns:
                continue
            values = 100 * pivot[group].to_numpy(float)
            ax.barh(
                np.arange(len(MAP_ORDER)),
                values,
                left=left,
                color=GROUP_COLORS[group],
                edgecolor="white",
                linewidth=0.35,
                label=group,
            )
            left += values
        ax.set_title(TARGET_LABELS[target], fontsize=10, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of absolute SHAP attribution (%)")
        ax.grid(axis="x", alpha=0.2)
        ax.set_yticks(np.arange(len(MAP_ORDER)))
        ax.set_yticklabels([SITE_LABELS[map_id] for map_id in MAP_ORDER])
        ax.invert_yaxis()

    handles = [
        Line2D([0], [0], color=GROUP_COLORS[group], linewidth=8, label=group)
        for group in GROUP_ORDER
        if group in set(group_table["feature_group"])
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=4,
        frameon=False,
        fontsize=8,
        columnspacing=1.6,
        handletextpad=0.6,
    )
    fig.suptitle(
        "Regional feature-group attribution from all valid low-OOD water pixels",
        fontsize=13,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.80, bottom=0.31, wspace=0.10)
    fig.savefig(FIGURE_DIR / "figure_regional_feature_group_attribution.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "figure_regional_feature_group_attribution.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_spatial_variation_attribution(group_table: pd.DataFrame):
    """Show which groups drive within-scene differences, not the scene baseline."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.8), sharey=True)
    for target_index, target in enumerate(TARGET_ORDER):
        ax = axes[target_index]
        subset = group_table[group_table["target"].eq(target)].copy()
        pivot = subset.pivot(
            index="map_id",
            columns="feature_group",
            values="spatial_variation_share",
        )
        pivot = pivot.reindex(MAP_ORDER).fillna(0)
        left = np.zeros(len(MAP_ORDER))
        for group in GROUP_ORDER:
            if group not in pivot.columns:
                continue
            values = 100 * pivot[group].to_numpy(float)
            ax.barh(
                np.arange(len(MAP_ORDER)),
                values,
                left=left,
                color=GROUP_COLORS[group],
                edgecolor="white",
                linewidth=0.35,
                label=group,
            )
            left += values
        ax.set_title(TARGET_LABELS[target], fontsize=10, fontweight="bold")
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of within-scene SHAP variation (%)")
        ax.grid(axis="x", alpha=0.2)
        ax.set_yticks(np.arange(len(MAP_ORDER)))
        ax.set_yticklabels([SITE_LABELS[map_id] for map_id in MAP_ORDER])
        ax.invert_yaxis()

    handles = [
        Line2D([0], [0], color=GROUP_COLORS[group], linewidth=8, label=group)
        for group in GROUP_ORDER
        if group in set(group_table["feature_group"])
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=4,
        frameon=False,
        fontsize=8,
    )
    fig.suptitle(
        "Feature groups associated with within-scene variation in model attribution",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "Shares use the mean absolute deviation of signed group SHAP values from each scene mean.",
        ha="center",
        fontsize=7,
    )
    fig.tight_layout(rect=[0, 0.23, 1, 0.93])
    fig.savefig(
        FIGURE_DIR / "figure_regional_within_scene_shap_variation.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        FIGURE_DIR / "figure_regional_within_scene_shap_variation.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def macro_top_features(feature_table: pd.DataFrame) -> dict[str, list[str]]:
    output = {}
    for target in TARGET_ORDER:
        ranked = (
            feature_table[feature_table["target"].eq(target)]
            .groupby("feature", as_index=False)["mean_abs_shap"]
            .mean()
            .sort_values("mean_abs_shap", ascending=False)
        )
        output[target] = ranked.head(TOP_FEATURES_PER_TARGET)["feature"].tolist()
    return output


def plot_regional_beeswarm(feature_table: pd.DataFrame):
    top_features = macro_top_features(feature_table)
    display_frames = {}
    x_limits = {}
    rng = np.random.default_rng(RANDOM_STATE)

    for target in TARGET_ORDER:
        target_chunks = []
        for map_id in MAP_ORDER:
            frame = load_map_target(map_id, target)
            if len(frame) > DISPLAY_MAX_PER_PANEL:
                selected = np.sort(
                    rng.choice(len(frame), size=DISPLAY_MAX_PER_PANEL, replace=False)
                )
                display_frame = frame.iloc[selected].copy()
            else:
                display_frame = frame.copy()
            display_frames[(map_id, target)] = display_frame
            target_chunks.append(
                np.abs(
                    display_frame[[f"shap__{feature}" for feature in top_features[target]]]
                    .to_numpy(float)
                    .ravel()
                )
            )
        pooled_abs = np.concatenate(target_chunks)
        limit = float(np.quantile(pooled_abs[np.isfinite(pooled_abs)], 0.995))
        x_limits[target] = max(limit, 1e-9)

    fig, axes = plt.subplots(3, 3, figsize=(10.2, 11.8), squeeze=False)
    cmap = plt.get_cmap("coolwarm")

    for row_index, map_id in enumerate(MAP_ORDER):
        for column_index, target in enumerate(TARGET_ORDER):
            ax = axes[row_index, column_index]
            frame = display_frames[(map_id, target)]
            features = top_features[target]
            for display_index, feature in enumerate(features):
                y_base = len(features) - 1 - display_index
                shap_values = frame[f"shap__{feature}"].to_numpy(float)
                raw_values = frame[f"value__{feature}"].to_numpy(float)
                finite_raw = raw_values[np.isfinite(raw_values)]
                if len(finite_raw):
                    low, high = np.quantile(finite_raw, [0.05, 0.95])
                    if high <= low:
                        normalized = np.full(len(raw_values), 0.5)
                    else:
                        normalized = np.clip((raw_values - low) / (high - low), 0, 1)
                        normalized = np.where(np.isfinite(normalized), normalized, 0.5)
                else:
                    normalized = np.full(len(raw_values), 0.5)
                jitter = rng.normal(0, 0.105, size=len(frame))
                ax.scatter(
                    shap_values,
                    y_base + jitter,
                    c=normalized,
                    cmap=cmap,
                    vmin=0,
                    vmax=1,
                    s=4,
                    alpha=0.42,
                    linewidths=0,
                    rasterized=True,
                )
            ax.axvline(0, color="#555555", linewidth=0.7)
            ax.set_xlim(-x_limits[target], x_limits[target])
            ax.set_yticks(np.arange(len(features)))
            ax.set_yticklabels(list(reversed(features)), fontsize=6.5)
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(axis="x", alpha=0.14)
            if row_index == 0:
                ax.set_title(TARGET_LABELS[target], fontsize=10, fontweight="bold")
            if column_index == 0:
                ax.set_ylabel(SITE_LABELS[map_id], fontsize=9, fontweight="bold")
            if row_index == len(MAP_ORDER) - 1:
                ax.set_xlabel(f"SHAP value [{TARGET_LABELS[target].split('(')[-1].rstrip(')')}]", fontsize=8)

    color_axis = fig.add_axes([0.30, 0.035, 0.40, 0.012])
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)),
        cax=color_axis,
        orientation="horizontal",
    )
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["Low feature value", "High feature value"])
    colorbar.ax.tick_params(labelsize=7)
    fig.suptitle(
        "Regional SHAP distributions across all mapped water pixels",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        (
            "SHAP statistics use every valid low-OOD pixel; at most "
            f"{DISPLAY_MAX_PER_PANEL:,} points per panel are displayed only for legibility."
        ),
        ha="center",
        fontsize=7,
    )
    fig.tight_layout(rect=[0.02, 0.07, 1, 0.96])
    fig.savefig(FIGURE_DIR / "figure_regional_shap_beeswarm_3x3.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "figure_regional_shap_beeswarm_3x3.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_dominant_group_maps(wf, bundles):
    used_groups = []
    map_arrays = {}
    map_shapes = {}

    for map_id in MAP_ORDER:
        feature_path = wf.find_raster(map_id, "moving_window_features.tif")
        with rasterio.open(feature_path) as src:
            map_shapes[map_id] = (src.height, src.width)
        for target in TARGET_ORDER:
            frame = load_map_target(map_id, target)
            features = list(bundles[target]["feature_names"])
            s2_features = wf.s2_feature_set(target, features)
            feature_groups = {
                feature: wf.target_feature_group(target, feature, s2_features)
                for feature in features
            }
            groups = [group for group in GROUP_ORDER if group in set(feature_groups.values())]
            for group in groups:
                if group not in used_groups:
                    used_groups.append(group)
            group_values = []
            for group in groups:
                group_features = [feature for feature in features if feature_groups[feature] == group]
                group_values.append(
                    np.sum(
                        np.abs(frame[[f"shap__{feature}" for feature in group_features]].to_numpy(float)),
                        axis=1,
                    )
                )
            dominant = np.argmax(np.column_stack(group_values), axis=1)
            dominant_group = np.asarray([groups[index] for index in dominant], dtype=object)
            map_arrays[(map_id, target)] = (frame, dominant_group)

    group_to_code = {group: index for index, group in enumerate(used_groups)}
    cmap = ListedColormap([GROUP_COLORS[group] for group in used_groups])
    norm = BoundaryNorm(np.arange(-0.5, len(used_groups) + 0.5), cmap.N)
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9.2), squeeze=False)

    for row_index, map_id in enumerate(MAP_ORDER):
        height, width = map_shapes[map_id]
        for column_index, target in enumerate(TARGET_ORDER):
            ax = axes[row_index, column_index]
            frame, dominant_group = map_arrays[(map_id, target)]
            raster = np.full((height, width), np.nan, dtype=float)
            codes = np.asarray([group_to_code[group] for group in dominant_group], dtype=float)
            valid_rows = frame["row"].to_numpy(int)
            valid_cols = frame["col"].to_numpy(int)
            raster[valid_rows, valid_cols] = codes

            row_pad = max(3, int(0.03 * (valid_rows.max() - valid_rows.min() + 1)))
            col_pad = max(3, int(0.03 * (valid_cols.max() - valid_cols.min() + 1)))
            row_start = max(0, int(valid_rows.min()) - row_pad)
            row_stop = min(height, int(valid_rows.max()) + row_pad + 1)
            col_start = max(0, int(valid_cols.min()) - col_pad)
            col_stop = min(width, int(valid_cols.max()) + col_pad + 1)
            cropped = raster[row_start:row_stop, col_start:col_stop]

            ax.imshow(cropped, cmap=cmap, norm=norm, interpolation="nearest")
            ax.set_axis_off()
            if row_index == 0:
                ax.set_title(TARGET_LABELS[target], fontsize=10, fontweight="bold")
            if column_index == 0:
                ax.text(
                    -0.04,
                    0.5,
                    SITE_LABELS[map_id],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=9,
                    fontweight="bold",
                )

    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markersize=8,
            markerfacecolor=GROUP_COLORS[group],
            markeredgecolor="none",
            label=group,
        )
        for group in used_groups
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=False,
        fontsize=7.5,
    )
    fig.suptitle(
        "Dominant feature group at each valid low-OOD water pixel",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0.03, 0.09, 1, 0.95])
    fig.savefig(FIGURE_DIR / "figure_regional_dominant_feature_group_maps.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "figure_regional_dominant_feature_group_maps.pdf", bbox_inches="tight")
    plt.close(fig)


def write_method_summary(
    wf,
    pixel_audit,
    batch_manifest,
    additivity_table,
    missingness_table,
):
    total_pixels = int(pixel_audit["common_valid_low_ood_pixels"].sum())
    completed_runtime = batch_manifest.loc[
        batch_manifest["status"].eq("completed"), "elapsed_seconds"
    ].sum()
    batch_files = list(BATCH_DIR.rglob("batch_*.parquet"))
    creation_span_minutes = np.nan
    if len(batch_files) > 1:
        timestamps = [path.stat().st_mtime for path in batch_files]
        creation_span_minutes = (max(timestamps) - min(timestamps)) / 60

    feature_audit = (
        missingness_table.groupby(["map_id", "waterbody", "target"], as_index=False)
        .agg(
            n_model_features=("feature", "size"),
            all_missing_features=("all_missing", "sum"),
            constant_features=("constant_within_scene", "sum"),
            mean_missing_fraction=("missing_fraction", "mean"),
        )
    )
    feature_audit["mean_missing_fraction"] = feature_audit["mean_missing_fraction"].round(4)
    lines = [
        "# Regional all-pixel SHAP workflow",
        "",
        "## Analysis frame",
        "",
        (
            f"SHAP values were calculated for all {total_pixels:,} common valid water pixels "
            "inside WB02, WB07, and WB05. Pixels were retained when prediction_support > 0, "
            f"all three predictions were finite, and every target-specific OOD fraction was <= {wf.OOD_MAX}."
        ),
        "",
        dataframe_to_markdown(
            pixel_audit[["map_id", "waterbody", "sat_date", "common_valid_low_ood_pixels"]]
        ),
        "",
        "## Locked model explanations",
        "",
        "- Turbidity: end-to-end Permutation SHAP in original NTU.",
        "- Suspended solids: end-to-end Permutation SHAP for the two-stage soft-gated model in mg/L.",
        "- DO proxy: native CatBoost TreeSHAP in mg/L.",
        "- The training background was fixed across all mapping scenes.",
        "- Every-pixel statistics use the complete retained frame; display subsampling is used only in beeswarm figures.",
        "",
        "## Runtime and verification",
        "",
        f"Newly computed SHAP batches in this invocation required {completed_runtime / 60:.1f} minutes.",
        (
            f"The cached batch files span approximately {creation_span_minutes:.1f} minutes of creation time."
            if np.isfinite(creation_span_minutes)
            else "Batch creation-time span was unavailable."
        ),
        "",
        dataframe_to_markdown(additivity_table),
        "",
        "## Scene-level feature audit",
        "",
        dataframe_to_markdown(feature_audit),
        "",
        "All-missing values are handled by the locked pipeline's fitted imputer. "
        "Features that are constant within a scene can alter the scene baseline but cannot explain fine-scale spatial variation.",
        "",
        "## Interpretation",
        "",
        "The outputs describe behavior of the locked predictive models over the mapped scenes. "
        "They do not establish causal environmental effects and do not replace same-date field validation. "
        "Adjacent moving-window pixels are spatially autocorrelated and must not be treated as independent observations.",
    ]
    (OUTPUT_DIR / "regional_all_pixel_shap_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_manifest(wf, bundles, pixel_audit):
    manifest = {
        "workflow": "regional_all_pixel_shap_v1",
        "created": pd.Timestamp.now().isoformat(),
        "base_workflow": str(BASE_WORKFLOW_PATH),
        "output_directory": str(OUTPUT_DIR),
        "batch_size": BATCH_SIZE,
        "display_max_per_panel": DISPLAY_MAX_PER_PANEL,
        "random_state": RANDOM_STATE,
        "ood_threshold": wf.OOD_MAX,
        "maps": pixel_audit.to_dict(orient="records"),
        "models": {
            target: {
                "artifact_id": bundles[target]["artifact_id"],
                "feature_count": len(bundles[target]["feature_names"]),
                "prediction_contract": bundles[target]["prediction_contract"],
            }
            for target in TARGET_ORDER
        },
        "methods": {
            "Turbidity": "End-to-end Permutation SHAP in NTU",
            "SS": "End-to-end Permutation SHAP for two-stage soft-gated prediction in mg/L",
            "DO proxy": "CatBoost native TreeSHAP in mg/L",
        },
    }
    (OUTPUT_DIR / "regional_all_pixel_shap_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def main():
    ensure_dirs()
    wf = load_base_workflow()
    _, bundles, splits, _ = wf.load_inputs()
    pixel_audit, batch_manifest = explain_all_pixels(wf, bundles, splits)
    feature_table, group_table, _, additivity_table, missingness_table = summarize_all_pixels(
        wf,
        bundles,
    )
    plot_group_attribution(group_table)
    plot_spatial_variation_attribution(group_table)
    plot_regional_beeswarm(feature_table)
    plot_dominant_group_maps(wf, bundles)
    write_method_summary(
        wf,
        pixel_audit,
        batch_manifest,
        additivity_table,
        missingness_table,
    )
    write_manifest(wf, bundles, pixel_audit)
    print(f"Regional all-pixel SHAP outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
