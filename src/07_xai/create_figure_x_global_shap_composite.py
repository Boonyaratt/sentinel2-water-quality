"""Create Figure X: global SHAP interpretation of the three locked models.

The figure is generated only from the final tabular SHAP outputs. It does not
reload, refit, or alter any predictive model.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd


ROOT = Path(r".\Results\XAI\v1")
GROUP_PATH = ROOT / "tables" / "feature_group_attribution.csv"
SHAP_PATH = ROOT / "tables" / "global_shap_long.csv"
FIGURE_DIR = ROOT / "figures"

TARGET_ORDER = ["Turbidity", "SS", "DO proxy"]
TARGET_TITLES = {
    "Turbidity": "Turbidity",
    "SS": "Suspended solids",
    "DO proxy": "DO proxy",
}
TARGET_XLABELS = {
    "Turbidity": "SHAP value [log1p(NTU)]",
    "SS": "SHAP value [mg/L; end-to-end prediction]",
    "DO proxy": "SHAP value [mg/L]",
}

# Fixed ordering keeps the stacked bars comparable across targets.
GROUP_ORDER = [
    "Sentinel-2 spectral",
    "Land cover",
    "Seasonality",
    "Water-pixel QA",
    "Rainfall",
    "Air temperature",
    "Cloud QA",
    "Wind",
    "Surrounding-water fraction",
    "Runoff",
    "Historical water",
    "Solar radiation",
]
GROUP_COLORS = {
    "Sentinel-2 spectral": "#2F6B9A",
    "Land cover": "#8C6D31",
    "Seasonality": "#E69F00",
    "Water-pixel QA": "#756BB1",
    "Rainfall": "#3A9D5D",
    "Air temperature": "#D55E00",
    "Cloud QA": "#A76DCA",
    "Wind": "#56B4E9",
    "Surrounding-water fraction": "#6BAED6",
    "Runoff": "#1B998B",
    "Historical water": "#4C956C",
    "Solar radiation": "#E6C229",
}

FEATURE_VALUE_CMAP = LinearSegmentedColormap.from_list(
    "shap_feature_value", ["#277DA1", "#F7F7F7", "#D73058"]
)


def pretty_feature_name(name: str) -> str:
    """Shorten machine column names without changing their meaning."""
    replacements = {
        "B1_median": "B1",
        "B2_median": "B2",
        "B3_median": "B3",
        "B4_median": "B4",
        "B5_median": "B5",
        "B6_median": "B6",
        "B7_median": "B7",
        "B8_median": "B8",
        "B8A_median": "B8A",
        "B9_median": "B9",
        "B11_median": "B11",
        "B12_median": "B12",
        "NDTI_median": "NDTI",
        "NDWI_median": "NDWI",
        "valid_water_pixel_count": "Valid water-pixel count",
        "day_of_year": "Day of year",
        "wc_tree_frac_1000m": "Tree cover fraction (1 km)",
        "wc_shrub_frac_5000m": "Shrubland fraction (5 km)",
        "wc_bare_frac_500m": "Bare-land fraction (500 m)",
        "wc_shrub_frac_1000m": "Shrubland fraction (1 km)",
        "wc_grass_frac_1000m": "Grassland fraction (1 km)",
        "era5h_air_temp_prior_72h_C": "Air temperature, prior 72 h",
        "era5h_wind_max_prior_168h_m_s": "Maximum wind, prior 168 h",
    }
    return replacements.get(name, name.replace("_", " "))


def validate_inputs(groups: pd.DataFrame, shap_long: pd.DataFrame) -> None:
    group_required = {"target", "feature_group", "group_attribution_share"}
    shap_required = {
        "target",
        "split",
        "observation_id",
        "feature",
        "raw_feature_value",
        "shap_value",
    }
    missing_group = group_required.difference(groups.columns)
    missing_shap = shap_required.difference(shap_long.columns)
    if missing_group:
        raise ValueError(f"Missing columns in {GROUP_PATH.name}: {sorted(missing_group)}")
    if missing_shap:
        raise ValueError(f"Missing columns in {SHAP_PATH.name}: {sorted(missing_shap)}")

    if set(TARGET_ORDER).difference(groups["target"].unique()):
        raise ValueError("Feature-group table does not contain all three locked targets.")
    if set(TARGET_ORDER).difference(shap_long["target"].unique()):
        raise ValueError("Global SHAP table does not contain all three locked targets.")


def symmetric_jitter(values: np.ndarray, max_jitter: float = 0.25) -> np.ndarray:
    """Create deterministic beeswarm-like vertical offsets by x-value bins."""
    values = np.asarray(values, dtype=float)
    if len(values) <= 1 or np.nanmax(values) == np.nanmin(values):
        return np.zeros(len(values), dtype=float)

    # Rank bins keep the display stable and avoid dependence on random jitter.
    order = np.argsort(values, kind="mergesort")
    bins = np.empty(len(values), dtype=int)
    bins[order] = np.minimum((np.arange(len(values)) * 10) // len(values), 9)
    offsets = np.zeros(len(values), dtype=float)
    for bin_id in np.unique(bins):
        members = np.flatnonzero(bins == bin_id)
        pattern = [0]
        for level in range(1, len(members)):
            magnitude = (level + 1) // 2
            sign = 1 if level % 2 else -1
            pattern.append(sign * magnitude)
        pattern = np.asarray(pattern, dtype=float)
        if np.max(np.abs(pattern)) > 0:
            pattern *= max_jitter / np.max(np.abs(pattern))
        offsets[members] = pattern
    return offsets


def normalized_feature_values(values: np.ndarray) -> np.ndarray:
    """Robust within-feature normalization used only for point color."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    normalized = np.full(len(values), 0.5, dtype=float)
    if not finite.any():
        return normalized
    low, high = np.nanpercentile(values[finite], [5, 95])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return normalized
    normalized[finite] = np.clip((values[finite] - low) / (high - low), 0, 1)
    return normalized


def plot_group_attribution(ax: plt.Axes, groups: pd.DataFrame) -> None:
    groups = groups.copy()
    groups["feature_group"] = groups["feature_group"].replace(
        {"Historical-water occurrence": "Historical water"}
    )
    pivot = (
        groups.pivot_table(
            index="target",
            columns="feature_group",
            values="group_attribution_share",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(TARGET_ORDER)
        .fillna(0)
    )

    left = np.zeros(len(TARGET_ORDER), dtype=float)
    handles = []
    labels = []
    for group in GROUP_ORDER:
        if group not in pivot.columns:
            continue
        values = pivot[group].to_numpy(dtype=float)
        bars = ax.barh(
            TARGET_ORDER,
            values,
            left=left,
            height=0.58,
            color=GROUP_COLORS[group],
            edgecolor="white",
            linewidth=0.55,
            label=group,
        )
        handles.append(bars[0])
        labels.append(group)
        for row_index, (start, width) in enumerate(zip(left, values)):
            if width >= 0.08:
                ax.text(
                    start + width / 2,
                    row_index,
                    f"{width * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    color="white" if group in {"Sentinel-2 spectral", "Land cover", "Water-pixel QA"} else "#202020",
                    fontweight="bold",
                )
        left += values

    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of absolute SHAP attribution", labelpad=3)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.38),
        ncol=4,
        frameon=False,
        fontsize=5.8,
        handlelength=1.25,
        columnspacing=1.0,
        handletextpad=0.4,
    )


def plot_beeswarm(ax: plt.Axes, shap_long: pd.DataFrame, target: str, top_n: int = 8) -> None:
    target_df = shap_long.loc[
        shap_long["target"].eq(target) & shap_long["split"].eq("evaluation")
    ].copy()
    target_df["shap_value"] = pd.to_numeric(target_df["shap_value"], errors="coerce")
    target_df["raw_feature_value"] = pd.to_numeric(target_df["raw_feature_value"], errors="coerce")
    target_df = target_df.dropna(subset=["shap_value"])

    feature_order = (
        target_df.assign(abs_shap=target_df["shap_value"].abs())
        .groupby("feature", as_index=False)["abs_shap"]
        .mean()
        .sort_values("abs_shap", ascending=False)
        .head(top_n)["feature"]
        .tolist()
    )
    if len(feature_order) != top_n:
        raise ValueError(f"{target}: expected {top_n} features, found {len(feature_order)}")

    for row, feature in enumerate(feature_order):
        feature_df = target_df.loc[target_df["feature"].eq(feature)].copy()
        feature_df = feature_df.sort_values("observation_id", kind="stable")
        x = feature_df["shap_value"].to_numpy(dtype=float)
        raw = feature_df["raw_feature_value"].to_numpy(dtype=float)
        y = row + symmetric_jitter(x)
        color = normalized_feature_values(raw)
        ax.scatter(
            x,
            y,
            c=color,
            cmap=FEATURE_VALUE_CMAP,
            norm=Normalize(0, 1),
            s=25,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.28,
            zorder=3,
        )

    ax.axvline(0, color="#666666", linewidth=0.8, zorder=1)
    ax.set_yticks(np.arange(len(feature_order)))
    ax.set_yticklabels([pretty_feature_name(feature) for feature in feature_order])
    ax.set_ylim(len(feature_order) - 0.45, -0.55)
    ax.set_xlabel(TARGET_XLABELS[target], labelpad=3)
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, labelsize=7.6)
    ax.tick_params(axis="x", labelsize=7.2)
    ax.spines[["top", "right", "left"]].set_visible(False)


def build_figure(groups: pd.DataFrame, shap_long: pd.DataFrame) -> plt.Figure:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    figure = plt.figure(figsize=(7.2, 10.5), facecolor="white")
    grid = figure.add_gridspec(
        4,
        1,
        height_ratios=[1.35, 1.85, 1.85, 1.85],
        left=0.31,
        right=0.91,
        top=0.935,
        bottom=0.065,
        hspace=0.79,
    )
    axes = [figure.add_subplot(grid[index, 0]) for index in range(4)]

    plot_group_attribution(axes[0], groups)
    plot_beeswarm(axes[1], shap_long, "Turbidity")
    plot_beeswarm(axes[2], shap_long, "SS")
    plot_beeswarm(axes[3], shap_long, "DO proxy")

    panel_titles = [
        "(A) Feature-group attribution",
        "(B) Turbidity: top eight features",
        "(C) Suspended solids: top eight features",
        "(D) DO proxy: top eight features",
    ]
    for ax, title in zip(axes, panel_titles):
        ax.set_title(title, loc="left", fontweight="bold", pad=7)

    # One shared color scale is valid because raw values are normalized within feature.
    color_axis = figure.add_axes([0.935, 0.065, 0.018, 0.61])
    scalar = mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap=FEATURE_VALUE_CMAP)
    colorbar = figure.colorbar(scalar, cax=color_axis)
    colorbar.set_ticks([0, 1], labels=["Low", "High"])
    colorbar.ax.tick_params(labelsize=7, length=2)
    colorbar.set_label("Relative feature value\n(normalized within feature)", fontsize=7, labelpad=5)
    colorbar.outline.set_linewidth(0.6)

    figure.suptitle(
        "Global SHAP interpretation of the three locked water-quality models",
        x=0.51,
        y=0.982,
        fontsize=12,
        fontweight="bold",
    )
    return figure


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    groups = pd.read_csv(GROUP_PATH)
    shap_long = pd.read_csv(SHAP_PATH, low_memory=False)
    validate_inputs(groups, shap_long)

    figure = build_figure(groups, shap_long)
    output_stem = FIGURE_DIR / "figure_X_global_SHAP_locked_models_composite"
    figure.savefig(output_stem.with_suffix(".png"), dpi=600, facecolor="white")
    figure.savefig(output_stem.with_suffix(".pdf"), facecolor="white")
    figure.savefig(output_stem.with_suffix(".svg"), facecolor="white")
    plt.close(figure)

    caption = (
        "Figure X. Global SHAP interpretation of the three locked water-quality models. "
        "(A) Feature-group attribution, expressed as each group's share of total mean absolute "
        "SHAP attribution. (B-D) SHAP beeswarm plots for the eight most influential features "
        "on the 2024-2025 temporal evaluation data for turbidity, suspended solids (SS), and "
        "the dissolved-oxygen (DO) proxy, respectively. Point color denotes the relative raw "
        "feature value normalized within each feature. Positive SHAP values increase the model "
        "output and negative values decrease it. Turbidity SHAP values are reported on the "
        "log1p(NTU) model-output scale; SS attributions describe the end-to-end two-stage "
        "prediction in mg/L; and DO-proxy attributions are in mg/L. Attributions are "
        "model-specific associations and should not be interpreted as causal effects."
    )
    (FIGURE_DIR / "figure_X_global_SHAP_locked_models_caption.txt").write_text(
        caption,
        encoding="utf-8",
    )

    eval_counts = (
        shap_long.loc[shap_long["split"].eq("evaluation")]
        .groupby("target")["observation_id"]
        .nunique()
        .reindex(TARGET_ORDER)
    )
    print(f"Created: {output_stem.with_suffix('.png')}")
    print(f"Created: {output_stem.with_suffix('.pdf')}")
    print(f"Created: {output_stem.with_suffix('.svg')}")
    print("Evaluation observations:")
    print(eval_counts.to_string())


if __name__ == "__main__":
    main()
