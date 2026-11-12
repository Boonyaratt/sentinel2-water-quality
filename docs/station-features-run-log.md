# Sentinel-2 station datasets and Colab run log

[Notebook 01](../notebooks/01_Sentinel2_Station_Features.ipynb) matches cleaned field observations with Sentinel-2 scenes and extracts median features from water pixels within a 100 m station buffer. It uses water mask v4 and requests a 20 m reduction scale. The three tables below are written before environmental context is joined.

## Full, strict, and relaxed station tables

| Table | Filename prefix | What it contains | Archived ZIP v2 rows | Live Colab rows |
| --- | --- | --- | ---: | ---: |
| Full (unfiltered) | `sentinel2_model_dataset_` | One row per cleaned field observation, with ground labels, scene metadata, Sentinel-2 features and QA. Unmatched or unusable observations remain in this table and may have missing feature values. | 826 | 826 |
| Strict candidates | `sentinel2_train_strict_candidates_` | A subset of the full table selected by the strict rules below. | 175 | 174 |
| Relaxed candidates | `sentinel2_train_relaxed_candidates_` | A subset of the full table selected by the relaxed rules below. | 340 | 342 |

The three filenames end in `epo14_2562_2568_b100_watermask_v4_s2formula.csv`. They each have 109 columns in the verified run. The archive stores them under `data_ready/station_features/`.

Both candidate sets require a nonmissing B3 median, at least one clear pixel, and at least one water pixel. The date difference is calculated from the calendar dates of the survey and the selected image.

| Filter | Strict | Relaxed |
| --- | --- | --- |
| Scene match type | `window_3d` | `window_3d` or `fallback_7d` |
| Absolute date difference | ≤3 days | ≤7 days |
| Local water fraction | ≥10% | >0% |
| Local cloud fraction | ≤50% | ≤70% |

The scene selector prefers candidates with at least three water pixels. If no scene meets that preference, it can retain a lower-water scene. The candidate-table filters allow such a row if the other conditions pass. See [the scene selector](https://github.com/Boonyaratt/sentinel2-water-quality/blob/509a13a2847d045de63b1d7b2405adaaac2995ad/notebooks/01_Sentinel2_Station_Features.ipynb#L877-L906) and [the table filters](https://github.com/Boonyaratt/sentinel2-water-quality/blob/509a13a2847d045de63b1d7b2405adaaac2995ad/notebooks/01_Sentinel2_Station_Features.ipynb#L2080-L2105).

[The assembly notebook](../src/03_assembly/02_Merge_rswq_dataset.ipynb) left-joins each table with environmental context by `sample_id`, keeping the station table's metadata when columns overlap. The archived relaxed table **after this join** has 340 rows × 226 columns. Historical model evaluation applies an additional ±1-day filter to that archived table, retaining 112 rows before target-specific filtering. See the [workflow](workflow.md#research-workflow) for the historical split.

## 2026-11-13 Colab run without export

The run used the cleaned ground CSV from the SHA-256-verified v2 handoff and the public Notebook 01 extraction code at commit [`509a13a`](https://github.com/Boonyaratt/sentinel2-water-quality/commit/509a13a2847d045de63b1d7b2405adaaac2995ad). It retrieved live Earth Engine results in requests of up to 12 observations using `FeatureCollection.getInfo()`, checkpointed the results in the Colab runtime, and ran the notebook's QA and table-building cells on the combined data. The live extraction took 175 seconds in that session. Earth Engine batch exports and the final Drive copy were disabled.

All 826 `sample_id` values matched the archive. Across 44 feature and QA columns, 811 rows agreed in every checked field and 15 differed in at least one field. Of the 15, six selected a different `image_id` and nine retained the archived `image_id`. Numeric comparisons used `rtol=1e-8` and `atol=1e-10`, with paired missing values treated as equal.

The live run produced 491 `window_3d` matches, 280 `fallback_7d` matches, and 55 `no_match` rows. One row left the strict set when its water fraction fell from 14.29% to 8.97%. Two rows entered the relaxed set and all archived relaxed members remained eligible in the live run.

The live table still reported more water pixels than clear pixels in 83 rows. A separate two-row check held the scene and buffer fixed and used one native 20 m B11 grid for both counts. Water/clear changed from 98/77 to 77/77 for the anomalous row and from 10/15 to 4/17 for the control. This check leaves the production scene-selection and table-filter rules as they were during the run. Further checks are needed to identify the causes of the 15 row differences and to evaluate a grid change across the full dataset.

Use the archived tables to reproduce the historical study. Record the date, inputs and comparison results when using a fresh Earth Engine run for further analysis. See the [handoff inventory](data-inventory.md) for the archived inputs and reference outputs.
