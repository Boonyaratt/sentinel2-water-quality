# Research Workflow

| Stage | Input | Method and output | Evidence |
|---|---|---|---|
| 1. Field observations | Cleaned field workbook/CSV | Standardized dates, station IDs, units, and coordinate exclusions. The canonical ready table has 826 rows. | Manuscript Tables 1 and 4 |
| 2. Station features | Field dates and locations | Sentinel-2 `100 m / watermask v4`; environmental context `GSMaP / ERA5-Land strict v2`. Archived exports substitute for live Earth Engine access. | Tables 2–3 and the [station dataset guide](station-features-run-log.md) |
| 3. Model table | Full, strict, and relaxed S2 tables and one context table | Join by `sample_id`, preserving Sentinel-2 metadata. The reference relaxed table is 340 × 226. | CSV assembly audit |
| 4. Model evaluation | Relaxed model table and target-specific filters | Fit 2019–2023 and evaluate 2024–2025. Within ±1 day there are 112 rows; the SS cutoff ≤130 mg/L retains 110. Train/evaluation counts are 79/33 for turbidity and DO proxy and 77/33 for SS. | Tables 4–5 |
| 5. Spatial prediction | Locked bundles, polygon 62-band rasters, RGB, and AOI | Apply the model and water/applicability masks, then draw prediction maps and S2-only versus S2 + Context differences. | Figs. 3–4 |
| 6. Temporal contrast | Two WB02 scenes, matching RGB and reference grid | Compare 19 June minus 4 June 2024 on 1,862,953 common valid pixels; summarize environmental context separately. | Fig. 5 and Table 6 |
| 7. Explainability | Locked models and reference tables/rasters | Global SHAP from evaluation observations. Regional attribution uses the archived v1 grid and common low-OOD water pixels. | Figs. 6–7 and Table 7 |

## Distinguish the raster families

| Folder | Use |
|---|---|
| `data_ready/rasters/polygon/` | Figs. 3–4. Use the archived scene manifest, AOI, and raster affine transform. Reproject RGB to the prediction grid when row orientation differs. |
| `data_ready/rasters/two_dates/` | Fig. 5. Two 1680 × 3065 scenes in EPSG:4326 with matching feature/RGB grids. |
| `data_ready/rasters/regional_shap_v1/` | Fig. 7. Use the original v1 feature and prediction rasters with OOD ≤0.25. |

## Reporting and evaluation notes

- The manuscript's Table 5 is a historical evaluation. Its Turbidity reporting notebook combines training-only LOYO choices with fixed-context overrides selected after holdout comparisons. The LinearSVR map-export notebook is a separate source.
- The independent file audit found three Table 5 rows that still require prediction-level tracing. The submitted CSV reference is preserved in the team archive.
- The re-run receipt verifies local table assembly, six locked models, sample-pixel predictions, spatial differences, and Global SHAP. Full historical retraining, fresh Earth Engine exports, full Colab rendering and Regional SHAP over every pixel remain separate checks.
