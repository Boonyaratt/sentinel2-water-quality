# Agent Guide: Sentinel-2 Water Quality Research

Start here when helping with this repository or its team handoff. The research estimates turbidity, suspended solids (SS), and an indirect dissolved oxygen **DO proxy** in southern Thailand. It compares Sentinel-2-only and Sentinel-2 + environmental context, then applies locked models to archived scenes. The research pipeline and the web prototype are separate.

## Locate the workspace

- **Public GitHub checkout:** Read [README.md](README.md), [the data inventory](docs/data-inventory.md), and [the Colab index](notebooks/README.md). Public notebooks are adapted copies; the checkout alone cannot reproduce the results.
- **Extracted private team ZIP:** Start with `START_HERE.md`, then `PIPELINE.md`, `COLAB.md`, `MANIFEST.csv`, and `audit/PROVENANCE.md`. Refer to [the public repository](https://github.com/Boonyaratt/sentinel2-water-quality) for the Colab launch links if local links in this guide are not present.
- The full team snapshot contains cleaned field observations, precomputed exports, original notebooks, fitted bundles, and reference outputs. The small map-support ZIP contains **no TIFFs** and cannot replace the full snapshot. If data are missing, ask for the authorized snapshot or the exact missing file. Do not silently substitute newer or similarly named Drive exports.

## Choose the task and inputs

Paths in the second and third columns are relative to the extracted full ZIP unless noted otherwise.

| Task | Source | Input / output |
|---|---|---|
| Start from cleaned observations | `src/01_ground/observation_data1.ipynb` inspects the cleaned data | `data_ready/ground/wq_14_2562_2568_rs_ready.csv`, 826 rows. Original raw Excel is not in the team ZIP. |
| Rebuild Sentinel-2 station features | `src/02_station_features/01_CAL_make_gg_ee_rswq_clean_2562_2568_b100_v4.ipynb`; public Colab 01 | Use the cleaned field CSV. Existing 100 m / watermask v4 tables and chunks are in `data_ready/station_features/`. |
| Rebuild environmental context | `src/02_station_features/03_make_rswq_context_features.ipynb`; public Colab 02 | Use the cleaned field CSV. Existing strict v2 CSV and export chunks are in `data_ready/station_features/`. |
| Reassemble the model table | `src/03_assembly/02_Merge_rswq_dataset.ipynb` or the verification runner | Join archived station features and context by `sample_id`. Reference: `data_ready/model/rswq_model_input_relaxed_s2_context_2562_2568.csv`, 340 rows × 226 columns. |
| Revisit model evaluation | `src/04_models/`: Turbidity `01_tur_within1_feature_model_compare_before_artifact_export.ipynb` (cells 24–32), SS `SS_model.ipynb` (cells 54–57), DO `01_DO_within1_temporal_model_compare.ipynb` | Start from the relaxed model CSV. The Turbidity notebook **without** the suffix is the LinearSVR artifact source for mapping. |
| Apply locked models | `models/mapping_v1/model_registry.json`, `artifacts/`, `contracts/` | Six bundles; use registry feature order and transforms. Compare to `models/mapping_v1/validation/test_predictions.csv`. |
| Rebuild polygon feature rasters | `src/05_mapping/03_multi_waterbody_moving_window_features.ipynb`; public Colab 03 | Reviewed GeoJSON, coverage and scene metadata; existing 62-band features and RGB are in `data_ready/rasters/polygon/`. |
| Rebuild the two-date feature rasters | `src/06_two_dates/03_prepare_figure2_contrasting_dates_colab.ipynb`; public Colab 03b | Keep the same WB02 polygon and grid. Use the saved manifest in `data_ready/maps/two_date_preparation/`; existing features and RGB are in `data_ready/rasters/two_dates/`. |

For historical model evaluation, the notebooks filter to ±1 day and use 2019–2023 for training and 2024–2025 for evaluation. Turbidity and DO proxy have 79/33 rows; SS has 77/33 after applying SS ≤130 mg/L. Derive the split from code and data; do not assume there are separate train/test CSVs. For a new experiment, keep configuration selection inside training data and report it separately from the historical results.

## Reproduce a figure

| Manuscript figure | Source and dependencies |
|---|---|
| Figs. 1–2 | Published PNGs in GitHub `figures/`; editable originals have not been verified. |
| Fig. 3 | ZIP `src/05_mapping/04_predict_multi_waterbody_water_quality_maps_paper.ipynb`; public Colab 04. Use the archived polygon grid, RGB and locked mapping bundles for WB02, WB07 and WB05. |
| Fig. 4 | ZIP `src/05_mapping/05_figure1_s2_vs_context_comparison_colab.ipynb`; public Colab 05. Compare S2 + Context minus S2-only on common supported water pixels. |
| Fig. 5 | ZIP `src/06_two_dates/06b_figure2_fast_counterfactual_colab.ipynb`; public Colab 06b. Use WB02, 4 and 19 June 2024, the archived common grid and the common low-OOD mask. Show 19 June minus 4 June. |
| Fig. 6 | ZIP `src/07_xai/05_final_model_shap_analysis.ipynb`, `final_model_shap_workflow.py` and `create_figure_x_global_shap_composite.py`; the Python scripts are also public. |
| Fig. 7 | `src/07_xai/regional_all_pixel_shap_workflow.py` in ZIP and GitHub. It requires `data_ready/rasters/regional_shap_v1/` from the ZIP. |

Working notebook names `Figure 1` and `Figure 2` mean manuscript Figs. 4 and 5. Do not interchange `polygon`, `two_dates`, and `regional_shap_v1` rasters, even where filenames match. Check CRS, transform, dimensions, band order, masks and nodata before comparing pixels. Preserve reviewed polygons and original-unit conventions.

## Verify before changing results

1. From the extracted full ZIP, create a Python 3.11 environment and install `requirements.txt`. Run `python -B tools/reproduce.py --output run_output/agent_check`. The output receipt is `run_output/agent_check/verification.json`. Alternatively open public Colab `00_Start_Here.ipynb` with the full ZIP.
2. Expect four checks: `assembly`, `locked_models`, `spatial_inputs_and_samples`, and `spatial_comparisons`. The default run uses local files and does not need Earth Engine. `--global-shap` is optional; `--regional-shap` is a separate, slow all-pixel run. Do not use `--ground` with this team ZIP.
3. Keep inputs and `expected_outputs/` immutable. Write new runs to `run_output/`. Compare to references and report the exact inputs, commands, outputs and any failures. Do not call a sampled check or a smoke test a full figure or training reproduction.
4. The submitted Table 5 CSV in `expected_outputs/Table05/` records what the manuscript reported; it is not generated by the smoke test. Some configurations used 2024–2025 scores for selection, and some rows have unresolved provenance or numerical differences. Read `audit/PROVENANCE.md` before claiming exact retraining or an untouched test set.
5. Report DO as an indirect proxy and SHAP as model attribution, not causation. The two-date pair was selected from contextual contrast and coverage; the prediction changes are results, not selection criteria. OOD screening is not a calibrated uncertainty interval.

## Data and execution boundaries

- Prefer the archived exports for reproduction. Earth Engine regeneration is a separate task: use the operator's authorized project, retain the reviewed input versions, and leave export switches off until explicitly requested.
- Do not put row-level data, private Drive links, credentials, fitted bundles, raster files, or notebook outputs into the public repo. If editing the public Colab adaptations, update `tools/build_colab.py` and regenerate; do not publish the private embedded artifact cells.
- Do not rename or overwrite source datasets, alter shared Drive permissions, or change cloud resources as part of a local reproduction. Point out missing access or differences and distinguish verified observations from assumptions.
