# Handoff Data Inventory

The public GitHub repository is the entry point for the project, figures and Colab notebooks. The restricted `Water_Quality_Processed_Handoff_Windows_v2.zip` is the input for a real reproduction. Keep the archive intact when sharing it with the team; after extraction the paths below are relative to the ZIP root. Agents should first read [AGENTS.md](../AGENTS.md), also available inside the v2 ZIP.

| Step | Input in team archive | Original source in team archive | Output or reference |
|---|---|---|---|
| 1. Read cleaned observations | `data_ready/ground/wq_14_2562_2568_rs_ready.csv`, with coordinate reference and QC CSVs beside it | `src/01_ground/observation_data1.ipynb` documents cleaning; raw Excel is not distributed | 826 × 116 clean CSV, openable in Excel |
| 2. Inspect satellite/context extraction | `data_ready/station_features/`, including the precomputed Sentinel-2 candidates, matching points, context CSV, and the original export chunks | `src/02_station_features/01_CAL_make_gg_ee_rswq_clean_2562_2568_b100_v4.ipynb` and `03_make_rswq_context_features.ipynb` | Station-scale water-pixel Sentinel-2 and environmental features |
| 3. Assemble the model table | Cleaned observations and the precomputed station features | `src/03_assembly/02_Merge_rswq_dataset.ipynb` or `tools/reproduce.py` | `data_ready/model/rswq_model_input_relaxed_s2_context_2562_2568.csv`, 340 × 226 |
| 4. Evaluate models or apply locked ones | Model input CSV; `models/mapping_v1/` contains bundles and contracts | `src/04_models/` and `tools/reproduce.py` | Six locked configurations; historical Table 5 in `expected_outputs/Table05/` |
| 5. Build Fig. 3–4 maps | `data_ready/rasters/polygon/{features,rgb}/`; `data_ready/maps/{polygon_metadata,polygons,reference}/` | Screening, polygon, moving-window and prediction notebooks in `src/05_mapping/`; public Colab 04 and 05 | Waterbody predictions and common-pixel S2 vs S2 + Context comparisons; `expected_outputs/Fig03/`, `Fig04/` |
| 6. Build Fig. 5 / Table 6 | `data_ready/rasters/two_dates/{features,rgb}/`; date pair, reference grid and QA in `data_ready/maps/two_date_preparation/` | `src/06_two_dates/03_prepare_figure2_contrasting_dates_colab.ipynb`, `06b_figure2_fast_counterfactual_colab.ipynb`; public Colab 06b | Prediction differences and contextual comparison for WB02; `expected_outputs/Fig05_Table06/` |
| 7. Reproduce SHAP | Model CSV and locked artifacts for Global SHAP; also `data_ready/rasters/regional_shap_v1/` for Regional SHAP | `src/07_xai/` and `tools/reproduce.py` | `expected_outputs/Fig06_Table07/`, `Fig07/` |

Braces in the table indicate separate directories. The polygon, two-date and regional v1 rasters serve different figure grids and cannot be substituted for one another.

## Original notebook index

All 19 original notebooks are in the team ZIP under `src/`. This is where to find them after extraction:

| Folder | Notebook filenames (add `.ipynb`) |
|---|---|
| `01_ground/` | `observation_data1` |
| `02_station_features/` | `01_CAL_make_gg_ee_rswq_clean_2562_2568_b100_v4`, `03_make_rswq_context_features` |
| `03_assembly/` | `02_Merge_rswq_dataset` |
| `04_models/` | `01_DO_within1_temporal_model_compare`, `01_tur_within1_feature_model_compare_before_artifact_export`, `01_tur_within1_feature_model_compare`, `SS_best_two_stage_xgb_ridge`, `SS_model` |
| `05_mapping/` | `01_lock_models_and_select_map_candidates`, `02_screen_all_waterbodies_clear_water_coverage`, `02b_draw_waterbody_polygons_colab`, `03_multi_waterbody_moving_window_features`, `03b_validate_and_register_locked_artifacts`, `04_predict_multi_waterbody_water_quality_maps_paper`, `05_figure1_s2_vs_context_comparison_colab` |
| `06_two_dates/` | `03_prepare_figure2_contrasting_dates_colab`, `06b_figure2_fast_counterfactual_colab` |
| `07_xai/` | `05_final_model_shap_analysis` |

## Version and access

- Current agent handoff: `Water_Quality_Processed_Handoff_Windows_v2.zip`, 701,271,208 bytes. SHA-256: `bff939d07db4ca710e62b3af1ba55386d498768cfc82bda3afe58c7e59ce0adc`. It has 200 files: the original research content plus `AGENTS.md`, with `START_HERE.md` and `MANIFEST.csv` updated.
- Snapshot: `Water_Quality_Processed_Handoff.zip`, 699,886,253 bytes. SHA-256: `1f1039b7a6d113685c2c95f840f2a475be5e936e02090bcf5283f2ba9b506773`.
- File Explorer-compatible copy: `Water_Quality_Processed_Handoff_Windows.zip`, 701,267,394 bytes. SHA-256: `324437bb6d0bb84c6bc1865a8ea1a172171974c6f039355deacc6a15ad7acc70`. Both packages have the same 199 files and manifest; the ZIP hashes differ because the archive paths and compression changed.
- Use the snapshot, not a similarly named Drive file. `MANIFEST.csv` inside the archive identifies individual inputs by path, size, and SHA-256. `PIPELINE.md` provides the more detailed dependency map.
- Share the ZIP to named team accounts using restricted Drive access. Send the [Colab notebook index](../notebooks/README.md) alongside the private file link. Confirm a recipient can open both links and send back their `verification.json`.
- Keep each recipient's new analyses in a separate results folder. Do not replace the snapshot with an edited ZIP or mix it with ongoing experiments.
- The verified path starts at the cleaned CSV and archived exports. It does not require the original Excel, an Earth Engine account, or recreation of past Earth Engine tasks. Historical Table 5 has provenance caveats in `audit/PROVENANCE.md`; its entire training/evaluation sweep is not part of the first check.
