# Run in Google Colab

## Start here

Get `Water_Quality_Processed_Handoff_Windows_v2.zip` from the project owner. It opens in File Explorer and includes [AGENTS.md](../AGENTS.md). The earlier full ZIPs also work in Colab. Add the shared ZIP to **My Drive** as a shortcut, or upload your copy. Open the appropriate notebook below and run from the top. Keep the ZIP filename unchanged and set `ARCHIVE_PATH` in the first code cell if the ZIP is in a subfolder.

**Already have the original TIFFs on Drive?** For notebooks 04, 05 and 06b, use the small `Water_Quality_Map_Support.zip` and switch on `USE_EXISTING_DRIVE_RASTERS`. The [Drive guide (ภาษาไทย)](../docs/drive-features-th.md) shows which folders to use. Notebook 00 still requires the full archive.

| Notebook | What it does | Expected result |
|---|---|---|
| [00 · Start Here](https://colab.research.google.com/github/Boonyaratt/sentinel2-water-quality/blob/main/notebooks/00_Start_Here.ipynb) | Reassembles the modeling dataset and verifies six locked configurations, spatial inputs and sample comparisons | `verification.json`, 340 × 226 assembled table |
| [04 · Representative Maps](https://colab.research.google.com/github/Boonyaratt/sentinel2-water-quality/blob/main/notebooks/04_Fig03_Representative_Maps.ipynb) | Predicts on the reviewed polygon grids with the historical mapping artifacts | Manuscript Fig. 3, WB02/WB07/WB05 |
| [05 · S2 vs Context](https://colab.research.google.com/github/Boonyaratt/sentinel2-water-quality/blob/main/notebooks/05_Fig04_S2_vs_Context.ipynb) | Compares the two locked configurations on common supported pixels, in a separate cell for each site | Manuscript Fig. 4, three site panels |
| [06b · Two-Date Comparison](https://colab.research.google.com/github/Boonyaratt/sentinel2-water-quality/blob/main/notebooks/06b_Fig05_Two_Date_Comparison.ipynb) | Compares 4 and 19 June 2024 on the same Thale Luang grid, using the archived context and RGB data | Manuscript Fig. 5, context tables and two-block analysis |

Each notebook initializes its own runtime; running `00` is recommended as a check, not a prerequisite for the others. The original working names **Figure 1** and **Figure 2** in notebooks 05 and 06b correspond to **manuscript Figs. 4 and 5**.

Outputs are printed in the last cell and stored temporarily under `run_output/`. The final cell shows how to save to Drive. Do not share a notebook with saved outputs that expose team data. Notebook 05 uses a satellite tile basemap; set `USE_SATELLITE_BASEMAP = False` in its configuration cell for a run without those tiles.

## Feature Generation (Google Earth Engine)

These public copies contain the extraction code. They are separate from the prediction notebooks above. Use an authorized Earth Engine project and your own input paths when regenerating data. Export switches are off by default.

| Open in Colab | What it produces |
|---|---|
| [01 · Sentinel-2 Station Features](https://colab.research.google.com/github/Boonyaratt/sentinel2-water-quality/blob/main/notebooks/01_Sentinel2_Station_Features.ipynb) | Station matchup CSVs, using the 100 m and watermask v4 configuration |
| [02 · Environmental Context Features](https://colab.research.google.com/github/Boonyaratt/sentinel2-water-quality/blob/main/notebooks/02_Environmental_Context_Features.ipynb) | Station-level contextual CSVs from the strict v2 extraction |
| [03 · Moving-Window Polygon Features](https://colab.research.google.com/github/Boonyaratt/sentinel2-water-quality/blob/main/notebooks/03_Moving_Window_Polygon_Features.ipynb) | Polygon-based 62-band feature GeoTIFFs and Sentinel-2 RGB for mapping |
| [03b · Two-Date Feature Preparation](https://colab.research.google.com/github/Boonyaratt/sentinel2-water-quality/blob/main/notebooks/03b_Two_Date_Feature_Preparation.ipynb) | Candidate-date audit and paired feature/RGB GeoTIFFs on a common grid |

## Other notebooks

- [Merge field, satellite, and context tables](../src/03_assembly/02_Merge_rswq_dataset.ipynb) and [model-evaluation sources](../src/04_models/) are public source copies. Use the team archive versions and its `COLAB.md` for the original run configurations.
- The unchanged extraction, scene screening, polygon and moving-window notebooks are included in the team ZIP. Use the [data inventory](../docs/data-inventory.md) to find each one.
- For Global SHAP, set `RUN_GLOBAL_SHAP = True` in notebook 00. Regional SHAP uses the separate `regional_shap_v1` grid and the local runner in the [team runbook](../docs/handoff-th.md).
