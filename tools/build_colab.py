"""Build public Colab notebooks from the matching private team snapshot."""

import argparse
import csv
import hashlib
import io
import json
import re
import textwrap
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ARCHIVE_SHA = "1f1039b7a6d113685c2c95f840f2a475be5e936e02090bcf5283f2ba9b506773"
ARCHIVE_NAME = "Water_Quality_Processed_Handoff.zip"
WINDOWS_ARCHIVE_SHA = "324437bb6d0bb84c6bc1865a8ea1a172171974c6f039355deacc6a15ad7acc70"
WINDOWS_ARCHIVE_NAME = "Water_Quality_Processed_Handoff_Windows.zip"
AGENT_ARCHIVE_SHA = "bff939d07db4ca710e62b3af1ba55386d498768cfc82bda3afe58c7e59ce0adc"
AGENT_ARCHIVE_NAME = "Water_Quality_Processed_Handoff_Windows_v2.zip"
MAP_SUPPORT_NAME = "Water_Quality_Map_Support.zip"
DRIVE_RASTER_FOLDERS = {
    "polygon/features": "Water_Quality_RS_Moving_Window_Polygon",
    "polygon/rgb": "Water_Quality_RS_Sentinel2_RGB_Polygon",
    "two_dates/features": "Water_Quality_RS_Figure2_Features",
    "two_dates/rgb": "Water_Quality_RS_Figure2_RGB",
}
COLAB_METADATA = {
    "colab": {"provenance": []},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
}


def cell(kind, source):
    value = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def source_cell(kind, source):
    return cell(kind, textwrap.dedent(source).strip() + "\n")


def write(name, cells):
    notebook = {"cells": cells, "metadata": COLAB_METADATA, "nbformat": 4, "nbformat_minor": 0}
    destination = REPO / "notebooks" / name
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def archive_cells(map_support_sha=None):
    full_archive_options = textwrap.dedent(f"""
    MY_DRIVE = Path("/content/drive/MyDrive")
    FULL_ARCHIVES = {{
        "{AGENT_ARCHIVE_NAME}": "{AGENT_ARCHIVE_SHA}",
        "{WINDOWS_ARCHIVE_NAME}": "{WINDOWS_ARCHIVE_SHA}",
        "{ARCHIVE_NAME}": "{ARCHIVE_SHA}",
    }}
    full_archive = next(
        (MY_DRIVE / name for name in FULL_ARCHIVES if (MY_DRIVE / name).is_file()),
        MY_DRIVE / "{AGENT_ARCHIVE_NAME}",
    )
    """).strip()
    if map_support_sha:
        instructions = """
        Choose the full team ZIP (default), or set `USE_EXISTING_DRIVE_RASTERS = True` to use the small map-support ZIP plus existing Drive folders. Set `ARCHIVE_PATH` if the ZIP is in a subfolder. Both routes use verified inputs and neither signs into Earth Engine. Save generated results to Drive before disconnecting.
        """
        options = textwrap.dedent(f"""
        USE_EXISTING_DRIVE_RASTERS = False
        ARCHIVE_PATH = (
            MY_DRIVE / "{MAP_SUPPORT_NAME}" if USE_EXISTING_DRIVE_RASTERS else full_archive
        )
        EXPECTED_SHA256 = (
            "{map_support_sha}" if USE_EXISTING_DRIVE_RASTERS
            else FULL_ARCHIVES.get(ARCHIVE_PATH.name, "{ARCHIVE_SHA}")
        )
        """).strip()
    else:
        instructions = """
        Open the authorized Drive folder and add the shared ZIP to **My Drive** as a shortcut, or upload your copy there. Set `ARCHIVE_PATH` if the file is in a subfolder. This cell verifies and unpacks the exact team snapshot to temporary Colab storage. No Earth Engine login is used. Save generated results to Drive before disconnecting.
        """
        options = (
            'ARCHIVE_PATH = full_archive\n'
            f'EXPECTED_SHA256 = FULL_ARCHIVES.get(ARCHIVE_PATH.name, "{ARCHIVE_SHA}")'
        )
    return [
        cell("markdown", "## Load the team data\n\n" + textwrap.dedent(instructions).strip() + "\n"),
        cell("code", textwrap.dedent("""
        import hashlib
        import os
        import tempfile
        import zipfile
        from pathlib import Path
        from google.colab import drive

        drive.mount("/content/drive")
        """).lstrip() + "\n" + full_archive_options + "\n\n" + options + "\n" + textwrap.dedent("""

        HANDOFF_ROOT = Path("/content/wqr-handoff-" + EXPECTED_SHA256[:12])
        marker = HANDOFF_ROOT / ".verified-archive-sha256"

        if not (marker.is_file() and marker.read_text().strip() == EXPECTED_SHA256):
            if not ARCHIVE_PATH.is_file():
                raise FileNotFoundError(
                    f"{ARCHIVE_PATH} was not found. Ask the project owner for the selected ZIP "
                    "and point ARCHIVE_PATH to its location on the mounted Drive."
                )
            with tempfile.TemporaryDirectory(dir=HANDOFF_ROOT.parent) as tmp:
                tmp = Path(tmp)
                local_zip = tmp / ARCHIVE_PATH.name
                digest = hashlib.sha256()
                with ARCHIVE_PATH.open("rb") as src, local_zip.open("wb") as dst:
                    for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                        dst.write(chunk)
                if digest.hexdigest() != EXPECTED_SHA256:
                    raise ValueError("This ZIP differs from the verified snapshot. Check the filename and download.")
                extracted = tmp / "snapshot"
                with zipfile.ZipFile(local_zip) as archive:
                    archive.extractall(extracted)
                (extracted / marker.name).write_text(EXPECTED_SHA256 + "\\n")
                os.replace(extracted, HANDOFF_ROOT)

        print("Team data ready:", HANDOFF_ROOT)
        """)),
    ]


def save_results():
    return source_cell("code", """
    # Optional: choose your own private destination and uncomment to persist results.
    # import shutil
    # destination = Path("/content/drive/MyDrive/wqr_my_results")
    # shutil.copytree(OUTPUT_ROOT, destination, dirs_exist_ok=True)
    print("Results for this run:", OUTPUT_ROOT)
    """)


def build_start():
    write("00_Start_Here.ipynb", [
        source_cell("markdown", """
        # Start Here | Water Quality Research

        สำหรับผู้รับช่วง: รันจากข้อมูลภาคสนามที่คลีนแล้วและไฟล์ดาวเทียมที่ทีม export ไว้ ไม่ต้องมีสิทธิ์ Earth Engine

        This first check assembles the model dataset, validates predictions from the six locked model configurations, and tests the archived spatial inputs and selected pixel comparisons. It does not render every manuscript figure. Use a fresh CPU runtime.
        """),
        *archive_cells(),
        source_cell("markdown", """
        ## Install and verify

        Python 3.11 is the reference environment. The runner executes in a separate environment so package changes do not affect the notebook kernel. For a different runtime, the same pinned dependencies must be available; otherwise use the Python 3.11 local route in the team runbook.
        """),
        source_cell("code", """
        import subprocess
        import sys

        OUTPUT_ROOT = HANDOFF_ROOT / "run_output" / "first_check"
        RUN_GLOBAL_SHAP = False

        print("Runtime Python:", sys.version.split()[0])
        env_dir = HANDOFF_ROOT / ".verification-env"
        python = env_dir / "bin" / "python"
        if not python.exists():
            subprocess.run([sys.executable, "-m", "venv", str(env_dir)], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "-q",
                        "-r", str(HANDOFF_ROOT / "requirements.txt")], check=True)
        command = [str(python), "-B", str(HANDOFF_ROOT / "tools/reproduce.py"),
                   "--output", str(OUTPUT_ROOT)]
        if RUN_GLOBAL_SHAP:
            command.append("--global-shap")
        subprocess.run(command, cwd=HANDOFF_ROOT, check=True)
        """),
        source_cell("code", """
        from IPython.display import JSON, display

        receipt = OUTPUT_ROOT / "verification.json"
        display(JSON(data=receipt.read_text(encoding="utf-8")))
        """),
        save_results(),
    ])


FIGURES = (
    ("05_mapping/04_predict_multi_waterbody_water_quality_maps_paper.ipynb",
     "04_Fig03_Representative_Maps.ipynb", 2, 3, 4),
    ("05_mapping/05_figure1_s2_vs_context_comparison_colab.ipynb",
     "05_Fig04_S2_vs_Context.ipynb", 1, 2, 3),
    ("06_two_dates/06b_figure2_fast_counterfactual_colab.ipynb",
     "06b_Fig05_Two_Date_Comparison.ipynb", 1, 3, 5),
)


def patch_config(source, stage):
    if stage == 0:
        source = (
            'METADATA_ROOT = HANDOFF_ROOT / "data_ready/maps/polygon_metadata"\n'
            'FEATURE_INPUT_ROOT = HANDOFF_ROOT / "data_ready/rasters/polygon/features"\n'
            'RGB_INPUT_ROOT = HANDOFF_ROOT / "data_ready/rasters/polygon/rgb"\n'
            'OUTPUT_ROOT_OVERRIDE = HANDOFF_ROOT / "run_output/colab_fig03"\n\n'
            + source[source.index("MODEL_ROOT = "):]
        )
        source = source.replace('Path("/content/rswq_mapping_v1")', 'OUTPUT_ROOT_OVERRIDE / "locked_models"')
        source = source.replace('Path("/content/rswq_prediction_temporary_tiles")', 'OUTPUT_ROOT / "temporary_tiles"')
        source = re.sub(r"OUTPUT_ROOT = \(.*?\n\)", "OUTPUT_ROOT = OUTPUT_ROOT_OVERRIDE", source, count=1, flags=re.S)
        source = source.replace("MAP_IDS_TO_RUN = None", 'MAP_IDS_TO_RUN = ["WB02", "WB07", "WB05"]')
    elif stage == 1:
        source = source[:source.index("if IN_COLAB:")] + source[source.index("OUTPUT_ROOT = Path"):].replace(
            "OUTPUT_ROOT = Path(OUTPUT_ROOT_OVERRIDE) if OUTPUT_ROOT_OVERRIDE else default_output_root",
            "OUTPUT_ROOT = OUTPUT_ROOT_OVERRIDE",
            1,
        )
        source = source.replace("FEATURE_INPUT_ROOT_OVERRIDE = None",
                                'FEATURE_INPUT_ROOT_OVERRIDE = HANDOFF_ROOT / "data_ready/rasters/polygon/features"\n'
                                'feature_root_candidates = [FEATURE_INPUT_ROOT_OVERRIDE]')
        source = source.replace("OUTPUT_ROOT_OVERRIDE = None",
                                'OUTPUT_ROOT_OVERRIDE = HANDOFF_ROOT / "run_output/colab_fig04"')
        source = re.sub(r'    # Example:\n    # "WB05": .*\n', "", source)
    else:
        start = source.index("if IN_COLAB:")
        end = source.index("OUTPUT_ROOT = OUTPUT_DIR", start)
        source = source[:start] + (
            "PREPARATION_DIR = PREPARATION_DIR_OVERRIDE\n"
            "FEATURE_DIR = FEATURE_DIR_OVERRIDE\n"
            "RGB_DIR = RGB_DIR_OVERRIDE\n"
            "OUTPUT_DIR = OUTPUT_DIR_OVERRIDE\n"
        ) + source[end:]
        for key, relative in {
            "MODEL_DATA_PATH": "data_ready/model/rswq_model_input_relaxed_s2_context_2562_2568.csv",
            "PREPARATION_DIR": "data_ready/maps/two_date_preparation",
            "FEATURE_DIR": "data_ready/rasters/two_dates/features",
            "RGB_DIR": "data_ready/rasters/two_dates/rgb",
            "OUTPUT_DIR": "run_output/colab_fig05",
        }.items():
            source = source.replace(f"{key}_OVERRIDE = None",
                                    f'{key}_OVERRIDE = HANDOFF_ROOT / "{relative}"')
    return source.replace('PROJECT_DIR / "Maps" / "polygons" / "waterbody_polygons.geojson"',
                          'HANDOFF_ROOT / "data_ready/maps/polygons/waterbody_polygons.geojson"')


def drive_raster_cell(stage):
    groups = ("two_dates/features", "two_dates/rgb") if stage == 2 else (
        ("polygon/features",) if stage == 1 else ("polygon/features", "polygon/rgb")
    )
    folders = {group: DRIVE_RASTER_FOLDERS[group] for group in groups}
    sites = ["WB02_20240719", "WB05_20250704", "WB07_20250120"]
    prefixes = ["WB02_20240604", "WB02_20240619"] if stage == 2 else sites
    return source_cell("code", f"""
    # Only needed for the optional direct-Drive route. Edit paths for Drive shortcuts.
    import csv

    raster_folders = {folders!r}
    expected_prefixes = {prefixes!r}
    DRIVE_RASTER_PATHS = {{group: MY_DRIVE / name for group, name in raster_folders.items()}}

    if USE_EXISTING_DRIVE_RASTERS:
        with (HANDOFF_ROOT / "RASTER_MANIFEST.csv").open(newline="", encoding="utf-8") as handle:
            raster_manifest = list(csv.DictReader(handle))
        for group, drive_folder in DRIVE_RASTER_PATHS.items():
            for prefix in expected_prefixes:
                suffix = "_sentinel2_rgb.tif" if group.endswith("/rgb") else "_moving_window_features.tif"
                relative = f"data_ready/rasters/{{group}}/{{prefix}}{{suffix}}"
                matches = [row for row in raster_manifest if row["path"] == relative]
                if len(matches) != 1:
                    raise ValueError(f"No unique reference file for {{relative}}")
                row = matches[0]
                destination = HANDOFF_ROOT / relative
                if destination.is_file() and destination.stat().st_size == int(row["bytes"]):
                    continue  # Only verified files are moved to this runtime-local folder.
                original_raster = drive_folder / destination.name
                if not original_raster.is_file():
                    raise FileNotFoundError(f"{{original_raster}} not found. Check DRIVE_RASTER_PATHS.")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".part")
                try:
                    digest = hashlib.sha256()
                    with original_raster.open("rb") as src, temporary.open("wb") as dst:
                        for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
                            digest.update(chunk)
                            dst.write(chunk)
                    if digest.hexdigest() != row["sha256"]:
                        raise ValueError(f"{{original_raster.name}} differs from the paper snapshot.")
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
        print("Verified selected Drive rasters:", ", ".join(expected_prefixes))
    """)


def build_figures(root, map_support_sha):
    for stage, (original, name, setup, config, private_cell) in enumerate(FIGURES):
        notebook = json.loads((root / "src" / original).read_text(encoding="utf-8-sig"))
        originals = notebook["cells"]
        private_source = "".join(originals[private_cell]["source"])
        assert "base64.b64decode" in private_source
        private_sha = hashlib.sha256(private_source.encode()).hexdigest()
        cells = [
            source_cell("markdown", f"""
            # Manuscript Figure {stage + 3}

            This is a public Colab adaptation of `src/{original}` from the team snapshot. It uses the original figure and inference cells with paths set to the archived data. Fitted models and cell outputs are not published. Start with a fresh CPU runtime and run from the top.
            """),
            *archive_cells(map_support_sha),
            drive_raster_cell(stage),
        ]
        for index, original_cell in enumerate(originals):
            kind = original_cell["cell_type"]
            source = "".join(original_cell["source"])
            if index == setup:
                source = re.sub(r'from google\.colab import drive\n(?:    )?drive\.mount\("/content/drive"\)\n', "", source)
                source = re.sub(r"^if IN_COLAB:\s*\n(?=\n?required_versions)", "", source, flags=re.M)
            elif index == config:
                source = patch_config(source, stage)
            elif index == private_cell:
                source = textwrap.dedent(f"""
                # Load the exact historical setup cell from the verified team snapshot.
                import hashlib
                import json

                original = HANDOFF_ROOT / "src/{original}"
                private_notebook = json.loads(original.read_text(encoding="utf-8-sig"))
                private_setup = "".join(private_notebook["cells"][{private_cell}]["source"])
                if hashlib.sha256(private_setup.encode()).hexdigest() != "{private_sha}":
                    raise ValueError("The locked-model setup cell does not match the team snapshot.")
                exec(compile(private_setup, str(original), "exec"))
                del private_setup, private_notebook
                """).lstrip()
            elif stage == 2 and "def resolve_model_data():" in source:
                start = source.index("def resolve_model_data():")
                end = source.index("MODEL_DATA_PATH = resolve_model_data()", start)
                source = source[:start] + (
                    "def resolve_model_data():\n"
                    "    path = Path(MODEL_DATA_PATH_OVERRIDE)\n"
                    "    if not path.is_file():\n"
                    "        raise FileNotFoundError(path)\n"
                    "    return path\n\n\n"
                ) + source[end:]
            source = source.replace('PROJECT_DIR / "Maps" / "reference"', 'HANDOFF_ROOT / "data_ready/maps/reference"')
            if kind == "code":
                compile(source, f"{name}:cell {index}", "exec")
            cells.append(cell(kind, source))
        cells.append(save_results())
        write(name, cells)


FEATURE_SOURCES = (
    ("02_station_features/01_CAL_make_gg_ee_rswq_clean_2562_2568_b100_v4.ipynb",
     "01_Sentinel2_Station_Features.ipynb"),
    ("02_station_features/03_make_rswq_context_features.ipynb",
     "02_Environmental_Context_Features.ipynb"),
    ("05_mapping/03_multi_waterbody_moving_window_features.ipynb",
     "03_Moving_Window_Polygon_Features.ipynb"),
    ("06_two_dates/03_prepare_figure2_contrasting_dates_colab.ipynb",
     "03b_Two_Date_Feature_Preparation.ipynb"),
)


def build_feature_sources(root):
    for stage, (original, name) in enumerate(FEATURE_SOURCES):
        notebook = json.loads((root / "src" / original).read_text(encoding="utf-8-sig"))
        cells = [source_cell("markdown", f"""
        # Feature generation | Google Earth Engine

        Public adaptation of `src/{original}` in the team archive. Use this to inspect how features were produced or to regenerate them with your own Earth Engine project and authorized inputs. To use the existing exports, go directly to the prediction notebooks instead. Exports are disabled by default. Details: [Feature sources and Drive inputs](https://github.com/Boonyaratt/sentinel2-water-quality/blob/main/docs/drive-features-th.md).
        """)]
        for index, original_cell in enumerate(notebook["cells"]):
            if stage == 0 and (13 <= index <= 21 or 34 <= index <= 72 or 87 <= index <= 118 or index >= 135):
                # Keep feature extraction and the final candidate tables, not site-specific QA and OWT explorations.
                continue
            if stage == 1 and index == 0:
                continue  # Link to an older repo.
            kind = original_cell["cell_type"]
            source = "".join(original_cell["source"])
            if (stage == 0 and index == 3) or (stage == 1 and index == 2):
                source = "from IPython.display import display\n" + source
            if "EE_PROJECT = " in source:
                source = re.sub(r'^EE_PROJECT = "[^"]+"',
                                'EE_PROJECT = "YOUR_EARTH_ENGINE_PROJECT_ID"\n'
                                'if EE_PROJECT == "YOUR_EARTH_ENGINE_PROJECT_ID":\n'
                                '    raise ValueError("Set EE_PROJECT to a project you can use in Earth Engine.")',
                                source, flags=re.M)
            source = re.sub(r"^(START_(?:EE_EXPORT|CONTEXT_EXPORT|EXPORT_TASKS)|COPY_OUTPUTS_TO_FINAL_DRIVE) = True",
                            r"\1 = False", source, flags=re.M)
            if stage >= 2:
                source = source.replace('PROJECT_DIR = MYDRIVE / "Water Quality RS"',
                                        'PROJECT_DIR = MYDRIVE / "wqr_feature_rebuild"')
                source = re.sub(r'^(DRIVE_EXPORT_FOLDER|RGB_DRIVE_EXPORT_FOLDER) = "([^"]+)"',
                                r'\1 = "\2_rebuild"', source, flags=re.M)
            elif stage == 0:
                source = source.replace('f"rswq_exports_s2_chunks_{RUN_TAG}"',
                                        'f"rswq_exports_s2_chunks_{RUN_TAG}_rebuild"')
                source = source.replace("/MyDrive/Water Quality RS/Dataset/rswq_exports_s2_chunks",
                                        "/MyDrive/wqr_feature_rebuild/station_sentinel2")
            else:
                source = source.replace('f"rswq_context_features_{CONTEXT_RUN_TAG}"',
                                        'f"rswq_context_features_{CONTEXT_RUN_TAG}_rebuild"')
            cells.append(cell(kind, source))
        write(name, cells)


def write_map_support(root, destination):
    """Package map inputs not already provided by the original Drive TIFF folders."""
    with (root / "MANIFEST.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["path"].startswith(tuple(f"data_ready/rasters/{group}/" for group in DRIVE_RASTER_FOLDERS))]
    manifest = io.StringIO()
    writer = csv.DictWriter(manifest, fieldnames=["path", "bytes", "sha256"], lineterminator="\n",
                            extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    data = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for group in ("data_ready/maps", "data_ready/model")
        for path in (root / group).rglob("*")
        if path.is_file()
    }
    for original, _, _, _, _ in FIGURES:
        key = f"src/{original}"
        notebook = json.loads((root / key).read_text(encoding="utf-8-sig"))
        for item in notebook["cells"]:
            if item["cell_type"] == "code":
                item["outputs"] = []
                item["execution_count"] = None
        data[key] = json.dumps(notebook, ensure_ascii=False, separators=(",", ":")).encode()
    data["RASTER_MANIFEST.csv"] = manifest.getvalue().encode()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in sorted(data.items()):
            entry = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            archive.writestr(entry, payload, compress_type=zipfile.ZIP_DEFLATED)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team_snapshot", type=Path, help="Root of the extracted, verified team ZIP")
    parser.add_argument("--map-support", type=Path, required=True, help="Local path for the private map-support ZIP")
    args = parser.parse_args()
    (REPO / "notebooks").mkdir(exist_ok=True)
    support_sha = write_map_support(args.team_snapshot, args.map_support)
    build_start()
    build_figures(args.team_snapshot, support_sha)
    build_feature_sources(args.team_snapshot)
