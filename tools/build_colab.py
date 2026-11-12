"""Build public Colab notebooks from the matching private team snapshot."""

import argparse
import hashlib
import json
import re
import textwrap
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ARCHIVE_SHA = "1f1039b7a6d113685c2c95f840f2a475be5e936e02090bcf5283f2ba9b506773"
ARCHIVE_NAME = "Water_Quality_Processed_Handoff.zip"
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


def archive_cells():
    return [
        source_cell("markdown", """
        ## Load the team data

        Open the authorized Drive folder and add the shared ZIP to **My Drive** as a shortcut, or upload your copy there. Set `ARCHIVE_PATH` if the file is in a subfolder. This cell verifies and unpacks the exact team snapshot to temporary Colab storage. No Earth Engine login is used. Save generated results to Drive before disconnecting.
        """),
        source_cell("code", f"""
        import hashlib
        import os
        import tempfile
        import zipfile
        from pathlib import Path
        from google.colab import drive

        drive.mount("/content/drive")

        ARCHIVE_PATH = Path("/content/drive/MyDrive/{ARCHIVE_NAME}")
        EXPECTED_SHA256 = "{ARCHIVE_SHA}"
        HANDOFF_ROOT = Path("/content/wqr-handoff-" + EXPECTED_SHA256[:12])
        marker = HANDOFF_ROOT / ".verified-archive-sha256"

        if not (marker.is_file() and marker.read_text().strip() == EXPECTED_SHA256):
            if not ARCHIVE_PATH.is_file():
                raise FileNotFoundError(
                    f"{{ARCHIVE_PATH}} was not found. Ask the project owner for the team ZIP "
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
        """),
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


def build_figures(root):
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
            *archive_cells(),
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("team_snapshot", type=Path, help="Root of the extracted, verified team ZIP")
    args = parser.parse_args()
    (REPO / "notebooks").mkdir(exist_ok=True)
    build_start()
    build_figures(args.team_snapshot)
