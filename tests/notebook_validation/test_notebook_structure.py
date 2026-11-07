"""Validates notebook format and enforces the mandatory Markdown-before-code-cell rule.

Scans all .ipynb files under `tracks/`. For each notebook:
1. Validates valid JSON structure via nbformat.
2. Asserts that EVERY code cell is immediately preceded by an explanatory Markdown cell.
"""

from pathlib import Path
import nbformat
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TRACKS_DIR = ROOT_DIR / "tracks"


def get_all_notebooks():
    return sorted(list(TRACKS_DIR.glob("**/*.ipynb")))


def test_markdown_before_every_code_cell():
    notebooks = get_all_notebooks()
    if not notebooks:
        pytest.skip("No notebooks present yet (Stage 0). Notebook checks will execute in Stage 1.")

    for notebook_path in notebooks:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        previous_cell_type = None
        for idx, cell in enumerate(nb.cells):
            current_cell_type = cell.cell_type
            if current_cell_type == "code":
                assert previous_cell_type == "markdown", (
                    f"Rule violation in {notebook_path.name} at cell index {idx}: "
                    f"Code cell is not preceded by a Markdown cell. Previous cell type: {previous_cell_type}"
                )
            previous_cell_type = current_cell_type
