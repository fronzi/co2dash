"""Shared test fixtures and canonical paths to the shipped example files.

Tests previously referenced example files by bare relative path
("examples/scenario_co_real.yaml"), which broke twice over: the paths only
resolved when pytest was invoked from the repository root, and they went stale
silently when the example files were renamed. Both failure modes are removed by
resolving paths from this file's location and asserting existence at import
time, so a rename fails loudly and immediately instead of ten tests down.
"""
import os

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")


def example(name: str) -> str:
    """Absolute path to a shipped example file; raises if it has been renamed."""
    path = os.path.join(EXAMPLES_DIR, name)
    if not os.path.exists(path):
        available = sorted(os.listdir(EXAMPLES_DIR)) if os.path.isdir(EXAMPLES_DIR) else []
        raise FileNotFoundError(
            f"example file '{name}' not found in {EXAMPLES_DIR}. "
            f"Available: {available}")
    return path


# Canonical names of the shipped examples. Update HERE if a file is renamed --
# every test resolves through these constants, so one edit fixes all of them.
SCENARIO_CO = "example_SCENARIO_CO2-to-CO.yaml"
# Deliberately differs from GENERIC_DEFAULTS on grid, scale and release fraction,
# so a test can tell "came from the YAML" apart from "came from the defaults".
SCENARIO_CO_FAVOURABLE = "example_SCENARIO_CO2-to-CO_favourable.yaml"
SCENARIO_METHANOL = "example_SCENARIO_methanol.yaml"
LITERATURE_CO_CSV = "example_YOUR-DATA-tab_AgCO_literature.csv"
MEASUREMENTS_CSV = "example_YOUR-DATA-tab_measurements.csv"


@pytest.fixture(scope="session")
def scenario_co_path() -> str:
    return example(SCENARIO_CO)


@pytest.fixture(scope="session")
def scenario_methanol_path() -> str:
    return example(SCENARIO_METHANOL)


@pytest.fixture(scope="session")
def literature_co_csv_path() -> str:
    return example(LITERATURE_CO_CSV)
