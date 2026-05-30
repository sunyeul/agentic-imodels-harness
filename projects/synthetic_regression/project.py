from __future__ import annotations

from pathlib import Path

from projects.synthetic_regression.datasets import (
    DEFAULT_DATA_DIR,
    load_competition_data,
)
from projects.synthetic_regression.spec import DefaultEvaluationSpec
from toy_imodels.core.project import Project
from toy_imodels.spec import EvaluationSpec

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results"


def synthetic_regression_project(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    spec: EvaluationSpec | None = None,
) -> Project:
    return Project(
        project_id="synthetic_regression",
        package_name="projects.synthetic_regression",
        spec=spec or DefaultEvaluationSpec(),
        data_dir=Path(data_dir),
        results_dir=Path(results_dir),
        load_data=load_competition_data,
    )
