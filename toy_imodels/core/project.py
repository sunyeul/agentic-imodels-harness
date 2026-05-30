from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from toy_imodels.spec import EvaluationSpec

PROJECT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, slots=True, kw_only=True)
class EvaluationData:
    """Project-provided supervised data consumed by the fixed harness."""

    x_labeled: pd.DataFrame
    y_labeled: pd.Series
    x_test: pd.DataFrame
    feature_columns: list[str]

    def __post_init__(self) -> None:
        feature_columns = list(self.feature_columns)
        if not feature_columns:
            raise ValueError("EvaluationData.feature_columns must not be empty")
        missing_by_frame = {
            "x_labeled": sorted(set(feature_columns) - set(self.x_labeled.columns)),
            "x_test": sorted(set(feature_columns) - set(self.x_test.columns)),
        }
        missing = {
            frame_name: columns
            for frame_name, columns in missing_by_frame.items()
            if columns
        }
        if missing:
            raise ValueError(f"EvaluationData missing feature columns: {missing}")
        if len(self.x_labeled) != len(self.y_labeled):
            raise ValueError("EvaluationData x_labeled and y_labeled lengths differ")
        object.__setattr__(self, "feature_columns", feature_columns)


@dataclass(frozen=True, slots=True, kw_only=True)
class Project:
    """Project-level mapping between data, predictive spec, and artifacts."""

    project_id: str
    package_name: str
    spec: EvaluationSpec
    data_dir: Path
    results_dir: Path
    load_data: Callable[[str | Path], EvaluationData]
    protected_paths: tuple[str, ...]
    forbidden_candidate_source_fragments: tuple[str, ...]
    write_submission: Callable[[Path, str, EvaluationData, Any], Path]
    write_diagnostics: (
        Callable[[Path, str, str, EvaluationData, pd.Series], Path | None] | None
    ) = None

    def __post_init__(self) -> None:
        if not PROJECT_ID_PATTERN.fullmatch(self.project_id):
            raise ValueError("Project.project_id must match ^[a-z][a-z0-9_]*$")
        if not self.package_name.strip():
            raise ValueError("Project.package_name must not be empty")
        package_parts = self.package_name.split(".")
        if package_parts[-1] != self.project_id or not all(
            part.isidentifier() for part in package_parts
        ):
            raise ValueError(
                "Project.package_name must be a dotted package ending in project_id"
            )
        if not callable(self.load_data):
            raise TypeError("Project.load_data must be callable")
        if not callable(self.write_submission):
            raise TypeError("Project.write_submission must be callable")
        if self.write_diagnostics is not None and not callable(self.write_diagnostics):
            raise TypeError("Project.write_diagnostics must be callable")
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "results_dir", Path(self.results_dir))
        object.__setattr__(self, "protected_paths", tuple(self.protected_paths))
        object.__setattr__(
            self,
            "forbidden_candidate_source_fragments",
            tuple(self.forbidden_candidate_source_fragments),
        )

    @property
    def default_candidate_module(self) -> str:
        return f"{self.package_name}.experiments.candidate_model"
