"""Synthetic regression project for the toy AGENTIC-IMODELS harness."""

from projects.synthetic_regression.project import synthetic_regression_project
from projects.synthetic_regression.spec import DefaultEvaluationSpec

__all__ = [
    "DefaultEvaluationSpec",
    "synthetic_regression_project",
]
