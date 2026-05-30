from __future__ import annotations

from projects.synthetic_regression import synthetic_regression_project
from toy_imodels.core.evaluation import run_experiment


def main() -> None:
    result = run_experiment(project=synthetic_regression_project())
    print(
        f"{result.status}: {result.model_name} "
        f"cv_rmse={result.cv_rmse_mean:.6f}+/-{result.cv_rmse_std:.6f} "
        f"interp={result.interpretability_score:.4f}"
    )


if __name__ == "__main__":
    main()
