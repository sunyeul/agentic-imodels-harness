from __future__ import annotations

import math

from projects.synthetic_regression import synthetic_regression_project
from toy_imodels.core.evaluation import run_experiment


def main() -> None:
    result = run_experiment(project=synthetic_regression_project())
    interp = (
        "pending_agent_judgment"
        if math.isnan(result.interpretability_score)
        else f"{result.interpretability_score:.4f}"
    )
    print(
        f"{result.status}: {result.model_name} "
        "cv_rmse="
        f"{result.result_metrics['cv_rmse_mean']:.6f}+/-"
        f"{result.result_metrics['cv_rmse_std']:.6f} "
        f"interp={interp}"
    )


if __name__ == "__main__":
    main()
