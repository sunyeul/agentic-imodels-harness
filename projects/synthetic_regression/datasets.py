from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from toy_imodels.core.project import CompetitionData

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
FEATURE_COLUMNS = [f"x{i}" for i in range(7)]
TARGET_COLUMN = "target"


def _target_function(x: np.ndarray) -> np.ndarray:
    linear = 2.8 * x[:, 0] - 1.7 * x[:, 1]
    nonlinear_hinge = 1.9 * np.maximum(0.0, x[:, 2] - 0.25)
    interaction_x3_x4 = 1.4 * x[:, 3] * x[:, 4]
    sinusoidal_x5 = 0.8 * np.sin(np.pi * x[:, 5])
    weak_quadratic_x6 = -0.5 * x[:, 6] ** 2
    return (
        linear + nonlinear_hinge + interaction_x3_x4 + sinusoidal_x5 + weak_quadratic_x6
    )


def make_synthetic_frame(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=0.0, scale=1.0, size=(n_samples, len(FEATURE_COLUMNS)))
    noise = rng.normal(loc=0.0, scale=0.15, size=n_samples)
    y = _target_function(x) + noise

    frame = pd.DataFrame(x, columns=FEATURE_COLUMNS)
    frame.insert(0, "id", np.arange(n_samples, dtype=int))
    frame[TARGET_COLUMN] = y
    return frame


def generate_synthetic_competition(
    output_dir: str | Path = DEFAULT_DATA_DIR,
    *,
    seed: int = 42,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    frame = make_synthetic_frame(seed=seed)
    train = frame.iloc[:600].reset_index(drop=True)
    valid = frame.iloc[600:800].reset_index(drop=True)
    test_with_target = frame.iloc[800:1000].reset_index(drop=True)
    test = test_with_target.drop(columns=[TARGET_COLUMN])
    sample_submission = pd.DataFrame(
        {"id": test["id"].to_numpy(), "prediction": np.zeros(len(test))}
    )

    train.to_csv(output_path / "train.csv", index=False)
    valid.to_csv(output_path / "valid.csv", index=False)
    test.to_csv(output_path / "test.csv", index=False)
    sample_submission.to_csv(output_path / "sample_submission.csv", index=False)

    metadata = {
        "name": "synthetic_agentic_imodels_regression",
        "seed": seed,
        "n_samples": 1000,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "splits": {"train": 600, "valid": 200, "test": 200},
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


def load_competition_data(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> CompetitionData:
    data_path = Path(data_dir)
    if not (data_path / "train.csv").exists():
        generate_synthetic_competition(data_path)

    train = pd.read_csv(data_path / "train.csv")
    valid = pd.read_csv(data_path / "valid.csv")
    test = pd.read_csv(data_path / "test.csv")

    return CompetitionData(
        x_train=cast(pd.DataFrame, train[FEATURE_COLUMNS]),
        y_train=cast(pd.Series, train[TARGET_COLUMN]),
        x_valid=cast(pd.DataFrame, valid[FEATURE_COLUMNS]),
        y_valid=cast(pd.Series, valid[TARGET_COLUMN]),
        x_test=cast(pd.DataFrame, test[["id", *FEATURE_COLUMNS]]),
        feature_columns=list(FEATURE_COLUMNS),
        target_column=TARGET_COLUMN,
    )


if __name__ == "__main__":
    generate_synthetic_competition()
    print(f"Wrote synthetic competition files to {DEFAULT_DATA_DIR}")
