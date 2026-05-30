from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd

from toy_imodels.core.project import EvaluationData

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
FEATURE_COLUMNS = [f"x{i}" for i in range(15)]
TARGET_COLUMN = "target"


def load_evaluation_data(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> EvaluationData:
    data_path = Path(data_dir)
    required_files = ("train.csv", "valid.csv", "test.csv", "metadata.json")
    missing_files = [name for name in required_files if not (data_path / name).exists()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing competition data files in {data_path}: {missing_files}"
        )

    metadata = json.loads((data_path / "metadata.json").read_text())
    if metadata.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("metadata feature_columns do not match benchmark schema")
    if metadata.get("target_column") != TARGET_COLUMN:
        raise ValueError("metadata target_column does not match benchmark schema")

    train = pd.read_csv(data_path / "train.csv")
    valid = pd.read_csv(data_path / "valid.csv")
    test = pd.read_csv(data_path / "test.csv")
    if TARGET_COLUMN in test.columns:
        raise ValueError("test.csv must not contain the target column")

    return EvaluationData(
        x_labeled=cast(
            pd.DataFrame, pd.concat([train[FEATURE_COLUMNS], valid[FEATURE_COLUMNS]])
        ).reset_index(drop=True),
        y_labeled=cast(
            pd.Series, pd.concat([train[TARGET_COLUMN], valid[TARGET_COLUMN]])
        ).reset_index(drop=True),
        x_test=cast(pd.DataFrame, test[["id", *FEATURE_COLUMNS]]),
        feature_columns=list(FEATURE_COLUMNS),
    )


if __name__ == "__main__":
    data = load_evaluation_data()
    print(
        "Loaded synthetic competition files from "
        f"{DEFAULT_DATA_DIR} with {len(data.feature_columns)} features"
    )
