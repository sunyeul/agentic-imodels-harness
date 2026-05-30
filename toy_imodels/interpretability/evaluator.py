from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

RUBRIC_VERSION = "agentic-imodels-agent-facing-v1"

SCORING_DIMENSIONS = [
    {
        "name": "prediction",
        "question": "Can an agent infer how the model computes predictions?",
        "rubric": (
            "Score high when the model string exposes an equation, rules, or other "
            "usable prediction procedure."
        ),
    },
    {
        "name": "feature_effects",
        "question": "Can an agent identify feature directions or relative importance?",
        "rubric": (
            "Score high when feature effects, coefficients, importances, or signs "
            "are clear from the model string."
        ),
    },
    {
        "name": "sensitivity",
        "question": "Can an agent reason about how changing a feature changes output?",
        "rubric": (
            "Score high when the model string supports local sensitivity or "
            "marginal-effect reasoning."
        ),
    },
    {
        "name": "counterfactual",
        "question": "Can an agent answer counterfactual input-change questions?",
        "rubric": (
            "Score high when the model string makes it possible to reason about "
            "increase/decrease or threshold counterfactuals."
        ),
    },
    {
        "name": "structure",
        "question": "Can an agent describe the model structure?",
        "rubric": (
            "Score high when the model family, transformations, interactions, "
            "rules, or simplicity constraints are explicit."
        ),
    },
]

DIMENSION_NAMES = tuple(dimension["name"] for dimension in SCORING_DIMENSIONS)

CALIBRATION_GUIDE = {
    "prediction": {
        "high": "Explicit equation, rule list, tree path, or calculation procedure.",
        "medium": "Model family is named but exact output calculation is partial.",
        "low": "Only a vague predictor name or black-box description is available.",
    },
    "feature_effects": {
        "high": (
            "Feature signs, coefficients, importances, or ranked effects are shown."
        ),
        "medium": (
            "Some relevant features are named without clear direction or magnitude."
        ),
        "low": "No usable feature-level attribution is provided.",
    },
    "sensitivity": {
        "high": "A feature change can be tied to output direction or magnitude.",
        "medium": "Direction can be guessed but magnitude/local effect is unclear.",
        "low": "The representation does not support sensitivity reasoning.",
    },
    "counterfactual": {
        "high": "Increase/decrease or threshold changes can be answered from the text.",
        "medium": "Some counterfactual direction is possible but incomplete.",
        "low": "Input-change questions cannot be answered from the representation.",
    },
    "structure": {
        "high": "Model family and internal structure are explicit.",
        "medium": "Model family is named but structure is mostly hidden.",
        "low": "No model structure is exposed.",
    },
}

AUDIT_EVIDENCE_TERMS = {
    "prediction": ("=", "equation", "predict", "prediction", "rule", "y ="),
    "feature_effects": (
        "coefficient",
        "coefficients",
        "importance",
        "top",
        "feature",
        "sign",
    ),
    "sensitivity": ("unit", "marginal", "coefficient", "change", "sensitivity"),
    "counterfactual": (
        "increase",
        "decrease",
        "threshold",
        "counterfactual",
        "coefficient",
    ),
    "structure": ("linear", "ridge", "tree", "rule", "interaction", "structure"),
}


def build_interpretability_packet(
    *, run_id: str, model_string: str, project_id: str | None = None
) -> dict[str, Any]:
    """Build the fixed packet a judge uses to score agent-facing interpretability."""

    packet: dict[str, Any] = {
        "run_id": run_id,
        "rubric_version": RUBRIC_VERSION,
        "model_string": model_string,
        "calibration_guide": CALIBRATION_GUIDE,
        "questions": [
            {
                "dimension": dimension["name"],
                "question": dimension["question"],
                "rubric": dimension["rubric"],
            }
            for dimension in SCORING_DIMENSIONS
        ],
        "scoring_dimensions": list(DIMENSION_NAMES),
        "output_schema": {
            "run_id": "string matching this packet",
            "rubric_version": RUBRIC_VERSION,
            "dimension_scores": {
                dimension_name: {
                    "score": "float in [0.0, 1.0]",
                    "rationale": "short evidence-based explanation",
                }
                for dimension_name in DIMENSION_NAMES
            },
            "interpretability_score": (
                "mean of dimension score values, rounded to 4 decimals"
            ),
        },
    }
    if project_id is not None:
        packet["project_id"] = project_id
        packet["output_schema"]["project_id"] = "string matching this packet"
    return packet


def write_interpretability_packet(
    run_dir: str | Path,
    *,
    run_id: str,
    model_string: str,
    project_id: str | None = None,
) -> Path:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    packet_path = run_path / "interpretability_packet.json"
    packet_path.write_text(
        json.dumps(
            build_interpretability_packet(
                run_id=run_id, model_string=model_string, project_id=project_id
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return packet_path


def validate_interpretability_judgment(
    judgment: dict[str, Any], *, expected_run_id: str
) -> dict[str, Any]:
    """Validate and normalize a judge artifact before harness-owned persistence."""

    if judgment.get("run_id") != expected_run_id:
        raise ValueError("Interpretability judgment run_id does not match target run")
    if judgment.get("rubric_version") != RUBRIC_VERSION:
        raise ValueError("Interpretability judgment rubric_version is not supported")

    dimension_scores = judgment.get("dimension_scores")
    if not isinstance(dimension_scores, dict):
        raise ValueError("Interpretability judgment must include dimension_scores")

    normalized_dimensions: dict[str, dict[str, float | str]] = {}
    for dimension_name in DIMENSION_NAMES:
        raw_dimension = dimension_scores.get(dimension_name)
        if not isinstance(raw_dimension, dict):
            raise ValueError(
                f"Interpretability judgment missing dimension {dimension_name}"
            )
        score = raw_dimension.get("score")
        if not isinstance(score, int | float):
            raise ValueError(
                f"Interpretability judgment dimension {dimension_name} score "
                "must be numeric"
            )
        score = float(score)
        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"Interpretability judgment dimension {dimension_name} score "
                "must be between 0 and 1"
            )
        rationale = raw_dimension.get("rationale")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError(
                f"Interpretability judgment dimension {dimension_name} "
                "must include rationale"
            )
        normalized_dimensions[dimension_name] = {
            "score": score,
            "rationale": rationale.strip(),
        }

    expected_score = round(
        sum(
            cast(float, dimension["score"])
            for dimension in normalized_dimensions.values()
        )
        / len(normalized_dimensions),
        4,
    )
    supplied_score = judgment.get("interpretability_score")
    if not isinstance(supplied_score, int | float):
        raise ValueError(
            "Interpretability judgment must include numeric interpretability_score"
        )
    if round(float(supplied_score), 4) != expected_score:
        raise ValueError(
            "Interpretability judgment interpretability_score must equal the "
            "mean dimension score rounded to 4 decimals"
        )

    return {
        "run_id": expected_run_id,
        "rubric_version": RUBRIC_VERSION,
        "dimension_scores": normalized_dimensions,
        "interpretability_score": expected_score,
    }


def audit_interpretability_judgment(
    *, packet: dict[str, Any], judgment: dict[str, Any]
) -> dict[str, Any]:
    """Create a non-blocking quality audit for retrospective judge review."""

    model_string = packet.get("model_string", "")
    if not isinstance(model_string, str):
        model_string = ""
    model_text = model_string.lower()
    judged_score = float(judgment["interpretability_score"])
    warnings: list[str] = []
    dimension_audits: dict[str, dict[str, object]] = {}

    if packet.get("run_id") != judgment.get("run_id"):
        warnings.append("packet run_id does not match judgment run_id")
    if packet.get("rubric_version") != judgment.get("rubric_version"):
        warnings.append("packet rubric_version does not match judgment rubric_version")

    dimension_scores = cast(
        dict[str, dict[str, float | str]], judgment["dimension_scores"]
    )
    for dimension_name in DIMENSION_NAMES:
        score = cast(float, dimension_scores[dimension_name]["score"])
        terms = AUDIT_EVIDENCE_TERMS[dimension_name]
        evidence_terms = [term for term in terms if term in model_text]
        dimension_warnings: list[str] = []
        if score >= 0.9 and not evidence_terms:
            dimension_warnings.append(
                "high score without obvious model-string evidence terms"
            )
        dimension_audits[dimension_name] = {
            "score": score,
            "evidence_terms": evidence_terms,
            "warnings": dimension_warnings,
        }
        warnings.extend(
            f"{dimension_name}: {warning}" for warning in dimension_warnings
        )

    return {
        "run_id": judgment["run_id"],
        "rubric_version": judgment["rubric_version"],
        "audit_status": "review" if warnings else "pass",
        "warnings": warnings,
        "judged_score": judged_score,
        "dimension_audits": dimension_audits,
    }
