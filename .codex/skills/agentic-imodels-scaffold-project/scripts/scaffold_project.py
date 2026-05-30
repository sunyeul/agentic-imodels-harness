#!/usr/bin/env python3
"""Scaffold a new AGENTIC-IMODELS project from an existing sample project."""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

PROJECT_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
DEFAULT_SOURCE = Path("projects/synthetic_regression")
TEXT_EXTENSIONS = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create projects/<name> from a sample project directory."
    )
    parser.add_argument(
        "--name",
        required=True,
        help="New project name. Use lowercase letters, numbers, and underscores.",
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help=(
            "Sample project directory to copy. "
            "Defaults to projects/synthetic_regression."
        ),
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def validate_project_name(name: str) -> None:
    if not PROJECT_NAME_RE.fullmatch(name):
        raise ValueError(
            "--name must start with a lowercase letter and contain only "
            "lowercase letters, numbers, and underscores"
        )


def ensure_relative_to(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path must stay inside the repository: {path}") from exc
    return resolved


def copy_project(source: Path, target: Path) -> None:
    if not source.exists() or not source.is_dir():
        raise FileNotFoundError(f"source project does not exist: {source}")
    if target.exists():
        raise FileExistsError(f"target project already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )


def replace_text_in_project(target: Path, source_name: str, new_name: str) -> int:
    changed = 0
    for path in target.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_EXTENSIONS:
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = original.replace(source_name, new_name)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed += 1
    return changed


def likely_files(target: Path) -> list[Path]:
    candidates = [
        target / "README.md",
        target / "spec.py",
        target / "project.py",
        target / "datasets.py",
        target / "run_experiment.py",
        target / "experiments" / "candidate_model.py",
    ]
    return [path for path in candidates if path.exists()]


def print_success(
    root: Path,
    source: Path,
    target: Path,
    changed_files: int,
) -> None:
    rel_source = source.relative_to(root)
    rel_target = target.relative_to(root)
    print(f"Created project: {rel_target}")
    print(f"Sample source: {rel_source}")
    print(f"Text files updated: {changed_files}")
    files = likely_files(target)
    if files:
        print("\nReview next:")
        for path in files:
            print(f"- {path.relative_to(root)}")
    print("\nSmoke/check:")
    print("python -m compileall -q projects")


def main() -> int:
    args = parse_args()
    root = repo_root()

    try:
        validate_project_name(args.name)
        source = ensure_relative_to(root / args.source, root)
        target = ensure_relative_to(root / "projects" / args.name, root)
        copy_project(source, target)
        changed_files = replace_text_in_project(target, source.name, args.name)
        print_success(root, source, target, changed_files)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
