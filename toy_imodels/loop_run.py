from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

from toy_imodels.loops import (
    init_loop_run,
    prepare_iteration,
    record_iteration,
    verify_loop_run,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Manage condition-isolated loop runs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--project-module", required=True)
    init_parser.add_argument("--condition", default="representation")
    init_parser.add_argument("--budget", required=True, type=int)
    init_parser.add_argument("--agent-model", required=True)
    init_parser.add_argument("--loop-run-id")

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("loop_run_id")
    prepare_parser.add_argument("--iteration", required=True, type=int)
    prepare_parser.add_argument("--results-dir", required=True)

    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("loop_run_id")
    record_parser.add_argument("--iteration", required=True, type=int)
    record_parser.add_argument("--project-module", required=True)
    record_parser.add_argument("--results-dir", required=True)
    record_parser.add_argument("--run-id")

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("loop_run_id")
    verify_parser.add_argument("--results-dir", required=True)

    args = parser.parse_args(argv)
    if args.command == "init":
        project = _load_project(args.project_module)
        manifest = init_loop_run(
            project,
            condition=args.condition,
            budget=args.budget,
            agent_model=args.agent_model,
            loop_run_id=args.loop_run_id,
        )
        print(manifest.loop_run_id)
        return 0
    if args.command == "prepare":
        path = prepare_iteration(
            args.results_dir,
            args.loop_run_id,
            iteration_index=args.iteration,
        )
        print(path)
        return 0
    if args.command == "record":
        project = _load_project(args.project_module, results_dir=Path(args.results_dir))
        result = record_iteration(
            project,
            args.loop_run_id,
            iteration_index=args.iteration,
            run_id=args.run_id,
        )
        print(result.run_id)
        return 0
    if args.command == "verify":
        findings = verify_loop_run(args.results_dir, args.loop_run_id)
        for finding in findings:
            print(finding)
        return 0 if all(finding.startswith("PASS") for finding in findings) else 1
    raise AssertionError(f"Unhandled command: {args.command}")


def _load_project(project_module: str, *, results_dir: Path | None = None) -> Any:
    module_name, separator, factory_name = project_module.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("--project-module must have the form module:function")
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name)
    if results_dir is None:
        return factory()
    return factory(results_dir=results_dir)


if __name__ == "__main__":
    raise SystemExit(main())
