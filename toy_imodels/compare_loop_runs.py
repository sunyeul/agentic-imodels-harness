from __future__ import annotations

import argparse
import json

from toy_imodels.loops import compare_loop_runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two condition loop runs.")
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--target", type=float)
    args = parser.parse_args(argv)

    comparison = compare_loop_runs(
        args.results_dir,
        args.left,
        args.right,
        target=args.target,
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
