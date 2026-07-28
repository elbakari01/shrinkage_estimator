"""Command-line interface for the simulation study."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shrinkage-sim",
        description="Run the adaptive pivot-based Poisson shrinkage simulation.",
    )
    parser.add_argument(
        "--scenario",
        default="D",
        choices=("A", "B", "C", "D", "all"),
        help="Scenario to run (default: D).",
    )
    parser.add_argument(
        "--kappa",
        nargs="+",
        type=float,
        default=(0.08,),
        metavar="VALUE",
        help="Signal proportions, for example --kappa 0.08 0.12.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run the short diagnostic configuration.",
    )
    parser.add_argument(
        "--replications",
        type=int,
        help="Override the number of Monte Carlo replications.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        help="Maximum parallel worker processes.",
    )
    parser.add_argument(
        "--axes",
        nargs="+",
        choices=("axis1", "axis2", "axis3", "axis3b", "coef_path", "axis4", "baseline"),
        help="Main axes to run. Omit to use the study defaults.",
    )
    parser.add_argument(
        "--post-run-axes",
        nargs="*",
        choices=("axis3", "axis3b"),
        help="Axes to run after the main outputs are saved.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory (default: ./results).",
    )
    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Disable Lasso, Elastic Net, and MNet comparisons.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from . import simulation

    config = simulation.ExperimentConfig(
        scenario_selector=args.scenario,
        fast_mode=args.fast,
        kappa_grid=tuple(args.kappa),
        output_dir=args.output_dir,
    )
    if args.replications is not None:
        if args.replications < 1:
            raise SystemExit("--replications must be at least 1")
        config.n_replications = args.replications
    if args.jobs is not None:
        if args.jobs < 1:
            raise SystemExit("--jobs must be at least 1")
        config.max_parallel_jobs = args.jobs
    if args.axes is not None:
        config.run_axes = tuple(args.axes)
    if args.post_run_axes is not None:
        config.post_run_axes = tuple(args.post_run_axes)
    if args.no_baselines:
        config.include_baselines = False

    simulation.CFG = config
    simulation.main()


if __name__ == "__main__":
    main()
