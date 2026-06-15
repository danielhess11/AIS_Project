"""Command-line interface for simulations and artifact generation."""

from __future__ import annotations

import argparse
import json

from .artifacts import generate_all_artifacts
from .experiment import compare_strategies, run_full_experiment
from .simulation import SimulationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sustainable tourism recommender simulator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    quick = subparsers.add_parser("quick", help="Compare all strategies for one seed")
    quick.add_argument("--population", type=int, default=1000)
    quick.add_argument("--seed", type=int, default=42)

    full = subparsers.add_parser("full", help="Run paired replicated experiments")
    full.add_argument("--population", type=int, default=5000)
    full.add_argument("--replications", type=int, default=30)
    full.add_argument("--base-seed", type=int, default=1000)
    full.add_argument("--output-dir", default="outputs")

    artifacts = subparsers.add_parser("artifacts", help="Generate detailed CSVs, figures, MP4, and notebook")
    artifacts.add_argument("--population", type=int, default=5000)
    artifacts.add_argument("--seed", type=int, default=2026)
    artifacts.add_argument("--output-dir", default="outputs")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "quick":
        results = compare_strategies(SimulationConfig(population=args.population, seed=args.seed))
        print(json.dumps({strategy: result.metrics for strategy, result in results.items()}, indent=2))
    elif args.command == "full":
        run_full_experiment(args.replications, args.population, args.base_seed, args.output_dir)
        print(f"Full experiment written to {args.output_dir}")
    else:
        generated = generate_all_artifacts(args.population, args.seed, args.output_dir)
        print(f"Generated {len(generated['figures'])} figures, {generated['video']}, and {generated['notebook']}")


if __name__ == "__main__":
    main()
