"""Repeated paired experiments and statistical summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .data import load_pois
from .profiles import generate_profiles
from .recommenders import STRATEGIES
from .simulation import SimulationConfig, SimulationResult, run_strategy


def compare_strategies(
    config: SimulationConfig | None = None,
    pois: pd.DataFrame | None = None,
) -> dict[str, SimulationResult]:
    config = config or SimulationConfig()
    pois = load_pois() if pois is None else pois.copy()
    profiles = generate_profiles(
        config.population,
        sorted(pois["category"].unique().tolist()),
        config.seed,
        config.slots,
        config.visits_per_tourist,
    )
    return {
        strategy: run_strategy(strategy, config, pois, profiles)
        for strategy in STRATEGIES
    }


def paired_statistics(run_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    metric_columns = [
        column
        for column in run_metrics.columns
        if column not in {"replication", "seed", "strategy", "population"}
    ]
    baseline = run_metrics[run_metrics["strategy"] == "popularity"].sort_values("replication")
    for strategy in ("personalized", "sustainability"):
        comparison = run_metrics[run_metrics["strategy"] == strategy].sort_values("replication")
        for metric in metric_columns:
            differences = comparison[metric].to_numpy() - baseline[metric].to_numpy()
            count = len(differences)
            mean_difference = float(np.mean(differences))
            standard_error = float(stats.sem(differences)) if count > 1 else 0.0
            critical = float(stats.t.ppf(0.975, count - 1)) if count > 1 else 0.0
            standard_deviation = float(np.std(differences, ddof=1)) if count > 1 else 0.0
            test = stats.ttest_rel(comparison[metric], baseline[metric]) if count > 1 else None
            rows.append(
                {
                    "strategy": strategy,
                    "baseline": "popularity",
                    "metric": metric,
                    "mean_difference": mean_difference,
                    "ci_95_low": mean_difference - critical * standard_error,
                    "ci_95_high": mean_difference + critical * standard_error,
                    "cohens_dz": mean_difference / standard_deviation if standard_deviation > 0 else 0.0,
                    "p_value": float(test.pvalue) if test is not None else np.nan,
                }
            )
    return pd.DataFrame(rows)


def run_full_experiment(
    replications: int = 30,
    population: int = 5000,
    base_seed: int = 1000,
    output_dir: str | Path = "outputs",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pois = load_pois()
    rows: list[dict] = []
    for replication in range(replications):
        seed = base_seed + replication
        config = SimulationConfig(population=population, seed=seed, sample_recommendations=0)
        results = compare_strategies(config, pois)
        for strategy, result in results.items():
            rows.append(
                {
                    "replication": replication,
                    "seed": seed,
                    "population": population,
                    "strategy": strategy,
                    **result.metrics,
                }
            )
    run_metrics = pd.DataFrame(rows)
    statistics = paired_statistics(run_metrics)
    run_metrics.to_csv(output / "run_metrics.csv", index=False)
    statistics.to_csv(output / "paired_statistics.csv", index=False)
    return run_metrics, statistics

