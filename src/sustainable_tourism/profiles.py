"""Synthetic tourist population generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_profiles(
    population: int,
    categories: list[str],
    seed: int,
    slots: int = 6,
    visits_per_tourist: int = 4,
) -> pd.DataFrame:
    if population <= 0:
        raise ValueError("Population must be positive")
    if not 1 <= visits_per_tourist <= slots:
        raise ValueError("visits_per_tourist must be between one and the number of slots")

    rng = np.random.default_rng(seed)
    primary = rng.choice(categories, population)
    secondary = np.empty(population, dtype=object)
    for index, interest in enumerate(primary):
        alternatives = [category for category in categories if category != interest]
        secondary[index] = rng.choice(alternatives)

    budgets = np.clip(rng.lognormal(mean=3.0, sigma=0.55, size=population), 5, 100)
    visit_slots = [
        tuple(sorted(rng.choice(slots, visits_per_tourist, replace=False).tolist()))
        for _ in range(population)
    ]
    return pd.DataFrame(
        {
            "tourist_id": np.arange(population, dtype=int),
            "primary_interest": primary,
            "secondary_interest": secondary,
            "daily_budget": budgets.round(2),
            "mobility_mode": rng.choice(
                ["walk", "transit", "bike"], population, p=[0.42, 0.43, 0.15]
            ),
            "walking_tolerance": rng.beta(2.2, 1.8, population).round(3),
            "crowd_aversion": rng.beta(2.0, 2.0, population).round(3),
            "sustainability_sensitivity": rng.beta(2.0, 1.8, population).round(3),
            "outdoor_preference": rng.beta(1.8, 1.8, population).round(3),
            "with_kids": rng.random(population) < 0.22,
            "visit_slots": visit_slots,
        }
    )

