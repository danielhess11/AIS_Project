"""Evaluation metrics for recommender and urban-management outcomes."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or np.allclose(values.sum(), 0):
        return 0.0
    values = np.sort(np.clip(values, 0, None))
    index = np.arange(1, len(values) + 1)
    return float((2 * np.sum(index * values) / (len(values) * values.sum())) - (len(values) + 1) / len(values))


def normalized_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if len(counts) <= 1:
        return 0.0
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)) / math.log(len(counts)))


def calculate_metrics(
    pois: pd.DataFrame,
    profiles: pd.DataFrame,
    visits: pd.DataFrame,
    occupancy: pd.DataFrame,
    decisions: pd.DataFrame,
    recommendations: pd.DataFrame | None = None,
) -> dict[str, float]:
    visit_counts = visits["poi_id"].value_counts().reindex(pois["poi_id"], fill_value=0)
    district_counts = visits["district"].value_counts().reindex(sorted(pois["district"].unique()), fill_value=0)
    popular_ids = set(pois.nlargest(10, "popularity")["poi_id"])
    recommendation_counts = (
        recommendations["poi_id"].value_counts().reindex(pois["poi_id"], fill_value=0)
        if recommendations is not None and not recommendations.empty
        else visit_counts
    )
    metrics = {
        "peak_load_ratio": float(occupancy["load_ratio"].max()),
        "overcrowded_poi_time_pct": float((occupancy["load_ratio"] > 1).mean() * 100),
        "top_10_visit_share": float(visits["poi_id"].isin(popular_ids).mean()),
        "visit_gini": gini(visit_counts.to_numpy()),
        "district_entropy": normalized_entropy(district_counts.to_numpy()),
        "catalog_coverage": float((visit_counts > 0).mean()),
        "tourist_satisfaction": float(visits["satisfaction"].mean()),
        "recommendation_acceptance": float(visits["accepted_top_1"].mean()),
        "mean_distance_km": float(visits["distance_km"].mean()),
        "budget_compliance": float(visits["budget_compliant"].mean()),
        "precision_at_5": float(decisions["precision_at_5"].mean()),
        "recall_at_5": float(decisions["recall_at_5"].mean()),
        "diversity": float(decisions["diversity"].mean()),
        "novelty": float(decisions["novelty"].mean()),
        "exposure_fairness": 1 - gini(recommendation_counts.to_numpy()),
        "mean_sustainability": float(visits["sustainability"].mean()),
        "mean_local_economic": float(visits["local_economic"].mean()),
        "completed_visit_share": float(len(visits) / (len(profiles) * len(profiles.iloc[0]["visit_slots"]))),
    }
    return metrics

