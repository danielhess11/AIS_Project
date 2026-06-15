"""Recommendation scoring and diversity reranking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


STRATEGIES = ("popularity", "personalized", "sustainability")


@dataclass(frozen=True)
class Recommendation:
    indices: np.ndarray
    scores: np.ndarray
    components: dict[str, np.ndarray]


@dataclass(frozen=True)
class PreparedPOIs:
    category: np.ndarray
    district: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    price: np.ndarray
    popularity: np.ndarray
    sustainability: np.ndarray
    local_economic: np.ndarray
    capacity: np.ndarray
    outdoor: np.ndarray
    kid_friendly: np.ndarray
    accessibility: np.ndarray
    walking_effort: np.ndarray


def prepare_pois(pois: pd.DataFrame) -> PreparedPOIs:
    return PreparedPOIs(
        category=pois["category"].to_numpy(),
        district=pois["district"].to_numpy(),
        lat=pois["lat"].to_numpy(float),
        lon=pois["lon"].to_numpy(float),
        price=pois["price"].to_numpy(float),
        popularity=pois["popularity"].to_numpy(float),
        sustainability=pois["sustainability"].to_numpy(float),
        local_economic=pois["local_economic"].to_numpy(float),
        capacity=pois["capacity"].to_numpy(float),
        outdoor=pois["outdoor"].to_numpy(float),
        kid_friendly=pois["kid_friendly"].to_numpy(float),
        accessibility=pois["accessibility"].to_numpy(float),
        walking_effort=pois["walking_effort"].to_numpy(float),
    )


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    radius = 6371.0
    lat1_r = np.radians(lat1)
    lat2_r = np.radians(lat2)
    delta_lat = lat2_r - lat1_r
    delta_lon = np.radians(lon2 - lon1)
    a = np.sin(delta_lat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(delta_lon / 2) ** 2
    return 2 * radius * np.arcsin(np.sqrt(a))


def component_scores(
    profile: dict,
    pois: pd.DataFrame,
    occupancy: np.ndarray,
    current_location: tuple[float, float] | None,
    district_visits: dict[str, int],
    prepared: PreparedPOIs | None = None,
) -> dict[str, np.ndarray]:
    prepared = prepared or prepare_pois(pois)
    count = len(prepared.category)
    interest = np.full(count, 0.2)
    interest[prepared.category == profile["secondary_interest"]] = 0.6
    interest[prepared.category == profile["primary_interest"]] = 1.0

    budget = np.clip(1 - prepared.price / max(profile["daily_budget"], 1), 0, 1)
    mobility_bonus = {"walk": 0.0, "bike": 0.15, "transit": 0.25}[profile["mobility_mode"]]
    accessibility = np.clip(
        0.6 * prepared.accessibility
        + 0.4 * (profile["walking_tolerance"] + mobility_bonus - prepared.walking_effort + 0.5),
        0,
        1,
    )
    family = prepared.kid_friendly if profile["with_kids"] else np.ones(count)
    outdoor = 1 - np.abs(profile["outdoor_preference"] - prepared.outdoor)
    context = 0.55 * family + 0.45 * outdoor
    load = occupancy / prepared.capacity
    crowd_comfort = np.clip(1 - load / 1.5, 0, 1)

    if current_location is None:
        distance = np.zeros(count)
        distance_fit = np.ones(count)
    else:
        distance = haversine_km(
            current_location[0],
            current_location[1],
            prepared.lat,
            prepared.lon,
        )
        mode_range = {"walk": 4.0, "bike": 8.0, "transit": 12.0}[profile["mobility_mode"]]
        distance_fit = np.exp(-distance / mode_range)

    counts = np.fromiter(
        (district_visits.get(district, 0) for district in prepared.district),
        dtype=float,
        count=count,
    )
    dispersion = 1 / (1 + counts)
    if dispersion.max() > dispersion.min():
        dispersion = (dispersion - dispersion.min()) / (dispersion.max() - dispersion.min())
    else:
        dispersion = np.ones(count)

    personal_relevance = (
        0.55 * interest
        + 0.15 * budget
        + 0.10 * accessibility
        + 0.10 * context
        + 0.10 * crowd_comfort
    )
    return {
        "interest": interest,
        "budget_fit": budget,
        "accessibility": accessibility,
        "context": context,
        "crowd_comfort": crowd_comfort,
        "distance_km": distance,
        "distance_fit": distance_fit,
        "dispersion": dispersion,
        "personal_relevance": personal_relevance,
    }


def feasible_mask(
    profile: dict,
    pois: pd.DataFrame,
    visited_indices: set[int],
    prepared: PreparedPOIs | None = None,
) -> np.ndarray:
    prepared = prepared or prepare_pois(pois)
    mask = prepared.price <= profile["budget_remaining"]
    tolerance = profile["walking_tolerance"] + {"walk": 0.0, "bike": 0.2, "transit": 0.35}[profile["mobility_mode"]]
    mask &= prepared.walking_effort <= min(tolerance + 0.25, 1.0)
    if profile["with_kids"]:
        mask &= prepared.kid_friendly > 0
    if visited_indices:
        mask[list(visited_indices)] = False
    return mask


def _diversity_rerank(
    candidates: np.ndarray,
    base_scores: np.ndarray,
    prepared: PreparedPOIs,
    top_k: int,
) -> np.ndarray:
    selected: list[int] = []
    remaining = candidates.tolist()
    while remaining and len(selected) < top_k:
        adjusted = []
        for index in remaining:
            category_penalty = 0.10 * sum(
                prepared.category[index] == prepared.category[chosen] for chosen in selected
            )
            district_penalty = 0.06 * sum(
                prepared.district[index] == prepared.district[chosen] for chosen in selected
            )
            adjusted.append(base_scores[index] - category_penalty - district_penalty)
        choice = remaining[int(np.argmax(adjusted))]
        selected.append(choice)
        remaining.remove(choice)
    return np.asarray(selected, dtype=int)


def recommend(
    strategy: str,
    profile: dict,
    pois: pd.DataFrame,
    occupancy: np.ndarray,
    current_location: tuple[float, float] | None,
    visited_indices: set[int],
    district_visits: dict[str, int],
    top_k: int = 5,
    sustainability_weights: tuple[float, float, float, float, float] = (0.40, 0.20, 0.15, 0.15, 0.10),
    prepared: PreparedPOIs | None = None,
) -> Recommendation:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    prepared = prepared or prepare_pois(pois)
    components = component_scores(profile, pois, occupancy, current_location, district_visits, prepared)
    mask = feasible_mask(profile, pois, visited_indices, prepared)
    if not mask.any():
        return Recommendation(np.array([], dtype=int), np.array([]), components)

    popularity = prepared.popularity
    if strategy == "popularity":
        scores = popularity.copy()
    elif strategy == "personalized":
        scores = components["personal_relevance"]
    else:
        personal, sustainable, local, crowd, dispersion = sustainability_weights
        scores = (
            personal * components["personal_relevance"]
            + sustainable * prepared.sustainability
            + local * prepared.local_economic
            + crowd * components["crowd_comfort"]
            + dispersion * components["dispersion"]
        )
    scores = 0.9 * scores + 0.1 * components["distance_fit"]
    scores[~mask] = -np.inf
    candidates = np.flatnonzero(mask)
    ordered = candidates[np.argsort(scores[candidates])[::-1]]
    if strategy == "sustainability":
        ranked = _diversity_rerank(ordered[: min(20, len(ordered))], scores, prepared, top_k)
    else:
        ranked = ordered[:top_k]
    return Recommendation(ranked, scores[ranked], components)
