"""Mesa agent-based tourist simulation."""

from __future__ import annotations

from dataclasses import dataclass, replace

import mesa
import numpy as np
import pandas as pd

from .data import load_pois
from .profiles import generate_profiles
from .recommenders import STRATEGIES, Recommendation, prepare_pois, recommend


@dataclass(frozen=True)
class SimulationConfig:
    population: int = 5000
    seed: int = 42
    slots: int = 6
    visits_per_tourist: int = 4
    top_k: int = 5
    sample_recommendations: int = 200
    comfortable_capacity_factor: float = 0.55
    sustainability_weights: tuple[float, float, float, float, float] = (0.40, 0.20, 0.15, 0.15, 0.10)


@dataclass
class SimulationResult:
    strategy: str
    config: SimulationConfig
    pois: pd.DataFrame
    profiles: pd.DataFrame
    visits: pd.DataFrame
    occupancy: pd.DataFrame
    recommendations: pd.DataFrame
    decision_metrics: pd.DataFrame
    metrics: dict[str, float]


class TouristAgent(mesa.Agent):
    def __init__(self, model: "TourismModel", profile: dict):
        super().__init__(model)
        self.profile = profile
        self.profile["budget_remaining"] = float(profile["daily_budget"])
        self.visited: set[int] = set()
        self.current_location: tuple[float, float] | None = None

    def is_active(self, slot: int) -> bool:
        return slot in self.profile["visit_slots"]

    def step(self, slot: int) -> None:
        recommendation = recommend(
            self.model.strategy,
            self.profile,
            self.model.pois,
            self.model.current_occupancy,
            self.current_location,
            self.visited,
            self.model.district_visits,
            self.model.config.top_k,
            self.model.config.sustainability_weights,
            self.model.prepared_pois,
        )
        if len(recommendation.indices) == 0:
            return
        chosen_rank = self._choose(recommendation)
        poi_index = int(recommendation.indices[chosen_rank])
        poi = self.model.poi_records[poi_index]
        components = recommendation.components
        self.model.current_occupancy[poi_index] += 1
        self.model.district_visits[poi["district"]] = self.model.district_visits.get(poi["district"], 0) + 1
        self.profile["budget_remaining"] -= float(poi["price"])
        self.visited.add(poi_index)
        self.current_location = (float(poi["lat"]), float(poi["lon"]))

        satisfaction = float(
            0.55 * components["interest"][poi_index]
            + 0.20 * components["crowd_comfort"][poi_index]
            + 0.15 * components["budget_fit"][poi_index]
            + 0.10 * components["context"][poi_index]
        )
        self.model.visit_rows.append(
            {
                "tourist_id": self.profile["tourist_id"],
                "strategy": self.model.strategy,
                "slot": slot,
                "poi_id": poi["poi_id"],
                "poi_name": poi["name"],
                "district": poi["district"],
                "category": poi["category"],
                "price": poi["price"],
                "distance_km": components["distance_km"][poi_index],
                "satisfaction": satisfaction,
                "accepted_top_1": int(chosen_rank == 0),
                "budget_compliant": int(poi["price"] <= self.profile["budget_remaining"] + poi["price"]),
                "sustainability": poi["sustainability"],
                "local_economic": poi["local_economic"],
            }
        )
        self._record_recommendation(slot, recommendation, chosen_rank)

    def _choose(self, recommendation: Recommendation) -> int:
        indices = recommendation.indices
        components = recommendation.components
        utility = (
            0.55 * recommendation.scores
            + 0.20 * components["personal_relevance"][indices]
            + 0.15 * components["crowd_comfort"][indices]
            + 0.10 * components["distance_fit"][indices]
        )
        probabilities = np.exp((utility - utility.max()) / 0.12)
        probabilities /= probabilities.sum()
        return int(self.model.rng.choice(len(indices), p=probabilities))

    def _record_recommendation(self, slot: int, recommendation: Recommendation, chosen_rank: int) -> None:
        indices = recommendation.indices
        components = recommendation.components
        relevant = (
            (self.model.prepared_pois.category == self.profile["primary_interest"])
            | (self.model.prepared_pois.category == self.profile["secondary_interest"])
        )
        feasible_relevant = relevant & (self.model.prepared_pois.price <= self.profile["budget_remaining"])
        relevant_recommended = relevant[indices].sum()
        categories = len(np.unique(self.model.prepared_pois.category[indices]))
        pair_count = max(len(indices) * (len(indices) - 1) / 2, 1)
        same_category_pairs = 0
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                same_category_pairs += int(
                    self.model.prepared_pois.category[indices[left]]
                    == self.model.prepared_pois.category[indices[right]]
                )
        self.model.decision_rows.append(
            {
                "tourist_id": self.profile["tourist_id"],
                "slot": slot,
                "precision_at_5": relevant_recommended / len(indices),
                "recall_at_5": relevant_recommended / max(feasible_relevant.sum(), 1),
                "diversity": 1 - same_category_pairs / pair_count,
                "category_coverage": categories / len(indices),
                "novelty": float((1 - self.model.prepared_pois.popularity[indices]).mean()),
            }
        )
        if self.profile["tourist_id"] < self.model.config.sample_recommendations:
            for rank, index in enumerate(indices, start=1):
                poi = self.model.poi_records[index]
                self.model.recommendation_rows.append(
                    {
                        "tourist_id": self.profile["tourist_id"],
                        "strategy": self.model.strategy,
                        "slot": slot,
                        "rank": rank,
                        "poi_id": poi["poi_id"],
                        "poi_name": poi["name"],
                        "final_score": recommendation.scores[rank - 1],
                        "interest": components["interest"][index],
                        "personal_relevance": components["personal_relevance"][index],
                        "crowd_comfort": components["crowd_comfort"][index],
                        "distance_fit": components["distance_fit"][index],
                        "selected": int(rank - 1 == chosen_rank),
                    }
                )


class TourismModel(mesa.Model):
    def __init__(
        self,
        strategy: str,
        config: SimulationConfig,
        pois: pd.DataFrame,
        profiles: pd.DataFrame,
    ):
        super().__init__(seed=config.seed)
        self.strategy = strategy
        self.config = config
        self.pois = pois
        prepared = prepare_pois(pois)
        self.prepared_pois = replace(
            prepared,
            capacity=prepared.capacity * config.comfortable_capacity_factor,
        )
        self.poi_records = pois.to_dict(orient="records")
        self.profiles = profiles
        self.rng = np.random.default_rng(config.seed + 10_000)
        self.current_occupancy = np.zeros(len(pois), dtype=int)
        self.district_visits: dict[str, int] = {}
        self.visit_rows: list[dict] = []
        self.occupancy_rows: list[dict] = []
        self.recommendation_rows: list[dict] = []
        self.decision_rows: list[dict] = []
        self.tourists = [TouristAgent(self, row.to_dict()) for _, row in profiles.iterrows()]

    def run(self) -> None:
        for slot in range(self.config.slots):
            self.current_occupancy.fill(0)
            active = [agent for agent in self.tourists if agent.is_active(slot)]
            self.rng.shuffle(active)
            for agent in active:
                agent.step(slot)
            for index, poi in self.pois.iterrows():
                occupancy = int(self.current_occupancy[index])
                effective_capacity = float(self.prepared_pois.capacity[index])
                self.occupancy_rows.append(
                    {
                        "strategy": self.strategy,
                        "slot": slot,
                        "poi_id": poi["poi_id"],
                        "poi_name": poi["name"],
                        "district": poi["district"],
                        "lat": poi["lat"],
                        "lon": poi["lon"],
                        "nominal_capacity": poi["capacity"],
                        "capacity": effective_capacity,
                        "occupancy": occupancy,
                        "load_ratio": occupancy / effective_capacity,
                    }
                )


def run_strategy(
    strategy: str,
    config: SimulationConfig | None = None,
    pois: pd.DataFrame | None = None,
    profiles: pd.DataFrame | None = None,
) -> SimulationResult:
    from .metrics import calculate_metrics

    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    config = config or SimulationConfig()
    pois = load_pois() if pois is None else pois.copy()
    if profiles is None:
        profiles = generate_profiles(
            config.population,
            sorted(pois["category"].unique().tolist()),
            config.seed,
            config.slots,
            config.visits_per_tourist,
        )
    model = TourismModel(strategy, config, pois, profiles)
    model.run()
    visits = pd.DataFrame(model.visit_rows)
    occupancy = pd.DataFrame(model.occupancy_rows)
    recommendations = pd.DataFrame(model.recommendation_rows)
    decisions = pd.DataFrame(model.decision_rows)
    metrics = calculate_metrics(pois, profiles, visits, occupancy, decisions, recommendations)
    return SimulationResult(
        strategy,
        config,
        pois,
        profiles,
        visits,
        occupancy,
        recommendations,
        decisions,
        metrics,
    )
