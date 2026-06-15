"""POI data loading and validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POI_PATH = ROOT / "data" / "pois.csv"
DISTRICTS = {
    "Ciutat Vella",
    "Eixample",
    "Sants-Montjuic",
    "Les Corts",
    "Sarria-Sant Gervasi",
    "Gracia",
    "Horta-Guinardo",
    "Nou Barris",
    "Sant Andreu",
    "Sant Marti",
}
NORMALIZED_COLUMNS = [
    "popularity",
    "sustainability",
    "local_economic",
    "outdoor",
    "kid_friendly",
    "accessibility",
    "walking_effort",
]
REQUIRED_COLUMNS = {
    "poi_id",
    "name",
    "district",
    "neighborhood",
    "category",
    "lat",
    "lon",
    "price",
    "capacity",
    *NORMALIZED_COLUMNS,
}


def validate_pois(pois: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(pois.columns)
    if missing:
        raise ValueError(f"POI dataset is missing columns: {sorted(missing)}")
    if len(pois) < 50:
        raise ValueError("At least 50 POIs are required")
    if set(pois["district"]) != DISTRICTS:
        missing_districts = DISTRICTS.difference(pois["district"])
        extra_districts = set(pois["district"]).difference(DISTRICTS)
        raise ValueError(
            f"District coverage mismatch; missing={missing_districts}, extra={extra_districts}"
        )
    if not pois["poi_id"].is_unique:
        raise ValueError("POI identifiers must be unique")
    if not pois["lat"].between(41.30, 41.50).all() or not pois["lon"].between(2.05, 2.25).all():
        raise ValueError("POI coordinates must be within the Barcelona study area")
    if not pois[NORMALIZED_COLUMNS].apply(lambda column: column.between(0, 1).all()).all():
        raise ValueError("Normalized POI attributes must be in [0, 1]")
    if not (pois["capacity"] > 0).all() or not (pois["price"] >= 0).all():
        raise ValueError("POI capacities must be positive and prices non-negative")


def load_pois(path: str | Path = DEFAULT_POI_PATH) -> pd.DataFrame:
    pois = pd.read_csv(path)
    validate_pois(pois)
    return pois.reset_index(drop=True)

