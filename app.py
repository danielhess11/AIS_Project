"""Streamlit dashboard for the sustainable tourism simulation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from sustainable_tourism.experiment import compare_strategies
from sustainable_tourism.simulation import SimulationConfig


st.set_page_config(page_title="Sustainable Tourism Simulator", layout="wide")
st.title("Barcelona Sustainable Tourism Recommender Evaluation")
st.caption("Agent-based comparison of popularity, personalized, and sustainability-aware recommendations")


@st.cache_data(show_spinner=False)
def run_comparison(population: int, seed: int, weights: tuple[float, ...]):
    config = SimulationConfig(population=population, seed=seed, sustainability_weights=weights)
    return compare_strategies(config)


with st.sidebar:
    st.header("Simulation controls")
    population = st.slider("Tourists", 100, 5000, 1000, 100)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    st.subheader("Sustainability weights")
    raw_weights = [
        st.slider("Personal relevance", 0.0, 1.0, 0.40, 0.05),
        st.slider("POI sustainability", 0.0, 1.0, 0.20, 0.05),
        st.slider("Local economic benefit", 0.0, 1.0, 0.15, 0.05),
        st.slider("Crowd relief", 0.0, 1.0, 0.15, 0.05),
        st.slider("Spatial dispersion", 0.0, 1.0, 0.10, 0.05),
    ]
    total = sum(raw_weights)
    weights = tuple(weight / total for weight in raw_weights) if total else (0.4, 0.2, 0.15, 0.15, 0.1)
    run = st.button("Run simulation", type="primary", use_container_width=True)

if run or "results" not in st.session_state:
    with st.spinner("Simulating tourist itineraries..."):
        st.session_state.results = run_comparison(int(population), int(seed), weights)

results = st.session_state.results
metric_frame = pd.DataFrame({strategy: result.metrics for strategy, result in results.items()}).T
visits = pd.concat([result.visits for result in results.values()], ignore_index=True)
occupancy = pd.concat([result.occupancy for result in results.values()], ignore_index=True)
recommendations = pd.concat([result.recommendations for result in results.values()], ignore_index=True)

overview_tab, map_tab, crowd_tab, quality_tab, fairness_tab, explanation_tab = st.tabs(
    ["Overview", "Animated map", "Crowding", "Recommendation quality", "Fairness", "Tourist explanation"]
)

with overview_tab:
    columns = st.columns(3)
    for column, strategy in zip(columns, results):
        with column:
            st.subheader(strategy.title())
            st.metric("Peak load ratio", f"{metric_frame.loc[strategy, 'peak_load_ratio']:.2f}")
            st.metric("Tourist satisfaction", f"{metric_frame.loc[strategy, 'tourist_satisfaction']:.3f}")
            st.metric("District entropy", f"{metric_frame.loc[strategy, 'district_entropy']:.3f}")
    st.dataframe(metric_frame.round(4), use_container_width=True)

with map_tab:
    map_data = occupancy.copy()
    map_data["marker_size"] = 5 + 25 * map_data["load_ratio"].clip(upper=1.5)
    figure = px.scatter(
        map_data,
        x="lon",
        y="lat",
        animation_frame="slot",
        animation_group="poi_id",
        facet_col="strategy",
        color="load_ratio",
        size="marker_size",
        hover_name="poi_name",
        hover_data=["district", "occupancy", "capacity"],
        color_continuous_scale="RdYlGn_r",
        range_color=[0, max(1.0, map_data["load_ratio"].max())],
        height=650,
        title="Crowding evolution across the six time slots",
    )
    figure.update_yaxes(matches="y")
    st.plotly_chart(figure, use_container_width=True)

with crowd_tab:
    crowd_summary = occupancy.groupby(["strategy", "slot"], as_index=False).agg(
        peak_load_ratio=("load_ratio", "max"),
        overcrowded_pois=("load_ratio", lambda values: int((values > 1).sum())),
    )
    st.plotly_chart(
        px.line(crowd_summary, x="slot", y="peak_load_ratio", color="strategy", markers=True),
        use_container_width=True,
    )
    top_crowded = occupancy.nlargest(25, "load_ratio")[
        ["strategy", "slot", "poi_name", "district", "occupancy", "capacity", "load_ratio"]
    ]
    st.dataframe(top_crowded, use_container_width=True, hide_index=True)

with quality_tab:
    quality = metric_frame[["precision_at_5", "recall_at_5", "diversity", "novelty", "tourist_satisfaction"]]
    quality_long = quality.reset_index(names="strategy").melt("strategy", var_name="metric", value_name="score")
    st.plotly_chart(px.bar(quality_long, x="metric", y="score", color="strategy", barmode="group"), use_container_width=True)

with fairness_tab:
    fairness = metric_frame[["exposure_fairness", "catalog_coverage", "district_entropy", "visit_gini", "top_10_visit_share"]]
    fairness_long = fairness.reset_index(names="strategy").melt("strategy", var_name="metric", value_name="score")
    st.plotly_chart(px.bar(fairness_long, x="metric", y="score", color="strategy", barmode="group"), use_container_width=True)
    district = visits.groupby(["strategy", "district"]).size().reset_index(name="visits")
    district["share"] = district["visits"] / district.groupby("strategy")["visits"].transform("sum")
    st.plotly_chart(px.bar(district, x="district", y="share", color="strategy", barmode="group"), use_container_width=True)

with explanation_tab:
    available_ids = sorted(recommendations["tourist_id"].unique().tolist())
    tourist_id = st.selectbox("Tourist", available_ids)
    profile = next(iter(results.values())).profiles.query("tourist_id == @tourist_id")
    st.dataframe(profile.drop(columns=["visit_slots"]), use_container_width=True, hide_index=True)
    selected = recommendations.query("tourist_id == @tourist_id").sort_values(["strategy", "slot", "rank"])
    st.dataframe(selected.round(3), use_container_width=True, hide_index=True)
    st.caption("Score components expose why each POI was ranked and which recommendation the tourist selected.")

st.download_button(
    "Download current metrics as CSV",
    metric_frame.reset_index(names="strategy").to_csv(index=False),
    file_name="dashboard_metrics.csv",
    mime="text/csv",
)

