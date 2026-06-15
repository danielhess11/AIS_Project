# Sustainable Tourism Recommender Evaluation

This project evaluates whether a sustainability-aware recommender can distribute tourists more evenly across Barcelona than popularity-based and conventionally personalized alternatives. It uses a Mesa agent-based model, 60 curated real POIs, synthetic tourist profiles, paired replicated experiments, an interactive Streamlit dashboard, and reproducible visual evidence.

## Key Results

The completed experiment contains 30 paired replications per strategy with 5,000 tourists in each run. The same tourist population and random seed are reused across strategies within every replication.

| Mean metric | Popularity | Personalized | Sustainability-aware |
|---|---:|---:|---:|
| Peak load / comfortable capacity | 1.385 | 0.692 | 0.675 |
| Overcrowded POI-time cells | 1.667% | 0.000% | 0.000% |
| Visits to top 10 POIs | 77.71% | 13.65% | 7.67% |
| Visit Gini | 0.856 | 0.358 | 0.408 |
| District entropy | 0.699 | 0.975 | 1.000 |
| Tourist satisfaction | 0.531 | 0.787 | 0.718 |
| Mean sustainability | 0.678 | 0.807 | 0.820 |

Against popularity, the sustainability-aware strategy reduced mean peak load by 0.710 (95% paired CI: -0.721 to -0.698), reduced top-10 visit share by 70.0 percentage points, and increased district entropy by 0.302. These are modeled outcomes, not causal claims about real Barcelona.

## Run The Project

```powershell
python -m pip install -e .
python -m pytest -q
tourism-sim quick --population 1000 --seed 42
streamlit run app.py
```

Reproduce the submitted full experiment and evidence package:

```powershell
tourism-sim full --population 5000 --replications 30 --base-seed 1000 --output-dir outputs
tourism-sim artifacts --population 5000 --seed 2026 --output-dir outputs
```

The artifact command uses ImageIO's packaged FFmpeg binary, so a system FFmpeg installation is not required.

## Model

Each tourist receives two interests, a daily budget, mobility mode, walking tolerance, crowd aversion, sustainability sensitivity, outdoor preference, family status, and four visit slots within a six-slot day. At each active slot, the recommender produces five feasible POIs and the tourist probabilistically selects one based on recommendation score, personal relevance, crowd comfort, and travel fit.

- **Popularity:** baseline popularity plus travel fit after budget, mobility, family, and repeat-visit filtering.
- **Personalized:** 55% interest, 15% budget fit, 10% accessibility, 10% family/outdoor context, and 10% crowd comfort.
- **Sustainability-aware:** 40% personal relevance, 20% POI sustainability, 15% local-economic benefit, 15% crowd relief, and 10% spatial dispersion, followed by category and district diversity reranking.

The CSV contains modeled nominal capacities. The simulation treats 55% of nominal capacity as the comfortable operating threshold, representing the point where crowding begins to reduce visitor comfort rather than a legal venue limit.

## Outputs

- `outputs/run_metrics.csv`: all 90 run-level metric records.
- `outputs/paired_statistics.csv`: paired mean differences, 95% confidence intervals, Cohen's dz, and p-values.
- `outputs/visits.csv`: 60,000 visits from the shared-seed demonstration run.
- `outputs/occupancy_timeseries.csv`: POI crowding by strategy and time slot.
- `outputs/recommendations.csv`: explainable recommendation samples with score components.
- `outputs/figures/*.png`: six report-ready figures.
- `outputs/crowd_flow.mp4`: 1280x720 H.264 comparison over all six time slots.
- `notebooks/evaluation.ipynb`: executed academic report with tables and embedded figures.


## Project Layout

```text
app.py                         Streamlit dashboard
data/pois.csv                  Curated Barcelona POIs
src/sustainable_tourism/       Model, recommenders, metrics, CLI, artifacts
tests/                         Unit and integration tests
notebooks/evaluation.ipynb     Executed report
outputs/                       Reproducible data and visual evidence
```
