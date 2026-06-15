"""Generate report figures, detailed CSVs, video, and the executed notebook."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from nbclient import NotebookClient

from .data import load_pois
from .experiment import compare_strategies
from .simulation import SimulationConfig, SimulationResult


STRATEGY_COLORS = {
    "popularity": "#d95f02",
    "personalized": "#1b9e77",
    "sustainability": "#4c78a8",
}


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_figures(
    results: dict[str, SimulationResult],
    run_metrics: pd.DataFrame,
    figure_dir: str | Path = "outputs/figures",
) -> list[Path]:
    output = Path(figure_dir)
    output.mkdir(parents=True, exist_ok=True)
    pois = next(iter(results.values())).pois
    created: list[Path] = []

    fig, ax = plt.subplots(figsize=(10, 8))
    for district, group in pois.groupby("district"):
        ax.scatter(group["lon"], group["lat"], s=45, label=district, alpha=0.8)
    ax.set(title="Curated Barcelona POI coverage", xlabel="Longitude", ylabel="Latitude")
    ax.legend(fontsize=7, ncol=2)
    path = output / "poi_coverage.png"
    _save_figure(fig, path)
    created.append(path)

    selected_metrics = [
        "peak_load_ratio",
        "top_10_visit_share",
        "visit_gini",
        "district_entropy",
        "tourist_satisfaction",
        "mean_sustainability",
    ]
    means = run_metrics.groupby("strategy")[selected_metrics].mean()
    normalized = (means - means.min()) / (means.max() - means.min()).replace(0, 1)
    for lower_is_better in ["peak_load_ratio", "top_10_visit_share", "visit_gini"]:
        normalized[lower_is_better] = 1 - normalized[lower_is_better]
    fig, ax = plt.subplots(figsize=(11, 6))
    normalized.T.plot(kind="bar", ax=ax, color=[STRATEGY_COLORS.get(column) for column in normalized.index])
    ax.set(title="Relative strategy performance across experiment replications", ylabel="Normalized performance (higher is better)", xlabel="Metric")
    ax.tick_params(axis="x", rotation=30)
    path = output / "strategy_comparison.png"
    _save_figure(fig, path)
    created.append(path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    for ax, (strategy, result) in zip(axes, results.items()):
        pivot = result.occupancy.pivot(index="poi_name", columns="slot", values="load_ratio")
        top = pivot.max(axis=1).nlargest(15).index
        image = ax.imshow(pivot.loc[top], aspect="auto", cmap="magma", vmin=0, vmax=max(1, pivot.max().max()))
        ax.set_title(strategy.title())
        ax.set_xlabel("Time slot")
        ax.set_yticks(range(len(top)), top, fontsize=6)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Occupancy / capacity", shrink=0.8)
    path = output / "crowding_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    created.append(path)

    district = pd.concat(
        [result.visits.groupby("district").size().rename(strategy) for strategy, result in results.items()],
        axis=1,
    )
    district = district.div(district.sum(axis=0), axis=1)
    fig, ax = plt.subplots(figsize=(12, 6))
    district.plot(kind="bar", ax=ax, color=[STRATEGY_COLORS.get(column) for column in district.columns])
    ax.set(title="Distribution of visits across districts", ylabel="Share of visits", xlabel="District")
    ax.tick_params(axis="x", rotation=35)
    path = output / "district_distribution.png"
    _save_figure(fig, path)
    created.append(path)

    quality_metrics = ["precision_at_5", "recall_at_5", "diversity", "novelty"]
    quality = run_metrics.groupby("strategy")[quality_metrics].mean().T
    fig, ax = plt.subplots(figsize=(10, 6))
    quality.plot(kind="bar", ax=ax, color=[STRATEGY_COLORS.get(column) for column in quality.columns])
    ax.set(title="Recommendation quality", ylabel="Mean score", xlabel="Metric", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=0)
    path = output / "recommendation_quality.png"
    _save_figure(fig, path)
    created.append(path)

    fairness_metrics = ["exposure_fairness", "catalog_coverage", "district_entropy", "visit_gini"]
    fairness = run_metrics.groupby("strategy")[fairness_metrics].mean().T
    fig, ax = plt.subplots(figsize=(10, 6))
    fairness.plot(kind="bar", ax=ax, color=[STRATEGY_COLORS.get(column) for column in fairness.columns])
    ax.set(title="Exposure, coverage, and distribution metrics", ylabel="Mean score", xlabel="Metric", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=20)
    path = output / "fairness.png"
    _save_figure(fig, path)
    created.append(path)
    return created


def generate_video(
    results: dict[str, SimulationResult],
    output_path: str | Path = "outputs/crowd_flow.mp4",
    fps: int = 2,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    strategies = list(results)
    writer = imageio.get_writer(output, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p")
    try:
        for slot in range(next(iter(results.values())).config.slots):
            fig, axes = plt.subplots(1, 3, figsize=(12.8, 7.2), dpi=100, sharex=True, sharey=True)
            fig.suptitle(f"Barcelona tourist crowding, time slot {slot + 1}", fontsize=16)
            for ax, strategy in zip(axes, strategies):
                frame = results[strategy].occupancy
                frame = frame[frame["slot"] == slot]
                loads = frame["load_ratio"].to_numpy()
                scatter = ax.scatter(
                    frame["lon"],
                    frame["lat"],
                    s=30 + 350 * np.clip(loads, 0, 1.5),
                    c=loads,
                    cmap="RdYlGn_r",
                    vmin=0,
                    vmax=max(1, loads.max()),
                    alpha=0.8,
                    edgecolors="black",
                    linewidths=0.3,
                )
                ax.set_title(strategy.title())
                ax.set_xlabel("Longitude")
                ax.grid(alpha=0.2)
            axes[0].set_ylabel("Latitude")
            fig.colorbar(scatter, ax=axes.ravel().tolist(), label="Occupancy / capacity", shrink=0.75)
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            frame_array = np.asarray(canvas.buffer_rgba())[:, :, :3]
            for _ in range(fps * 2):
                writer.append_data(frame_array)
            plt.close(fig)
    finally:
        writer.close()
    return output


def write_detailed_outputs(results: dict[str, SimulationResult], output_dir: str | Path = "outputs") -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.concat([result.visits for result in results.values()], ignore_index=True).to_csv(output / "visits.csv", index=False)
    pd.concat([result.occupancy for result in results.values()], ignore_index=True).to_csv(output / "occupancy_timeseries.csv", index=False)
    pd.concat([result.recommendations for result in results.values()], ignore_index=True).to_csv(output / "recommendations.csv", index=False)


def build_and_execute_notebook(output_dir: str | Path = "outputs") -> Path:
    output = Path(output_dir)
    metrics = pd.read_csv(output / "run_metrics.csv")
    means = metrics.groupby("strategy").mean(numeric_only=True)
    baseline = means.loc["popularity"]
    sustainable = means.loc["sustainability"]
    findings = (
        "## Main findings\n"
        f"Across 30 paired replications, the popularity baseline reached a mean peak load of **{baseline['peak_load_ratio']:.3f}** "
        f"times comfortable capacity, versus **{sustainable['peak_load_ratio']:.3f}** for the sustainability-aware strategy. "
        f"The share of visits assigned to the ten most popular POIs fell from **{baseline['top_10_visit_share']:.1%}** to "
        f"**{sustainable['top_10_visit_share']:.1%}**, while normalized district entropy rose from "
        f"**{baseline['district_entropy']:.3f}** to **{sustainable['district_entropy']:.3f}**. "
        f"Mean modeled tourist satisfaction increased from **{baseline['tourist_satisfaction']:.3f}** to "
        f"**{sustainable['tourist_satisfaction']:.3f}**. These are simulation outcomes under synthetic assumptions, not causal estimates for Barcelona."
    )
    notebook_path = Path("notebooks/evaluation.ipynb")
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.v4.new_notebook()
    notebook["cells"] = [
        nbformat.v4.new_markdown_cell(
            "# Sustainable Tourism Recommender Evaluation\n"
            "This notebook reports a paired agent-based experiment comparing popularity, personalized, and sustainability-aware recommendations in Barcelona."
        ),
        nbformat.v4.new_markdown_cell(
            "## Methodology and assumptions\n"
            "Each replication generates the same synthetic tourist population for all strategies. Tourists make four visits over six time slots. "
            "POI identities and approximate locations are real; capacity, popularity, sustainability, and local-economic attributes are modeled for evaluation."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\n"
            "root = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
            "metrics = pd.read_csv(root / 'outputs' / 'run_metrics.csv')\n"
            "statistics = pd.read_csv(root / 'outputs' / 'paired_statistics.csv')\n"
            "metrics.groupby('strategy').mean(numeric_only=True).round(4)"
        ),
        nbformat.v4.new_markdown_cell("## Paired statistical comparisons"),
        nbformat.v4.new_code_cell(
            "statistics.sort_values(['metric', 'strategy']).reset_index(drop=True)"
        ),
        nbformat.v4.new_markdown_cell(findings),
        nbformat.v4.new_markdown_cell("## Visual evidence"),
        nbformat.v4.new_code_cell(
            "for name in ['poi_coverage.png', 'strategy_comparison.png', 'crowding_heatmap.png', "
            "'district_distribution.png', 'recommendation_quality.png', 'fairness.png']:\n"
            "    display(Image(filename=str(root / 'outputs' / 'figures' / name), width=900))"
        ),
        nbformat.v4.new_markdown_cell(
            "## Interpretation and limitations\n"
            "The sustainability-aware strategy is evaluated on whether it reduces concentration and peak crowding without an unacceptable loss in satisfaction or relevance. "
            "Results are evidence about this modeled scenario, not causal estimates for real Barcelona. Capacities and sustainability attributes require calibration with operational data before policy use. "
            "The model also abstracts opening hours, queues, public-transport congestion, resident behavior, weather, and seasonal demand."
        ),
        nbformat.v4.new_markdown_cell(
            "## References\n"
            "- Mesa documentation: https://mesa.readthedocs.io/\n"
            "- Streamlit documentation: https://docs.streamlit.io/\n"
            "- Barcelona Open Data portal: https://opendata-ajuntament.barcelona.cat/"
        ),
    ]
    notebook["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.10"}
    nbformat.write(notebook, notebook_path)
    client = NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(Path.cwd())}})
    client.execute()
    nbformat.write(notebook, notebook_path)
    return notebook_path


def generate_all_artifacts(
    population: int = 5000,
    seed: int = 2026,
    output_dir: str | Path = "outputs",
) -> dict[str, object]:
    output = Path(output_dir)
    metrics_path = output / "run_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError("Run the full experiment before generating artifacts")
    run_metrics = pd.read_csv(metrics_path)
    results = compare_strategies(SimulationConfig(population=population, seed=seed), load_pois())
    write_detailed_outputs(results, output)
    figures = generate_figures(results, run_metrics, output / "figures")
    video = generate_video(results, output / "crowd_flow.mp4")
    notebook = build_and_execute_notebook(output)
    return {"results": results, "figures": figures, "video": video, "notebook": notebook}
