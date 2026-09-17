"""Plotly figures for the results notebook.

Colors follow a fixed categorical order, so a system keeps its color in every figure. The
palette was checked for color-vision deficiencies; two of its colors have low contrast on white,
which is why every figure also prints its values as text.
"""

import math

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import SYSTEM_NAMES

SYSTEM_COLORS: dict[str, str] = {
    "naive_rag": "#2a78d6",
    "lightrag_hybrid": "#eb6834",
    "graphrag_local": "#1baf7a",
    "graphrag_global": "#eda100",
}
SYSTEM_MARKER_SYMBOLS: dict[str, str] = {
    "naive_rag": "circle",
    "lightrag_hybrid": "square",
    "graphrag_local": "diamond",
    "graphrag_global": "triangle-up",
}
QUESTION_TYPE_LABELS: dict[str, str] = {
    "single_fact": "Single-Fact<br>(accuracy)",
    "multi_fact": "Multi-Fact<br>(accuracy)",
    "summary": "Summary<br>(statement F1)",
}
TEXT_COLOR = "#52514e"
GRID_COLOR = "#e6e5e0"


def _apply_common_layout(figure: go.Figure, title: str) -> go.Figure:
    """Apply the shared title, fonts, background and legend placement."""
    figure.update_layout(
        title={"text": title, "x": 0, "xanchor": "left"},
        template="plotly_white",
        font={"family": "Inter, Helvetica, Arial, sans-serif", "size": 13, "color": TEXT_COLOR},
        plot_bgcolor="#fcfcfb",
        paper_bgcolor="#fcfcfb",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 60, "r": 20, "t": 90, "b": 50},
        hoverlabel={"bgcolor": "white"},
    )
    figure.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)
    figure.update_xaxes(showgrid=False)
    return figure


def _ordered_systems(results_table: pd.DataFrame) -> list[str]:
    """Return the systems present in the table, in the display order of the config."""
    return [system for system in SYSTEM_NAMES if system in set(results_table["system_name"])]


def plot_quality_by_question_type(results_table: pd.DataFrame) -> go.Figure:
    """Grouped bars of answer quality per question type, one bar per system.

    Single-Fact and Multi-Fact show judged accuracy with its 95% Wilson interval (black
    whisker). Summary shows the statement-level F1, which has no interval. Values are printed
    at the base of each bar so they never collide with the whiskers.

    Args:
        results_table: Output of ``src.evaluation.build_results_table``.

    Returns:
        The Plotly figure.
    """
    quality_rows = results_table[
        ((results_table["metric"] == "accuracy") & results_table["question_type"].isin(["single_fact", "multi_fact"]))
        | ((results_table["metric"] == "f1") & (results_table["question_type"] == "summary"))
    ]
    figure = go.Figure()
    for system_name in _ordered_systems(results_table):
        system_rows = quality_rows[quality_rows["system_name"] == system_name].set_index("question_type")
        system_rows = system_rows.reindex(
            [question_type for question_type in QUESTION_TYPE_LABELS if question_type in system_rows.index]
        )
        figure.add_trace(
            go.Bar(
                name=system_name,
                x=[QUESTION_TYPE_LABELS[question_type] for question_type in system_rows.index],
                y=system_rows["value"],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": (system_rows["ci_high"] - system_rows["value"]).tolist(),
                    "arrayminus": (system_rows["value"] - system_rows["ci_low"]).tolist(),
                    "color": TEXT_COLOR,
                    "thickness": 1,
                    "width": 0,
                },
                marker={"color": SYSTEM_COLORS[system_name], "line": {"color": "#fcfcfb", "width": 2}},
                text=[f"{value:.0%}" for value in system_rows["value"]],
                textposition="inside",
                insidetextanchor="start",
                textangle=0,
                textfont={"color": "#0b0b0b", "size": 12},
                customdata=system_rows["question_count"],
                hovertemplate="%{x}<br>" + system_name + ": %{y:.1%}<br>n = %{customdata}<extra></extra>",
            )
        )
    figure.update_yaxes(title="Score", tickformat=".0%", range=[0, 1.1])
    figure.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    return _apply_common_layout(figure, "Answer quality by question type")


def plot_cost_versus_accuracy(results_table: pd.DataFrame) -> go.Figure:
    """Scatter of overall accuracy against the average query cost of one question.

    Each point is labelled with its system name, so the chart reads without the legend.

    Args:
        results_table: Output of ``src.evaluation.build_results_table``.

    Returns:
        The Plotly figure (log-scaled cost axis).
    """
    overall_rows = results_table[results_table["question_type"] == "all"]
    query_costs = overall_rows.loc[overall_rows["metric"] == "query_cost_usd_per_question", "value"]
    figure = go.Figure()
    for system_name in _ordered_systems(results_table):
        system_rows = overall_rows[overall_rows["system_name"] == system_name].set_index("metric")["value"]
        figure.add_trace(
            go.Scatter(
                name=system_name,
                x=[system_rows["query_cost_usd_per_question"]],
                y=[system_rows["accuracy"]],
                mode="markers+text",
                text=[system_name],
                textposition="top center",
                textfont={"color": TEXT_COLOR},
                marker={
                    "size": 14,
                    "color": SYSTEM_COLORS[system_name],
                    "symbol": SYSTEM_MARKER_SYMBOLS[system_name],
                    "line": {"color": "#fcfcfb", "width": 2},
                },
                hovertemplate=system_name + "<br>cost per question: $%{x:.5f}<br>accuracy: %{y:.1%}<extra></extra>",
            )
        )
    figure.update_xaxes(
        title="Average query cost per question (USD, log scale)",
        type="log",
        range=[math.log10(query_costs.min()) - 0.3, math.log10(query_costs.max()) + 0.3],
        dtick=1,
        tickprefix="$",
        showgrid=True,
        gridcolor=GRID_COLOR,
    )
    figure.update_yaxes(title="Accuracy, all question types", tickformat=".0%")
    return _apply_common_layout(figure, "Quality against query cost")


def plot_indexing_cost_and_time(results_table: pd.DataFrame) -> go.Figure:
    """Two side-by-side bar charts: indexing cost and indexing wall time per index.

    The two GraphRAG search modes share one index and therefore one bar.

    Args:
        results_table: Output of ``src.evaluation.build_results_table``.

    Returns:
        The Plotly figure.
    """
    indexing_rows = results_table[results_table["metric"].isin(["indexing_cost_usd", "indexing_time_seconds"])]
    indexing_rows = indexing_rows[indexing_rows["system_name"] != "graphrag_global"]
    systems = _ordered_systems(indexing_rows)
    display_names = {
        "naive_rag": "naive_rag",
        "lightrag_hybrid": "lightrag",
        "graphrag_local": "graphrag (local + global)",
    }

    figure = make_subplots(rows=1, cols=2, subplot_titles=("Indexing cost (USD)", "Indexing time (seconds)"))
    for column_index, (metric, value_format) in enumerate(
        [("indexing_cost_usd", "${:.3f}"), ("indexing_time_seconds", "{:.0f} s")], start=1
    ):
        metric_values = (
            indexing_rows[indexing_rows["metric"] == metric].set_index("system_name")["value"].reindex(systems)
        )
        figure.add_trace(
            go.Bar(
                x=[display_names.get(system, system) for system in systems],
                y=metric_values,
                marker={
                    "color": [SYSTEM_COLORS[system] for system in systems],
                    "line": {"color": "#fcfcfb", "width": 2},
                },
                text=[value_format.format(value) for value in metric_values],
                textposition="outside",
                textfont={"color": TEXT_COLOR},
                hovertemplate="%{x}: %{text}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=column_index,
        )
    figure.update_yaxes(rangemode="tozero")
    figure.update_yaxes(tickprefix="$", row=1, col=1)
    return _apply_common_layout(figure, "What it costs to build each index")
