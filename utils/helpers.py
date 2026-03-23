

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import base64
from io import BytesIO


# ─────────────────────────────────────────────────────────────────────────────
# Chart helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_feature_importance_chart(importance_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    df = importance_df.head(top_n).copy()
    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {min(top_n, len(df))} Feature Importances",
        color="importance",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        height=max(400, min(top_n, len(df)) * 28),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
    )
    return fig


def create_prediction_distribution(predictions: List[Any], title: str = "Prediction Distribution") -> go.Figure:
    pred_series = pd.Series(predictions)
    fig = px.histogram(
        pred_series,
        title=title,
        labels={"value": "Predicted Value", "count": "Frequency"},
        color_discrete_sequence=["#6C63FF"],
    )
    fig.update_layout(
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_confusion_matrix(
    y_true: List[Any], y_pred: List[Any], labels: Optional[List[str]] = None
) -> go.Figure:
    from sklearn.metrics import confusion_matrix as sk_cm

    # Get all unique labels from both y_true and y_pred
    unique_true = set(y_true)
    unique_pred = set(y_pred)
    all_labels = sorted(unique_true.union(unique_pred))
    
    # If labels provided, filter to only those in y_true
    if labels:
        all_labels = [l for l in labels if l in unique_true or l in unique_pred]
    
    if not all_labels:
        all_labels = None
    
    cm = sk_cm(y_true, y_pred, labels=all_labels)
    display_labels = [str(l) for l in all_labels] if all_labels else [f"Class {i}" for i in range(len(cm))]

    # Normalised version for annotation text
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    annotations = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            annotations.append(
                dict(
                    x=display_labels[j],
                    y=display_labels[i],
                    text=f"{cm[i, j]}<br>({cm_norm[i, j]:.1%})",
                    showarrow=False,
                    font=dict(color="white", size=13),
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=display_labels,
            y=display_labels,
            colorscale="Blues",
            showscale=True,
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        annotations=annotations,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def create_metrics_dashboard(metrics: Dict[str, Any]) -> go.Figure:
    numeric_metrics = {
        k: v
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }

    if not numeric_metrics:
        fig = go.Figure()
        fig.add_annotation(text="No numeric metrics available", showarrow=False)
        return fig

    names = [k.replace("_", " ").title() for k in numeric_metrics]
    values = list(numeric_metrics.values())

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "bar"}, {"type": "indicator"}]],
        subplot_titles=["All Metrics", "Primary Metric"],
    )

    fig.add_trace(
        go.Bar(
            x=names, y=values,
            marker=dict(color=values, colorscale="Viridis"),
            text=[f"{v:.4f}" for v in values],
            textposition="outside",
        ),
        row=1, col=1,
    )

    primary_value = values[0]
    primary_name = names[0]
    gauge_max = 1.0 if primary_value <= 1.0 else primary_value * 1.2

    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            value=primary_value,
            title={"text": primary_name},
            gauge={
                "axis": {"range": [0, gauge_max]},
                "bar": {"color": "#6C63FF"},
                "steps": [
                    {"range": [0, gauge_max * 0.5], "color": "#FF6B6B"},
                    {"range": [gauge_max * 0.5, gauge_max * 0.8], "color": "#FFD93D"},
                    {"range": [gauge_max * 0.8, gauge_max], "color": "#6BCB77"},
                ],
            },
        ),
        row=1, col=2,
    )

    fig.update_layout(
        height=420,
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
    )
    return fig


def create_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need at least 2 numeric columns for correlation.", showarrow=False)
        return fig

    corr = numeric_df.corr()
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu",
            zmid=0,
            text=corr.round(2).values,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Feature Correlation Matrix",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        height=500,
    )
    return fig


def create_time_series_chart(
    df: pd.DataFrame, date_col: str, value_col: str, title: str = "Time Series"
) -> go.Figure:
    fig = px.line(df, x=date_col, y=value_col, title=title)
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def create_scatter_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    title: str = "Scatter Plot",
) -> go.Figure:
    fig = px.scatter(
        df, x=x_col, y=y_col, color=color_col,
        title=title, opacity=0.7,
        template="plotly_dark",
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    return fig


def create_class_distribution(series: pd.Series, title: str = "Class Distribution") -> go.Figure:
    counts = series.value_counts()
    fig = px.pie(
        values=counts.values,
        names=counts.index.astype(str),
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.35,
    )
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def create_actual_vs_predicted(y_true, y_pred, title: str = "Actual vs Predicted") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(y_true), y=list(y_pred),
        mode="markers",
        marker=dict(color="#6C63FF", opacity=0.6, size=6),
        name="Predictions",
    ))
    mn, mx = min(float(np.min(y_true)), float(np.min(y_pred))), max(float(np.max(y_true)), float(np.max(y_pred)))
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines",
        line=dict(color="#FF6B6B", dash="dash"),
        name="Perfect Fit",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Actual",
        yaxis_title="Predicted",
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def save_plotly_html(fig: go.Figure, filename: str, path: str = ".") -> Path:
    output_path = Path(path) / f"{filename}.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# Chart Generator utility class
# ─────────────────────────────────────────────────────────────────────────────

class ChartGenerator:
    def __init__(self):
        self.charts: Dict[str, go.Figure] = {}

    def add_chart(self, name: str, fig: go.Figure):
        self.charts[name] = fig

    def get_all_charts(self) -> Dict[str, go.Figure]:
        return self.charts

    def export_all_html(self, output_dir: str = "./output/charts") -> List[Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        saved = []
        for name, fig in self.charts.items():
            p = save_plotly_html(fig, name, output_dir)
            saved.append(p)
        return saved
