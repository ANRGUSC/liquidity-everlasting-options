"""Plotly chart components for visualization."""

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_price_chart(
    prices: Sequence[float],
    dates: Optional[Sequence] = None,
    current_day: Optional[int] = None,
    strike: Optional[float] = None,
    title: str = "Asset Price",
) -> go.Figure:
    """
    Create asset price chart.

    Parameters
    ----------
    prices : Sequence[float]
        Price values
    dates : Sequence, optional
        Date labels for x-axis
    current_day : int, optional
        Current day to highlight (for playback)
    strike : float, optional
        Strike price to show as horizontal line
    title : str
        Chart title

    Returns
    -------
    go.Figure
        Plotly figure
    """
    x = dates if dates is not None else list(range(len(prices)))

    fig = go.Figure()

    # Full price path (dimmed if in playback mode)
    opacity = 0.3 if current_day is not None else 1.0
    fig.add_trace(go.Scatter(
        x=x,
        y=prices,
        mode="lines",
        name="Price",
        line=dict(color="#2196F3", width=2),
        opacity=opacity,
    ))

    # Highlighted portion up to current day
    if current_day is not None and current_day > 0:
        fig.add_trace(go.Scatter(
            x=x[:current_day + 1],
            y=prices[:current_day + 1],
            mode="lines",
            name="Current",
            line=dict(color="#2196F3", width=2),
        ))

        # Current point marker
        fig.add_trace(go.Scatter(
            x=[x[current_day]],
            y=[prices[current_day]],
            mode="markers",
            name="Now",
            marker=dict(color="#FF5722", size=12, symbol="circle"),
        ))

    # Strike line
    if strike is not None:
        fig.add_hline(
            y=strike,
            line_dash="dash",
            line_color="#9C27B0",
            annotation_text=f"Strike: ${strike:,.0f}",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date" if dates is not None else "Day",
        yaxis_title="Price ($)",
        hovermode="x unified",
        showlegend=False,
        height=350,
        margin=dict(l=50, r=20, t=40, b=40),
    )

    return fig


def create_equity_chart(
    equity_curve: Sequence[float],
    dates: Optional[Sequence] = None,
    current_day: Optional[int] = None,
    initial_capital: Optional[float] = None,
    title: str = "LP Equity Curve",
) -> go.Figure:
    """
    Create LP equity curve chart.

    Parameters
    ----------
    equity_curve : Sequence[float]
        Equity values
    dates : Sequence, optional
        Date labels
    current_day : int, optional
        Current day for playback
    initial_capital : float, optional
        Initial capital line
    title : str
        Chart title

    Returns
    -------
    go.Figure
        Plotly figure
    """
    x = dates if dates is not None else list(range(len(equity_curve)))

    fig = go.Figure()

    # Color based on profit/loss
    colors = ["#4CAF50" if e >= equity_curve[0] else "#F44336" for e in equity_curve]

    opacity = 0.3 if current_day is not None else 1.0
    fig.add_trace(go.Scatter(
        x=x,
        y=equity_curve,
        mode="lines",
        name="Equity",
        line=dict(color="#4CAF50", width=2),
        fill="tozeroy",
        fillcolor="rgba(76, 175, 80, 0.1)",
        opacity=opacity,
    ))

    if current_day is not None and current_day > 0:
        color = "#4CAF50" if equity_curve[current_day] >= equity_curve[0] else "#F44336"
        fig.add_trace(go.Scatter(
            x=x[:current_day + 1],
            y=equity_curve[:current_day + 1],
            mode="lines",
            name="Current",
            line=dict(color=color, width=2),
            fill="tozeroy",
            fillcolor=f"rgba({76 if color == '#4CAF50' else 244}, {175 if color == '#4CAF50' else 67}, {80 if color == '#4CAF50' else 54}, 0.2)",
        ))

        fig.add_trace(go.Scatter(
            x=[x[current_day]],
            y=[equity_curve[current_day]],
            mode="markers",
            name="Now",
            marker=dict(color="#FF5722", size=12),
        ))

    if initial_capital is not None:
        fig.add_hline(
            y=initial_capital,
            line_dash="dot",
            line_color="#9E9E9E",
            annotation_text=f"Initial: ${initial_capital:,.0f}",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Date" if dates is not None else "Day",
        yaxis_title="Equity ($)",
        hovermode="x unified",
        showlegend=False,
        height=350,
        margin=dict(l=50, r=20, t=40, b=40),
    )

    return fig


def create_pnl_attribution_chart(
    df: pd.DataFrame,
    title: str = "PnL Attribution",
) -> go.Figure:
    """
    Create stacked area chart for PnL attribution.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with cumulative_funding, cumulative_fees, cumulative_hedge_pnl columns
    title : str
        Chart title

    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = go.Figure()

    # Add each component as stacked area
    fig.add_trace(go.Scatter(
        x=df["date"] if "date" in df.columns else df.index,
        y=df["cumulative_funding"],
        mode="lines",
        name="Funding",
        stackgroup="one",
        fillcolor="rgba(33, 150, 243, 0.5)",
        line=dict(color="#2196F3", width=1),
    ))

    fig.add_trace(go.Scatter(
        x=df["date"] if "date" in df.columns else df.index,
        y=df["cumulative_fees"],
        mode="lines",
        name="Fees",
        stackgroup="one",
        fillcolor="rgba(76, 175, 80, 0.5)",
        line=dict(color="#4CAF50", width=1),
    ))

    fig.add_trace(go.Scatter(
        x=df["date"] if "date" in df.columns else df.index,
        y=df["cumulative_hedge_pnl"],
        mode="lines",
        name="Hedge",
        stackgroup="one",
        fillcolor="rgba(156, 39, 176, 0.5)",
        line=dict(color="#9C27B0", width=1),
    ))

    # Total line
    total = df["cumulative_funding"] + df["cumulative_fees"] + df["cumulative_hedge_pnl"]
    fig.add_trace(go.Scatter(
        x=df["date"] if "date" in df.columns else df.index,
        y=total,
        mode="lines",
        name="Total",
        line=dict(color="#FF5722", width=2, dash="dot"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative PnL ($)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=50, r=20, t=60, b=40),
    )

    return fig


def create_sensitivity_chart(
    param_name: str,
    param_range: Sequence[float],
    metric_values: Sequence[float],
    current_value: Optional[float] = None,
    metric_name: str = "Sharpe Ratio",
) -> go.Figure:
    """
    Create parameter sensitivity chart.

    Parameters
    ----------
    param_name : str
        Parameter name for x-axis
    param_range : Sequence[float]
        Parameter values
    metric_values : Sequence[float]
        Corresponding metric values
    current_value : float, optional
        Current parameter value to highlight
    metric_name : str
        Metric name for y-axis

    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=param_range,
        y=metric_values,
        mode="lines+markers",
        name=metric_name,
        line=dict(color="#2196F3", width=2),
        marker=dict(size=8),
    ))

    if current_value is not None:
        # Find closest metric value
        idx = np.argmin(np.abs(np.array(param_range) - current_value))
        fig.add_trace(go.Scatter(
            x=[param_range[idx]],
            y=[metric_values[idx]],
            mode="markers",
            name="Current",
            marker=dict(color="#FF5722", size=14, symbol="star"),
        ))

    fig.update_layout(
        title=f"{metric_name} vs {param_name}",
        xaxis_title=param_name,
        yaxis_title=metric_name,
        hovermode="x unified",
        showlegend=True,
        height=350,
        margin=dict(l=50, r=20, t=40, b=40),
    )

    return fig


def create_position_chart(
    df: pd.DataFrame,
    title: str = "Net Position & Funding",
) -> go.Figure:
    """
    Create position and funding direction chart.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with net_position and funding_payment columns
    title : str
        Chart title

    Returns
    -------
    go.Figure
        Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Net Position", "Funding Payment"),
        row_heights=[0.6, 0.4],
    )

    x = df["date"] if "date" in df.columns else df.index

    # Net position
    colors = ["#4CAF50" if p >= 0 else "#F44336" for p in df["net_position"]]
    fig.add_trace(go.Bar(
        x=x,
        y=df["net_position"],
        name="Net Position",
        marker_color=colors,
    ), row=1, col=1)

    # Funding payments
    funding_colors = ["#2196F3" if f >= 0 else "#FF9800" for f in df["funding_payment"]]
    fig.add_trace(go.Bar(
        x=x,
        y=df["funding_payment"],
        name="Funding",
        marker_color=funding_colors,
    ), row=2, col=1)

    fig.update_layout(
        title=title,
        height=450,
        showlegend=False,
        margin=dict(l=50, r=20, t=60, b=40),
    )

    fig.update_yaxes(title_text="Contracts", row=1, col=1)
    fig.update_yaxes(title_text="$", row=2, col=1)

    return fig


def create_combined_playback_chart(
    df: pd.DataFrame,
    current_day: int,
    strike: float,
    initial_capital: float,
) -> go.Figure:
    """
    Create combined chart for playback view.

    Parameters
    ----------
    df : pd.DataFrame
        Simulation history DataFrame
    current_day : int
        Current playback day
    strike : float
        Option strike price
    initial_capital : float
        Initial LP capital

    Returns
    -------
    go.Figure
        Combined Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Asset Price", "LP Equity"),
        row_heights=[0.5, 0.5],
    )

    dates = df["date"].tolist()
    prices = df["spot_price"].tolist()
    equity = df["lp_equity"].tolist()

    # Full paths (dimmed)
    fig.add_trace(go.Scatter(
        x=dates, y=prices,
        mode="lines",
        line=dict(color="#2196F3", width=1),
        opacity=0.3,
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=equity,
        mode="lines",
        line=dict(color="#4CAF50", width=1),
        opacity=0.3,
        showlegend=False,
    ), row=2, col=1)

    # Current paths
    if current_day > 0:
        fig.add_trace(go.Scatter(
            x=dates[:current_day + 1],
            y=prices[:current_day + 1],
            mode="lines",
            name="Price",
            line=dict(color="#2196F3", width=2),
        ), row=1, col=1)

        color = "#4CAF50" if equity[current_day] >= initial_capital else "#F44336"
        fig.add_trace(go.Scatter(
            x=dates[:current_day + 1],
            y=equity[:current_day + 1],
            mode="lines",
            name="Equity",
            line=dict(color=color, width=2),
        ), row=2, col=1)

    # Current markers
    fig.add_trace(go.Scatter(
        x=[dates[current_day]],
        y=[prices[current_day]],
        mode="markers",
        marker=dict(color="#FF5722", size=10),
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=[dates[current_day]],
        y=[equity[current_day]],
        mode="markers",
        marker=dict(color="#FF5722", size=10),
        showlegend=False,
    ), row=2, col=1)

    # Reference lines
    fig.add_hline(y=strike, line_dash="dash", line_color="#9C27B0", row=1, col=1)
    fig.add_hline(y=initial_capital, line_dash="dot", line_color="#9E9E9E", row=2, col=1)

    fig.update_layout(
        height=500,
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=40),
    )

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=2, col=1)

    return fig
