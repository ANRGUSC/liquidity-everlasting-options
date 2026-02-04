"""Metrics display components."""

from typing import Dict, Any, Optional

import streamlit as st
import pandas as pd


def render_metric_cards(metrics: dict, columns: int = 4):
    """
    Render metric cards in a grid layout.

    Parameters
    ----------
    metrics : dict
        Dictionary of metric name -> value pairs
    columns : int
        Number of columns in the grid
    """
    cols = st.columns(columns)

    metric_configs = {
        "total_return_pct": ("Total Return", "{:.2f}%", None),
        "sharpe_ratio": ("Sharpe Ratio", "{:.2f}", None),
        "max_drawdown_pct": ("Max Drawdown", "{:.2f}%", "inverse"),
        "win_rate_pct": ("Win Rate", "{:.1f}%", None),
        "sortino_ratio": ("Sortino Ratio", "{:.2f}", None),
        "profit_factor": ("Profit Factor", "{:.2f}", None),
        "var_95": ("VaR (95%)", "{:.2f}%", "inverse"),
        "calmar_ratio": ("Calmar Ratio", "{:.2f}", None),
    }

    i = 0
    for key, (label, fmt, delta_type) in metric_configs.items():
        if key in metrics:
            value = metrics[key]
            with cols[i % columns]:
                # Format the delta color
                if delta_type == "inverse":
                    delta_color = "inverse"
                else:
                    delta_color = "normal"

                st.metric(
                    label=label,
                    value=fmt.format(value) if not isinstance(value, str) else value,
                    delta=None,
                )
            i += 1


def render_metrics_table(metrics: dict, title: str = "Performance Metrics"):
    """
    Render detailed metrics table.

    Parameters
    ----------
    metrics : dict
        Dictionary of metrics
    title : str
        Table title
    """
    st.subheader(title)

    # Format metrics for display
    formatted = {}

    format_map = {
        "total_pnl": ("Total PnL", "${:,.0f}"),
        "total_return_pct": ("Total Return", "{:.2f}%"),
        "sharpe_ratio": ("Sharpe Ratio", "{:.2f}"),
        "sortino_ratio": ("Sortino Ratio", "{:.2f}"),
        "max_drawdown_pct": ("Max Drawdown", "{:.2f}%"),
        "max_drawdown_duration": ("Max DD Duration", "{} days"),
        "win_rate_pct": ("Win Rate", "{:.1f}%"),
        "profit_factor": ("Profit Factor", "{:.2f}"),
        "var_95": ("VaR (95%)", "{:.4f}"),
        "var_99": ("VaR (99%)", "{:.4f}"),
        "cvar_95": ("CVaR (95%)", "{:.4f}"),
        "calmar_ratio": ("Calmar Ratio", "{:.2f}"),
    }

    rows = []
    for key, (label, fmt) in format_map.items():
        if key in metrics:
            value = metrics[key]
            try:
                formatted_value = fmt.format(value)
            except (ValueError, TypeError):
                formatted_value = str(value)
            rows.append({"Metric": label, "Value": formatted_value})

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True)


def render_pnl_breakdown(pnl: dict, title: str = "PnL Breakdown"):
    """
    Render PnL breakdown display.

    Parameters
    ----------
    pnl : dict
        PnL breakdown with funding, fees, hedge, total
    title : str
        Display title
    """
    st.subheader(title)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Component**")
        st.markdown("Funding Income")
        st.markdown("Fee Income")
        st.markdown("Hedge PnL")
        st.markdown("---")
        st.markdown("**Total**")

    with col2:
        st.markdown("**Amount**")

        funding = pnl.get("funding_pnl", pnl.get("funding", 0))
        fees = pnl.get("fee_income", pnl.get("trading_fees", 0))
        hedge = pnl.get("hedge_pnl", pnl.get("hedge", 0))
        total = pnl.get("total_pnl", pnl.get("total", funding + fees + hedge))

        color_funding = "green" if funding >= 0 else "red"
        color_fees = "green" if fees >= 0 else "red"
        color_hedge = "green" if hedge >= 0 else "red"
        color_total = "green" if total >= 0 else "red"

        st.markdown(f":{color_funding}[${funding:,.0f}]")
        st.markdown(f":{color_fees}[${fees:,.0f}]")
        st.markdown(f":{color_hedge}[${hedge:,.0f}]")
        st.markdown("---")
        st.markdown(f"**:{color_total}[${total:,.0f}]**")


def render_live_metrics(state: Any):
    """
    Render live metrics during playback.

    Parameters
    ----------
    state : SimState
        Current simulation state
    """
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Live Metrics")
        st.metric("Spot Price", f"${state.spot_price:,.0f}")
        st.metric("Mark Price", f"${state.mark_price:,.2f}")
        st.metric("Delta", f"{state.delta:.3f}")
        st.metric("Net Position", f"{state.net_position:,.0f}")

    with col2:
        st.markdown("##### PnL (Day)")
        funding_color = "green" if state.funding_payment >= 0 else "red"
        total_color = "green" if state.daily_pnl >= 0 else "red"

        st.markdown(f"Funding: :{funding_color}[${state.funding_payment:,.0f}]")
        st.markdown(f"Total Funding: ${state.cumulative_funding:,.0f}")
        st.markdown(f"Total Fees: ${state.cumulative_fees:,.0f}")
        st.markdown(f"Daily PnL: :{total_color}[${state.daily_pnl:,.0f}]")


def render_position_metrics(stats: dict):
    """
    Render position-related metrics.

    Parameters
    ----------
    stats : dict
        Position statistics
    """
    st.subheader("Position Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Avg Net Exposure", f"{stats.get('avg_net_position', 0):,.1f}")
        st.metric("Peak Net Exposure", f"{stats.get('max_net_position', 0):,.1f}")

    with col2:
        receiving = stats.get("pct_receiving_funding", 0)
        st.metric("Receiving Funding", f"{receiving:.1f}%")
        st.metric("Paying Funding", f"{100 - receiving:.1f}%")
