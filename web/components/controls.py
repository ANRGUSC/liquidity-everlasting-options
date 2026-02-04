"""Input controls and widgets for parameter configuration."""

from datetime import date, timedelta
from typing import Tuple

import streamlit as st


def render_lp_config() -> dict:
    """
    Render LP configuration inputs.

    Returns
    -------
    dict
        LP configuration parameters
    """
    st.subheader("LP Configuration")

    col1, col2 = st.columns(2)

    with col1:
        capital = st.number_input(
            "Initial Capital ($)",
            min_value=10_000,
            max_value=100_000_000,
            value=1_000_000,
            step=100_000,
            format="%d",
            help="LP's initial capital to deploy",
        )

    with col2:
        hedge_ratio = st.slider(
            "Hedge Ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Fraction of delta to hedge (0 = no hedge, 1 = full hedge)",
        )

    return {
        "capital": capital,
        "hedge_ratio": hedge_ratio,
    }


def render_option_params() -> dict:
    """
    Render option parameters inputs.

    Returns
    -------
    dict
        Option configuration parameters
    """
    st.subheader("Option Parameters")

    col1, col2 = st.columns(2)

    with col1:
        option_type = st.selectbox(
            "Option Type",
            options=["call", "put"],
            index=0,
            help="Call or Put option",
        )

        strike = st.number_input(
            "Strike Price ($)",
            min_value=100,
            max_value=100_000,
            value=3000,
            step=100,
            help="Strike price of the everlasting option",
        )

    with col2:
        entry_date = st.date_input(
            "Entry Date",
            value=date(2024, 1, 1),
            help="Simulation start date",
        )

        exit_date = st.date_input(
            "Exit Date",
            value=date(2024, 6, 30),
            help="Simulation end date",
        )

    # Validate dates
    if exit_date <= entry_date:
        st.error("Exit date must be after entry date")

    return {
        "option_type": option_type,
        "strike": strike,
        "entry_date": entry_date,
        "exit_date": exit_date,
    }


def render_price_model_params() -> dict:
    """
    Render price model (Jump-Diffusion) parameters.

    Returns
    -------
    dict
        Price model configuration
    """
    st.subheader("Price Model (Jump-Diffusion)")

    col1, col2 = st.columns(2)

    with col1:
        initial_price = st.number_input(
            "Initial Price ($)",
            min_value=100,
            max_value=100_000,
            value=3000,
            step=100,
            help="Starting asset price",
        )

        volatility = st.slider(
            "Volatility (%)",
            min_value=10,
            max_value=200,
            value=80,
            step=5,
            help="Annualized volatility",
        ) / 100.0

        drift = st.slider(
            "Drift (%)",
            min_value=-50,
            max_value=100,
            value=5,
            step=5,
            help="Annualized drift rate",
        ) / 100.0

    with col2:
        jump_intensity = st.number_input(
            "Jump Intensity",
            min_value=0,
            max_value=100,
            value=10,
            step=1,
            help="Expected jumps per year",
        )

        jump_mean = st.slider(
            "Jump Mean",
            min_value=-0.20,
            max_value=0.20,
            value=-0.02,
            step=0.01,
            format="%.2f",
            help="Mean of log jump size",
        )

        jump_std = st.slider(
            "Jump Std",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            format="%.2f",
            help="Standard deviation of log jump size",
        )

    return {
        "initial_price": initial_price,
        "volatility": volatility,
        "drift": drift,
        "jump_intensity": jump_intensity,
        "jump_mean": jump_mean,
        "jump_std": jump_std,
    }


def render_trader_mix() -> dict:
    """
    Render trader mix configuration.

    Returns
    -------
    dict
        Trader mix configuration
    """
    st.subheader("Trader Mix")

    col1, col2 = st.columns(2)

    with col1:
        noise_weight = st.slider(
            "Noise Traders (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            help="Random trading with no signal",
        )

        momentum_weight = st.slider(
            "Momentum Traders (%)",
            min_value=0,
            max_value=100,
            value=30,
            step=5,
            help="Follow price trends",
        )

        mean_rev_weight = st.slider(
            "Mean-Reversion Traders (%)",
            min_value=0,
            max_value=100,
            value=20,
            step=5,
            help="Bet on price returning to mean",
        )

    with col2:
        trades_per_day = st.number_input(
            "Trades per Day",
            min_value=1,
            max_value=1000,
            value=100,
            step=10,
            help="Average number of trades per day",
        )

        avg_trade_size = st.number_input(
            "Avg Trade Size",
            min_value=1.0,
            max_value=1000.0,
            value=10.0,
            step=1.0,
            help="Average trade size in contracts",
        )

    # Normalize weights
    total = noise_weight + momentum_weight + mean_rev_weight
    if total > 0:
        noise_weight /= total
        momentum_weight /= total
        mean_rev_weight /= total
    else:
        noise_weight = 1.0
        momentum_weight = 0.0
        mean_rev_weight = 0.0

    return {
        "noise_weight": noise_weight,
        "momentum_weight": momentum_weight,
        "mean_reversion_weight": mean_rev_weight,
        "trades_per_day": trades_per_day,
        "avg_trade_size": avg_trade_size,
    }


def render_market_params() -> dict:
    """
    Render market parameters.

    Returns
    -------
    dict
        Market configuration
    """
    st.subheader("Market Parameters")

    col1, col2 = st.columns(2)

    with col1:
        pool_liquidity = st.number_input(
            "Pool Liquidity ($)",
            min_value=100_000,
            max_value=1_000_000_000,
            value=10_000_000,
            step=1_000_000,
            format="%d",
            help="Total liquidity in the pool",
        )

        funding_period = st.selectbox(
            "Funding Period",
            options=[1, 8],  # 1 day or 8 hours
            index=0,
            format_func=lambda x: f"{x} day" if x == 1 else f"{x} hours",
            help="Period between funding settlements",
        )

    with col2:
        trading_fee = st.slider(
            "Trading Fee (%)",
            min_value=0.01,
            max_value=1.0,
            value=0.1,
            step=0.01,
            format="%.2f",
            help="Fee per trade",
        ) / 100.0

        lp_fee_share = st.slider(
            "LP Fee Share (%)",
            min_value=0,
            max_value=100,
            value=20,
            step=5,
            help="LP's share of trading fees",
        ) / 100.0

    return {
        "pool_liquidity": pool_liquidity,
        "funding_period_days": funding_period if funding_period == 1 else funding_period / 24,
        "trading_fee_rate": trading_fee,
        "lp_fee_share": lp_fee_share,
    }


def render_playback_controls(
    max_day: int,
    current_day: int,
) -> Tuple[str, float, int]:
    """
    Render playback controls for simulation visualization.

    Parameters
    ----------
    max_day : int
        Maximum day in simulation
    current_day : int
        Current playback position

    Returns
    -------
    Tuple[str, float, int]
        (action, speed, selected_day)
        action: "play", "pause", "reset", "step_forward", "step_backward", or None
        speed: playback speed multiplier
        selected_day: day selected via slider
    """
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 2])

    action = None

    with col1:
        if st.button("⏮", help="Go to start"):
            action = "reset"

    with col2:
        if st.button("⏪", help="Step backward"):
            action = "step_backward"

    with col3:
        # Toggle play/pause based on session state
        is_playing = st.session_state.get("is_playing", False)
        if is_playing:
            if st.button("⏸", help="Pause"):
                action = "pause"
        else:
            if st.button("▶", help="Play"):
                action = "play"

    with col4:
        if st.button("⏩", help="Step forward"):
            action = "step_forward"

    with col5:
        if st.button("⏭", help="Go to end"):
            action = "end"

    with col6:
        speed = st.selectbox(
            "Speed",
            options=[0.5, 1.0, 2.0, 5.0, 10.0],
            index=1,
            format_func=lambda x: f"{x}x",
            label_visibility="collapsed",
        )

    # Day slider
    selected_day = st.slider(
        "Day",
        min_value=0,
        max_value=max_day,
        value=current_day,
        help="Drag to jump to specific day",
    )

    return action, speed, selected_day
