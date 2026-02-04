"""Everlasting Option Market Simulator - Single Page Animated Dashboard."""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

from src.config import (
    SimulationConfig,
    PriceModelConfig,
    TraderMixConfig,
    MarketConfig,
    LPConfig,
    OptionConfig,
    SimulationResult,
)
from src.simulation import EverlastingOptionSimulator
from src.analysis.reporting import SimulationReport

# Page config
st.set_page_config(
    page_title="Everlasting Option Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for animations and styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }

    /* Animated gradient header */
    .header-gradient {
        background: linear-gradient(-45deg, #1a1a2e, #16213e, #0f3460, #533483);
        background-size: 400% 400%;
        animation: gradient 8s ease infinite;
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Pulse animation for live indicator */
    .pulse {
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    /* Smooth transitions */
    .stButton button {
        transition: all 0.3s ease;
        border-radius: 25px;
        font-weight: bold;
    }

    .stButton button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(0,0,0,0.3);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    /* Progress bar animation */
    .stProgress > div > div {
        background: linear-gradient(90deg, #00d4ff, #7b2cbf, #00d4ff);
        background-size: 200% 100%;
        animation: progress-animation 2s linear infinite;
    }

    @keyframes progress-animation {
        0% { background-position: 100% 0; }
        100% { background-position: -100% 0; }
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-gradient">
    <h1 style="color: white; margin: 0; text-align: center;">📈 Everlasting Option Market Simulator</h1>
    <p style="color: #aaa; text-align: center; margin: 10px 0 0 0;">Backtest LP strategies on perpetual options using real or simulated price data</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for parameters
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # LP Settings
    st.markdown("### 💰 LP Settings")
    capital = st.number_input("Initial Capital ($)", 100_000, 10_000_000, 1_000_000, 100_000)

    hedge_mode = st.radio(
        "Hedging Strategy",
        ["None", "Delta Hedging"],
        horizontal=True,
        help="None: no hedging, Delta Hedging: hedge option delta with spot"
    )

    # Show relevant controls based on hedge mode
    if hedge_mode == "Delta Hedging":
        hedge_ratio_pct = st.slider("Hedge Ratio (%)", 0, 100, 100, 10,
                               help="0% = no hedge, 100% = full delta hedge")
        hedge_ratio = hedge_ratio_pct / 100.0
    else:
        hedge_ratio = 0.0

    st.markdown("---")

    # Option Settings
    st.markdown("### 📊 Option Settings")
    option_type = st.radio("Type", ["call", "put"], horizontal=True)
    strike = st.number_input("Strike Price ($)", 100, 10000, 3000, 100)

    col1, col2 = st.columns(2)
    with col1:
        entry_date = st.date_input("Entry", date(2024, 1, 1))
    with col2:
        exit_date = st.date_input("Exit", date(2024, 3, 31))

    st.markdown("---")

    # Price Model
    st.markdown("### 📉 Price Model")
    price_model_type = st.radio(
        "Price Evolution",
        ["Black-Scholes", "Jump Diffusion", "Historical Data"],
        horizontal=False,
        help="BS: Geometric Brownian Motion, Jump: Merton jump-diffusion, Historical: Real market data"
    )

    # Historical data options (show first if selected)
    historical_asset = None
    if price_model_type == "Historical Data":
        historical_asset = st.selectbox(
            "Asset",
            ["BTC", "ETH", "SPY"],
            help="Select asset to fetch real historical prices"
        )
        st.caption("📊 Fetches real prices from CoinGecko (BTC/ETH) or Yahoo Finance (SPY)")
        st.info(f"Will fetch {historical_asset} prices from {entry_date} to {exit_date}")

        # Use realistic volatility estimates for option pricing
        default_vols = {"BTC": 80, "ETH": 90, "SPY": 20}
        volatility = st.slider(
            "Implied Volatility (for option pricing)",
            10, 150, default_vols.get(historical_asset, 80), 5,
            format="%d%%",
            help="Used for Black-Scholes option pricing calculations"
        ) / 100

        # Use placeholder values - actual prices come from API
        initial_price = 0  # Will be ignored, real price used
        drift = 0.0
        jump_intensity = 0
        jump_mean = 0.0
        jump_std = 0.0
    else:
        # Show price model parameters for synthetic models
        initial_price = st.number_input("Initial Price ($)", 100, 10000, 3000, 100)
        volatility = st.slider("Volatility", 20, 150, 80, 5, format="%d%%") / 100
        drift = st.slider("Drift", -20, 50, 5, 5, format="%d%%") / 100

        # Jump parameters (only show for Jump Diffusion)
        if price_model_type == "Jump Diffusion":
            with st.expander("Jump Parameters", expanded=True):
                jump_intensity = st.slider("Jump Intensity", 0, 50, 10)
                jump_mean = st.slider("Jump Mean", -0.10, 0.10, -0.02, 0.01)
                jump_std = st.slider("Jump Std", 0.01, 0.10, 0.05, 0.01)
        else:
            jump_intensity = 0
            jump_mean = 0.0
            jump_std = 0.0

    st.markdown("---")

    # Trader Mix (linked sliders that sum to 100%)
    st.markdown("### 👥 Trader Mix")

    # Initialize session state for trader mix
    if "noise_pct" not in st.session_state:
        st.session_state.noise_pct = 50
    if "momentum_pct" not in st.session_state:
        st.session_state.momentum_pct = 30
    if "meanrev_pct" not in st.session_state:
        st.session_state.meanrev_pct = 20

    def update_trader_mix(changed):
        """Adjust other sliders when one changes to maintain sum of 100."""
        total = st.session_state.noise_pct + st.session_state.momentum_pct + st.session_state.meanrev_pct
        if total == 100:
            return

        diff = total - 100
        others = [k for k in ["noise_pct", "momentum_pct", "meanrev_pct"] if k != changed]
        other_total = sum(st.session_state[k] for k in others)

        if other_total > 0:
            for k in others:
                ratio = st.session_state[k] / other_total
                adjustment = int(diff * ratio)
                st.session_state[k] = max(0, st.session_state[k] - adjustment)

        # Fix rounding errors
        total = st.session_state.noise_pct + st.session_state.momentum_pct + st.session_state.meanrev_pct
        if total != 100:
            diff = total - 100
            for k in others:
                if st.session_state[k] >= diff:
                    st.session_state[k] -= diff
                    break

    noise_pct = st.slider("Noise Traders", 0, 100,
                          st.session_state.noise_pct, 5, format="%d%%",
                          key="noise_pct", on_change=update_trader_mix, args=("noise_pct",))
    momentum_pct = st.slider("Momentum Traders", 0, 100,
                             st.session_state.momentum_pct, 5, format="%d%%",
                             key="momentum_pct", on_change=update_trader_mix, args=("momentum_pct",))
    meanrev_pct = st.slider("Mean-Rev Traders", 0, 100,
                            st.session_state.meanrev_pct, 5, format="%d%%",
                            key="meanrev_pct", on_change=update_trader_mix, args=("meanrev_pct",))

    # Show total
    total_pct = noise_pct + momentum_pct + meanrev_pct
    if total_pct != 100:
        st.caption(f"⚠️ Total: {total_pct}% (adjusting...)")
    else:
        st.caption(f"✓ Total: {total_pct}%")

    trades_per_day = st.slider("Trades/Day", 10, 500, 100, 10)

    st.markdown("---")

    # Advanced
    with st.expander("🔧 Advanced"):
        pool_liquidity = st.number_input("Pool Liquidity ($)", 1_000_000, 100_000_000, 10_000_000, 1_000_000)
        trading_fee = st.slider("Trading Fee", 0.01, 0.50, 0.10, 0.01, format="%.2f%%") / 100
        random_seed = st.number_input("Random Seed", 0, 99999, 42)

# Main content area
main_col1, main_col2 = st.columns([2, 1])

# Initialize session state
if "sim_running" not in st.session_state:
    st.session_state.sim_running = False
if "sim_result" not in st.session_state:
    st.session_state.sim_result = None
if "current_step" not in st.session_state:
    st.session_state.current_step = 0

# Run button
with main_col2:
    st.markdown("### 🎮 Controls")

    run_col1, run_col2 = st.columns(2)
    with run_col1:
        run_button = st.button("🚀 Run", type="primary", use_container_width=True)
    with run_col2:
        reset_button = st.button("🔄 Reset", use_container_width=True)

    if reset_button:
        st.session_state.sim_result = None
        st.session_state.current_step = 0
        st.rerun()

    # Animation speed
    speed = st.select_slider("Animation Speed", ["Slow", "Normal", "Fast", "Instant"], value="Fast")
    speed_map = {"Slow": 0.1, "Normal": 0.05, "Fast": 0.01, "Instant": 0}

# Build config - convert percentages to ratios
noise_w = noise_pct / 100.0
momentum_w = momentum_pct / 100.0
meanrev_w = meanrev_pct / 100.0

# Map price model type to config value
price_model_map = {
    "Black-Scholes": "gbm",
    "Jump Diffusion": "jump_diffusion",
    "Historical Data": "historical"
}

config = SimulationConfig(
    entry_date=entry_date,
    exit_date=exit_date,
    random_seed=random_seed if random_seed > 0 else None,
    price_model=PriceModelConfig(
        model_type=price_model_map[price_model_type],
        initial_price=initial_price,
        drift=drift,
        volatility=volatility,
        jump_intensity=jump_intensity,
        jump_mean=jump_mean,
        jump_std=jump_std,
        historical_asset=historical_asset if price_model_type == "Historical Data" else None,
    ),
    trader_mix=TraderMixConfig(
        noise_weight=noise_w,
        momentum_weight=momentum_w,
        mean_reversion_weight=meanrev_w,
        trades_per_day=trades_per_day,
    ),
    market=MarketConfig(
        pool_liquidity=pool_liquidity,
        trading_fee_rate=trading_fee,
    ),
    lp=LPConfig(
        capital=capital,
        hedge_ratio=hedge_ratio,
    ),
    option=OptionConfig(option_type=option_type, strike=strike),
)

# Chart placeholder
with main_col1:
    chart_placeholder = st.empty()
    progress_placeholder = st.empty()

# Metrics placeholder
with main_col2:
    metrics_placeholder = st.empty()
    breakdown_placeholder = st.empty()

def create_animated_chart(history, current_idx, config):
    """Create animated dual-axis chart."""
    if not history:
        return go.Figure()

    df = pd.DataFrame([{
        'day': s.day,
        'date': s.date,
        'price': s.spot_price,
        'equity': s.lp_equity,
        'funding': s.cumulative_funding,
        'fees': s.cumulative_fees,
        'hedge': s.cumulative_hedge_pnl,
    } for s in history[:current_idx+1]])

    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.6, 0.4],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("", "")
    )

    # Price trace
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['price'],
        name='Asset Price',
        line=dict(color='#00d4ff', width=3),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 255, 0.1)',
    ), row=1, col=1)

    # Strike line
    fig.add_hline(y=config.option.strike, line_dash="dash",
                  line_color="#ff6b6b", row=1, col=1,
                  annotation_text=f"Strike: ${config.option.strike:,}")

    # Current price marker
    if len(df) > 0:
        fig.add_trace(go.Scatter(
            x=[df['date'].iloc[-1]], y=[df['price'].iloc[-1]],
            mode='markers',
            marker=dict(size=15, color='#00d4ff',
                       line=dict(width=3, color='white'),
                       symbol='circle'),
            showlegend=False,
        ), row=1, col=1)

    # Equity curve
    equity_color = '#4ade80' if df['equity'].iloc[-1] >= config.lp.capital else '#f87171'
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['equity'],
        name='LP Equity',
        line=dict(color=equity_color, width=3),
        fill='tozeroy',
        fillcolor=f'rgba({74 if equity_color=="#4ade80" else 248}, {222 if equity_color=="#4ade80" else 113}, {128 if equity_color=="#4ade80" else 113}, 0.1)',
    ), row=2, col=1)

    # Initial capital line
    fig.add_hline(y=config.lp.capital, line_dash="dot",
                  line_color="#888", row=2, col=1,
                  annotation_text=f"Initial: ${config.lp.capital:,}")

    # Current equity marker
    if len(df) > 0:
        fig.add_trace(go.Scatter(
            x=[df['date'].iloc[-1]], y=[df['equity'].iloc[-1]],
            mode='markers',
            marker=dict(size=15, color=equity_color,
                       line=dict(width=3, color='white'),
                       symbol='circle'),
            showlegend=False,
        ), row=2, col=1)

    # Layout
    fig.update_layout(
        height=500,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=20, t=30, b=30),
        font=dict(color='#888'),
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=2, col=1)

    return fig

def render_metrics(state, config):
    """Render live metrics."""
    if state is None:
        return

    pnl = state.lp_pnl
    ret = pnl / config.lp.capital * 100

    col1, col2 = st.columns(2)
    with col1:
        st.metric("💵 Spot Price", f"${state.spot_price:,.0f}")
        st.metric("📈 Mark Price", f"${state.mark_price:,.2f}")
        st.metric("📊 Delta", f"{state.delta:.3f}")
    with col2:
        delta_str = f"{ret:+.2f}%" if pnl != 0 else None
        st.metric("💰 Total PnL", f"${pnl:,.0f}", delta_str)
        st.metric("📍 Net Position", f"{state.net_position:,.0f}")
        st.metric("⏱️ Day", f"{state.day} / {config.n_days}")

def render_breakdown(state):
    """Render PnL breakdown."""
    if state is None:
        return

    st.markdown("#### 📊 PnL Breakdown")

    components = {
        "Funding": state.cumulative_funding,
        "Fees": state.cumulative_fees,
        "Hedge": state.cumulative_hedge_pnl,
    }

    # Mini bar chart
    fig = go.Figure()
    colors = ['#3b82f6', '#22c55e', '#a855f7']

    for i, (name, value) in enumerate(components.items()):
        fig.add_trace(go.Bar(
            x=[value], y=[name],
            orientation='h',
            marker_color=colors[i],
            text=f"${value:,.0f}",
            textposition='auto',
        ))

    fig.update_layout(
        height=150,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=20, t=10, b=10),
        font=dict(color='#888', size=12),
        xaxis=dict(showgrid=False, zeroline=True, zerolinecolor='#444'),
        yaxis=dict(showgrid=False),
    )

    st.plotly_chart(fig, use_container_width=True)

# Run simulation
if run_button and exit_date > entry_date:
    simulator = EverlastingOptionSimulator(config)
    simulator.setup()

    history = []
    delay = speed_map[speed]

    for i in range(config.n_steps + 1):
        state = simulator.run_step()
        history.append(state)

        # Update chart
        fig = create_animated_chart(history, i, config)
        chart_placeholder.plotly_chart(fig, use_container_width=True)

        # Update metrics
        with metrics_placeholder.container():
            render_metrics(state, config)

        with breakdown_placeholder.container():
            render_breakdown(state)

        # Progress
        progress = (i + 1) / (config.n_steps + 1)
        progress_placeholder.progress(progress, text=f"Day {i} of {config.n_days}")

        if delay > 0:
            time.sleep(delay)

    # Store result
    st.session_state.sim_result = SimulationResult(config=config, history=history)
    progress_placeholder.success(f"Simulation complete: {config.n_days} days")

elif st.session_state.sim_result is not None:
    # Show existing result
    result = st.session_state.sim_result
    fig = create_animated_chart(result.history, len(result.history)-1, result.config)
    chart_placeholder.plotly_chart(fig, use_container_width=True)

    with metrics_placeholder.container():
        render_metrics(result.final_state, result.config)

    with breakdown_placeholder.container():
        render_breakdown(result.final_state)

else:
    # Show placeholder
    chart_placeholder.markdown("""
    <div style="
        height: 500px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, rgba(30,58,95,0.3) 0%, rgba(45,90,135,0.3) 100%);
        border-radius: 15px;
        border: 2px dashed #444;
    ">
        <div style="text-align: center; color: #666;">
            <h2>📈 Configure parameters and click Run</h2>
            <p>The simulation will animate in real-time</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with metrics_placeholder.container():
        st.markdown("### 📊 Live Metrics")
        st.info("Metrics will appear here during simulation")

# Final analysis section (only show after simulation)
if st.session_state.sim_result is not None:
    st.markdown("---")
    st.markdown("## 📈 Final Analysis")

    result = st.session_state.sim_result
    report = SimulationReport(result)
    metrics = report.metrics

    # Summary metrics row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Return", f"{metrics['total_return_pct']:.2f}%")
    with m2:
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    with m3:
        st.metric("Max Drawdown", f"{metrics['max_drawdown_pct']:.1f}%")
    with m4:
        st.metric("Win Rate", f"{metrics['win_rate_pct']:.0f}%")
    with m5:
        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")

    # Detailed charts
    analysis_col1, analysis_col2 = st.columns(2)

    with analysis_col1:
        st.markdown("#### PnL Attribution Over Time")
        df = report.history_df

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['cumulative_funding'],
            name='Funding', stackgroup='one',
            line=dict(color='#3b82f6'),
        ))
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['cumulative_fees'],
            name='Fees', stackgroup='one',
            line=dict(color='#22c55e'),
        ))
        fig.add_trace(go.Scatter(
            x=df['date'], y=df['cumulative_hedge_pnl'],
            name='Hedge', stackgroup='one',
            line=dict(color='#a855f7'),
        ))

        fig.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=20, t=20, b=40),
            legend=dict(orientation='h', y=-0.2),
            font=dict(color='#888'),
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')

        st.plotly_chart(fig, use_container_width=True)

    with analysis_col2:
        st.markdown("#### Daily PnL Distribution")
        daily_pnl = df['daily_pnl'].dropna().values

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=daily_pnl,
            nbinsx=30,
            marker_color='#3b82f6',
            opacity=0.7,
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="#f87171")
        fig.add_vline(x=np.mean(daily_pnl), line_dash="dot", line_color="#4ade80",
                     annotation_text=f"Mean: ${np.mean(daily_pnl):,.0f}")

        fig.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=40, r=20, t=20, b=40),
            font=dict(color='#888'),
            xaxis_title="Daily PnL ($)",
            yaxis_title="Frequency",
        )
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')

        st.plotly_chart(fig, use_container_width=True)

    # Export
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        csv = report.history_df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "simulation.csv", "text/csv")
    with col2:
        import json
        summary = json.dumps(report.get_summary(), indent=2, default=str)
        st.download_button("📥 Download JSON", summary, "summary.json", "application/json")
