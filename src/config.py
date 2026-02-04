"""Simulation configuration and parameter dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Optional, List


@dataclass
class PriceModelConfig:
    """Configuration for price evolution model."""
    model_type: Literal["gbm", "jump_diffusion", "historical"] = "jump_diffusion"
    initial_price: float = 3000.0
    drift: float = 0.05  # Annual drift (mu)
    volatility: float = 0.80  # Annual volatility (sigma)
    # Jump parameters (only used for jump_diffusion)
    jump_intensity: float = 10.0  # Expected jumps per year (lambda)
    jump_mean: float = -0.02  # Mean jump size (log)
    jump_std: float = 0.05  # Jump size std (log)
    # Historical data (only used for historical)
    historical_asset: Optional[str] = None  # "BTC", "ETH", "SPY"


@dataclass
class TraderMixConfig:
    """Configuration for trader behavior mix."""
    noise_weight: float = 0.50
    momentum_weight: float = 0.30
    mean_reversion_weight: float = 0.20
    trades_per_day: int = 100
    avg_trade_size: float = 10.0  # In option contracts


@dataclass
class MarketConfig:
    """Configuration for market parameters."""
    pool_liquidity: float = 10_000_000.0
    funding_period_days: float = 1.0
    trading_fee_rate: float = 0.001  # 0.1%
    lp_fee_share: float = 0.20  # 20% of fees go to LP
    alpha_coefficient: float = 0.001  # DPMM price impact coefficient


@dataclass
class LPConfig:
    """Configuration for LP position."""
    capital: float = 1_000_000.0
    hedge_ratio: float = 0.0  # 0 = no hedge, 1 = full delta hedge


@dataclass
class OptionConfig:
    """Configuration for the everlasting option."""
    option_type: Literal["call", "put"] = "call"
    strike: float = 3000.0
    risk_free_rate: float = 0.05


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""
    entry_date: date = field(default_factory=lambda: date(2024, 1, 1))
    exit_date: date = field(default_factory=lambda: date(2024, 6, 30))
    time_step_days: float = 1.0  # 1 = daily, 1/24 = hourly
    random_seed: Optional[int] = None

    price_model: PriceModelConfig = field(default_factory=PriceModelConfig)
    trader_mix: TraderMixConfig = field(default_factory=TraderMixConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    lp: LPConfig = field(default_factory=LPConfig)
    option: OptionConfig = field(default_factory=OptionConfig)

    @property
    def n_days(self) -> int:
        """Total number of simulation days."""
        return (self.exit_date - self.entry_date).days

    @property
    def n_steps(self) -> int:
        """Total number of simulation steps."""
        return int(self.n_days / self.time_step_days)


@dataclass
class SimState:
    """State at a single simulation step."""
    day: int
    date: date
    spot_price: float
    mark_price: float
    theoretical_price: float
    payoff: float

    # Greeks
    delta: float
    gamma: float
    theta: float
    vega: float

    # Positions
    net_position: float  # Net trader position (LP is opposite)
    hedge_position: float  # LP's hedge in spot

    # Funding
    funding_rate: float
    funding_payment: float  # Positive = LP receives

    # PnL components
    cumulative_funding: float
    cumulative_fees: float
    cumulative_hedge_pnl: float
    mtm_pnl: float

    # LP totals
    lp_equity: float
    lp_pnl: float
    daily_pnl: float


@dataclass
class SimulationResult:
    """Complete simulation results."""
    config: SimulationConfig
    history: List[SimState]

    @property
    def final_state(self) -> SimState:
        return self.history[-1]

    @property
    def total_pnl(self) -> float:
        return self.final_state.lp_pnl

    @property
    def total_return(self) -> float:
        return self.total_pnl / self.config.lp.capital

    @property
    def equity_curve(self) -> List[float]:
        return [s.lp_equity for s in self.history]

    @property
    def daily_pnls(self) -> List[float]:
        return [s.daily_pnl for s in self.history]
