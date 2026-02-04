"""Main simulation engine for everlasting option markets."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional, List

import numpy as np

from ..config import SimulationConfig, SimState, SimulationResult
from ..pricing import EverlastingOption
from ..market import DPMMMarket, FundingEngine, PriceOracle
from .price_paths import JumpDiffusionSimulator, GBMSimulator, HistoricalDataLoader
from .traders import TraderMix
from .lp_position import LPPosition


class EverlastingOptionSimulator:
    """
    Main simulation engine for everlasting option markets.

    Simulates LP performance in an oracle-dependent everlasting option
    market with configurable trader behavior and optional delta hedging.

    Parameters
    ----------
    config : SimulationConfig
        Complete simulation configuration
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self._history: List[SimState] = []
        self._current_step = 0
        self._initialized = False

        # Components (initialized in setup)
        self._oracle: Optional[PriceOracle] = None
        self._option: Optional[EverlastingOption] = None
        self._market: Optional[DPMMMarket] = None
        self._funding: Optional[FundingEngine] = None
        self._traders: Optional[TraderMix] = None
        self._lp: Optional[LPPosition] = None

    def setup(self):
        """Initialize all simulation components."""
        cfg = self.config

        # Generate price path based on model type
        price_path = self._generate_price_path(cfg)
        self._oracle = PriceOracle(price_path)

        # Initialize option
        self._option = EverlastingOption(
            strike=cfg.option.strike,
            option_type=cfg.option.option_type,
            funding_period_days=cfg.market.funding_period_days,
        )

        # Initialize market
        self._market = DPMMMarket(
            pool_liquidity=cfg.market.pool_liquidity,
            alpha_coefficient=cfg.market.alpha_coefficient,
            trading_fee_rate=cfg.market.trading_fee_rate,
        )

        # Initialize funding engine
        self._funding = FundingEngine(
            funding_period_days=cfg.market.funding_period_days
        )

        # Initialize traders
        seed = cfg.random_seed + 1 if cfg.random_seed else None
        self._traders = TraderMix(
            noise_weight=cfg.trader_mix.noise_weight,
            momentum_weight=cfg.trader_mix.momentum_weight,
            mean_reversion_weight=cfg.trader_mix.mean_reversion_weight,
            trades_per_day=cfg.trader_mix.trades_per_day,
            avg_trade_size=cfg.trader_mix.avg_trade_size,
            random_seed=seed,
        )

        # Initialize LP
        self._lp = LPPosition(
            capital=cfg.lp.capital,
            hedge_ratio=cfg.lp.hedge_ratio,
        )

        self._history = []
        self._current_step = 0
        self._initialized = True
    
    def _generate_price_path(self, cfg: SimulationConfig) -> np.ndarray:
        """Generate price path based on configured model type."""
        model_type = cfg.price_model.model_type
        
        if model_type == "gbm":
            # Black-Scholes / Geometric Brownian Motion
            simulator = GBMSimulator(
                initial_price=cfg.price_model.initial_price,
                drift=cfg.price_model.drift,
                volatility=cfg.price_model.volatility,
                random_seed=cfg.random_seed,
            )
            return simulator.simulate_daily(cfg.n_days)
        
        elif model_type == "jump_diffusion":
            # Merton Jump-Diffusion
            simulator = JumpDiffusionSimulator(
                initial_price=cfg.price_model.initial_price,
                drift=cfg.price_model.drift,
                volatility=cfg.price_model.volatility,
                jump_intensity=cfg.price_model.jump_intensity,
                jump_mean=cfg.price_model.jump_mean,
                jump_std=cfg.price_model.jump_std,
                random_seed=cfg.random_seed,
            )
            return simulator.simulate_daily(cfg.n_days)
        
        elif model_type == "historical":
            # Historical data - fetch real prices for the simulation date range
            asset = cfg.price_model.historical_asset or "BTC"
            loader = HistoricalDataLoader(asset, random_seed=cfg.random_seed)
            # Don't pass initial_price - use actual historical prices
            return loader.load(
                n_days=cfg.n_days,
                initial_price=None,  # Use real prices, no scaling
                start_date=cfg.entry_date,
                end_date=cfg.exit_date,
            )
        
        else:
            raise ValueError(f"Unknown price model type: {model_type}")

    def run(self) -> SimulationResult:
        """
        Run the complete simulation.

        Returns
        -------
        SimulationResult
            Complete simulation results with history
        """
        if not self._initialized:
            self.setup()

        # Run all steps
        for _ in range(self.config.n_steps + 1):
            self.run_step()

        return SimulationResult(
            config=self.config,
            history=self._history,
        )

    def run_step(self) -> SimState:
        """
        Run a single simulation step.

        Returns
        -------
        SimState
            State after this step
        """
        if not self._initialized:
            self.setup()

        step = self._current_step
        cfg = self.config

        # Get current prices
        spot = self._oracle.get_price(step)
        price_history = self._oracle.get_history(step)

        # Calculate option pricing
        theo_price = self._option.price(
            spot,
            cfg.price_model.volatility,
            cfg.option.risk_free_rate,
        )
        greeks = self._option.greeks(
            spot,
            cfg.price_model.volatility,
            cfg.option.risk_free_rate,
        )
        payoff = self._option.payoff(spot)

        # Get mark price from DPMM
        mark_price = self._market.mark_price(theo_price)

        # Calculate funding
        funding_rate = self._option.funding_rate(mark_price, spot)

        # Previous values for daily PnL
        prev_pnl = self._lp.total_pnl if step > 0 else 0.0

        # Process trader activity
        daily_fees = 0.0
        if step > 0:
            trades = self._traders.generate_trades(
                price_history,
                self._market.net_position,
            )

            for trade in trades:
                _, fee, _ = self._market.execute_trade(
                    trade.size,
                    theo_price,
                    trade.is_long,
                )
                daily_fees += fee

            # Update LP position
            self._lp.update_position(self._market.net_position)
            self._lp.apply_trading_fee(daily_fees, cfg.market.lp_fee_share)

            # Calculate and apply funding
            funding_payment = self._funding.get_lp_funding(
                self._market.net_position,
                funding_rate,
                cfg.time_step_days,
            )
            self._lp.apply_funding(funding_payment)

            # Rebalance hedge
            self._lp.rebalance_hedge(spot, greeks["delta"])

        # Mark to market
        self._lp.mark_to_market(mark_price, spot)

        # Calculate current date
        current_date = cfg.entry_date + timedelta(days=step)

        # Create state
        pnl_breakdown = self._lp.get_pnl_breakdown()
        current_pnl = self._lp.total_pnl
        daily_pnl = current_pnl - prev_pnl

        state = SimState(
            day=step,
            date=current_date,
            spot_price=spot,
            mark_price=mark_price,
            theoretical_price=theo_price,
            payoff=payoff,
            delta=greeks["delta"],
            gamma=greeks["gamma"],
            theta=greeks["theta"],
            vega=greeks["vega"],
            net_position=self._market.net_position,
            hedge_position=self._lp.hedge_position,
            funding_rate=funding_rate,
            funding_payment=pnl_breakdown.funding - (
                self._history[-1].cumulative_funding if self._history else 0
            ),
            cumulative_funding=pnl_breakdown.funding,
            cumulative_fees=pnl_breakdown.trading_fees,
            cumulative_hedge_pnl=pnl_breakdown.hedge,
            mtm_pnl=pnl_breakdown.mtm,
            lp_equity=self._lp.current_equity,
            lp_pnl=current_pnl,
            daily_pnl=daily_pnl,
        )

        self._history.append(state)
        self._current_step += 1

        return state

    def get_history(self) -> List[SimState]:
        """Get full simulation history."""
        return self._history.copy()

    def get_state(self, step: int) -> Optional[SimState]:
        """Get state at a specific step."""
        if 0 <= step < len(self._history):
            return self._history[step]
        return None

    def reset(self):
        """Reset simulation to initial state."""
        self._history = []
        self._current_step = 0
        self._initialized = False

    @property
    def is_complete(self) -> bool:
        """Check if simulation is complete."""
        return self._current_step > self.config.n_steps
