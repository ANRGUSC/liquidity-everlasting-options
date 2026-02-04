"""Simulation reporting and visualization."""

from typing import Optional

import numpy as np
import pandas as pd

from ..config import SimulationResult, SimState
from .metrics import PerformanceMetrics


class SimulationReport:
    """
    Generate reports and summaries from simulation results.

    Parameters
    ----------
    result : SimulationResult
        Completed simulation result
    """

    def __init__(self, result: SimulationResult):
        self.result = result
        self._metrics: Optional[dict] = None

    @property
    def history_df(self) -> pd.DataFrame:
        """Convert history to DataFrame."""
        records = []
        for state in self.result.history:
            records.append({
                "day": state.day,
                "date": state.date,
                "spot_price": state.spot_price,
                "mark_price": state.mark_price,
                "theoretical_price": state.theoretical_price,
                "payoff": state.payoff,
                "delta": state.delta,
                "gamma": state.gamma,
                "theta": state.theta,
                "vega": state.vega,
                "net_position": state.net_position,
                "hedge_position": state.hedge_position,
                "funding_rate": state.funding_rate,
                "funding_payment": state.funding_payment,
                "cumulative_funding": state.cumulative_funding,
                "cumulative_fees": state.cumulative_fees,
                "cumulative_hedge_pnl": state.cumulative_hedge_pnl,
                "mtm_pnl": state.mtm_pnl,
                "lp_equity": state.lp_equity,
                "lp_pnl": state.lp_pnl,
                "daily_pnl": state.daily_pnl,
            })
        return pd.DataFrame(records)

    @property
    def metrics(self) -> dict:
        """Calculate and cache performance metrics."""
        if self._metrics is None:
            self._metrics = PerformanceMetrics.calculate_all(
                equity_curve=self.result.equity_curve,
                daily_pnl=self.result.daily_pnls,
                initial_capital=self.result.config.lp.capital,
            )
        return self._metrics

    def get_pnl_attribution(self) -> dict:
        """
        Get PnL attribution breakdown.

        Returns
        -------
        dict
            Dictionary with funding, fees, hedge, and total PnL
        """
        final = self.result.final_state
        return {
            "funding_pnl": final.cumulative_funding,
            "fee_income": final.cumulative_fees,
            "hedge_pnl": final.cumulative_hedge_pnl,
            "total_pnl": final.lp_pnl,
        }

    def get_position_stats(self) -> dict:
        """
        Get position-related statistics.

        Returns
        -------
        dict
            Position statistics
        """
        df = self.history_df

        net_positions = df["net_position"].values
        hedge_positions = df["hedge_position"].values
        funding_payments = df["funding_payment"].values

        return {
            "avg_net_position": np.mean(np.abs(net_positions)),
            "max_net_position": np.max(np.abs(net_positions)),
            "avg_hedge_position": np.mean(np.abs(hedge_positions)),
            "pct_receiving_funding": np.mean(funding_payments > 0) * 100,
            "pct_paying_funding": np.mean(funding_payments < 0) * 100,
            "total_funding_received": np.sum(funding_payments[funding_payments > 0]),
            "total_funding_paid": -np.sum(funding_payments[funding_payments < 0]),
        }

    def get_summary(self) -> dict:
        """
        Get complete summary report.

        Returns
        -------
        dict
            Summary containing metrics, attribution, and position stats
        """
        return {
            "config": {
                "entry_date": str(self.result.config.entry_date),
                "exit_date": str(self.result.config.exit_date),
                "n_days": self.result.config.n_days,
                "initial_capital": self.result.config.lp.capital,
                "option_type": self.result.config.option.option_type,
                "strike": self.result.config.option.strike,
                "volatility": self.result.config.price_model.volatility,
                "hedge_ratio": self.result.config.lp.hedge_ratio,
            },
            "performance": self.metrics,
            "pnl_attribution": self.get_pnl_attribution(),
            "position_stats": self.get_position_stats(),
        }

    def to_csv(self, filepath: str):
        """Export history to CSV."""
        self.history_df.to_csv(filepath, index=False)

    def print_summary(self):
        """Print formatted summary to console."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("EVERLASTING OPTION SIMULATION RESULTS")
        print("=" * 60)

        print("\nConfiguration:")
        print(f"  Period: {summary['config']['entry_date']} to {summary['config']['exit_date']}")
        print(f"  Days: {summary['config']['n_days']}")
        print(f"  Option: {summary['config']['option_type'].upper()} @ {summary['config']['strike']}")
        print(f"  Volatility: {summary['config']['volatility']*100:.0f}%")
        print(f"  Capital: ${summary['config']['initial_capital']:,.0f}")
        print(f"  Hedge Ratio: {summary['config']['hedge_ratio']*100:.0f}%")

        print("\nPerformance Metrics:")
        perf = summary["performance"]
        print(f"  Total Return: {perf['total_return_pct']:.2f}%")
        print(f"  Total PnL: ${perf['total_pnl']:,.0f}")
        print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
        print(f"  Sortino Ratio: {perf['sortino_ratio']:.2f}")
        print(f"  Max Drawdown: {perf['max_drawdown_pct']:.2f}%")
        print(f"  Win Rate: {perf['win_rate_pct']:.1f}%")
        print(f"  Profit Factor: {perf['profit_factor']:.2f}")

        print("\nPnL Attribution:")
        attr = summary["pnl_attribution"]
        print(f"  Funding Income: ${attr['funding_pnl']:,.0f}")
        print(f"  Fee Income: ${attr['fee_income']:,.0f}")
        print(f"  Hedge PnL: ${attr['hedge_pnl']:,.0f}")

        print("\nPosition Statistics:")
        pos = summary["position_stats"]
        print(f"  Avg Net Position: {pos['avg_net_position']:.1f}")
        print(f"  Max Net Position: {pos['max_net_position']:.1f}")
        print(f"  % Time Receiving Funding: {pos['pct_receiving_funding']:.1f}%")

        print("\n" + "=" * 60)
