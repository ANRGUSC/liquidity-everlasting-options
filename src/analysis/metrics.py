"""Performance metrics calculation."""

import numpy as np
from typing import Sequence


class PerformanceMetrics:
    """
    Performance metrics calculator for simulation results.

    All methods are static and can be used independently.
    """

    @staticmethod
    def sharpe_ratio(
        returns: Sequence[float],
        risk_free_rate: float = 0.0,
        periods_per_year: float = 252,
    ) -> float:
        """
        Calculate annualized Sharpe ratio.

        Parameters
        ----------
        returns : Sequence[float]
            Periodic returns
        risk_free_rate : float
            Annualized risk-free rate
        periods_per_year : float
            Number of periods per year (252 for daily)

        Returns
        -------
        float
            Annualized Sharpe ratio
        """
        returns = np.array(returns)
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate / periods_per_year
        mean_return = np.mean(excess_returns)
        std_return = np.std(excess_returns, ddof=1)

        if std_return == 0:
            return 0.0

        return float(mean_return / std_return * np.sqrt(periods_per_year))

    @staticmethod
    def sortino_ratio(
        returns: Sequence[float],
        risk_free_rate: float = 0.0,
        periods_per_year: float = 252,
    ) -> float:
        """
        Calculate annualized Sortino ratio.

        Parameters
        ----------
        returns : Sequence[float]
            Periodic returns
        risk_free_rate : float
            Annualized risk-free rate
        periods_per_year : float
            Number of periods per year

        Returns
        -------
        float
            Annualized Sortino ratio
        """
        returns = np.array(returns)
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - risk_free_rate / periods_per_year
        mean_return = np.mean(excess_returns)

        # Downside deviation (only negative returns)
        negative_returns = excess_returns[excess_returns < 0]
        if len(negative_returns) == 0:
            return float('inf') if mean_return > 0 else 0.0

        downside_std = np.sqrt(np.mean(negative_returns ** 2))

        if downside_std == 0:
            return 0.0

        return float(mean_return / downside_std * np.sqrt(periods_per_year))

    @staticmethod
    def max_drawdown(equity_curve: Sequence[float]) -> float:
        """
        Calculate maximum drawdown.

        Parameters
        ----------
        equity_curve : Sequence[float]
            Equity values over time

        Returns
        -------
        float
            Maximum drawdown as a positive fraction (e.g., 0.15 = 15%)
        """
        equity = np.array(equity_curve)
        if len(equity) < 2:
            return 0.0

        # Running maximum
        running_max = np.maximum.accumulate(equity)

        # Drawdown at each point
        drawdown = (running_max - equity) / running_max

        return float(np.max(drawdown))

    @staticmethod
    def max_drawdown_duration(equity_curve: Sequence[float]) -> int:
        """
        Calculate maximum drawdown duration in periods.

        Parameters
        ----------
        equity_curve : Sequence[float]
            Equity values over time

        Returns
        -------
        int
            Maximum number of periods in drawdown
        """
        equity = np.array(equity_curve)
        if len(equity) < 2:
            return 0

        running_max = np.maximum.accumulate(equity)
        in_drawdown = equity < running_max

        max_duration = 0
        current_duration = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    @staticmethod
    def win_rate(daily_pnl: Sequence[float]) -> float:
        """
        Calculate win rate (fraction of positive days).

        Parameters
        ----------
        daily_pnl : Sequence[float]
            Daily PnL values

        Returns
        -------
        float
            Win rate as a fraction
        """
        pnl = np.array(daily_pnl)
        if len(pnl) == 0:
            return 0.0

        return float(np.mean(pnl > 0))

    @staticmethod
    def profit_factor(daily_pnl: Sequence[float]) -> float:
        """
        Calculate profit factor (gross profits / gross losses).

        Parameters
        ----------
        daily_pnl : Sequence[float]
            Daily PnL values

        Returns
        -------
        float
            Profit factor (>1 is profitable)
        """
        pnl = np.array(daily_pnl)

        gross_profit = np.sum(pnl[pnl > 0])
        gross_loss = -np.sum(pnl[pnl < 0])

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 1.0

        return float(gross_profit / gross_loss)

    @staticmethod
    def var(
        returns: Sequence[float],
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate Value at Risk.

        Parameters
        ----------
        returns : Sequence[float]
            Return values
        confidence : float
            Confidence level (e.g., 0.95)

        Returns
        -------
        float
            VaR as a positive number (worst expected loss)
        """
        returns = np.array(returns)
        if len(returns) < 2:
            return 0.0

        percentile = (1 - confidence) * 100
        var = -np.percentile(returns, percentile)
        return float(max(var, 0))

    @staticmethod
    def cvar(
        returns: Sequence[float],
        confidence: float = 0.95,
    ) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall).

        Parameters
        ----------
        returns : Sequence[float]
            Return values
        confidence : float
            Confidence level

        Returns
        -------
        float
            CVaR as a positive number
        """
        returns = np.array(returns)
        if len(returns) < 2:
            return 0.0

        var = PerformanceMetrics.var(returns, confidence)
        tail_returns = returns[returns < -var]

        if len(tail_returns) == 0:
            return var

        return float(-np.mean(tail_returns))

    @staticmethod
    def calmar_ratio(
        returns: Sequence[float],
        equity_curve: Sequence[float],
        periods_per_year: float = 252,
    ) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).

        Parameters
        ----------
        returns : Sequence[float]
            Periodic returns
        equity_curve : Sequence[float]
            Equity curve
        periods_per_year : float
            Periods per year

        Returns
        -------
        float
            Calmar ratio
        """
        max_dd = PerformanceMetrics.max_drawdown(equity_curve)
        if max_dd == 0:
            return 0.0

        returns = np.array(returns)
        total_return = np.prod(1 + returns) - 1
        n_years = len(returns) / periods_per_year
        annualized_return = (1 + total_return) ** (1 / max(n_years, 1e-6)) - 1

        return float(annualized_return / max_dd)

    @staticmethod
    def calculate_all(
        equity_curve: Sequence[float],
        daily_pnl: Sequence[float],
        initial_capital: float,
        risk_free_rate: float = 0.0,
    ) -> dict:
        """
        Calculate all performance metrics.

        Parameters
        ----------
        equity_curve : Sequence[float]
            Equity values over time
        daily_pnl : Sequence[float]
            Daily PnL values
        initial_capital : float
            Initial capital
        risk_free_rate : float
            Annual risk-free rate

        Returns
        -------
        dict
            Dictionary of all metrics
        """
        equity = np.array(equity_curve)
        pnl = np.array(daily_pnl)

        # Calculate returns from equity curve
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]

        total_pnl = equity[-1] - initial_capital if len(equity) > 0 else 0
        total_return = total_pnl / initial_capital if initial_capital > 0 else 0

        return {
            "total_pnl": total_pnl,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "sharpe_ratio": PerformanceMetrics.sharpe_ratio(returns, risk_free_rate),
            "sortino_ratio": PerformanceMetrics.sortino_ratio(returns, risk_free_rate),
            "max_drawdown": PerformanceMetrics.max_drawdown(equity),
            "max_drawdown_pct": PerformanceMetrics.max_drawdown(equity) * 100,
            "max_drawdown_duration": PerformanceMetrics.max_drawdown_duration(equity),
            "win_rate": PerformanceMetrics.win_rate(pnl),
            "win_rate_pct": PerformanceMetrics.win_rate(pnl) * 100,
            "profit_factor": PerformanceMetrics.profit_factor(pnl),
            "var_95": PerformanceMetrics.var(returns, 0.95),
            "var_99": PerformanceMetrics.var(returns, 0.99),
            "cvar_95": PerformanceMetrics.cvar(returns, 0.95),
            "calmar_ratio": PerformanceMetrics.calmar_ratio(returns, equity),
        }
