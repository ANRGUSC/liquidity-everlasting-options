"""Configurable trader behavior simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np


@dataclass
class Trade:
    """Represents a single trade."""
    size: float  # Positive = long, negative = short
    is_long: bool
    trader_type: str  # "noise", "momentum", "mean_reversion"


class TraderMix:
    """
    Configurable mix of trader behaviors.

    Traders are categorized into three types:
    - Noise traders: Random trading with no signal
    - Momentum traders: Follow price trends
    - Mean-reversion traders: Bet on price returning to mean

    Parameters
    ----------
    noise_weight : float
        Proportion of noise traders (0-1)
    momentum_weight : float
        Proportion of momentum traders (0-1)
    mean_reversion_weight : float
        Proportion of mean-reversion traders (0-1)
    trades_per_day : int
        Average number of trades per day
    avg_trade_size : float
        Average trade size in contracts
    lookback_days : int
        Days of price history for signals
    random_seed : int, optional
        Random seed
    """

    def __init__(
        self,
        noise_weight: float = 0.5,
        momentum_weight: float = 0.3,
        mean_reversion_weight: float = 0.2,
        trades_per_day: int = 100,
        avg_trade_size: float = 10.0,
        lookback_days: int = 5,
        random_seed: Optional[int] = None,
    ):
        # Normalize weights
        total = noise_weight + momentum_weight + mean_reversion_weight
        if total <= 0:
            total = 1.0
            noise_weight = 1.0

        self.noise_weight = noise_weight / total
        self.momentum_weight = momentum_weight / total
        self.mean_reversion_weight = mean_reversion_weight / total

        self.trades_per_day = trades_per_day
        self.avg_trade_size = avg_trade_size
        self.lookback_days = lookback_days

        self._rng = np.random.default_rng(random_seed)

    def generate_trades(
        self,
        price_history: np.ndarray,
        current_net_position: float = 0.0,
        position_limit: float = 10000.0,
    ) -> List[Trade]:
        """
        Generate trades for the current period.

        Parameters
        ----------
        price_history : np.ndarray
            Array of historical prices (most recent last)
        current_net_position : float
            Current net trader position
        position_limit : float
            Maximum net position allowed

        Returns
        -------
        list[Trade]
            List of trades for this period
        """
        # Determine number of trades
        n_trades = self._rng.poisson(self.trades_per_day)
        if n_trades == 0:
            return []

        trades = []

        # Compute signals for informed traders
        momentum_signal = self._momentum_signal(price_history)
        mean_rev_signal = self._mean_reversion_signal(price_history)

        # Allocate trades to trader types
        n_noise = int(n_trades * self.noise_weight)
        n_momentum = int(n_trades * self.momentum_weight)
        n_mean_rev = n_trades - n_noise - n_momentum

        # Generate noise trades
        for _ in range(n_noise):
            size = self._rng.exponential(self.avg_trade_size)
            is_long = self._rng.random() > 0.5
            trades.append(Trade(size, is_long, "noise"))

        # Generate momentum trades
        for _ in range(n_momentum):
            size = self._rng.exponential(self.avg_trade_size)
            # Momentum traders follow the signal
            prob_long = 0.5 + 0.3 * momentum_signal  # Signal in [-1, 1]
            is_long = self._rng.random() < prob_long
            trades.append(Trade(size, is_long, "momentum"))

        # Generate mean-reversion trades
        for _ in range(n_mean_rev):
            size = self._rng.exponential(self.avg_trade_size)
            # Mean-rev traders fade the signal
            prob_long = 0.5 - 0.3 * mean_rev_signal
            is_long = self._rng.random() < prob_long
            trades.append(Trade(size, is_long, "mean_reversion"))

        # Apply position limits
        trades = self._apply_position_limits(
            trades, current_net_position, position_limit
        )

        return trades

    def _momentum_signal(self, prices: np.ndarray) -> float:
        """
        Calculate momentum signal from price history.

        Returns value in [-1, 1] where:
        - Positive = upward momentum
        - Negative = downward momentum
        """
        if len(prices) < 2:
            return 0.0

        lookback = min(self.lookback_days, len(prices) - 1)
        if lookback < 1:
            return 0.0

        recent_return = (prices[-1] - prices[-lookback - 1]) / prices[-lookback - 1]

        # Normalize to [-1, 1] using tanh
        signal = np.tanh(recent_return * 10)  # Scale factor for sensitivity
        return float(signal)

    def _mean_reversion_signal(self, prices: np.ndarray) -> float:
        """
        Calculate mean-reversion signal.

        Returns value in [-1, 1] where:
        - Positive = price above mean (expect down)
        - Negative = price below mean (expect up)
        """
        if len(prices) < 2:
            return 0.0

        lookback = min(self.lookback_days * 4, len(prices))  # Longer lookback for mean
        mean_price = np.mean(prices[-lookback:])

        deviation = (prices[-1] - mean_price) / mean_price

        # Normalize to [-1, 1]
        signal = np.tanh(deviation * 5)
        return float(signal)

    def _apply_position_limits(
        self,
        trades: List[Trade],
        current_position: float,
        limit: float,
    ) -> List[Trade]:
        """Apply position limits, reducing or rejecting trades as needed."""
        result = []
        position = current_position

        for trade in trades:
            delta = trade.size if trade.is_long else -trade.size

            # Check if trade would breach limit
            new_position = position + delta

            if abs(new_position) > limit:
                # Reduce trade size to stay within limits
                if delta > 0:  # Buying
                    max_buy = limit - position
                    if max_buy > 0:
                        trade = Trade(max_buy, True, trade.trader_type)
                        result.append(trade)
                        position += max_buy
                else:  # Selling
                    max_sell = position + limit
                    if max_sell > 0:
                        trade = Trade(max_sell, False, trade.trader_type)
                        result.append(trade)
                        position -= max_sell
            else:
                result.append(trade)
                position = new_position

        return result

    def set_seed(self, seed: int):
        """Set random seed."""
        self._rng = np.random.default_rng(seed)
