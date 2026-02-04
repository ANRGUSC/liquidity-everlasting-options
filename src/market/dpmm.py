"""DPMM (Proactive Market Making) implementation.

Based on Deri Protocol's DPMM mechanism:
https://docs.deri.io/how-it-works/dpmm-proactive-market-making

The DPMM adjusts the mark price based on net position to incentivize
balanced markets:
    Mark = Theoretical × (1 + k × NetPosition / Liquidity)

When traders are net long, mark price increases above theoretical;
when net short, mark price decreases below theoretical.

The price deviation is inversely proportional to pool liquidity -
larger pools have smaller price impact per unit of position.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple


class DPMMMarket:
    """
    DPMM-style market maker for everlasting options.

    Mark price formula (per Deri Protocol):
        Mark = Theoretical × (1 + k × NetPosition / Liquidity)

    Where:
        - k (alpha): price impact coefficient
        - NetPosition: total longs - total shorts
        - Liquidity: pool liquidity (higher = less price impact)

    At equilibrium (net_position = 0), mark price equals theoretical price.

    Parameters
    ----------
    pool_liquidity : float
        Total pool liquidity (in notional terms)
    alpha_coefficient : float
        Price impact coefficient k (higher = more impact per position)
    trading_fee_rate : float
        Fee rate per trade (e.g., 0.001 = 0.1%)
    """

    def __init__(
        self,
        pool_liquidity: float,
        alpha_coefficient: float = 0.001,
        trading_fee_rate: float = 0.001,
    ):
        self.pool_liquidity = pool_liquidity
        self.alpha = alpha_coefficient
        self.trading_fee_rate = trading_fee_rate
        self._net_position = 0.0
        self._total_volume = 0.0
        self._total_fees = 0.0

    @property
    def net_position(self) -> float:
        """Current net trader position."""
        return self._net_position

    def mark_price(self, theoretical_price: float, net_position: Optional[float] = None) -> float:
        """
        Calculate mark price adjusted by net position.

        Parameters
        ----------
        theoretical_price : float
            Theoretical option price
        net_position : float, optional
            Net position to use (defaults to current)

        Returns
        -------
        float
            Adjusted mark price
        """
        if net_position is None:
            net_position = self._net_position

        # Price adjustment based on net position
        # Positive net position (traders long) -> higher mark price
        adjustment = self.alpha * net_position / max(self.pool_liquidity, 1e-10)

        # Cap adjustment to prevent extreme prices
        adjustment = max(-0.5, min(0.5, adjustment))

        mark = theoretical_price * (1 + adjustment)

        # Mark price can't be negative
        return max(mark, 0.0)

    def price_impact(self, trade_size: float, theoretical_price: float) -> float:
        """
        Calculate price impact of a trade.

        Parameters
        ----------
        trade_size : float
            Trade size (positive = buy, negative = sell)
        theoretical_price : float
            Theoretical option price

        Returns
        -------
        float
            Average price impact as a fraction
        """
        # Average impact is half the final impact
        avg_adjustment = self.alpha * (self._net_position + trade_size / 2) / self.pool_liquidity
        avg_adjustment = max(-0.5, min(0.5, avg_adjustment))

        return avg_adjustment

    def execute_trade(
        self,
        trade_size: float,
        theoretical_price: float,
        is_long: Optional[bool] = None,
    ) -> Tuple[float, float, float]:
        """
        Execute a trade and return fill price.

        Parameters
        ----------
        trade_size : float
            Size of trade (always positive, direction from is_long)
        theoretical_price : float
            Current theoretical price
        is_long : bool, optional
            True = buy/long, False = sell/short.
            If None, sign of trade_size determines direction.

        Returns
        -------
        tuple[float, float, float]
            (fill_price, fee_amount, new_net_position)
        """
        if is_long is None:
            is_long = trade_size > 0
            trade_size = abs(trade_size)

        signed_size = trade_size if is_long else -trade_size

        # Calculate average fill price
        impact = self.price_impact(signed_size, theoretical_price)
        fill_price = theoretical_price * (1 + impact)
        fill_price = max(fill_price, 0.0)

        # Calculate fee
        notional = fill_price * trade_size
        fee = notional * self.trading_fee_rate

        # Update state
        self._net_position += signed_size
        self._total_volume += notional
        self._total_fees += fee

        return fill_price, fee, self._net_position

    def get_stats(self) -> dict:
        """Get market statistics."""
        return {
            "net_position": self._net_position,
            "total_volume": self._total_volume,
            "total_fees": self._total_fees,
        }

    def reset(self, initial_position: float = 0.0):
        """Reset market state."""
        self._net_position = initial_position
        self._total_volume = 0.0
        self._total_fees = 0.0
