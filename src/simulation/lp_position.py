"""LP position management and hedging."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PnLBreakdown:
    """Breakdown of PnL components."""
    funding: float = 0.0
    trading_fees: float = 0.0
    mtm: float = 0.0  # Mark-to-market on option position
    hedge: float = 0.0  # PnL from hedge position

    @property
    def total(self) -> float:
        return self.funding + self.trading_fees + self.mtm + self.hedge


class LPPosition:
    """
    LP position manager with optional delta hedging.

    The LP provides liquidity to the everlasting option market and is
    the counterparty to all traders. LP can optionally delta-hedge
    the option exposure.

    Parameters
    ----------
    capital : float
        Initial LP capital
    hedge_ratio : float
        Fraction of delta to hedge (0 = no hedge, 1 = full hedge)
    """

    def __init__(self, capital: float, hedge_ratio: float = 0.0):
        self.initial_capital = capital
        self.hedge_ratio = max(0.0, min(1.0, hedge_ratio))

        # Position state
        self._option_exposure = 0.0  # Net option position (opposite of traders)
        self._hedge_position = 0.0  # Spot position for hedging
        self._hedge_entry_price = 0.0  # Average entry price for hedge

        # PnL tracking
        self._cumulative_funding = 0.0
        self._cumulative_fees = 0.0
        self._cumulative_hedge_pnl = 0.0
        self._realized_hedge_pnl = 0.0

        # Current state
        self._last_spot_price = 0.0
        self._last_option_price = 0.0

    @property
    def option_exposure(self) -> float:
        """LP's net option exposure (negative of trader position)."""
        return self._option_exposure

    @property
    def hedge_position(self) -> float:
        """LP's spot hedge position."""
        return self._hedge_position

    @property
    def current_equity(self) -> float:
        """Current LP equity."""
        return self.initial_capital + self.total_pnl

    @property
    def total_pnl(self) -> float:
        """Total PnL."""
        return (
            self._cumulative_funding
            + self._cumulative_fees
            + self._cumulative_hedge_pnl
        )

    def update_position(self, net_trader_position: float):
        """
        Update LP position based on net trader position.

        Parameters
        ----------
        net_trader_position : float
            Current net position of all traders
        """
        self._option_exposure = -net_trader_position

    def apply_funding(self, funding_amount: float):
        """
        Apply funding payment to LP.

        Parameters
        ----------
        funding_amount : float
            Funding received (positive) or paid (negative)
        """
        self._cumulative_funding += funding_amount

    def apply_trading_fee(self, fee_amount: float, lp_share: float = 1.0):
        """
        Apply trading fee income to LP.

        Parameters
        ----------
        fee_amount : float
            Total fee amount
        lp_share : float
            LP's share of fees (0-1)
        """
        self._cumulative_fees += fee_amount * lp_share

    def set_hedge_ratio(self, ratio: float):
        """Dynamically set hedge ratio (for RL-based hedging)."""
        self.hedge_ratio = max(0.0, min(1.0, ratio))

    def rebalance_hedge(
        self,
        current_spot: float,
        option_delta: float,
        override_ratio: Optional[float] = None,
    ) -> float:
        """
        Rebalance the hedge position.

        Parameters
        ----------
        current_spot : float
            Current spot price
        option_delta : float
            Option delta (per unit)

        Returns
        -------
        float
            Trade size executed (positive = bought spot)
        """
        effective_ratio = override_ratio if override_ratio is not None else self.hedge_ratio

        if effective_ratio == 0:
            return 0.0

        # Target hedge position
        # LP is short options when traders are long
        # Delta hedge: buy spot if short calls (positive delta)
        target_delta = -self._option_exposure * option_delta
        target_hedge = target_delta * effective_ratio

        # Calculate trade needed
        trade_size = target_hedge - self._hedge_position

        if abs(trade_size) < 0.01:  # Minimum trade threshold
            return 0.0

        # Execute trade (update average entry price)
        old_value = self._hedge_position * self._hedge_entry_price

        if self._hedge_position * trade_size > 0:
            # Adding to position
            new_value = old_value + trade_size * current_spot
            self._hedge_entry_price = new_value / (self._hedge_position + trade_size)
        else:
            # Reducing or reversing position
            if abs(trade_size) >= abs(self._hedge_position):
                # Closing and possibly reversing
                closed_pnl = self._hedge_position * (
                    current_spot - self._hedge_entry_price
                )
                self._realized_hedge_pnl += closed_pnl
                remaining = trade_size + self._hedge_position
                self._hedge_entry_price = current_spot if remaining != 0 else 0
            else:
                # Partial close
                closed_pnl = -trade_size * (current_spot - self._hedge_entry_price)
                self._realized_hedge_pnl += closed_pnl

        self._hedge_position += trade_size
        self._last_spot_price = current_spot

        return trade_size

    def mark_to_market(
        self,
        option_price: float,
        spot_price: float,
    ) -> float:
        """
        Calculate current mark-to-market value.

        Parameters
        ----------
        option_price : float
            Current option mark price
        spot_price : float
            Current spot price

        Returns
        -------
        float
            Current MTM PnL
        """
        self._last_option_price = option_price
        self._last_spot_price = spot_price

        # Unrealized hedge PnL
        unrealized_hedge = self._hedge_position * (
            spot_price - self._hedge_entry_price
        )
        self._cumulative_hedge_pnl = self._realized_hedge_pnl + unrealized_hedge

        return self.total_pnl

    def get_pnl_breakdown(self) -> PnLBreakdown:
        """Get detailed PnL breakdown."""
        return PnLBreakdown(
            funding=self._cumulative_funding,
            trading_fees=self._cumulative_fees,
            mtm=0.0,  # MTM on options is implicit in funding
            hedge=self._cumulative_hedge_pnl,
        )

    def get_state(self) -> dict:
        """Get current state as dictionary."""
        return {
            "option_exposure": self._option_exposure,
            "hedge_position": self._hedge_position,
            "hedge_entry_price": self._hedge_entry_price,
            "cumulative_funding": self._cumulative_funding,
            "cumulative_fees": self._cumulative_fees,
            "cumulative_hedge_pnl": self._cumulative_hedge_pnl,
            "total_pnl": self.total_pnl,
            "equity": self.current_equity,
        }

    def reset(self):
        """Reset LP position to initial state."""
        self._option_exposure = 0.0
        self._hedge_position = 0.0
        self._hedge_entry_price = 0.0
        self._cumulative_funding = 0.0
        self._cumulative_fees = 0.0
        self._cumulative_hedge_pnl = 0.0
        self._realized_hedge_pnl = 0.0
        self._last_spot_price = 0.0
        self._last_option_price = 0.0
