"""Funding rate engine for everlasting options."""


class FundingEngine:
    """
    Funding rate calculation and accrual engine.

    The funding mechanism keeps the mark price aligned with the theoretical
    value by having longs/shorts periodically pay each other.

    Parameters
    ----------
    funding_period_days : float
        Period between funding settlements (default 1 day)
    """

    def __init__(self, funding_period_days: float = 1.0):
        self.funding_period_days = funding_period_days
        self._accumulated_funding = 0.0

    def calculate_funding(self, mark_price: float, payoff: float) -> float:
        """
        Calculate the funding rate for the current period.

        Funding = (Mark - Payoff) / Period

        Parameters
        ----------
        mark_price : float
            Current mark price
        payoff : float
            Current intrinsic payoff

        Returns
        -------
        float
            Funding rate (positive means longs pay shorts)
        """
        return (mark_price - payoff) / self.funding_period_days

    def accrue_funding(
        self,
        position_size: float,
        funding_rate: float,
        dt: float = 1.0,
    ) -> float:
        """
        Calculate funding payment for a position.

        Parameters
        ----------
        position_size : float
            Position size (positive = long, negative = short)
        funding_rate : float
            Current funding rate
        dt : float
            Time step in days

        Returns
        -------
        float
            Funding payment (positive = position pays, negative = receives)
        """
        # Longs pay funding when rate is positive
        # Shorts receive funding when rate is positive
        funding_payment = position_size * funding_rate * dt

        self._accumulated_funding += funding_payment
        return funding_payment

    def get_lp_funding(
        self,
        net_trader_position: float,
        funding_rate: float,
        dt: float = 1.0,
    ) -> float:
        """
        Calculate funding payment received by LP.

        LP is counterparty to all traders, so LP's position is opposite
        of net trader position.

        Parameters
        ----------
        net_trader_position : float
            Net trader position (positive = traders are net long)
        funding_rate : float
            Current funding rate
        dt : float
            Time step in days

        Returns
        -------
        float
            Funding received by LP (positive = LP receives)
        """
        # LP position is opposite of traders
        lp_position = -net_trader_position

        # If LP is short (traders are long) and funding is positive,
        # LP receives funding
        return -self.accrue_funding(lp_position, funding_rate, dt)

    @property
    def total_accumulated(self) -> float:
        """Total accumulated funding."""
        return self._accumulated_funding

    def reset(self):
        """Reset accumulated funding."""
        self._accumulated_funding = 0.0
