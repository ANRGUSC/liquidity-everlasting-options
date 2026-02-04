"""Everlasting option pricing model.

Everlasting options are perpetual options that use a funding mechanism
to keep the mark price aligned with the theoretical value. The price
is computed as a weighted sum of vanilla options with different maturities.

Based on the Paradigm Research paper "Everlasting Options":
https://www.paradigm.xyz/2021/05/everlasting-options

Price = (1/f) * sum_{i=1}^{inf} [(f/(f+1))^i * BS(t_i)]

where f = funding frequency (payments per period).
"""

import math
from typing import Literal

from .black_scholes import bs_price, bs_greeks


class EverlastingOption:
    """
    Everlasting option pricing and Greeks.

    The everlasting option price is computed as a weighted sum of
    Black-Scholes prices at increasing maturities:

        P_ever = (1/f) * sum_{i=1}^{n} [decay^i * BS(T_i)]

    where decay = f/(f+1) and f is funding frequency.

    For daily funding (f=1): decay = 0.5, so weights are 0.5, 0.25, 0.125, ...
    For weekly funding (f=7): decay = 7/8 = 0.875

    Reference: Paradigm "Everlasting Options" (2021)
    https://www.paradigm.xyz/2021/05/everlasting-options

    Parameters
    ----------
    strike : float
        Strike price
    option_type : str
        'call' or 'put'
    funding_period_days : float
        Funding period in days (default 1 day = daily funding)
    n_terms : int
        Number of terms in the series approximation (default 30)
    """

    def __init__(
        self,
        strike: float,
        option_type: Literal["call", "put"] = "call",
        funding_period_days: float = 1.0,
        n_terms: int = 30,
    ):
        self.strike = strike
        self.option_type = option_type
        self.funding_period_days = funding_period_days
        self.n_terms = n_terms

        # Funding frequency: payments per funding period
        # For daily funding, f = 1 (one payment per day)
        self._funding_frequency = 1.0
        self._funding_period_years = funding_period_days / 365.0

        # Decay factor: f/(f+1) per the Paradigm formula
        # For daily funding: 1/(1+1) = 0.5
        self._decay = self._funding_frequency / (self._funding_frequency + 1.0)

    def _compute_weights(self) -> list[float]:
        """
        Compute weights for the series expansion.

        Following Paradigm formula:
            weight_i = (1/f) * (f/(f+1))^i = (1/f) * decay^i

        For daily funding (f=1):
            weights = 0.5^1, 0.5^2, 0.5^3, ... = 0.5, 0.25, 0.125, ...
        """
        f = self._funding_frequency
        decay = self._decay

        weights = []
        for i in range(1, self.n_terms + 1):
            # weight_i = (1/f) * decay^i
            w = (1.0 / f) * (decay ** i)
            weights.append(w)

        # The infinite series sums to 1, but we truncate
        # Normalize to ensure weights sum to 1 for the truncated series
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]

        return weights

    def price(
        self,
        spot: float,
        volatility: float,
        risk_free_rate: float = 0.0,
    ) -> float:
        """
        Calculate the everlasting option price.

        Parameters
        ----------
        spot : float
            Current spot price
        volatility : float
            Annualized volatility
        risk_free_rate : float
            Risk-free rate (annualized)

        Returns
        -------
        float
            Everlasting option price
        """
        weights = self._compute_weights()
        price = 0.0

        for i, w in enumerate(weights, start=1):
            T = i * self._funding_period_years
            bs = bs_price(spot, self.strike, T, risk_free_rate, volatility, self.option_type)
            price += w * bs

        return price

    def greeks(
        self,
        spot: float,
        volatility: float,
        risk_free_rate: float = 0.0,
    ) -> dict:
        """
        Calculate the everlasting option Greeks.

        Parameters
        ----------
        spot : float
            Current spot price
        volatility : float
            Annualized volatility
        risk_free_rate : float
            Risk-free rate (annualized)

        Returns
        -------
        dict
            Dictionary with delta, gamma, theta, vega, rho
        """
        weights = self._compute_weights()

        greeks_sum = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}

        for i, w in enumerate(weights, start=1):
            T = i * self._funding_period_years
            g = bs_greeks(spot, self.strike, T, risk_free_rate, volatility, self.option_type)
            for key in greeks_sum:
                greeks_sum[key] += w * g[key]

        return greeks_sum

    def payoff(self, spot: float) -> float:
        """
        Calculate the intrinsic payoff.

        Parameters
        ----------
        spot : float
            Current spot price

        Returns
        -------
        float
            Intrinsic value (payoff if exercised now)
        """
        if self.option_type == "call":
            return max(spot - self.strike, 0.0)
        else:
            return max(self.strike - spot, 0.0)

    def funding_rate(self, mark_price: float, spot: float) -> float:
        """
        Calculate the funding rate.

        The funding rate ensures convergence between mark price and payoff.
        Funding = (Mark - Payoff) / Funding_Period

        Positive funding: longs pay shorts (LP receives if net short options)
        Negative funding: shorts pay longs (LP pays if net short options)

        Parameters
        ----------
        mark_price : float
            Current mark price (possibly adjusted by DPMM)
        spot : float
            Current spot price

        Returns
        -------
        float
            Funding rate for the period
        """
        payoff = self.payoff(spot)

        # Funding rate per period
        # Mark > Payoff: longs pay, shorts receive
        # Mark < Payoff: shorts pay, longs receive
        funding = (mark_price - payoff) / self.funding_period_days

        return funding

    def theoretical_funding_rate(
        self,
        spot: float,
        volatility: float,
        risk_free_rate: float = 0.0,
    ) -> float:
        """
        Calculate the theoretical funding rate.

        Uses the theoretical price instead of mark price.

        Parameters
        ----------
        spot : float
            Current spot price
        volatility : float
            Annualized volatility
        risk_free_rate : float
            Risk-free rate

        Returns
        -------
        float
            Theoretical funding rate
        """
        theo_price = self.price(spot, volatility, risk_free_rate)
        return self.funding_rate(theo_price, spot)
