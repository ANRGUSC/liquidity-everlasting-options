"""Price oracle interface for spot price feeds."""

from __future__ import annotations

from typing import Optional, Union, List

import numpy as np


class PriceOracle:
    """
    Price oracle that provides spot prices from a precomputed path.

    Parameters
    ----------
    price_path : array-like
        Array of spot prices indexed by time step
    """

    def __init__(self, price_path: Union[np.ndarray, List[float]]):
        self._prices = np.array(price_path)
        self._current_step = 0

    def __len__(self) -> int:
        return len(self._prices)

    @property
    def current_price(self) -> float:
        """Get current spot price."""
        return float(self._prices[self._current_step])

    def get_price(self, step: Optional[int] = None) -> float:
        """
        Get price at a specific step.

        Parameters
        ----------
        step : int, optional
            Time step to query (defaults to current)

        Returns
        -------
        float
            Spot price at the given step
        """
        if step is None:
            step = self._current_step

        if step < 0 or step >= len(self._prices):
            raise IndexError(f"Step {step} out of range [0, {len(self._prices)})")

        return float(self._prices[step])

    def get_history(self, up_to_step: Optional[int] = None) -> np.ndarray:
        """
        Get price history up to a step.

        Parameters
        ----------
        up_to_step : int, optional
            Last step to include (defaults to current)

        Returns
        -------
        np.ndarray
            Array of prices from step 0 to up_to_step
        """
        if up_to_step is None:
            up_to_step = self._current_step

        return self._prices[: up_to_step + 1].copy()

    def advance(self) -> float:
        """
        Advance to next time step.

        Returns
        -------
        float
            New current price
        """
        if self._current_step < len(self._prices) - 1:
            self._current_step += 1
        return self.current_price

    def set_step(self, step: int):
        """
        Set the current step.

        Parameters
        ----------
        step : int
            Step to set as current
        """
        if step < 0 or step >= len(self._prices):
            raise IndexError(f"Step {step} out of range [0, {len(self._prices)})")
        self._current_step = step

    def reset(self):
        """Reset oracle to step 0."""
        self._current_step = 0

    @property
    def all_prices(self) -> np.ndarray:
        """Get full price path."""
        return self._prices.copy()
