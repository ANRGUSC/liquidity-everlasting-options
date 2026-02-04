"""Price path simulation models: GBM, Jump-Diffusion, and Historical."""

import numpy as np
from typing import Optional
from pathlib import Path
from datetime import date, timedelta
import requests


class GBMSimulator:
    """
    Geometric Brownian Motion (Black-Scholes) price simulator.
    
    The standard Black-Scholes model:
        dS/S = mu * dt + sigma * dW
    
    Parameters
    ----------
    initial_price : float
        Starting asset price
    drift : float
        Annual drift rate (mu)
    volatility : float
        Annual volatility (sigma)
    random_seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(
        self,
        initial_price: float,
        drift: float = 0.05,
        volatility: float = 0.80,
        random_seed: Optional[int] = None,
    ):
        self.initial_price = initial_price
        self.drift = drift
        self.volatility = volatility
        self._rng = np.random.default_rng(random_seed)
    
    def simulate(
        self,
        n_days: int,
        dt: float = 1 / 252,
        n_paths: int = 1,
    ) -> np.ndarray:
        """
        Simulate price paths using GBM.
        
        Parameters
        ----------
        n_days : int
            Number of trading days to simulate
        dt : float
            Time step in years (default 1/252 = 1 trading day)
        n_paths : int
            Number of paths to simulate
            
        Returns
        -------
        np.ndarray
            Price paths of shape (n_steps+1, n_paths)
        """
        n_steps = int(n_days / (dt * 252))
        n_steps = max(n_steps, n_days)
        
        # Initialize paths
        paths = np.zeros((n_steps + 1, n_paths))
        paths[0, :] = self.initial_price
        
        for i in range(n_steps):
            dW = self._rng.normal(0, np.sqrt(dt), n_paths)
            log_return = (self.drift - 0.5 * self.volatility**2) * dt
            log_return += self.volatility * dW
            paths[i + 1, :] = paths[i, :] * np.exp(log_return)
        
        return paths
    
    def simulate_daily(self, n_days: int) -> np.ndarray:
        """Simulate daily prices."""
        paths = self.simulate(n_days, dt=1 / 252, n_paths=1)
        return paths[:, 0]
    
    def set_seed(self, seed: int):
        """Set random seed."""
        self._rng = np.random.default_rng(seed)


class HistoricalDataLoader:
    """
    Loader for real historical price data from APIs.

    Fetches actual historical prices for BTC, ETH (from CoinGecko) and SPY (from Yahoo Finance).
    """

    # CoinGecko IDs for crypto assets
    _COINGECKO_IDS = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
    }

    # Fallback parameters if API fails
    _FALLBACK_PARAMS = {
        "BTC": {"base_price": 40000, "daily_vol": 0.04, "drift": 0.001},
        "ETH": {"base_price": 2500, "daily_vol": 0.05, "drift": 0.0015},
        "SPY": {"base_price": 450, "daily_vol": 0.012, "drift": 0.0003},
    }

    SUPPORTED_ASSETS = ["BTC", "ETH", "SPY"]

    def __init__(self, asset: str, random_seed: Optional[int] = None):
        """
        Initialize historical data loader.

        Parameters
        ----------
        asset : str
            Asset symbol ("BTC", "ETH", "SPY")
        random_seed : int, optional
            Random seed for reproducibility (used for fallback synthetic data)
        """
        if asset not in self.SUPPORTED_ASSETS:
            raise ValueError(f"Unknown asset: {asset}. Choose from {self.SUPPORTED_ASSETS}")

        self.asset = asset
        self._rng = np.random.default_rng(random_seed)
        self._cached_data: Optional[np.ndarray] = None

    def _fetch_coingecko(self, start_date: date, end_date: date) -> Optional[np.ndarray]:
        """
        Fetch historical prices from CoinGecko API.

        Parameters
        ----------
        start_date : date
            Start date for historical data
        end_date : date
            End date for historical data

        Returns
        -------
        np.ndarray or None
            Daily prices array, or None if fetch fails
        """
        coin_id = self._COINGECKO_IDS.get(self.asset)
        if not coin_id:
            return None

        # Convert dates to Unix timestamps
        from_ts = int((start_date - date(1970, 1, 1)).total_seconds())
        to_ts = int((end_date - date(1970, 1, 1)).total_seconds()) + 86400  # Add one day

        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range"
        params = {
            "vs_currency": "usd",
            "from": from_ts,
            "to": to_ts,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "prices" not in data or len(data["prices"]) == 0:
                return None

            # Extract daily prices (CoinGecko returns [timestamp, price] pairs)
            prices_raw = data["prices"]

            # Resample to daily (CoinGecko may return hourly for shorter ranges)
            # Group by day and take the last price of each day
            daily_prices = {}
            for ts_ms, price in prices_raw:
                day = date.fromtimestamp(ts_ms / 1000)
                daily_prices[day] = price

            # Sort by date and extract prices
            sorted_days = sorted(daily_prices.keys())
            prices = np.array([daily_prices[d] for d in sorted_days])

            return prices

        except Exception as e:
            print(f"CoinGecko API error for {self.asset}: {e}")
            return None

    def _fetch_yahoo(self, start_date: date, end_date: date) -> Optional[np.ndarray]:
        """
        Fetch historical prices from Yahoo Finance.

        Parameters
        ----------
        start_date : date
            Start date for historical data
        end_date : date
            End date for historical data

        Returns
        -------
        np.ndarray or None
            Daily prices array, or None if fetch fails
        """
        try:
            import yfinance as yf

            ticker = yf.Ticker(self.asset)
            # Add buffer days to ensure we get enough data
            hist = ticker.history(start=start_date, end=end_date + timedelta(days=1))

            if hist.empty:
                return None

            prices = hist["Close"].values
            return prices

        except ImportError:
            print("yfinance not installed. Install with: pip install yfinance")
            return None
        except Exception as e:
            print(f"Yahoo Finance error for {self.asset}: {e}")
            return None

    def _generate_fallback(self, n_days: int, initial_price: Optional[float] = None) -> np.ndarray:
        """
        Generate synthetic fallback data when API fails.

        Parameters
        ----------
        n_days : int
            Number of days
        initial_price : float, optional
            Starting price

        Returns
        -------
        np.ndarray
            Synthetic price path
        """
        params = self._FALLBACK_PARAMS[self.asset]
        start_price = initial_price or params["base_price"]
        daily_vol = params["daily_vol"]
        drift = params["drift"]

        prices = np.zeros(n_days + 1)
        prices[0] = start_price

        for i in range(n_days):
            df = 5
            z = self._rng.standard_t(df)
            daily_return = drift + daily_vol * z * np.sqrt((df - 2) / df)
            if self._rng.random() < 0.02:
                daily_return *= self._rng.uniform(1.5, 3.0) * np.sign(daily_return)
            prices[i + 1] = prices[i] * np.exp(daily_return)

        return prices

    def load(
        self,
        n_days: int,
        initial_price: Optional[float] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> np.ndarray:
        """
        Load historical price data from APIs.

        Parameters
        ----------
        n_days : int
            Number of days of data needed
        initial_price : float, optional
            If provided, prices will be scaled so first price matches this value
        start_date : date, optional
            Start date for fetching (defaults to n_days ago from today)
        end_date : date, optional
            End date for fetching (defaults to today)

        Returns
        -------
        np.ndarray
            Daily price path of shape (n_days+1,)
        """
        # Default date range if not provided
        if end_date is None:
            end_date = date.today()
        if start_date is None:
            start_date = end_date - timedelta(days=n_days)

        # Fetch from appropriate API
        prices = None
        if self.asset in self._COINGECKO_IDS:
            prices = self._fetch_coingecko(start_date, end_date)
        else:
            prices = self._fetch_yahoo(start_date, end_date)

        # If API fetch failed, use fallback
        if prices is None or len(prices) < 2:
            print(f"Using synthetic fallback data for {self.asset}")
            return self._generate_fallback(n_days, initial_price)

        # Ensure we have exactly n_days+1 prices
        if len(prices) > n_days + 1:
            # Take the last n_days+1 prices
            prices = prices[-(n_days + 1):]
        elif len(prices) < n_days + 1:
            # Extend with synthetic continuation if not enough data
            print(f"Only {len(prices)} days of data available, extending with synthetic data")
            needed = n_days + 1 - len(prices)
            extension = self._extend_prices(prices[-1], needed)
            prices = np.concatenate([prices, extension[1:]])  # Skip first to avoid duplicate

        # Scale prices if initial_price is specified
        if initial_price is not None:
            scale_factor = initial_price / prices[0]
            prices = prices * scale_factor

        return prices

    def _extend_prices(self, last_price: float, n_days: int) -> np.ndarray:
        """Extend price series with synthetic data matching asset characteristics."""
        params = self._FALLBACK_PARAMS[self.asset]
        daily_vol = params["daily_vol"]
        drift = params["drift"]

        prices = np.zeros(n_days + 1)
        prices[0] = last_price

        for i in range(n_days):
            df = 5
            z = self._rng.standard_t(df)
            daily_return = drift + daily_vol * z * np.sqrt((df - 2) / df)
            prices[i + 1] = prices[i] * np.exp(daily_return)

        return prices

    def set_seed(self, seed: int):
        """Set random seed for fallback generation."""
        self._rng = np.random.default_rng(seed)


class JumpDiffusionSimulator:
    """
    Merton Jump-Diffusion model for asset price simulation.

    The model combines geometric Brownian motion with a compound Poisson
    jump process:

        dS/S = (mu - lambda*k) dt + sigma dW + (J - 1) dN

    where:
        - mu: drift rate
        - sigma: diffusion volatility
        - lambda: jump intensity (expected jumps per year)
        - J: jump size multiplier (log-normal)
        - k = E[J-1]: expected jump contribution
        - dW: Wiener process
        - dN: Poisson process

    Parameters
    ----------
    initial_price : float
        Starting asset price
    drift : float
        Annual drift rate (mu)
    volatility : float
        Annual diffusion volatility (sigma)
    jump_intensity : float
        Expected jumps per year (lambda)
    jump_mean : float
        Mean of log jump size (mu_J)
    jump_std : float
        Std of log jump size (sigma_J)
    random_seed : int, optional
        Random seed for reproducibility
    """

    def __init__(
        self,
        initial_price: float,
        drift: float = 0.05,
        volatility: float = 0.80,
        jump_intensity: float = 10.0,
        jump_mean: float = -0.02,
        jump_std: float = 0.05,
        random_seed: Optional[int] = None,
    ):
        self.initial_price = initial_price
        self.drift = drift
        self.volatility = volatility
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std

        self._rng = np.random.default_rng(random_seed)

    def _expected_jump_return(self) -> float:
        """Calculate E[J-1] = E[exp(mu_J + sigma_J^2/2)] - 1."""
        return np.exp(self.jump_mean + 0.5 * self.jump_std**2) - 1

    def simulate(
        self,
        n_days: int,
        dt: float = 1 / 252,
        n_paths: int = 1,
    ) -> np.ndarray:
        """
        Simulate price paths.

        Parameters
        ----------
        n_days : int
            Number of trading days to simulate
        dt : float
            Time step in years (default 1/252 = 1 trading day)
        n_paths : int
            Number of paths to simulate

        Returns
        -------
        np.ndarray
            Price paths of shape (n_steps+1, n_paths)
            First row is initial price
        """
        n_steps = int(n_days / (dt * 252))
        n_steps = max(n_steps, n_days)  # At least n_days steps

        # Compensated drift
        k = self._expected_jump_return()
        compensated_drift = self.drift - self.jump_intensity * k

        # Initialize paths
        paths = np.zeros((n_steps + 1, n_paths))
        paths[0, :] = self.initial_price

        for i in range(n_steps):
            # Diffusion component
            dW = self._rng.normal(0, np.sqrt(dt), n_paths)
            diffusion = (compensated_drift - 0.5 * self.volatility**2) * dt
            diffusion += self.volatility * dW

            # Jump component
            # Number of jumps in this interval (Poisson)
            n_jumps = self._rng.poisson(self.jump_intensity * dt, n_paths)

            # Total jump contribution
            jump_return = np.zeros(n_paths)
            for p in range(n_paths):
                if n_jumps[p] > 0:
                    jump_sizes = self._rng.normal(
                        self.jump_mean, self.jump_std, n_jumps[p]
                    )
                    jump_return[p] = np.sum(jump_sizes)

            # Update prices (log-space then exponentiate)
            log_return = diffusion + jump_return
            paths[i + 1, :] = paths[i, :] * np.exp(log_return)

        return paths

    def simulate_single_path(self, n_days: int, dt: float = 1 / 252) -> np.ndarray:
        """
        Simulate a single price path.

        Parameters
        ----------
        n_days : int
            Number of trading days
        dt : float
            Time step in years

        Returns
        -------
        np.ndarray
            Price path of shape (n_steps+1,)
        """
        paths = self.simulate(n_days, dt, n_paths=1)
        return paths[:, 0]

    def simulate_daily(self, n_days: int) -> np.ndarray:
        """
        Simulate daily prices.

        Parameters
        ----------
        n_days : int
            Number of days

        Returns
        -------
        np.ndarray
            Daily price path of shape (n_days+1,)
        """
        return self.simulate_single_path(n_days, dt=1 / 252)

    def set_seed(self, seed: int):
        """Set random seed."""
        self._rng = np.random.default_rng(seed)

    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            "initial_price": self.initial_price,
            "drift": self.drift,
            "volatility": self.volatility,
            "jump_intensity": self.jump_intensity,
            "jump_mean": self.jump_mean,
            "jump_std": self.jump_std,
        }
