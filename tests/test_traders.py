"""Tests for trader behavior module."""

import pytest
import numpy as np

from src.simulation.traders import TraderMix, Trade


class TestTraderMixWeights:
    """Tests for trader weight normalization."""

    def test_weights_normalize(self):
        """Weights should be normalized to sum to 1."""
        traders = TraderMix(
            noise_weight=1,
            momentum_weight=1,
            mean_reversion_weight=1,
        )

        assert abs(
            traders.noise_weight
            + traders.momentum_weight
            + traders.mean_reversion_weight
            - 1.0
        ) < 1e-10

    def test_zero_weights_handled(self):
        """Zero weights should default to noise only."""
        traders = TraderMix(
            noise_weight=0,
            momentum_weight=0,
            mean_reversion_weight=0,
        )

        assert traders.noise_weight == 1.0

    def test_single_trader_type(self):
        """Single trader type should get all weight."""
        traders = TraderMix(
            noise_weight=0,
            momentum_weight=100,
            mean_reversion_weight=0,
        )

        assert traders.momentum_weight == 1.0
        assert traders.noise_weight == 0.0
        assert traders.mean_reversion_weight == 0.0


class TestTraderMixSignals:
    """Tests for trader signal generation."""

    def test_momentum_signal_up(self):
        """Upward price movement should give positive momentum signal."""
        traders = TraderMix(random_seed=42)

        # Rising prices
        prices = np.array([100, 102, 104, 106, 108, 110])
        signal = traders._momentum_signal(prices)

        assert signal > 0

    def test_momentum_signal_down(self):
        """Downward price movement should give negative momentum signal."""
        traders = TraderMix(random_seed=42)

        # Falling prices
        prices = np.array([100, 98, 96, 94, 92, 90])
        signal = traders._momentum_signal(prices)

        assert signal < 0

    def test_mean_reversion_signal_above_mean(self):
        """Price above mean should give positive mean-rev signal."""
        traders = TraderMix(random_seed=42)

        # Price ended above the mean
        prices = np.array([100, 100, 100, 100, 120])
        signal = traders._mean_reversion_signal(prices)

        assert signal > 0  # Expect price to fall

    def test_mean_reversion_signal_below_mean(self):
        """Price below mean should give negative mean-rev signal."""
        traders = TraderMix(random_seed=42)

        # Price ended below the mean
        prices = np.array([100, 100, 100, 100, 80])
        signal = traders._mean_reversion_signal(prices)

        assert signal < 0  # Expect price to rise

    def test_signals_bounded(self):
        """Signals should be bounded in [-1, 1]."""
        traders = TraderMix(random_seed=42)

        # Extreme price movement
        prices = np.array([100, 200, 300, 400, 500])
        mom_signal = traders._momentum_signal(prices)
        mr_signal = traders._mean_reversion_signal(prices)

        assert -1 <= mom_signal <= 1
        assert -1 <= mr_signal <= 1


class TestTraderMixTradeGeneration:
    """Tests for trade generation."""

    def test_average_trade_count(self):
        """Average trades should be close to trades_per_day over many runs."""
        traders = TraderMix(
            trades_per_day=100,
            random_seed=42,
        )

        prices = np.array([100, 101, 102, 103, 104])
        trade_counts = []

        for i in range(100):
            traders.set_seed(i)
            trades = traders.generate_trades(prices)
            trade_counts.append(len(trades))

        avg = np.mean(trade_counts)
        # Should be within 20% of expected
        assert 80 < avg < 120

    def test_trade_types_distributed(self):
        """Trade types should follow weight distribution."""
        traders = TraderMix(
            noise_weight=0.5,
            momentum_weight=0.3,
            mean_reversion_weight=0.2,
            trades_per_day=1000,
            random_seed=42,
        )

        prices = np.array([100, 101, 102, 103, 104])
        trades = traders.generate_trades(prices)

        counts = {"noise": 0, "momentum": 0, "mean_reversion": 0}
        for t in trades:
            counts[t.trader_type] += 1

        total = len(trades)
        if total > 0:
            noise_pct = counts["noise"] / total
            mom_pct = counts["momentum"] / total
            mr_pct = counts["mean_reversion"] / total

            # Should be roughly proportional (within 10%)
            assert abs(noise_pct - 0.5) < 0.15
            assert abs(mom_pct - 0.3) < 0.15
            assert abs(mr_pct - 0.2) < 0.15

    def test_trades_have_positive_size(self):
        """All trade sizes should be positive."""
        traders = TraderMix(trades_per_day=100, random_seed=42)
        prices = np.array([100, 101, 102])

        trades = traders.generate_trades(prices)

        for trade in trades:
            assert trade.size > 0

    def test_reproducibility_with_seed(self):
        """Same seed should give same trades."""
        traders1 = TraderMix(trades_per_day=50, random_seed=42)
        traders2 = TraderMix(trades_per_day=50, random_seed=42)

        prices = np.array([100, 101, 102, 103])

        trades1 = traders1.generate_trades(prices)
        trades2 = traders2.generate_trades(prices)

        assert len(trades1) == len(trades2)
        for t1, t2 in zip(trades1, trades2):
            assert t1.size == t2.size
            assert t1.is_long == t2.is_long
            assert t1.trader_type == t2.trader_type


class TestTradeClass:
    """Tests for Trade dataclass."""

    def test_trade_creation(self):
        """Should create trades with correct attributes."""
        trade = Trade(size=10.0, is_long=True, trader_type="noise")

        assert trade.size == 10.0
        assert trade.is_long is True
        assert trade.trader_type == "noise"

    def test_trade_short(self):
        """Should handle short trades."""
        trade = Trade(size=5.0, is_long=False, trader_type="momentum")

        assert trade.size == 5.0
        assert trade.is_long is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
