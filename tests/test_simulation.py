"""Tests for simulation components."""

import pytest
import numpy as np
from datetime import date

from src.simulation.price_paths import JumpDiffusionSimulator
from src.simulation.traders import TraderMix, Trade
from src.simulation.lp_position import LPPosition
from src.simulation.simulator import EverlastingOptionSimulator
from src.config import SimulationConfig, PriceModelConfig, LPConfig, OptionConfig


class TestJumpDiffusionSimulator:
    """Tests for jump-diffusion price simulation."""

    def test_initial_price(self):
        """First price should equal initial price."""
        sim = JumpDiffusionSimulator(
            initial_price=100,
            random_seed=42,
        )
        path = sim.simulate_daily(30)

        assert path[0] == 100

    def test_path_length(self):
        """Path should have correct length."""
        sim = JumpDiffusionSimulator(
            initial_price=100,
            random_seed=42,
        )
        path = sim.simulate_daily(30)

        assert len(path) == 31  # n_days + 1

    def test_prices_positive(self):
        """All prices should be positive."""
        sim = JumpDiffusionSimulator(
            initial_price=100,
            volatility=0.8,
            jump_intensity=20,
            random_seed=42,
        )
        path = sim.simulate_daily(100)

        assert all(p > 0 for p in path)

    def test_reproducibility(self):
        """Same seed should give same path."""
        sim1 = JumpDiffusionSimulator(initial_price=100, random_seed=42)
        sim2 = JumpDiffusionSimulator(initial_price=100, random_seed=42)

        path1 = sim1.simulate_daily(30)
        path2 = sim2.simulate_daily(30)

        np.testing.assert_array_equal(path1, path2)

    def test_different_seeds_different_paths(self):
        """Different seeds should give different paths."""
        sim1 = JumpDiffusionSimulator(initial_price=100, random_seed=42)
        sim2 = JumpDiffusionSimulator(initial_price=100, random_seed=43)

        path1 = sim1.simulate_daily(30)
        path2 = sim2.simulate_daily(30)

        assert not np.allclose(path1, path2)

    def test_multiple_paths(self):
        """Should be able to generate multiple paths."""
        sim = JumpDiffusionSimulator(initial_price=100, random_seed=42)
        paths = sim.simulate(n_days=30, n_paths=10)

        assert paths.shape == (31, 10)
        assert all(paths[0, :] == 100)


class TestTraderMix:
    """Tests for trader behavior simulation."""

    def test_generate_trades(self):
        """Should generate trades."""
        traders = TraderMix(
            noise_weight=0.5,
            momentum_weight=0.3,
            mean_reversion_weight=0.2,
            trades_per_day=50,
            random_seed=42,
        )

        prices = np.array([100, 101, 102, 103, 104])
        trades = traders.generate_trades(prices)

        assert len(trades) > 0
        assert all(isinstance(t, Trade) for t in trades)

    def test_trade_attributes(self):
        """Trades should have proper attributes."""
        traders = TraderMix(trades_per_day=10, random_seed=42)
        prices = np.array([100, 101, 102])
        trades = traders.generate_trades(prices)

        for trade in trades:
            assert trade.size > 0
            assert isinstance(trade.is_long, bool)
            assert trade.trader_type in ["noise", "momentum", "mean_reversion"]

    def test_position_limit(self):
        """Should respect position limits."""
        traders = TraderMix(
            trades_per_day=100,
            avg_trade_size=100,
            random_seed=42,
        )

        prices = np.array([100] * 10)

        # Start with large position
        trades = traders.generate_trades(
            prices,
            current_net_position=9900,
            position_limit=10000,
        )

        # Calculate new position
        net_delta = sum(
            t.size if t.is_long else -t.size
            for t in trades
        )

        assert abs(9900 + net_delta) <= 10000


class TestLPPosition:
    """Tests for LP position management."""

    def test_initial_state(self):
        """LP should start with correct initial state."""
        lp = LPPosition(capital=1_000_000, hedge_ratio=0.5)

        assert lp.initial_capital == 1_000_000
        assert lp.hedge_ratio == 0.5
        assert lp.option_exposure == 0
        assert lp.current_equity == 1_000_000

    def test_apply_funding(self):
        """Funding should affect PnL."""
        lp = LPPosition(capital=1_000_000)

        lp.apply_funding(1000)
        assert lp.total_pnl == 1000
        assert lp.current_equity == 1_001_000

        lp.apply_funding(-500)
        assert lp.total_pnl == 500

    def test_apply_trading_fee(self):
        """Trading fees should affect PnL."""
        lp = LPPosition(capital=1_000_000)

        lp.apply_trading_fee(1000, lp_share=0.2)
        assert lp.total_pnl == 200

    def test_update_position(self):
        """Position update should set correct exposure."""
        lp = LPPosition(capital=1_000_000)

        lp.update_position(net_trader_position=100)
        assert lp.option_exposure == -100  # LP is opposite

    def test_hedge_ratio_clamped(self):
        """Hedge ratio should be clamped to [0, 1]."""
        lp1 = LPPosition(capital=1000, hedge_ratio=-0.5)
        lp2 = LPPosition(capital=1000, hedge_ratio=1.5)

        assert lp1.hedge_ratio == 0.0
        assert lp2.hedge_ratio == 1.0


class TestEverlastingOptionSimulator:
    """Tests for the main simulator."""

    @pytest.fixture
    def basic_config(self):
        """Create a basic simulation config."""
        return SimulationConfig(
            entry_date=date(2024, 1, 1),
            exit_date=date(2024, 1, 31),  # 30 days
            random_seed=42,
            price_model=PriceModelConfig(
                initial_price=100,
                volatility=0.5,
            ),
            lp=LPConfig(
                capital=100_000,
                hedge_ratio=0.0,
            ),
            option=OptionConfig(
                option_type="call",
                strike=100,
            ),
        )

    def test_setup(self, basic_config):
        """Simulator should set up without errors."""
        sim = EverlastingOptionSimulator(basic_config)
        sim.setup()

        assert sim._initialized
        assert sim._oracle is not None
        assert sim._option is not None

    def test_run_step(self, basic_config):
        """Running a step should produce valid state."""
        sim = EverlastingOptionSimulator(basic_config)
        sim.setup()

        state = sim.run_step()

        assert state.day == 0
        assert state.spot_price == basic_config.price_model.initial_price
        assert state.lp_equity == basic_config.lp.capital

    def test_run_full(self, basic_config):
        """Full simulation should complete."""
        sim = EverlastingOptionSimulator(basic_config)
        result = sim.run()

        assert len(result.history) == basic_config.n_steps + 1
        assert result.config == basic_config

    def test_history_accessible(self, basic_config):
        """History should be accessible after run."""
        sim = EverlastingOptionSimulator(basic_config)
        sim.run()

        history = sim.get_history()
        assert len(history) > 0
        assert history[0].day == 0

    def test_equity_curve(self, basic_config):
        """Equity curve should have correct length."""
        sim = EverlastingOptionSimulator(basic_config)
        result = sim.run()

        assert len(result.equity_curve) == basic_config.n_steps + 1

    def test_pnl_components_sum(self, basic_config):
        """PnL components should approximately sum to total."""
        sim = EverlastingOptionSimulator(basic_config)
        result = sim.run()

        final = result.final_state
        components_sum = (
            final.cumulative_funding
            + final.cumulative_fees
            + final.cumulative_hedge_pnl
        )

        # Should be close (MTM might cause small differences)
        assert abs(components_sum - final.lp_pnl) < 100


class TestSimulationIntegration:
    """Integration tests for full simulation flow."""

    def test_hedged_vs_unhedged(self):
        """Hedged LP should have different results than unhedged."""
        base_config = {
            "entry_date": date(2024, 1, 1),
            "exit_date": date(2024, 2, 1),
            "random_seed": 42,
            "price_model": PriceModelConfig(initial_price=100, volatility=0.5),
            "option": OptionConfig(strike=100),
        }

        # Unhedged
        config1 = SimulationConfig(
            **base_config,
            lp=LPConfig(capital=100_000, hedge_ratio=0.0),
        )
        sim1 = EverlastingOptionSimulator(config1)
        result1 = sim1.run()

        # Hedged
        config2 = SimulationConfig(
            **base_config,
            lp=LPConfig(capital=100_000, hedge_ratio=1.0),
        )
        sim2 = EverlastingOptionSimulator(config2)
        result2 = sim2.run()

        # Results should differ
        assert result1.total_pnl != result2.total_pnl

    def test_call_vs_put(self):
        """Call and put simulations should produce different results."""
        base_config = {
            "entry_date": date(2024, 1, 1),
            "exit_date": date(2024, 2, 1),
            "random_seed": 42,
            "price_model": PriceModelConfig(initial_price=100),
            "lp": LPConfig(capital=100_000),
        }

        config1 = SimulationConfig(
            **base_config,
            option=OptionConfig(option_type="call", strike=100),
        )
        config2 = SimulationConfig(
            **base_config,
            option=OptionConfig(option_type="put", strike=100),
        )

        result1 = EverlastingOptionSimulator(config1).run()
        result2 = EverlastingOptionSimulator(config2).run()

        # Different option types should give different funding
        assert result1.final_state.cumulative_funding != result2.final_state.cumulative_funding


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
