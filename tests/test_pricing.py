"""Tests for pricing modules."""

import pytest
import math
from src.pricing.black_scholes import bs_call_price, bs_put_price, bs_greeks
from src.pricing.everlasting import EverlastingOption


class TestBlackScholes:
    """Tests for Black-Scholes pricing."""

    def test_call_price_atm(self):
        """Test ATM call price."""
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.20

        price = bs_call_price(S, K, T, r, sigma)

        # ATM call should be roughly 10-12% of spot for these params
        assert 8 < price < 15
        assert price > 0

    def test_put_price_atm(self):
        """Test ATM put price."""
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.20

        price = bs_put_price(S, K, T, r, sigma)

        assert price > 0
        assert price < S  # Put can't be worth more than underlying

    def test_put_call_parity(self):
        """Verify put-call parity: C - P = S - K*exp(-rT)."""
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.20

        call = bs_call_price(S, K, T, r, sigma)
        put = bs_put_price(S, K, T, r, sigma)

        # C - P = S - K*exp(-rT)
        lhs = call - put
        rhs = S - K * math.exp(-r * T)

        assert abs(lhs - rhs) < 1e-10

    def test_call_intrinsic_at_expiry(self):
        """Call at expiry equals intrinsic value."""
        S = 110
        K = 100
        T = 0
        r = 0.05
        sigma = 0.20

        price = bs_call_price(S, K, T, r, sigma)
        assert abs(price - 10) < 1e-10

    def test_put_intrinsic_at_expiry(self):
        """Put at expiry equals intrinsic value."""
        S = 90
        K = 100
        T = 0
        r = 0.05
        sigma = 0.20

        price = bs_put_price(S, K, T, r, sigma)
        assert abs(price - 10) < 1e-10

    def test_greeks_call_delta_bounds(self):
        """Call delta should be between 0 and 1."""
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.20

        greeks = bs_greeks(S, K, T, r, sigma, "call")

        assert 0 <= greeks["delta"] <= 1

    def test_greeks_put_delta_bounds(self):
        """Put delta should be between -1 and 0."""
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.20

        greeks = bs_greeks(S, K, T, r, sigma, "put")

        assert -1 <= greeks["delta"] <= 0

    def test_greeks_gamma_positive(self):
        """Gamma should be positive for both calls and puts."""
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.20

        call_greeks = bs_greeks(S, K, T, r, sigma, "call")
        put_greeks = bs_greeks(S, K, T, r, sigma, "put")

        assert call_greeks["gamma"] > 0
        assert put_greeks["gamma"] > 0
        # Gamma is same for calls and puts
        assert abs(call_greeks["gamma"] - put_greeks["gamma"]) < 1e-10

    def test_greeks_vega_positive(self):
        """Vega should be positive."""
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.20

        greeks = bs_greeks(S, K, T, r, sigma, "call")
        assert greeks["vega"] > 0

    def test_deep_itm_call_delta(self):
        """Deep ITM call should have delta close to 1."""
        S = 150
        K = 100
        T = 0.5
        r = 0.05
        sigma = 0.20

        greeks = bs_greeks(S, K, T, r, sigma, "call")
        assert greeks["delta"] > 0.95

    def test_deep_otm_call_delta(self):
        """Deep OTM call should have delta close to 0."""
        S = 50
        K = 100
        T = 0.5
        r = 0.05
        sigma = 0.20

        greeks = bs_greeks(S, K, T, r, sigma, "call")
        assert greeks["delta"] < 0.05


class TestEverlastingOption:
    """Tests for everlasting option pricing."""

    def test_price_positive(self):
        """Everlasting option price should be positive."""
        option = EverlastingOption(strike=100, option_type="call")
        price = option.price(spot=100, volatility=0.5)

        assert price > 0

    def test_call_price_increases_with_spot(self):
        """Call price should increase with spot price."""
        option = EverlastingOption(strike=100, option_type="call")

        price_low = option.price(spot=90, volatility=0.5)
        price_high = option.price(spot=110, volatility=0.5)

        assert price_high > price_low

    def test_put_price_decreases_with_spot(self):
        """Put price should decrease with spot price."""
        option = EverlastingOption(strike=100, option_type="put")

        price_low = option.price(spot=90, volatility=0.5)
        price_high = option.price(spot=110, volatility=0.5)

        assert price_low > price_high

    def test_price_increases_with_volatility(self):
        """Option price should increase with volatility."""
        option = EverlastingOption(strike=100, option_type="call")

        price_low_vol = option.price(spot=100, volatility=0.2)
        price_high_vol = option.price(spot=100, volatility=0.8)

        assert price_high_vol > price_low_vol

    def test_payoff_call(self):
        """Call payoff should be max(S - K, 0)."""
        option = EverlastingOption(strike=100, option_type="call")

        assert option.payoff(spot=110) == 10
        assert option.payoff(spot=90) == 0

    def test_payoff_put(self):
        """Put payoff should be max(K - S, 0)."""
        option = EverlastingOption(strike=100, option_type="put")

        assert option.payoff(spot=110) == 0
        assert option.payoff(spot=90) == 10

    def test_funding_rate_positive_when_mark_above_payoff(self):
        """Funding rate should be positive when mark > payoff."""
        option = EverlastingOption(strike=100, option_type="call")

        # When ATM, mark price (with time value) > payoff (0)
        spot = 100
        mark_price = 15  # Some time value
        payoff = option.payoff(spot)

        funding = option.funding_rate(mark_price, spot)

        assert funding > 0  # Longs pay shorts

    def test_greeks_delta_bounds(self):
        """Everlasting option delta should have reasonable bounds."""
        call = EverlastingOption(strike=100, option_type="call")
        put = EverlastingOption(strike=100, option_type="put")

        call_greeks = call.greeks(spot=100, volatility=0.5)
        put_greeks = put.greeks(spot=100, volatility=0.5)

        assert 0 < call_greeks["delta"] < 1
        assert -1 < put_greeks["delta"] < 0


class TestPricingIntegration:
    """Integration tests for pricing consistency."""

    def test_everlasting_higher_than_vanilla(self):
        """Everlasting option should be worth more than short-dated vanilla."""
        from src.pricing.black_scholes import bs_call_price

        ever = EverlastingOption(strike=100, option_type="call")
        ever_price = ever.price(spot=100, volatility=0.5)

        # 1-week vanilla should be worth less
        vanilla_price = bs_call_price(S=100, K=100, T=7/365, r=0, sigma=0.5)

        assert ever_price > vanilla_price

    def test_greeks_finite_difference(self):
        """Verify delta using finite difference."""
        option = EverlastingOption(strike=100, option_type="call")

        spot = 100
        vol = 0.5
        h = 0.01

        price_up = option.price(spot=spot + h, volatility=vol)
        price_down = option.price(spot=spot - h, volatility=vol)

        fd_delta = (price_up - price_down) / (2 * h)
        greeks = option.greeks(spot=spot, volatility=vol)

        # Should be within 1% of finite difference
        assert abs(greeks["delta"] - fd_delta) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
