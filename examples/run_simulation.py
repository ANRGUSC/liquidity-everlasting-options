#!/usr/bin/env python3
"""Example script demonstrating CLI usage of the everlasting option simulator."""

import sys
from pathlib import Path
from datetime import date

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    SimulationConfig,
    PriceModelConfig,
    TraderMixConfig,
    MarketConfig,
    LPConfig,
    OptionConfig,
)
from src.simulation import EverlastingOptionSimulator
from src.analysis.reporting import SimulationReport


def main():
    """Run a sample simulation and print results."""
    print("=" * 60)
    print("Everlasting Option Market Simulator - CLI Example")
    print("=" * 60)

    # Configure simulation
    config = SimulationConfig(
        entry_date=date(2024, 1, 1),
        exit_date=date(2024, 6, 30),
        random_seed=42,
        price_model=PriceModelConfig(
            initial_price=3000.0,
            drift=0.05,
            volatility=0.80,
            jump_intensity=10.0,
            jump_mean=-0.02,
            jump_std=0.05,
        ),
        trader_mix=TraderMixConfig(
            noise_weight=0.50,
            momentum_weight=0.30,
            mean_reversion_weight=0.20,
            trades_per_day=100,
            avg_trade_size=10.0,
        ),
        market=MarketConfig(
            pool_liquidity=10_000_000.0,
            funding_period_days=1,
            trading_fee_rate=0.001,
            lp_fee_share=0.20,
        ),
        lp=LPConfig(
            capital=1_000_000.0,
            hedge_ratio=0.5,  # 50% delta hedge
        ),
        option=OptionConfig(
            option_type="call",
            strike=3000.0,
        ),
    )

    print(f"\nSimulation Period: {config.entry_date} to {config.exit_date}")
    print(f"Days: {config.n_days}")
    print(f"Option: {config.option.option_type.upper()} @ ${config.option.strike:,.0f}")
    print(f"Initial Capital: ${config.lp.capital:,.0f}")
    print(f"Hedge Ratio: {config.lp.hedge_ratio * 100:.0f}%")

    # Run simulation
    print("\nRunning simulation...")
    simulator = EverlastingOptionSimulator(config)
    result = simulator.run()

    # Generate report
    report = SimulationReport(result)
    report.print_summary()

    # Save to CSV
    output_path = Path(__file__).parent / "simulation_results.csv"
    report.to_csv(str(output_path))
    print(f"\nResults saved to: {output_path}")

    return result


def run_comparison():
    """Run comparison of hedged vs unhedged LP."""
    print("\n" + "=" * 60)
    print("Hedge Comparison: 0% vs 50% vs 100% Delta Hedge")
    print("=" * 60)

    results = {}

    for hedge_ratio in [0.0, 0.5, 1.0]:
        config = SimulationConfig(
            entry_date=date(2024, 1, 1),
            exit_date=date(2024, 6, 30),
            random_seed=42,  # Same seed for comparison
            price_model=PriceModelConfig(
                initial_price=3000.0,
                volatility=0.80,
            ),
            lp=LPConfig(
                capital=1_000_000.0,
                hedge_ratio=hedge_ratio,
            ),
            option=OptionConfig(
                option_type="call",
                strike=3000.0,
            ),
        )

        simulator = EverlastingOptionSimulator(config)
        result = simulator.run()
        report = SimulationReport(result)
        metrics = report.metrics

        results[hedge_ratio] = {
            "return": metrics["total_return_pct"],
            "sharpe": metrics["sharpe_ratio"],
            "max_dd": metrics["max_drawdown_pct"],
        }

        print(f"\nHedge Ratio: {hedge_ratio * 100:.0f}%")
        print(f"  Return: {metrics['total_return_pct']:.2f}%")
        print(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max DD: {metrics['max_drawdown_pct']:.2f}%")

    return results


if __name__ == "__main__":
    main()
    run_comparison()
