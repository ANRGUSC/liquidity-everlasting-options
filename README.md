# Everlasting Option Market Simulator

Backtest LP strategies on perpetual options using real or simulated price data.

**[Live Demo](https://liquidityeverlastingoptions.streamlit.app/)**

## Overview

A simulator for oracle-dependent everlasting option markets (based on [Deri Protocol](https://docs.deri.io/)) that allows liquidity providers to backtest strategies with different hedging approaches, trader behaviors, and market conditions.

## Features

- **Real Historical Data**: Fetch actual prices from CoinGecko (BTC/ETH) or Yahoo Finance (SPY)
- **Multiple Price Models**: GBM (Black-Scholes), Jump-Diffusion, or Historical Data
- **DPMM Market Maker**: Mark price adjustment based on net position ([Deri Protocol spec](https://docs.deri.io/how-it-works/dpmm-proactive-market-making))
- **Everlasting Option Pricing**: Weighted Black-Scholes series ([Paradigm paper](https://www.paradigm.xyz/2021/05/everlasting-options))
- **Delta Hedging**: Configurable hedge ratio (0-100%)
- **Configurable Trader Mix**: Noise, momentum, and mean-reversion traders
- **Real-Time Animation**: Watch simulations unfold day by day
- **Comprehensive Analytics**: Sharpe ratio, max drawdown, PnL attribution

## Quick Start

### Live Demo
Visit **[liquidityeverlastingoptions.streamlit.app](https://liquidityeverlastingoptions.streamlit.app/)**

### Local Installation

```bash
git clone https://github.com/ANRGUSC/liquidity-everlasting-options.git
cd liquidity-everlasting-options
pip install -r requirements.txt
streamlit run web/app.py
```

## How It Works

### Everlasting Options
Perpetual options using funding instead of expiration:
- **Pricing**: Weighted sum of Black-Scholes prices at increasing maturities
- **Funding**: `(Mark - Payoff) / Period` — longs pay shorts when mark > payoff
- **Mark Price**: Adjusted by DPMM based on net position

### LP Economics
The LP (counterparty to all traders) earns from:
1. **Funding Income**: When mark > payoff and traders are net long
2. **Trading Fees**: Share of fees from trade execution
3. **Hedge PnL**: Delta hedging gains/losses (if enabled)

### Key Formulas

**Everlasting Option Price** ([Paradigm](https://www.paradigm.xyz/2021/05/everlasting-options)):
```
Price = Σ (0.5^i) × BlackScholes(T=i×period)
```

**DPMM Mark Price** ([Deri Protocol](https://docs.deri.io/how-it-works/dpmm-proactive-market-making)):
```
Mark = Theoretical × (1 + k × NetPosition / Liquidity)
```

**Funding Rate** ([Deri Protocol](https://docs.deri.io/how-it-works/funding-fee)):
```
Funding = (Mark - Payoff) / FundingPeriod
```

**Delta Hedge**:
```
HedgePosition = -OptionExposure × Delta × HedgeRatio
```

## Project Structure

```
├── src/
│   ├── pricing/
│   │   ├── black_scholes.py    # BS pricing + Greeks
│   │   └── everlasting.py      # Everlasting option pricing
│   ├── market/
│   │   ├── dpmm.py             # DPMM market maker
│   │   ├── funding.py          # Funding rate engine
│   │   └── oracle.py           # Price oracle
│   ├── simulation/
│   │   ├── price_paths.py      # GBM, Jump-Diffusion, Historical
│   │   ├── traders.py          # Trader behavior models
│   │   ├── lp_position.py      # LP position + hedging
│   │   └── simulator.py        # Main simulation engine
│   ├── analysis/
│   │   ├── metrics.py          # Performance metrics
│   │   └── reporting.py        # Reports + export
│   └── config.py               # Configuration
├── web/
│   └── app.py                  # Streamlit dashboard
├── tests/                      # Unit tests
└── requirements.txt
```

## Configuration Options

| Parameter | Description |
|-----------|-------------|
| **Price Model** | Black-Scholes (GBM), Jump-Diffusion, or Historical Data |
| **Historical Asset** | BTC, ETH, or SPY (fetches real prices) |
| **Hedge Ratio** | 0% (no hedge) to 100% (full delta hedge) |
| **Trader Mix** | Noise / Momentum / Mean-Reversion weights |
| **Pool Liquidity** | Higher = less price impact |
| **Trading Fee** | Fee rate per trade |

## Example: Programmatic Usage

```python
from datetime import date
from src.config import SimulationConfig, PriceModelConfig, LPConfig
from src.simulation import EverlastingOptionSimulator

config = SimulationConfig(
    entry_date=date(2024, 1, 1),
    exit_date=date(2024, 6, 30),
    price_model=PriceModelConfig(
        model_type="historical",
        historical_asset="BTC",
    ),
    lp=LPConfig(capital=1_000_000, hedge_ratio=0.5),
)

simulator = EverlastingOptionSimulator(config)
result = simulator.run()
print(f"Total Return: {result.total_return:.2%}")
```

## References

- [Everlasting Options - Paradigm Research](https://www.paradigm.xyz/2021/05/everlasting-options)
- [Deri Protocol Documentation](https://docs.deri.io/)
- [DPMM Mechanism](https://docs.deri.io/how-it-works/dpmm-proactive-market-making)
- [Funding Fee](https://docs.deri.io/how-it-works/funding-fee)

## Citation

If you use this simulator for replication, extension, or further research, please cite:

```bibtex
@inproceedings{mohanty2025proactive,
  title={Proactive Market Making and Liquidity Analysis for Everlasting Options in DeFi Ecosystems},
  author={Mohanty, Hardhik and Zaarour, Giovanni and Krishnamachari, Bhaskar},
  booktitle={2025 IEEE International Conference on Blockchain and Cryptocurrency (ICBC)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}
```

## License

MIT License

## Disclaimer

This simulator is intended for educational and research purposes. While it models key mechanisms of everlasting option markets, it operates under simplified assumptions and may not fully capture the complexities of real-world DeFi ecosystems, including network latency, gas costs, oracle delays, liquidity fragmentation, and extreme market conditions. Results should be interpreted as directional insights rather than precise predictions.
