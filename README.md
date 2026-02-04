# Everlasting Option Market Simulator

A research-grade simulator for oracle-dependent everlasting option markets (similar to Deri Protocol) with an interactive Streamlit web interface for LP performance analysis.

## Features

- **Jump-Diffusion Price Model**: Merton model with configurable drift, volatility, and jump parameters
- **Configurable Trader Mix**: Noise, momentum, and mean-reversion traders
- **DPMM Market Maker**: Price impact based on net position
- **Funding Mechanism**: Periodic funding to align mark price with payoff
- **Optional Delta Hedging**: Configure LP hedge ratio from 0% to 100%
- **Real-Time Playback**: Watch simulations unfold day by day
- **Comprehensive Analytics**: Performance metrics, PnL attribution, and parameter sensitivity

## Installation

```bash
# Clone or navigate to the project directory
cd everlasting_option_markets

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Web Interface (Recommended)

```bash
# Run the Streamlit app
streamlit run web/app.py
```

Then navigate to `http://localhost:8501` in your browser.

### CLI Usage

```bash
# Run the example simulation
python examples/run_simulation.py
```

## Project Structure

```
everlasting_option_markets/
├── src/
│   ├── pricing/
│   │   ├── black_scholes.py      # BS pricing + Greeks
│   │   └── everlasting.py        # Everlasting option pricing
│   ├── market/
│   │   ├── dpmm.py               # DPMM market maker logic
│   │   ├── funding.py            # Funding rate engine
│   │   └── oracle.py             # Price oracle interface
│   ├── simulation/
│   │   ├── price_paths.py        # Jump-Diffusion simulation
│   │   ├── traders.py            # Configurable trader behavior
│   │   ├── lp_position.py        # LP position + hedging
│   │   └── simulator.py          # Main simulation engine
│   ├── analysis/
│   │   ├── metrics.py            # Performance metrics
│   │   └── reporting.py          # PnL reports
│   └── config.py                 # Configuration dataclasses
├── web/
│   ├── app.py                    # Main Streamlit app
│   ├── pages/
│   │   ├── 1_simulation.py       # Simulation setup page
│   │   ├── 2_playback.py         # Real-time playback
│   │   └── 3_analysis.py         # Analysis & metrics
│   └── components/               # Reusable UI components
├── examples/
│   └── run_simulation.py         # CLI example
├── tests/                        # Unit tests
└── requirements.txt
```

## Web Interface Pages

### 1. Simulation Setup
Configure all simulation parameters:
- LP capital and hedge ratio
- Option type (call/put) and strike price
- Entry and exit dates
- Price model parameters (volatility, drift, jump intensity)
- Trader mix weights
- Market parameters (liquidity, fees)

### 2. Playback
Watch the simulation unfold with:
- Animated price and equity charts
- Live metrics display
- Playback controls (play, pause, speed, jump to day)
- Real-time PnL breakdown

### 3. Analysis
Comprehensive analysis including:
- Key performance metrics (Sharpe, Sortino, Max DD, Win Rate)
- PnL attribution charts
- Position and Greeks analysis
- Parameter sensitivity explorer
- Export results to CSV/JSON

## Key Concepts

### Everlasting Options
Perpetual options that use a funding mechanism instead of expiration:
- **Mark Price**: Adjusted by DPMM based on net trader position
- **Funding**: Periodic payment = (Mark - Payoff) / Period
- **LP Role**: Provides liquidity, counterparty to all trades

### LP Economics
The LP earns from:
1. **Funding Income**: When mark > payoff and traders are long
2. **Trading Fees**: Portion of fees from trade execution
3. **Hedge PnL**: Delta hedging gains/losses (if enabled)

### DPMM (Derivative Pricing Market Maker)
Adjusts mark price based on net position:
```
mark = theoretical * (1 + alpha * net_position / liquidity)
```

## Running Tests

```bash
pytest tests/ -v
```

## Example: Programmatic Usage

```python
from datetime import date
from src.config import SimulationConfig, PriceModelConfig, LPConfig, OptionConfig
from src.simulation import EverlastingOptionSimulator
from src.analysis.reporting import SimulationReport

# Configure simulation
config = SimulationConfig(
    entry_date=date(2024, 1, 1),
    exit_date=date(2024, 6, 30),
    random_seed=42,
    price_model=PriceModelConfig(
        initial_price=3000,
        volatility=0.80,
    ),
    lp=LPConfig(
        capital=1_000_000,
        hedge_ratio=0.5,
    ),
    option=OptionConfig(
        option_type="call",
        strike=3000,
    ),
)

# Run simulation
simulator = EverlastingOptionSimulator(config)
result = simulator.run()

# Generate report
report = SimulationReport(result)
report.print_summary()
```

## Dependencies

- numpy >= 1.24.0
- scipy >= 1.10.0
- pandas >= 2.0.0
- streamlit >= 1.28.0
- plotly >= 5.18.0
- pytest >= 7.0.0

## License

MIT License
