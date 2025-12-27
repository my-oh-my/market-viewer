# Market Viewer

A Python-based market analysis application providing technical analysis tools for financial markets. This tool fetches data (yahoo finance), computes various indicators, and generates interactive dashboards for visualization.

## Features

- **Candlestick Charts**: Interactive OHLC charts with adjustable timeframes.
- **Stochastic Oscillator**: Momentum indicator comparing a particular closing price of a security to a range of its prices over a certain period of time.
- **Volume Profile (VP)**: Displays trading activity over a specified time period at specified price levels, highlighting the Value Area (VA) and Point of Control (POC).
- **Consolidation Detection**: Automatically detects and highlights consolidation zones (sideways market movement) based on ATR (Average True Range).
- **VWAP (Volume Weighted Average Price)**: Displays the VWAP line with Standard Deviation bands (2SD) to assess trend and volatility.
- **Support & Resistance**: Automatically detects and plots potential support and resistance levels.
- **Trend Indicator**: Dynamic support/resistance lines that ratchet based on trend direction (Uptrend logic based on Lows, Downtrend logic based on Highs).
- **Rate of Return (RoR) Distribution**: Visualizes the distribution of returns to understand market volatility.
- **Volume Distribution**: Histogram of volume traded.

## Setup

1.  **Create a virtual environment:**

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    pip install -r requirements-dev.txt
    ```

## Usage

The application is run via the command line.

### Basic Usage

Fetch data and plot basic charts for default symbols:

```bash
python -m src.main
```

### specific Symbols and Timeframes

```bash
python -m src.main --symbols "BTC-USD,ETH-USD" --period "1mo" --intervals "1h,4h"
```

### Advanced Analysis

Enable advanced features like Volume Profile, Consolidation, VWAP, Trend and Support/Resistance:

```bash
python -m src.main --symbols "SPY" --period "6mo" --intervals "1d" \
    --volume-profile --vp-bins 100 \
    --consolidation --consolidation-window 20 --consolidation-atr-multiplier 1.5 \
    --vwap \
    --trend --trend-shift 5 \
    --support-resistance \
    --ror
```

### CLI Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--symbols` | Comma-separated list of symbols (Yahoo Finance format). | Polish indices (W20/mWIG40/sWIG80) |
| `--period` | Data fetching period (e.g., '1d', '1mo', '1y'). | '1d' |
| `--intervals` | Comma-separated list of intervals (e.g., '1m', '5m', '1h', '1d'). | '1h' |
| `--k-window` | Window size for Stochastic %K line. | 14 |
| `--d-window` | Window size for Stochastic %D line. | 3 |
| `--volume-profile` | Enable Volume Profile calculation. | False |
| `--vp-bins` | Number of bins for Volume Profile. | 50 |
| `--consolidation` | Enable Consolidation zone detection. | False |
| `--consolidation-window` | Window size for consolidation check. | 20 |
| `--consolidation-atr-multiplier` | ATR multiplier for consolidation threshold. | 1.5 |
| `--vwap` | Enable VWAP with Standard Deviation bands. | False |
| `--trend` | Enable Trend Indictor. | False |
| `--trend-shift` | Shift parameter for Trend Indicator. | 5 |
| `--support-resistance` | Enable Support & Resistance detection. | False |
| `--ror` | Enable Rate of Return distribution plot. | False |
| `--save-html-dir` | Directory to save HTML charts instead of showing them. | None (Show chart) |
| `--plot-all` | Plot a chart for every symbol regardless of filters. | False |

## Development

This project uses Black for code formatting and Pylint for linting.

*   **To format the code:**

    ```bash
    black src/ tests/
    ```

*   **To run the linter:**

    ```bash
    pylint src/ tests/
    ```
