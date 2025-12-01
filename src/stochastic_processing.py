"""Module for calculating stochastic indicators."""

import pandas as pd
from ta.momentum import StochasticOscillator


def calculate_stochastic(
    data: pd.DataFrame, k_window: int = 14, d_window: int = 3
) -> pd.DataFrame:
    """Calculates the Stochastic Oscillator for the given data using the 'ta' library."""
    if data is None or data.empty:
        raise ValueError("Input data cannot be empty.")

    if not all(col in data.columns for col in ["High", "Low", "Close"]):
        raise ValueError("Input data must contain 'High', 'Low', and 'Close' columns.")

    stoch = StochasticOscillator(
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        window=k_window,
        smooth_window=d_window,
    )
    data["%K"] = stoch.stoch()
    data["%D"] = stoch.stoch_signal()

    return data
