"""Module for price-related analysis."""

import pandas as pd


def detect_consolidation(
    data: pd.DataFrame, window: int = 20, threshold_percent: float = 2.0
) -> pd.Series:
    """Detects consolidation periods where price stays within a narrow range.

    Args:
        data: DataFrame containing 'High' and 'Low' (or 'Close') columns.
        window: The rolling window size to check for consolidation.
        threshold_percent: The maximum percentage difference between max and min price
                           in the window to be considered consolidation.

    Returns:
        Boolean Series indicating if the corresponding period is in consolidation.
    """
    if data.empty:
        return pd.Series(dtype=bool)

    # Use High and Low if available, otherwise fallback to Close
    if "High" in data.columns and "Low" in data.columns:
        rolling_max = data["High"].rolling(window=window).max()
        rolling_min = data["Low"].rolling(window=window).min()
    else:
        rolling_max = data["Close"].rolling(window=window).max()
        rolling_min = data["Close"].rolling(window=window).min()

    price_range_percent = (rolling_max - rolling_min) / rolling_min * 100

    return price_range_percent <= threshold_percent


def calculate_ror(data: pd.DataFrame) -> pd.Series:
    """Calculates the Rate of Return (percentage change) of the Close price.

    Args:
        data: DataFrame containing 'Close' column.

    Returns:
        Series containing the percentage change (e.g., 1.5 for 1.5%).
    """
    if data.empty or "Close" not in data.columns:
        return pd.Series(dtype=float)

    return data["Close"].pct_change() * 100
