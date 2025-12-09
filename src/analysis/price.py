"""
Module for price-related analysis.

This module contains functions for analyzing price patterns, including
consolidation detection, support and resistance level identification,
and Rate of Return (RoR) calculation.
"""

import pandas as pd


def detect_consolidation(
    data: pd.DataFrame,
    window: int = 20,
    threshold_multiplier: float = 1.5,
    use_atr: bool = True,
) -> pd.Series:
    """
    Detects consolidation periods where price stays within a narrow range.

    Consolidation is identified when the price range (High - Low or variation in Close)
    over a specified window is below a certain threshold. The threshold can be dynamic
    (based on ATR) or percentage-based.

    Args:
        data (pd.DataFrame): DataFrame containing 'High', 'Low', 'Close' and optionally 'ATR'.
        window (int, optional): The rolling window size to check for consolidation. Defaults to 20.
        threshold_multiplier (float, optional): Multiplier for ATR to define the range threshold,
                                                or the percentage value if use_atr is False.
                                                Defaults to 1.5.
        use_atr (bool, optional): Whether to use ATR for dynamic thresholding. Defaults to True.

    Returns:
        pd.Series: A Boolean Series indicating if the corresponding period is in consolidation.
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

    price_range = rolling_max - rolling_min

    if use_atr and "ATR" in data.columns:
        # Dynamic threshold: Range < Multiplier * ATR
        # We use the ATR at the end of the window (current candle)
        threshold = data["ATR"] * threshold_multiplier
        return price_range <= threshold

    # Fallback to percentage based
    # Note: threshold_multiplier is treated as percentage here (e.g. 2.0 for 2%)
    price_range_percent = (price_range / rolling_min) * 100
    return price_range_percent <= threshold_multiplier


def detect_support_resistance(data: pd.DataFrame, window: int = 5) -> list[float]:
    """
    Detects Support and Resistance levels using local extrema (fractals).

    Uses a rolling window to identify local maximums (resistance) and local minimums (support).

    Args:
        data (pd.DataFrame): DataFrame containing 'High' and 'Low' columns.
        window (int, optional): Window size for local extrema detection. Should be an odd number.
                                Defaults to 5.

    Returns:
        list[float]: A sorted list of unique price levels identified as support or resistance.
    """
    if data.empty or "High" not in data.columns or "Low" not in data.columns:
        return []

    levels = []

    # Simple fractal / local extrema detection
    # A high is a resistance if it's higher than 'n' neighbors
    # A low is a support if it's lower than 'n' neighbors

    # We can use rolling max/min with center=True
    # But rolling with center=True looks ahead, which is fine for historical analysis
    # For live trading, we'd only know it 'window//2' bars later.

    # Resistance (Local Max)
    # We shift to align the window so that the comparison is valid for the centered element
    # Actually, simpler: iterate or use shift logic
    # is_max = data['High'] == data['High'].rolling(window=window, center=True).max()

    # Using rolling with center=True requires future data (fine for this viewer)
    rolling_max = data["High"].rolling(window=window, center=True).max()
    rolling_min = data["Low"].rolling(window=window, center=True).min()

    resistance_mask = data["High"] == rolling_max
    support_mask = data["Low"] == rolling_min

    # Extract levels
    res_levels = data.loc[resistance_mask, "High"].tolist()
    sup_levels = data.loc[support_mask, "Low"].tolist()

    levels = sorted(list(set(res_levels + sup_levels)))

    # Optional: Cluster nearby levels (simple version: round to nearest 0.5 or 1% diff)
    # For now, return all raw levels.
    return levels


def calculate_ror(data: pd.DataFrame) -> pd.Series:
    """
    Calculates the Rate of Return (percentage change) of the Close price.

    Args:
        data (pd.DataFrame): DataFrame containing the 'Close' column.

    Returns:
        pd.Series: A Series containing the percentage change (e.g., 1.5 for 1.5%).
    """
    if data.empty or "Close" not in data.columns:
        return pd.Series(dtype=float)

    return data["Close"].pct_change() * 100
