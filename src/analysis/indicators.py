"""
Module for general technical indicators.

This module provides functions to calculate common technical indicators such as
Volume Weighted Average Price (VWAP) and Average True Range (ATR).
"""

import pandas as pd


def calculate_vwap(
    data: pd.DataFrame, window: int | None = None, num_std: float = 2.0
) -> pd.DataFrame:
    """
    Calculates Volume Weighted Average Price (VWAP) and Standard Deviation Bands.

    VWAP is calculated as the cumulative sum of price * volume divided by cumulative volume.
    Standard deviation bands are approximated using the standard deviation of the close price.

    Args:
        data (pd.DataFrame): DataFrame containing 'High', 'Low', 'Close', and 'Volume' columns.
        window (int | None, optional): Rolling window size. If None, calculates cumulative VWAP
                                       anchored to the start of the data. Defaults to None.
        num_std (float, optional): Number of standard deviations for the upper/lower bands.
                                   Defaults to 2.0.

    Returns:
        pd.DataFrame: A DataFrame with columns 'VWAP', 'VWAP_Upper', and 'VWAP_Lower',
                      indexed by the same index as the input data.
    """
    if data.empty:
        return pd.DataFrame()

    # Typical Price
    tp = (data["High"] + data["Low"] + data["Close"]) / 3
    tp_v = tp * data["Volume"]

    if window:
        # Rolling VWAP
        cum_vol = data["Volume"].rolling(window=window).sum()
        cum_tp_v = tp_v.rolling(window=window).sum()
    else:
        # Cumulative VWAP (Anchored to start of data)
        cum_vol = data["Volume"].cumsum()
        cum_tp_v = tp_v.cumsum()

    vwap = cum_tp_v / cum_vol

    # Standard Deviation Calculation
    # VWAP variance is weighted variance
    # Var = Sum(Vol * (TP - VWAP)^2) / Sum(Vol)
    # This is computationally expensive to do rolling exactly without a loop or complex vectorization
    # Approximation: Standard deviation of the Close price relative to VWAP

    # Let's use a simpler approximation for bands often used in trading:
    # Std Dev of the price itself over the same window/cumulative
    if window:
        std = data["Close"].rolling(window=window).std()
    else:
        std = data["Close"].expanding().std()

    upper = vwap + (std * num_std)
    lower = vwap - (std * num_std)

    return pd.DataFrame(
        {"VWAP": vwap, "VWAP_Upper": upper, "VWAP_Lower": lower}, index=data.index
    )


def calculate_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Calculates Average True Range (ATR).

    ATR is a measure of volatility. It is calculated as the rolling mean of the True Range (TR).
    True Range is the maximum of:
    - High - Low
    - |High - Previous Close|
    - |Low - Previous Close|

    Args:
        data (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
        window (int, optional): The smoothing window size. Defaults to 14.

    Returns:
        pd.Series: A Series containing the ATR values.
    """
    if data.empty:
        return pd.Series(dtype=float)

    high = data["High"]
    low = data["Low"]
    close = data["Close"]

    # TR = Max(High - Low, |High - PrevClose|, |Low - PrevClose|)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR is usually a smoothed average of TR (Wilder's smoothing or EMA)
    # Here we use simple rolling mean for simplicity, or EMA
    atr = tr.rolling(window=window).mean()

    return atr
