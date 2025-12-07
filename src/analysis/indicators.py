"""Module for general technical indicators."""

import pandas as pd


def calculate_vwap(
    data: pd.DataFrame, window: int | None = None, num_std: float = 2.0
) -> pd.DataFrame:
    """Calculates Volume Weighted Average Price (VWAP) and Standard Deviation Bands.

    Args:
        data: DataFrame containing 'High', 'Low', 'Close', 'Volume'.
              If 'Datetime' is present, it can be used for anchoring (not implemented yet,
              defaulting to rolling or full series).
        window: Rolling window size. If None, calculates cumulative VWAP (anchored to start).
        num_std: Number of standard deviations for the bands.

    Returns:
        DataFrame with columns 'VWAP', 'VWAP_Upper', 'VWAP_Lower'.
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
    """Calculates Average True Range (ATR).

    Args:
        data: DataFrame containing 'High', 'Low', 'Close'.
        window: Smoothing window.

    Returns:
        Series containing ATR values.
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
