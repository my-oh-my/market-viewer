"""Module for volume-related analysis."""

import pandas as pd
import numpy as np


def calculate_volume_profile(data: pd.DataFrame, bins: int = 50) -> pd.DataFrame:
    """Calculates the Market Profile (Volume Profile) with Bullish/Bearish breakdown.

    Args:
        data: DataFrame containing 'Open', 'Close', and 'Volume' columns.
        bins: Number of bins to divide the price range into.

    Returns:
        DataFrame with columns:
            - 'Price_Bin_Mid': Midpoint of the price bin.
            - 'Bullish_Volume': Volume where Close > Open.
            - 'Bearish_Volume': Volume where Close <= Open.
            - 'Total_Volume': Sum of Bullish and Bearish volume.
            - 'POC': Boolean indicating if this bin is the Point of Control (max volume).
    """
    if data.empty:
        return pd.DataFrame()

    # Determine Candle Type
    data = data.copy()
    data["Candle_Type"] = np.where(data["Close"] > data["Open"], "Bullish", "Bearish")

    # Define price bins
    price_min = data["Close"].min()
    price_max = data["Close"].max()
    price_range = price_max - price_min

    if price_range == 0:
        # Handle case with no price movement
        bin_edges = np.array([price_min - 0.5, price_max + 0.5])
    else:
        # Use linspace to ensure we cover the exact range
        bin_edges = np.linspace(price_min, price_max, bins + 1)

    # Assign each row to a bin based on Close price
    data["Price_Bin"] = pd.cut(data["Close"], bins=bin_edges, include_lowest=True)

    # Group by Price Bin and Candle Type
    market_profile = (
        data.groupby(["Price_Bin", "Candle_Type"], observed=False)["Volume"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Calculate Price Mid
    market_profile["Price_Bin_Mid"] = market_profile["Price_Bin"].apply(lambda x: x.mid)

    # Ensure Bullish and Bearish columns exist
    if "Bullish" not in market_profile.columns:
        market_profile["Bullish"] = 0
    if "Bearish" not in market_profile.columns:
        market_profile["Bearish"] = 0

    # Rename for clarity
    market_profile.rename(
        columns={"Bullish": "Bullish_Volume", "Bearish": "Bearish_Volume"}, inplace=True
    )

    # Calculate Total Volume
    market_profile["Total_Volume"] = (
        market_profile["Bullish_Volume"] + market_profile["Bearish_Volume"]
    )

    # Identify Point of Control (POC)
    max_vol_idx = market_profile["Total_Volume"].idxmax()
    market_profile["POC"] = False
    market_profile.loc[max_vol_idx, "POC"] = True

    # Select and order columns
    return market_profile[
        [
            "Price_Bin_Mid",
            "Bullish_Volume",
            "Bearish_Volume",
            "Total_Volume",
            "POC",
        ]
    ]


def calculate_volume_percentiles(data: pd.DataFrame, window: int = 50) -> pd.Series:
    """Calculates the rolling percentile of the current volume relative to the past window.

    Args:
        data: DataFrame containing 'Volume' column.
        window: The lookback window for percentile calculation.

    Returns:
        Series containing the volume percentile (0.0 to 1.0).
    """
    if data.empty or "Volume" not in data.columns:
        return pd.Series(dtype=float)

    # Calculate rolling rank (percentile)
    # pct=True returns values between 0 and 1
    return data["Volume"].rolling(window=window).rank(pct=True)
