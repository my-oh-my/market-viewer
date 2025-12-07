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
    # pylint: disable=too-many-locals
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
    profile = market_profile[
        [
            "Price_Bin_Mid",
            "Bullish_Volume",
            "Bearish_Volume",
            "Total_Volume",
            "POC",
        ]
    ].copy()

    # Calculate Value Area (VA) - 70% of volume
    total_volume = profile["Total_Volume"].sum()
    value_area_vol = total_volume * 0.70

    # Sort by Total Volume descending to accumulate highest volume nodes first
    # This is a simplified approximation. Real Market Profile expands from POC out.
    # Let's implement the standard way: Start at POC, expand up/down by 1 row, pick larger, repeat.

    profile["In_VA"] = False

    # Find POC index in the original dataframe (not sorted)
    poc_idx = profile[profile["POC"]].index[0]

    # Initialize VA with POC
    current_vol = profile.loc[poc_idx, "Total_Volume"]
    profile.loc[poc_idx, "In_VA"] = True

    # Pointers
    up_idx = poc_idx + 1
    down_idx = poc_idx - 1

    while current_vol < value_area_vol:
        vol_up = 0
        vol_down = 0

        # Check bounds
        if up_idx < len(profile):
            vol_up = profile.loc[up_idx, "Total_Volume"]

        if down_idx >= 0:
            vol_down = profile.loc[down_idx, "Total_Volume"]

        if vol_up == 0 and vol_down == 0:
            break

        # Add the larger neighbor
        if vol_up > vol_down:
            current_vol += vol_up
            profile.loc[up_idx, "In_VA"] = True
            up_idx += 1
        else:
            current_vol += vol_down
            profile.loc[down_idx, "In_VA"] = True
            down_idx -= 1

    return profile


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
