"""
Module for trend-following indicators.

This module provides functions to calculate dynamic trend indicators that
adjust support and resistance levels based on price movement, similar to
SuperTrend or Chandelier Exit logic.
"""

import pandas as pd
import numpy as np


# pylint: disable=too-many-locals
def calculate_trend_indicator(data: pd.DataFrame, shift: int = 5) -> pd.DataFrame:
    """
    Calculates a trend-following indicator with ratcheting support/resistance levels.

    The indicator plots a line that acts as dynamic support in an uptrend and
    dynamic resistance in a downtrend.

    Logic:
    - Uptrend: Indicator is based on the Lowest Low of the last 'shift' periods.
      It ratchets up (can only go higher or flat).
      If the current Low drops below the indicator, the trend switches to Downtrend.
    - Downtrend: Indicator is based on the Highest High of the last 'shift' periods.
      It ratchets down (can only go lower or flat).
      If the current High rises above the indicator, the trend switches to Uptrend.

    Args:
        data (pd.DataFrame): DataFrame containing 'High', 'Low'.
        shift (int): The window size for the rolling min/max calculation.

    Returns:
        pd.DataFrame: A DataFrame with columns:
                      - 'Trend_Indicator': The value of the indicator.
                      - 'Trend_Direction': 1 for Uptrend, -1 for Downtrend.
    """
    if data.empty:
        return pd.DataFrame(
            columns=["Trend_Indicator", "Trend_Direction"], index=data.index
        )

    high = data["High"]
    low = data["Low"]

    # Pre-calculate Rolling Min/Max
    # We look at the past 'shift' periods excluding the current one.
    # So we shift by 1 first, then apply rolling min/max.
    # e.g. at index t, we want min(Low[t-shift : t-1])

    # Note: rolling(window=shift) on a shifted series:
    # shift(1) moves t-1 to t.
    # rolling(N) at t looks at t, t-1, ... t-N+1.
    # Because we shifted 1, this corresponds to original indices t-1, t-2, ... t-N.
    # Which is exactly "last 'shift' periods excluding current".

    rolling_min = low.shift(1).rolling(window=shift).min()
    rolling_max = high.shift(1).rolling(window=shift).max()

    # Output arrays
    n = len(data)
    trend_indicator = np.zeros(n)
    trend_direction = np.zeros(n, dtype=int)

    # Initialize state
    # We need a valid starting point.
    # Using 'shift' + 1 or similar to ensure we have enough data?
    # rolling() produces NaNs for first 'shift' elements (indices 0 to shift-1).
    # So index 'shift' is the first valid one?
    # Wait: shift(1) adds a NaN at 0. rolling(k) adds k-1 NaNs. Total k NaNs?
    # series: [1, 2, 3, 4, 5]
    # shift(1): [NaN, 1, 2, 3, 4]
    # rolling(2):
    #   i=0: NaN
    #   i=1: NaN (needs 2 vals, has NaN, 1. Min is NaN?) -> usually yes if min_periods=None (default window size)
    #   i=2: min(1, 2) = 1. Valid.
    # So first valid index is 'shift'.

    start_index = shift
    if n <= start_index:
        return pd.DataFrame(
            {"Trend_Indicator": [np.nan] * n, "Trend_Direction": [0] * n},
            index=data.index,
        )

    # Initial state determination at start_index
    # Default to Uptrend using the Rolling Min
    current_trend = 1
    current_level = rolling_min.iloc[start_index]

    if pd.isna(current_level):
        # Fallback if somehow still NaN
        current_level = low.iloc[start_index]

    trend_indicator[start_index] = current_level
    trend_direction[start_index] = current_trend

    # Check initial validity
    if low.iloc[start_index] < current_level:
        current_trend = -1
        current_level = rolling_max.iloc[start_index]
        trend_direction[start_index] = current_trend
        trend_indicator[start_index] = current_level

    # Convert to numpy for iteration speed
    low_vals = low.values
    high_vals = high.values
    rolling_min_vals = rolling_min.values
    rolling_max_vals = rolling_max.values

    # Fill pre-start with NaNs
    trend_indicator[:start_index] = np.nan
    trend_direction[:start_index] = 0

    for i in range(start_index + 1, n):
        curr_low = low_vals[i]
        curr_high = high_vals[i]
        r_min = rolling_min_vals[i]
        r_max = rolling_max_vals[i]

        if pd.isna(r_min) or pd.isna(r_max):
            trend_indicator[i] = np.nan
            trend_direction[i] = 0
            continue

        if current_trend == 1:  # Uptrend
            # Ratchet logic: Can only go up or stay flat

            # Check for switch FIRST
            if curr_low < current_level:
                # BREAK: Trend switches to Downtrend
                current_trend = -1
                # Reset level to the rolling max
                current_level = r_max
            else:
                # Continue Uptrend
                # Logic: max(candidate, previous_level)
                # Candidate is rolling min
                current_level = max(r_min, current_level)

        else:  # Downtrend
            # Downtrend logic

            # Check for switch
            if curr_high > current_level:
                # BREAK: Trend switches to Uptrend
                current_trend = 1
                # Reset level to the rolling min
                current_level = r_min
            else:
                # Continue Downtrend
                current_level = min(r_max, current_level)

        trend_indicator[i] = current_level
        trend_direction[i] = current_trend

    return pd.DataFrame(
        {"Trend_Indicator": trend_indicator, "Trend_Direction": trend_direction},
        index=data.index,
    )
