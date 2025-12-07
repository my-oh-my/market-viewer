"""Module for orchestrating market data processing and analysis."""

import argparse
import pandas as pd
from src.data_fetcher import fetch_market_data
from src.stochastic_processing import calculate_stochastic
from src.analysis import volume, price, indicators
from src.chart_generator import generate_analysis_chart


def process_symbol(symbol: str, args: argparse.Namespace):
    """Processes a single symbol: fetches data, runs analysis, and generates charts.

    Args:
        symbol: The market symbol to process.
        args: Parsed CLI arguments.
    """
    print(f"\nProcessing symbol: {symbol}")
    intervals = [interval.strip() for interval in args.intervals.split(",")]

    # Dictionary to store processed data for each interval
    # Key: interval, Value: DataFrame with all indicators
    processed_data = {}

    # We also need to store Volume Profile separately as it's not a time-series aligned with the main DF in the same way
    # (It's price-based, not time-based). But for simplicity, we can calculate it on the lowest interval data
    # or the first interval. Let's store it in a separate dict or just pass it along.
    # For now, let's calculate volume profile on the first interval (usually the most granular if sorted).
    volume_profiles = {}
    support_resistance_levels = []

    # Re-implementing the loop with the helper function properly
    for interval in intervals:
        try:
            print(f"Fetching data for {symbol} with interval {interval}...")
            market_data = fetch_market_data(symbol, args.period, interval)

            # Process the data
            market_data, vp = _process_single_interval_logic(
                market_data, interval, args
            )

            processed_data[interval] = market_data
            if vp is not None:
                volume_profiles[interval] = vp

            # Support & Resistance (only on first interval)
            if args.support_resistance and interval == intervals[0]:
                print(f"Detecting Support & Resistance for {interval}...")
                support_resistance_levels = price.detect_support_resistance(market_data)

            print(f"Successfully processed interval {interval}.")

        except ValueError as e:
            print(f"Error processing {symbol} for interval {interval}: {e}")
            continue

    if processed_data:
        # Check conditions for plotting
        # For now, we plot if plot_all is True or if we want to replicate the old logic
        # The old logic checked for overbought/oversold on the last candle.
        # We can keep that logic or just plot if plot_all is True.
        # The user didn't explicitly say to remove the condition, but implied adding features.
        # Let's keep the condition check from stochastic_processing but refactored.

        should_plot = args.plot_all
        if not should_plot:
            # Re-implement the check locally or import it.
            # Since we want to avoid mono-scripts, let's just check it here simply.
            # Check last candle of the first interval (or all).
            # The old logic checked ALL intervals.
            is_condition_met = False
            last_k_values = [
                df["%K"].iloc[-1] for df in processed_data.values() if not df.empty
            ]
            if last_k_values:
                is_overbought = all(k > 80 for k in last_k_values)
                is_oversold = all(k < 20 for k in last_k_values)
                is_condition_met = is_overbought or is_oversold

            should_plot = is_condition_met

        if should_plot:
            print(f"Generating chart for {symbol}...")
            generate_analysis_chart(
                symbol,
                processed_data,
                volume_profiles=volume_profiles,
                support_resistance_levels=support_resistance_levels,
                output_dir=args.save_html_dir,
                consolidation_window=args.consolidation_window,
            )
        else:
            print(f"Condition not met for {symbol}. Skipping chart.")


def _process_single_interval_logic(
    market_data: pd.DataFrame, interval: str, args: argparse.Namespace
):
    """Helper function to process a single interval's data."""
    # 1. Stochastic
    print(f"Calculating stochastic for {interval}...")
    market_data = calculate_stochastic(
        market_data, k_window=args.k_window, d_window=args.d_window
    )

    # Calculate VWAP
    if args.vwap:
        print(f"Calculating VWAP for {interval}...")
        vwap_df = indicators.calculate_vwap(market_data)
        market_data = pd.concat([market_data, vwap_df], axis=1)

    # Calculate ATR
    if args.consolidation:
        print(f"Calculating ATR for {interval}...")
        market_data["ATR"] = indicators.calculate_atr(market_data)

    # 2. Volume Analysis
    vp = None
    if args.volume_profile:
        print(f"Calculating Volume Profile for {interval}...")
        vp = volume.calculate_volume_profile(market_data, bins=args.vp_bins)

    print(f"Calculating Volume Percentiles for {interval}...")
    market_data["Volume_Percentile"] = volume.calculate_volume_percentiles(market_data)

    # 3. Price Analysis
    if args.consolidation:
        print(f"Detecting Consolidation for {interval}...")
        market_data["Is_Consolidation"] = price.detect_consolidation(
            market_data,
            window=args.consolidation_window,
            threshold_multiplier=args.consolidation_atr_multiplier,
            use_atr=True,
        )

    if args.ror:
        print(f"Calculating RoR for {interval}...")
        market_data["RoR"] = price.calculate_ror(market_data)

    return market_data, vp
