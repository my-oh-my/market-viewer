"""
Module for generating financial charts using Plotly.

This module provides functions to create interactive dashboards that visualize
price data, technical indicators (Stochastic, VWAP), volume profiles,
consolidation zones, and potential support/resistance levels.
"""

import os
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.analysis.trend import calculate_trend_indicator


@dataclass
class ChartConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration and data for generating the analysis chart."""

    symbol: str
    candlestick_data: pd.DataFrame
    lowest_interval: str
    consolidation_window: int
    support_resistance_levels: list[float] | None
    trend_data: pd.DataFrame
    volume_profiles: dict


@dataclass
class SubplotConfig:
    """Configuration for chart subplots."""

    rows: int
    row_heights: list[float]
    specs: list[list]
    subplot_titles: list[str]
    indices: dict[str, int]


def _parse_interval_to_minutes(interval_str: str) -> int:
    """
    Converts an interval string to minutes.

    Args:
        interval_str (str): The interval string (e.g., '1m', '1h', '1d', '1wk').

    Returns:
        int: The equivalent duration in minutes.

    Raises:
        ValueError: If the interval format is invalid or unknown.
    """
    unit_map = {"m": 1, "h": 60, "d": 24 * 60, "wk": 7 * 24 * 60}
    sorted_units = sorted(unit_map.keys(), key=len, reverse=True)

    for unit in sorted_units:
        if interval_str.endswith(unit):
            value_str = interval_str[: -len(unit)]
            if value_str.isdigit():
                return int(value_str) * unit_map[unit]

    raise ValueError(f"Unknown or invalid interval format: {interval_str}")


def _plot_candlestick(
    fig: go.Figure, data: pd.DataFrame, interval: str, row: int, col: int
):
    """
    Adds a candlestick trace to the figure.

    Args:
        fig (go.Figure): The Plotly figure to add the trace to.
        data (pd.DataFrame): The DataFrame containing OHLC data.
        interval (str): The time interval label for the legend.
        row (int): The row index for the subplot.
        col (int): The column index for the subplot.
    """
    fig.add_trace(
        go.Candlestick(
            x=data["Datetime"],
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name=f"Candlestick ({interval})",
        ),
        row=row,
        col=col,
    )


def _plot_stochastic(
    fig: go.Figure,
    data: pd.DataFrame,
    interval: str,
    row: int | None = None,
    col: int | None = None,
    yaxis: str | None = None,
):  # pylint: disable=too-many-arguments
    """
    Adds stochastic oscillator traces (%K and %D) to the figure.

    Args:
        fig (go.Figure): The Plotly figure to add traces to.
        data (pd.DataFrame): The DataFrame containing '%K' and '%D' columns.
        interval (str): The time interval label for the legend.
        row (int, optional): The row index for the subplot. Defaults to None.
        col (int, optional): The column index for the subplot. Defaults to None.
        yaxis (str, optional): The y-axis reference (e.g., 'y2') if overlaying. Defaults to None.
    """
    # Common args for traces
    trace_args = {
        "x": data["Datetime"],
        "mode": "lines",
    }
    if yaxis:
        trace_args["yaxis"] = yaxis
        # If yaxis is specified, we don't use row/col for placement
        # as we are likely overlaying.

    fig.add_trace(
        go.Scatter(
            y=data["%K"],
            name=f"%K ({interval})",
            line={"width": 2},
            **trace_args,
        ),
        row=row if not yaxis else None,
        col=col if not yaxis else None,
    )
    fig.add_trace(
        go.Scatter(
            y=data["%D"],
            name=f"%D ({interval})",
            line={"width": 1, "dash": "dash"},
            **trace_args,
        ),
        row=row if not yaxis else None,
        col=col if not yaxis else None,
    )


def _plot_vwap(fig: go.Figure, data: pd.DataFrame, row: int, col: int):
    """
    Adds VWAP and Standard Deviation bands to the figure.

    Args:
        fig (go.Figure): The Plotly figure to add traces to.
        data (pd.DataFrame): The DataFrame containing 'VWAP', 'VWAP_Upper', and 'VWAP_Lower'.
        row (int): The row index for the subplot.
        col (int): The column index for the subplot.
    """
    if "VWAP" not in data.columns:
        return

    # Plot VWAP
    fig.add_trace(
        go.Scatter(
            x=data["Datetime"],
            y=data["VWAP"],
            name="VWAP",
            line={"color": "orange", "width": 2},
        ),
        row=row,
        col=col,
    )

    # Plot Bands
    if "VWAP_Upper" in data.columns and "VWAP_Lower" in data.columns:
        # Upper Band
        fig.add_trace(
            go.Scatter(
                x=data["Datetime"],
                y=data["VWAP_Upper"],
                name="VWAP Upper (2SD)",
                line={"color": "gray", "width": 1, "dash": "dot"},
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        # Lower Band
        fig.add_trace(
            go.Scatter(
                x=data["Datetime"],
                y=data["VWAP_Lower"],
                name="VWAP Lower (2SD)",
                line={"color": "gray", "width": 1, "dash": "dot"},
                fill="tonexty",  # Fill area between upper and lower
                fillcolor="rgba(128, 128, 128, 0.1)",
                showlegend=False,
            ),
            row=row,
            col=col,
        )


def _plot_trend_indicator(
    fig: go.Figure, data: pd.DataFrame, trend_data: pd.DataFrame, row: int, col: int
):
    """
    Adds the trend indicator line to the chart.

    Args:
        fig (go.Figure): The Plotly figure.
        data (pd.DataFrame): The original OHLC data.
        trend_data (pd.DataFrame): DataFrame with 'Trend_Indicator' and 'Trend_Direction'.
        row (int): Subplot row.
        col (int): Subplot column.
    """
    if trend_data.empty or "Trend_Indicator" not in trend_data.columns:
        return

    # Create a single trace, but maybe color it differently based on trend?
    # Plotly lines are single color per trace usually.
    # To show Green for Up, Red for Down, we can split into two traces or use segments.
    # Simpler: Split into Uptrend segments and Downtrend segments.

    # However, to avoid gaps, we might want to plot the whole line as one trace (e.g. gray or white)
    # AND overlay colored segments? Or just use one line with a color array (complex in simple go.Scatter).

    # Approach: Two traces.
    # Uptrend Points: value if Direction == 1 else NaN
    # Downtrend Points: value if Direction == -1 else NaN

    up_trend = trend_data["Trend_Indicator"].copy()
    down_trend = trend_data["Trend_Indicator"].copy()

    up_trend[trend_data["Trend_Direction"] != 1] = None
    down_trend[trend_data["Trend_Direction"] != -1] = None

    fig.add_trace(
        go.Scatter(
            x=data["Datetime"],
            y=up_trend,
            mode="markers",
            name="Trend Support",
            marker={"color": "green", "size": 4, "symbol": "triangle-up"},
            opacity=0.8,
            legendgroup="Trend",
            showlegend=True,
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Scatter(
            x=data["Datetime"],
            y=down_trend,
            mode="markers",
            name="Trend Resistance",
            marker={"color": "red", "size": 4, "symbol": "triangle-down"},
            opacity=0.8,
            legendgroup="Trend",
            showlegend=True,
        ),
        row=row,
        col=col,
    )

    # Also line for continuity?
    # Let's try just markers first as it might be cleaner than connected lines with gaps.
    # Or connected lines with `connectgaps=False`.

    fig.add_trace(
        go.Scatter(
            x=data["Datetime"],
            y=up_trend,
            mode="lines",
            name="Trend Up Line",
            line={"color": "green", "width": 2},
            connectgaps=False,
            showlegend=False,
            legendgroup="Trend",
        ),
        row=row,
        col=col,
    )

    fig.add_trace(
        go.Scatter(
            x=data["Datetime"],
            y=down_trend,
            mode="lines",
            name="Trend Down Line",
            line={"color": "red", "width": 2},
            connectgaps=False,
            showlegend=False,
            legendgroup="Trend",
        ),
        row=row,
        col=col,
    )


def _plot_support_resistance(fig: go.Figure, levels: list[float], row: int, col: int):
    """
    Adds horizontal lines for Support and Resistance levels.

    Args:
        fig (go.Figure): The Plotly figure to add shapes to.
        levels (list[float]): A list of price levels to plot.
        row (int): The row index for the subplot.
        col (int): The column index for the subplot.
    """
    if not levels:
        return

    for level in levels:
        fig.add_hline(
            y=level,
            line_dash="dot",
            line_color="rgba(0, 0, 255, 0.4)",
            line_width=1,
            row=row,
            col=col,
        )


def _plot_consolidation(
    fig: go.Figure, data: pd.DataFrame, row: int, col: int, window: int = 20
):
    """
    Highlights consolidation regions on the chart.

    Args:
        fig (go.Figure): The Plotly figure to add shapes to.
        data (pd.DataFrame): The DataFrame containing 'Is_Consolidation' flag.
        row (int): The row index for the subplot.
        col (int): The column index for the subplot.
        window (int, optional): The lookback window used for detection. Defaults to 20.
    """
    if "Is_Consolidation" not in data.columns:
        return

    # Find ranges where Is_Consolidation is True
    consolidation_mask = data["Is_Consolidation"]
    starts = data.index[
        consolidation_mask & ~consolidation_mask.shift(fill_value=False)
    ]
    ends = data.index[
        consolidation_mask & ~consolidation_mask.shift(-1, fill_value=False)
    ]

    for start, end in zip(starts, ends):
        # Adjust start to include the window that triggered the consolidation
        # The consolidation flag at 'start' means the window [start-window+1, start] is consolidated
        adjusted_start = max(0, start - window + 1)
        x0 = data.loc[adjusted_start, "Datetime"]

        # End of the range
        end_iloc = data.index.get_loc(end)
        if end_iloc + 1 < len(data):
            x1 = data["Datetime"].iloc[end_iloc + 1]
        else:
            x1 = data.loc[end, "Datetime"]

        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor="rgba(255, 165, 0, 0.2)",  # Orange for consolidation
            layer="below",
            line_width=0,
            row=row,
            col=col,
        )


def _plot_volume_profile(fig: go.Figure, volume_profile: pd.DataFrame):
    """
    Adds a Volume Profile histogram to the chart.

    Overlaying is achieved by using a secondary x-axis (xaxis5) on top of the main chart.

    Args:
        fig (go.Figure): The Plotly figure to add traces to.
        volume_profile (pd.DataFrame): DataFrame containing 'Price_Bin_Mid', 'Bullish_Volume',
                                       'Bearish_Volume', and optionally 'In_VA', 'POC'.
    """
    if volume_profile.empty:
        return

    # Configure secondary x-axis for volume profile overlay
    # We use 'xaxis5' to avoid conflict with axes generated by make_subplots
    # (x, y, x2, y2, etc. are used by the subplots)
    fig.update_layout(
        xaxis5={
            "title": "Volume",
            "overlaying": "x",  # Overlay on the first subplot's x-axis
            "side": "top",
            "showgrid": False,
            "showticklabels": False,
        }
    )

    # Plot Bullish Volume
    # Split into Value Area and Non-Value Area for coloring
    # This is a bit complex with stacked bars.
    # Simpler approach: Plot all, then overlay VA highlight or just use different colors if we can.
    # Let's use the 'In_VA' column if available.

    if "In_VA" in volume_profile.columns:
        # We need to plot 4 traces: Bullish In VA, Bullish Out VA, Bearish In VA, Bearish Out VA

        # Bullish In VA
        fig.add_trace(
            go.Bar(
                y=volume_profile[volume_profile["In_VA"]]["Price_Bin_Mid"],
                x=volume_profile[volume_profile["In_VA"]]["Bullish_Volume"],
                orientation="h",
                name="Bullish Vol (VA)",
                marker_color="rgba(0, 255, 0, 0.6)",  # Darker Green
                xaxis="x5",
                yaxis="y",
                legendgroup="Volume Profile",
                showlegend=False,
            )
        )
        # Bullish Out VA
        fig.add_trace(
            go.Bar(
                y=volume_profile[~volume_profile["In_VA"]]["Price_Bin_Mid"],
                x=volume_profile[~volume_profile["In_VA"]]["Bullish_Volume"],
                orientation="h",
                name="Bullish Vol",
                marker_color="rgba(0, 255, 0, 0.2)",  # Lighter Green
                xaxis="x5",
                yaxis="y",
                legendgroup="Volume Profile",
            )
        )
        # Bearish In VA
        fig.add_trace(
            go.Bar(
                y=volume_profile[volume_profile["In_VA"]]["Price_Bin_Mid"],
                x=volume_profile[volume_profile["In_VA"]]["Bearish_Volume"],
                orientation="h",
                name="Bearish Vol (VA)",
                marker_color="rgba(255, 0, 0, 0.6)",  # Darker Red
                xaxis="x5",
                yaxis="y",
                legendgroup="Volume Profile",
                showlegend=False,
            )
        )
        # Bearish Out VA
        fig.add_trace(
            go.Bar(
                y=volume_profile[~volume_profile["In_VA"]]["Price_Bin_Mid"],
                legendgroup="Volume Profile",
            )
        )

    else:
        # Fallback to old simple plotting
        fig.add_trace(
            go.Bar(
                y=volume_profile["Price_Bin_Mid"],
                x=volume_profile["Bullish_Volume"],
                orientation="h",
                name="Bullish Volume",
                marker_color="rgba(0, 255, 0, 0.3)",  # Green with transparency
                xaxis="x5",
                yaxis="y",
                legendgroup="Volume Profile",
            )
        )

        fig.add_trace(
            go.Bar(
                y=volume_profile["Price_Bin_Mid"],
                x=volume_profile["Bearish_Volume"],
                orientation="h",
                name="Bearish Volume",
                marker_color="rgba(255, 0, 0, 0.3)",  # Red with transparency
                xaxis="x5",
                yaxis="y",
                legendgroup="Volume Profile",
            )
        )

    # Plot POC Line
    if "POC" in volume_profile.columns:
        poc_row = volume_profile[volume_profile["POC"]]
        if not poc_row.empty:
            poc_price = poc_row["Price_Bin_Mid"].values[0]
            max_vol = volume_profile["Total_Volume"].max()

            # Add a line for POC
            fig.add_trace(
                go.Scatter(
                    x=[0, max_vol],
                    y=[poc_price, poc_price],
                    mode="lines",
                    line={"color": "blue", "width": 2, "dash": "dash"},
                    name=f"POC ({poc_price:.2f})",
                    xaxis="x5",
                    yaxis="y",
                    legendgroup="Volume Profile",
                )
            )


def _plot_distribution(
    fig: go.Figure,
    data: pd.Series,
    name: str,
    row: int,
    col: int,
    color: str = "blue",
):  # pylint: disable=too-many-arguments
    """
    Adds a histogram distribution to the figure.

    Args:
        fig (go.Figure): The Plotly figure to add the histogram to.
        data (pd.Series): The numerical data series to calculate the distribution from.
        name (str): The name of the distribution (e.g., 'RoR', 'Volume').
        row (int): The row index for the subplot.
        col (int): The column index for the subplot.
        color (str, optional): The color of the histogram bars. Defaults to "blue".
    """
    if data.empty:
        return

    # Calculate histogram
    fig.add_trace(
        go.Histogram(
            x=data,
            name=f"{name} Dist",
            marker_color=color,
            opacity=0.7,
            showlegend=False,
        ),
        row=row,
        col=col,
    )

    # Highlight the most recent value
    last_val = data.iloc[-1]
    fig.add_vline(
        x=last_val,
        line_width=2,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Last: {last_val:.2f}",
        annotation_position="top right",
        row=row,
        col=col,
    )


def _configure_subplot_specs(symbol: str, lowest_interval: str) -> SubplotConfig:
    """
    Configures subplot specifications based on enabled features.

    Args:
        symbol (str): The ticker symbol.
        lowest_interval (str): The interval string for the price chart (e.g., '1h').

    Returns:
        SubplotConfig: Configuration object for subplots.
    """
    rows = 3
    row_heights = [0.5, 0.2, 0.3]
    specs = [
        [{"colspan": 2}, None],
        [{"colspan": 2}, None],
        [{}, {}],
    ]
    subplot_titles = [
        f"{symbol} Price ({lowest_interval})",
        "",
        "RoR Distribution",
        "Volume Distribution",
    ]
    indices = {"stoch": 2, "dist": 3}

    return SubplotConfig(
        rows=rows,
        row_heights=row_heights,
        specs=specs,
        subplot_titles=subplot_titles,
        indices=indices,
    )


def _plot_price_chart_components(fig: go.Figure, config: ChartConfig):
    """Plots the main price chart and its overlays."""
    # 1. Price Chart (Row 1, Col 1 - spans 2)
    _plot_candlestick(
        fig, config.candlestick_data, config.lowest_interval, row=1, col=1
    )
    _plot_vwap(fig, config.candlestick_data, row=1, col=1)
    _plot_consolidation(
        fig,
        config.candlestick_data,
        row=1,
        col=1,
        window=config.consolidation_window,
    )
    if config.support_resistance_levels:
        _plot_support_resistance(fig, config.support_resistance_levels, row=1, col=1)

    if not config.trend_data.empty:
        _plot_trend_indicator(
            fig, config.candlestick_data, config.trend_data, row=1, col=1
        )

    if config.volume_profiles and config.lowest_interval in config.volume_profiles:
        _plot_volume_profile(fig, config.volume_profiles[config.lowest_interval])


def _create_chart_config(  # pylint: disable=too-many-arguments
    symbol: str,
    data_dict: dict[str, pd.DataFrame],
    lowest_interval: str,
    consolidation_window: int,
    support_resistance_levels: list[float] | None,
    trend_shift: int | None,
    volume_profiles: dict | None,
) -> ChartConfig:
    """Creates the ChartConfig object, calculating necessary data."""
    candlestick_data = data_dict[lowest_interval].copy()

    # Calculate Trend if requested
    if trend_shift is not None:
        trend_data = calculate_trend_indicator(candlestick_data, shift=trend_shift)
    else:
        trend_data = pd.DataFrame()

    return ChartConfig(
        symbol=symbol,
        candlestick_data=candlestick_data,
        lowest_interval=lowest_interval,
        consolidation_window=consolidation_window,
        support_resistance_levels=support_resistance_levels,
        trend_data=trend_data,
        volume_profiles=volume_profiles or {},
    )


def _calculate_y_axis_range(data: pd.DataFrame) -> list[float]:
    """Calculates the Y-axis range based on price data to prevent 0-scaling issues."""
    y_min = data["Low"].min()
    y_max = data["High"].max()
    y_padding = (y_max - y_min) * 0.05
    return [y_min - y_padding, y_max + y_padding]


def generate_analysis_chart(
    symbol: str,
    data_dict: dict[str, pd.DataFrame],
    volume_profiles: dict[str, pd.DataFrame] | None = None,
    support_resistance_levels: list[float] | None = None,
    output_dir: str | None = None,
    consolidation_window: int = 20,
    trend_shift: int | None = None,
):  # pylint: disable=too-many-arguments
    """
    Generates and saves/displays a comprehensive market analysis dashboard.

    The dashboard includes:
    1. Main Price Chart (Candlestick) with VWAP, Consolidation zones, and Support/Resistance levels.
    2. Volume Profile overlay on the price chart.
    3. Stochastic Oscillator subplot.
    4. Rate of Return (RoR) Distribution subplot.
    5. Volume Distribution subplot.
    6. Trend Indicator (Optional).

    Args:
        symbol (str): The ticker symbol.
        data_dict (dict[str, pd.DataFrame]): Dictionary mapping intervals to DataFrames.
        volume_profiles (dict[str, pd.DataFrame] | None, optional): Dictionary of volume profiles. Defaults to None.
        support_resistance_levels (list[float] | None, optional): List of S/R price levels. Defaults to None.
        output_dir (str | None, optional): Directory to save the HTML chart. If None, shows the chart. Defaults to None.
        consolidation_window (int, optional): Window size used for consolidation text. Defaults to 20.
        trend_shift (int | None, optional): If provided, calculates and plots the trend indicator with this shift.
        daily_volume_profile (bool, optional): If True, plots daily volume profiles. Defaults to False.
    """
    if not data_dict:
        print("No data to plot.")
        return

    lowest_interval = min(data_dict.keys(), key=_parse_interval_to_minutes)

    # Configure subplots
    subplot_config = _configure_subplot_specs(symbol, lowest_interval)

    fig = make_subplots(
        rows=subplot_config.rows,
        cols=2,
        shared_xaxes=False,
        vertical_spacing=0.1,
        row_heights=subplot_config.row_heights,
        specs=subplot_config.specs,
        subplot_titles=subplot_config.subplot_titles,
    )

    config = _create_chart_config(
        symbol=symbol,
        data_dict=data_dict,
        lowest_interval=lowest_interval,
        consolidation_window=consolidation_window,
        support_resistance_levels=support_resistance_levels,
        trend_shift=trend_shift,
        volume_profiles=volume_profiles,
    )

    _plot_price_chart_components(fig, config)

    # 2. Stochastic (Row 2 or 3)
    for interval, data in data_dict.items():
        _plot_stochastic(
            fig, data, interval, row=subplot_config.indices["stoch"], col=1
        )

    # Link Stochastic x-axis to Price x-axis
    fig.update_xaxes(matches="x", row=subplot_config.indices["stoch"], col=1)

    # Stochastic levels
    fig.add_hline(
        y=80,
        line_dash="dot",
        line_color="red",
        row=subplot_config.indices["stoch"],
        col=1,
    )
    fig.add_hline(
        y=20,
        line_dash="dot",
        line_color="green",
        row=subplot_config.indices["stoch"],
        col=1,
    )

    # 3. Distributions (Row 3 or 4)
    if "RoR" in config.candlestick_data.columns:
        _plot_distribution(
            fig,
            config.candlestick_data["RoR"].dropna(),
            "RoR",
            row=subplot_config.indices["dist"],
            col=1,
            color="purple",
        )

    if "Volume" in config.candlestick_data.columns:
        _plot_distribution(
            fig,
            config.candlestick_data["Volume"],
            "Volume",
            row=subplot_config.indices["dist"],
            col=2,
            color="orange",
        )

    # Calculate Y-axis range to prevent 0-scaling issue with POC
    y_range = _calculate_y_axis_range(config.candlestick_data)

    # Layout updates
    fig.update_layout(
        title_text=f"Market Analysis Dashboard for {symbol}",
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        legend_title="Indicators",
        bargap=0,
        barmode="stack",
        height=1000,
        width=1400,
        xaxis_range=[
            config.candlestick_data["Datetime"].iloc[0],
            config.candlestick_data["Datetime"].iloc[-1],
        ],
        yaxis_range=y_range,  # Apply calculated Y range
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Stoch", row=subplot_config.indices["stoch"], col=1)
    fig.update_xaxes(title_text="RoR %", row=subplot_config.indices["dist"], col=1)
    fig.update_xaxes(title_text="Volume", row=subplot_config.indices["dist"], col=2)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        chart_filename = os.path.join(
            output_dir, f"{symbol.replace('/', '_')}_analysis_dashboard.html"
        )
        fig.write_html(chart_filename)
        print(f"Dashboard saved to {chart_filename}")
    else:
        fig.show()
