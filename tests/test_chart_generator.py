"""Unit tests for the chart_generator module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.chart_generator import (
    _parse_interval_to_minutes,
    generate_analysis_chart,
)


@pytest.mark.parametrize(
    "interval_str, expected_minutes",
    [
        ("1m", 1),
        ("5m", 5),
        ("1h", 60),
        ("4h", 240),
        ("1d", 1440),
        ("1wk", 10080),
    ],
)
def test_parse_interval_to_minutes_valid(interval_str, expected_minutes):
    """Test that _parse_interval_to_minutes converts valid interval strings correctly."""
    assert _parse_interval_to_minutes(interval_str) == expected_minutes


@pytest.mark.parametrize(
    "invalid_interval",
    ["1y", "1s", "abc", "1", "1M"],
)
def test_parse_interval_to_minutes_invalid(invalid_interval):
    """Test that _parse_interval_to_minutes raises ValueError for invalid inputs."""
    with pytest.raises(ValueError):
        _parse_interval_to_minutes(invalid_interval)


@patch("src.chart_generator.make_subplots")
def test_generate_analysis_chart_save_only(mock_make_subplots, tmp_path):
    """Test that generate_analysis_chart saves the chart when output_dir is provided."""
    # Arrange
    mock_fig = MagicMock()
    mock_make_subplots.return_value = mock_fig
    output_dir = tmp_path / "charts"
    output_dir.mkdir()

    data_dict = {
        "1h": pd.DataFrame(
            {
                "Datetime": pd.to_datetime(["2023-01-01"]),
                "Open": [100],
                "High": [110],
                "Low": [90],
                "Close": [105],
                "Volume": [1000],
                "%K": [80],
                "%D": [75],
            }
        )
    }

    # Act
    generate_analysis_chart("BTC/USD", data_dict, output_dir=str(output_dir))

    # Assert
    expected_path = output_dir / "BTC_USD_analysis_dashboard.html"
    mock_fig.write_html.assert_called_once_with(str(expected_path))
    mock_fig.show.assert_not_called()


@patch("src.chart_generator.make_subplots")
def test_generate_analysis_chart_show(mock_make_subplots):
    """Test that generate_analysis_chart shows the chart when output_dir is None."""
    # Arrange
    mock_fig = MagicMock()
    mock_make_subplots.return_value = mock_fig

    data_dict = {
        "1h": pd.DataFrame(
            {
                "Datetime": pd.to_datetime(["2023-01-01"]),
                "Open": [100],
                "High": [110],
                "Low": [90],
                "Close": [105],
                "Volume": [1000],
                "%K": [80],
                "%D": [75],
            }
        )
    }

    # Act
    generate_analysis_chart("BTC-USD", data_dict)

    # Assert
    mock_fig.show.assert_called_once()
    mock_fig.write_html.assert_not_called()


def test_generate_analysis_chart_no_data(tmp_path):
    """Test that generate_analysis_chart handles an empty data_dict gracefully."""
    # Act & Assert (should not raise an error)
    generate_analysis_chart("BTC-USD", {}, output_dir=str(tmp_path))


@patch("src.chart_generator.make_subplots")
def test_generate_analysis_chart_with_consolidation(mock_make_subplots):
    """Test that generate_analysis_chart plots consolidation zones."""
    # Arrange
    mock_fig = MagicMock()
    mock_make_subplots.return_value = mock_fig

    # Create data with consolidation
    # 10 points, last 5 are consolidation
    dates = pd.date_range(start="2023-01-01", periods=10, freq="1h")
    data = pd.DataFrame(
        {
            "Datetime": dates,
            "Open": [100] * 10,
            "High": [101] * 10,
            "Low": [99] * 10,
            "Close": [100] * 10,
            "Volume": [1000] * 10,
            "%K": [50] * 10,
            "%D": [50] * 10,
            "Is_Consolidation": [False] * 5 + [True] * 5,
        }
    )

    data_dict = {"1h": data}

    # Act
    # Window 3, so if consolidation starts at index 5, visual start should be 5-3+1 = 3
    generate_analysis_chart("BTC-USD", data_dict, consolidation_window=3)

    # Assert
    # add_vrect should be called for the consolidation zone
    assert mock_fig.add_vrect.called

    # Verify arguments of the call if possible, but just checking it's called is a good start
    # The first call args:
    _, kwargs = mock_fig.add_vrect.call_args
    # x0 should be dates[3] (index 5 - 3 + 1 = 3)
    assert kwargs["x0"] == dates[3]
