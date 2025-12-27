"""Unit tests for the stochastic_processing module."""

import pandas as pd
import pytest
from src.stochastic_processing import calculate_stochastic


@pytest.fixture
def _sample_market_data() -> pd.DataFrame:
    """Provides a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "High": [
                110,
                112,
                111,
                109,
                108,
                107,
                106,
                105,
                104,
                103,
                102,
                101,
                100,
                99,
                98,
                115,
                112,
                111,
                109,
                108,
            ],
            "Low": [
                100,
                101,
                102,
                103,
                104,
                103,
                102,
                101,
                100,
                99,
                98,
                97,
                96,
                95,
                94,
                99,
                101,
                102,
                103,
                104,
            ],
            "Close": [
                105,
                111,
                103,
                108,
                105,
                106,
                105,
                104,
                103,
                102,
                101,
                100,
                99,
                98,
                97,
                110,
                111,
                103,
                108,
                105,
            ],
        }
    )


def test_calculate_stochastic_success(_sample_market_data):
    """Test successful calculation of the stochastic oscillator."""
    # Arrange
    data = _sample_market_data

    # Act
    result = calculate_stochastic(data, k_window=14, d_window=3)

    # Assert
    assert "%K" in result.columns
    assert "%D" in result.columns
    assert not result["%K"].isnull().all()
    assert not result["%D"].isnull().all()

    # Check the last values
    pd.testing.assert_series_equal(
        result[["%K", "%D"]].iloc[-1],
        pd.Series({"%K": 52.380952, "%D": 53.968254}),
        check_names=False,
        atol=1e-6,
    )


def test_calculate_stochastic_empty_data():
    """Test ValueError when input data is empty."""
    # Arrange
    empty_df = pd.DataFrame()

    # Act & Assert
    with pytest.raises(ValueError, match="Input data cannot be empty."):
        calculate_stochastic(empty_df)


def test_calculate_stochastic_missing_columns():
    """Test ValueError when required columns are missing."""
    # Arrange
    data = pd.DataFrame({"Open": [1, 2, 3]})

    # Act & Assert
    with pytest.raises(
        ValueError, match="Input data must contain 'High', 'Low', and 'Close' columns."
    ):
        calculate_stochastic(data)
