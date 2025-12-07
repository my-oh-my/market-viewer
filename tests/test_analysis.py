"""Tests for the analysis package."""

# pylint: disable=redefined-outer-name

import pandas as pd
import pytest
from src.analysis import volume, price


@pytest.fixture
def sample_data():
    """Creates a sample DataFrame for testing."""
    data = {
        "Close": [100, 101, 102, 101, 100, 99, 98, 99, 100, 101],
        "Open": [99, 100, 101, 102, 101, 100, 99, 98, 99, 100],
        "High": [102, 103, 104, 103, 102, 101, 100, 101, 102, 103],
        "Low": [98, 99, 100, 99, 98, 97, 96, 97, 98, 99],
        "Volume": [1000, 1100, 1200, 1100, 1000, 900, 800, 900, 1000, 1100],
    }
    return pd.DataFrame(data)


def test_calculate_volume_profile(sample_data):
    """Tests volume profile calculation."""
    bins = 5
    vp = volume.calculate_volume_profile(sample_data, bins=bins)

    assert not vp.empty
    assert len(vp) == bins
    assert "Price_Bin_Mid" in vp.columns
    assert "Total_Volume" in vp.columns
    assert vp["Total_Volume"].sum() == sample_data["Volume"].sum()


def test_calculate_volume_percentiles(sample_data):
    """Tests volume percentile calculation."""
    window = 5
    percentiles = volume.calculate_volume_percentiles(sample_data, window=window)

    assert len(percentiles) == len(sample_data)
    # First few should be NaN due to window
    assert pd.isna(percentiles.iloc[0])
    # Last one should be valid
    assert not pd.isna(percentiles.iloc[-1])
    assert 0 <= percentiles.iloc[-1] <= 1


def test_detect_consolidation():
    """Tests consolidation detection."""
    # Create data that is definitely consolidating
    consolidation_data = pd.DataFrame(
        {
            "Close": [100] * 10,
            "High": [100.1] * 10,
            "Low": [99.9] * 10,
        }
    )

    is_consolidation = price.detect_consolidation(
        consolidation_data, window=5, threshold_multiplier=1.0, use_atr=False
    )

    assert is_consolidation.iloc[-1]

    # Create data that is trending (not consolidating)
    trending_data = pd.DataFrame(
        {
            "Close": [100, 105, 110, 115, 120],
            "High": [101, 106, 111, 116, 121],
            "Low": [99, 104, 109, 114, 119],
        }
    )

    is_consolidation_trend = price.detect_consolidation(
        trending_data, window=3, threshold_multiplier=1.0, use_atr=False
    )

    assert not is_consolidation_trend.iloc[-1]


def test_calculate_ror(sample_data):
    """Tests RoR calculation."""
    ror = price.calculate_ror(sample_data)

    assert len(ror) == len(sample_data)
    assert pd.isna(ror.iloc[0])
    # 100 -> 101 is 1% increase
    assert ror.iloc[1] == pytest.approx(1.0)
