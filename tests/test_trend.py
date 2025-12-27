"""
Unit tests for the trend indication analysis module.
"""

import unittest
import pandas as pd
import numpy as np
from src.analysis.trend import calculate_trend_indicator


class TestTrendIndicator(unittest.TestCase):
    """
    Test suite for the calculate_trend_indicator function.
    """

    def test_uptrend_ratchet(self):
        """
        Tests the ratcheting behavior in an uptrend (support only moves up or flat).
        """
        # Create data for a clear uptrend
        # Highs don't matter much for uptrend logic unless trend switches
        # Data:
        # i=0: Low=10
        # i=1: Low=11
        # i=2: Low=12
        # i=3: Low=11.5
        # i=4: Low=13
        # i=5: Low=14

        data = pd.DataFrame(
            {
                "High": [20] * 10,
                "Low": [10, 11, 12, 11.5, 13, 14, 15, 16, 17, 18],
                "Close": [15] * 10,
            }
        )
        shift = 2

        # Logic: Rolling min of last 2.
        # i=0, 1: NaN (shifted rolling window not full)
        # i=2: Shifted Lows at 2 (excluding 2) are Low[0], Low[1]. Min(10, 11) = 10.
        # Initial Level = 10. Uptrend.

        result = calculate_trend_indicator(data, shift=shift)

        self.assertEqual(result["Trend_Direction"].iloc[2], 1)
        self.assertEqual(result["Trend_Indicator"].iloc[2], 10.0)

        # i=3: Prev Level=10. Rolling Min of [1, 2] -> Min(11, 12) = 11.
        # Level = max(11, 10) = 11.
        self.assertEqual(result["Trend_Indicator"].iloc[3], 11.0)

        # i=4: Prev Level=11. Rolling Min of [2, 3] -> Min(12, 11.5) = 11.5.
        # Level = max(11.5, 11) = 11.5.
        self.assertEqual(result["Trend_Indicator"].iloc[4], 11.5)

    def test_trend_switch_downtrend(self):
        """
        Tests the switch from Uptrend to Downtrend when Low breaks below support.
        """
        # Uptrend starts, then Low drops below indicator
        # i=0: Low=10
        # i=1: Low=12
        # i=2: Low=13
        # i=3: Low=5 (Crash)

        data = pd.DataFrame(
            {"High": [20, 20, 20, 15, 14], "Low": [10, 12, 13, 5, 4], "Close": [15] * 5}
        )
        shift = 2
        # i=2: Rolling Min(Low[0], Low[1]) = Min(10, 12) = 10.
        # Level=10. Dir=1.

        result = calculate_trend_indicator(data, shift=shift)

        self.assertEqual(result["Trend_Direction"].iloc[2], 1)
        self.assertEqual(result["Trend_Indicator"].iloc[2], 10.0)

        # i=3: Current Low=5. Prev Level=10.
        # Rolling Max (for switch reset) -> Max(High[1], High[2]) = Max(20, 20) = 20.
        # 5 < 10 -> Switch!
        # New Dir = -1. New Level = 20.

        self.assertEqual(result["Trend_Direction"].iloc[3], -1)
        self.assertEqual(result["Trend_Indicator"].iloc[3], 20.0)

    def test_trend_switch_uptrend(self):
        """
        Tests switch from Downtrend to Uptrend when High breaks above resistance.
        """
        # Force a downtrend start by violation
        # i=0: Low=10, High=20
        # i=1: Low=12, High=20
        # i=2: Low=5 (below expected start level? No, let's explicit force)

        # Let's construct a scenario where it flows naturally from Down to Up.
        # Start in Uptrend, switch to Down, then switch to Up.

        # Lows: 10, 12, 13, 5 (Switch D), 4, 3, 25 (Switch U)
        # Highs: 20, 20, 20, 15, 14, 13, 30

        data = pd.DataFrame(
            {
                "High": [20, 20, 20, 15, 14, 13, 30],
                "Low": [10, 12, 13, 5, 4, 3, 25],
                "Close": [15] * 7,
            }
        )
        shift = 2

        result = calculate_trend_indicator(data, shift=shift)

        # i=3: Switch to Downtrend. Level=20 (Max of High[1,2]).
        self.assertEqual(result["Trend_Direction"].iloc[3], -1)
        self.assertEqual(result["Trend_Indicator"].iloc[3], 20.0)

        # i=4: Curr High=14. Prev Level=20.
        # Rolling Max(High[2,3]) = Max(20, 15) = 20.
        # Ratchet Down: min(20, 20) = 20.
        self.assertEqual(result["Trend_Indicator"].iloc[4], 20.0)

        # i=5: Curr High=13. Prev Level=20.
        # Rolling Max(High[3,4]) = Max(15, 14) = 15.
        # Ratchet Down: min(15, 20) = 15.
        self.assertEqual(result["Trend_Indicator"].iloc[5], 15.0)

        # i=6: Curr High=30. Prev Level=15.
        # 30 > 15 -> Switch to UPTREND.
        # New Level = Rolling Min(Low[4,5]) = Min(4, 3) = 3.
        self.assertEqual(result["Trend_Direction"].iloc[6], 1)
        self.assertEqual(result["Trend_Indicator"].iloc[6], 3.0)

    def test_strict_ratcheting(self):
        """
        Tests that strict monotonicity is maintained (strict ratcheting).
        """
        # Verify strict ratcheting: Indicator can only go up in Uptrend.
        # We construct a scenario where rolling min drops, but indicator stays flat.

        # To avoid "Break" logic, we need the Rolling Min to drop, but the new Lows
        # must still be >= current level? No.
        # If Rolling Min drops, it means a Low value entered the window that is smaller than previous rolling min.
        # But for the indicator to NOT break, that Low value must have been >= Level at the time it occurred.

        # Let's force it by overriding values manually or just trust the logic?
        # Let's trust the logic `max(r_min, current_level)` guarantees monotonicity.
        # We can test simple monotonicity on random data.

        np.random.seed(42)
        random_lows = np.random.randint(10, 50, 100)
        random_highs = random_lows + 10
        data = pd.DataFrame(
            {"High": random_highs, "Low": random_lows, "Close": random_lows}
        )
        shift = 5
        result = calculate_trend_indicator(data, shift=shift)

        trend_dirs = result["Trend_Direction"].values
        trend_vals = result["Trend_Indicator"].values

        # Verify monotonicity per sequence
        for i in range(1, len(data)):
            if trend_dirs[i] == 1 and trend_dirs[i - 1] == 1:
                # Uptrend continuation: Value must be >= Prev Value
                self.assertGreaterEqual(trend_vals[i], trend_vals[i - 1])

            if trend_dirs[i] == -1 and trend_dirs[i - 1] == -1:
                # Downtrend continuation: Value must be <= Prev Value
                self.assertLessEqual(trend_vals[i], trend_vals[i - 1])


if __name__ == "__main__":
    unittest.main()
