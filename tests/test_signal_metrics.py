from __future__ import annotations

import unittest

import numpy as np

from serdes_model.signal_metrics import voltage_papr


class SignalMetricsTests(unittest.TestCase):
    def test_balanced_pam4_papr_is_nine_over_five(self) -> None:
        result = voltage_papr([-3.0, -1.0, 1.0, 3.0])
        self.assertAlmostEqual(result.papr_linear, 9.0 / 5.0, places=15)
        self.assertAlmostEqual(result.papr_db, 10.0 * np.log10(9.0 / 5.0))
        self.assertAlmostEqual(result.rms_v, np.sqrt(5.0), places=15)
        self.assertEqual(result.peak_abs_v, 3.0)

    def test_papr_is_invariant_to_scalar_gain(self) -> None:
        samples_v = np.asarray([-0.3, -0.1, 0.1, 0.3])
        before = voltage_papr(samples_v)
        after = voltage_papr(0.217 * samples_v)
        self.assertAlmostEqual(before.papr_db, after.papr_db, places=14)

    def test_all_zero_waveform_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "all-zero"):
            voltage_papr(np.zeros(8))


if __name__ == "__main__":
    unittest.main()
