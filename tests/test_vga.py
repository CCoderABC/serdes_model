from __future__ import annotations

import unittest

import numpy as np

from serdes_model.vga import (
    flat_vga_response,
    peak_target_gain_db,
    worst_case_pam_waveform_peak_v,
)


class FlatVgaTests(unittest.TestCase):
    def test_worst_case_bound_checks_all_sample_phases(self) -> None:
        pulse_v = np.asarray([0.10, 0.02, -0.03, 0.01, 0.04, -0.01])
        peak_v = worst_case_pam_waveform_peak_v(
            pulse_v,
            samples_per_ui=2,
            maximum_symbol_magnitude=3.0,
        )
        expected_phase_zero_v = 3.0 * (0.10 + 0.03 + 0.04)
        self.assertAlmostEqual(peak_v, expected_phase_zero_v, places=15)

    def test_peak_target_gain_maps_bound_to_requested_headroom(self) -> None:
        gain_db, requested_gain_db, limited = peak_target_gain_db(
            0.8,
            output_half_scale_v=0.2,
            target_fraction=0.95,
            minimum_gain_db=-30.0,
            maximum_gain_db=10.0,
        )
        self.assertFalse(limited)
        self.assertAlmostEqual(gain_db, requested_gain_db, places=15)
        gain_linear = 10.0 ** (gain_db / 20.0)
        self.assertAlmostEqual(0.8 * gain_linear, 0.19, places=15)

    def test_gain_limit_is_reported(self) -> None:
        gain_db, _, limited = peak_target_gain_db(
            1.0,
            output_half_scale_v=0.2,
            target_fraction=0.95,
            minimum_gain_db=-10.0,
            maximum_gain_db=10.0,
        )
        self.assertTrue(limited)
        self.assertEqual(gain_db, -10.0)

    def test_flat_vga_has_constant_magnitude_and_zero_group_delay(self) -> None:
        frequency_hz = np.linspace(0.0, 80e9, 101)
        response = flat_vga_response(
            frequency_hz,
            gain_db=-9.0,
            differential_z0_ohm=100.0,
        )
        np.testing.assert_allclose(response.magnitude_db, -9.0, atol=1e-14)
        np.testing.assert_allclose(response.phase_rad, 0.0)
        np.testing.assert_allclose(response.group_delay_s, 0.0)


if __name__ == "__main__":
    unittest.main()
