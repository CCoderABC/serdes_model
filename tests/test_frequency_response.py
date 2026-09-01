from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from serdes_model.frequency_response import (
    analytic_channel_response,
    analytic_high_loss_channel_response,
    cascade_responses,
    ctle_afe_response,
    frequency_to_impulse,
    load_touchstone_sdd21,
    uniform_frequency_grid,
)


class FrequencyResponseTests(unittest.TestCase):
    def test_high_loss_channel_hits_nyquist_target(self) -> None:
        nyquist_hz = 28e9
        frequency_hz = np.linspace(0.0, 2.0 * nyquist_hz, 2001)
        response = analytic_high_loss_channel_response(
            frequency_hz,
            delay_s=500e-12,
            reference_frequency_hz=nyquist_hz,
            insertion_loss_db_at_reference=36.0,
            loss_dc_db=0.2,
            skin_effect_fraction=0.45,
        )
        nyquist_index = int(np.argmin(np.abs(frequency_hz - nyquist_hz)))
        self.assertAlmostEqual(response.magnitude_db[nyquist_index], -36.0, places=10)
        self.assertEqual(response.metadata["model"], "analytic_high_loss_target")

    def test_analytic_channel_loss_and_delay_at_reference(self) -> None:
        reference_hz = 28e9
        # Keep adjacent phase steps below pi so numerical phase unwrapping is unique.
        frequency_hz = np.linspace(0.0, 2.0 * reference_hz, 1001)
        response = analytic_channel_response(
            frequency_hz,
            delay_s=500e-12,
            loss_reference_hz=reference_hz,
            loss_dc_db=0.2,
            loss_sqrt_db_at_reference=10.0,
            loss_linear_db_at_reference=18.0,
        )
        reference_index = int(np.argmin(np.abs(frequency_hz - reference_hz)))
        self.assertAlmostEqual(response.magnitude_db[reference_index], -28.2, places=10)
        self.assertGreater(float(response.group_delay_s[reference_index]), 450e-12)
        self.assertLess(float(response.group_delay_s[reference_index]), 550e-12)

    def test_ctle_dc_gain_and_peaking(self) -> None:
        frequency_hz = np.array([0.0, 28e9])
        response = ctle_afe_response(
            frequency_hz,
            dc_gain_db=6.0,
            zero_frequencies_hz=[5e9, 15e9],
            pole_frequencies_hz=[30e9, 50e9, 90e9],
        )
        self.assertAlmostEqual(response.magnitude_db[0], 6.0, places=12)
        self.assertGreater(response.magnitude_db[1], response.magnitude_db[0] + 10.0)

    def test_ctle_retains_complex_pole_zero_phase(self) -> None:
        frequency_hz = np.array([0.0, 10e9, 20e9])
        dc_gain_db = 6.0
        response = ctle_afe_response(
            frequency_hz,
            dc_gain_db=dc_gain_db,
            zero_frequencies_hz=[5e9, 15e9],
            pole_frequencies_hz=[30e9, 50e9, 90e9],
        )
        s = 1j * 2.0 * np.pi * frequency_hz
        expected = np.full(
            frequency_hz.shape, 10.0 ** (dc_gain_db / 20.0), dtype=complex
        )
        for zero_hz in (5e9, 15e9):
            expected *= 1.0 + s / (2.0 * np.pi * zero_hz)
        for pole_hz in (30e9, 50e9, 90e9):
            expected /= 1.0 + s / (2.0 * np.pi * pole_hz)
        np.testing.assert_allclose(response.transfer, expected)
        self.assertGreater(response.phase_rad[1], 0.0)

    def test_cascade_rejects_gain_semantic_mismatch(self) -> None:
        frequency_hz = np.array([0.0, 1e9])
        channel = analytic_channel_response(
            frequency_hz,
            delay_s=0.0,
            loss_reference_hz=1e9,
            loss_dc_db=0.0,
            loss_sqrt_db_at_reference=0.0,
            loss_linear_db_at_reference=0.0,
        )
        afe = ctle_afe_response(
            frequency_hz,
            dc_gain_db=0.0,
            zero_frequencies_hz=[],
            pole_frequencies_hz=[10e9],
        )
        total = cascade_responses(channel, afe)
        np.testing.assert_allclose(total.transfer, afe.transfer)
        self.assertEqual(total.gain_kind, "Sdd21_x_Av_port")

    def test_real_ifft_places_integer_sample_delay(self) -> None:
        max_frequency_hz = 16e9
        frequency_hz = uniform_frequency_grid(max_frequency_hz, 1025)
        expected_dt_s = 1.0 / (2.0 * max_frequency_hz)
        delay_samples = 9
        response = analytic_channel_response(
            frequency_hz,
            delay_s=delay_samples * expected_dt_s,
            loss_reference_hz=1e9,
            loss_dc_db=0.0,
            loss_sqrt_db_at_reference=0.0,
            loss_linear_db_at_reference=0.0,
        )
        time_s, impulse = frequency_to_impulse(response)
        self.assertAlmostEqual(time_s[1] - time_s[0], expected_dt_s, places=22)
        self.assertEqual(int(np.argmax(impulse)), delay_samples)
        self.assertAlmostEqual(float(impulse[delay_samples]), 1.0, places=12)

    def test_touchstone_mixed_mode_extracts_sdd21(self) -> None:
        import skrf as rf

        frequency = rf.Frequency.from_f([1e9, 2e9, 3e9], unit="hz")
        s = np.zeros((3, 4, 4), dtype=complex)
        s[:, 2, 0] = 1.0
        s[:, 3, 1] = 1.0
        network = rf.Network(frequency=frequency, s=s, z0=50.0)
        with tempfile.TemporaryDirectory() as temporary_directory:
            base_path = Path(temporary_directory) / "ideal_pair"
            network.write_touchstone(str(base_path))
            response = load_touchstone_sdd21(
                base_path.with_suffix(".s4p"),
                expected_differential_z0_ohm=100.0,
            )
        np.testing.assert_allclose(response.transfer, 1.0, atol=1e-12)
        self.assertAlmostEqual(response.differential_z0_ohm, 100.0)


if __name__ == "__main__":
    unittest.main()
