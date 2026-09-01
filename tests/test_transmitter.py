from __future__ import annotations

import unittest

import numpy as np

from serdes_model.frequency_response import FrequencyResponse
from serdes_model.transmitter import (
    apply_tx_pulse_shaping,
    apply_tx_pulse_shaping_to_channel,
    centered_symbol_pulse_response,
    tx_pulse_shaping_response,
    zoh_frequency_response,
)


class TransmitterTests(unittest.TestCase):
    def test_zoh_spectrum_has_expected_dc_nyquist_and_symbol_rate_values(self) -> None:
        symbol_period_s = 1e-9
        frequency_hz = np.array([0.0, 0.5e9, 1e9])
        response = zoh_frequency_response(
            frequency_hz, symbol_period_s=symbol_period_s
        )
        self.assertAlmostEqual(abs(response[0]), 1.0, places=14)
        self.assertAlmostEqual(abs(response[1]), 2.0 / np.pi, places=14)
        self.assertAlmostEqual(abs(response[2]), 0.0, places=14)

    def test_centered_ideal_zoh_pulse_has_one_ui_hold(self) -> None:
        frequency_hz = np.linspace(0.0, 4e9, 1025)
        response = tx_pulse_shaping_response(frequency_hz, kind="ideal_zoh")
        _, pulse, samples_per_ui = centered_symbol_pulse_response(
            response, symbol_period_s=1e-9
        )
        self.assertEqual(samples_per_ui, 8)
        self.assertEqual(int(np.count_nonzero(np.isclose(pulse, 1.0))), 8)
        self.assertAlmostEqual(float(np.max(pulse)), 1.0, places=14)

    def test_ideal_zoh_has_unity_frequency_multiplier(self) -> None:
        frequency_hz = np.array([0.0, 1e9, 2e9])
        response = tx_pulse_shaping_response(frequency_hz, kind="ideal_zoh")
        np.testing.assert_allclose(response.transfer, 1.0)
        self.assertEqual(response.metadata["rise_time_20_80_s"], 0.0)

    def test_gaussian_rise_time_matches_com_equation(self) -> None:
        frequency_hz = np.array([0.0, 28e9])
        rise_time_s = 7.5e-12
        response = tx_pulse_shaping_response(
            frequency_hz,
            kind="gaussian_rise_time",
            rise_time_20_80_s=rise_time_s,
        )
        expected = np.exp(-(np.pi * frequency_hz * rise_time_s / 1.6832) ** 2)
        np.testing.assert_allclose(response.transfer, expected)
        self.assertAlmostEqual(response.magnitude_db[0], 0.0, places=14)

    def test_tx_shaping_applies_to_signal_path(self) -> None:
        frequency_hz = np.array([0.0, 1e9])
        tx = tx_pulse_shaping_response(
            frequency_hz,
            kind="gaussian_rise_time",
            rise_time_20_80_s=20e-12,
        )
        channel_afe = FrequencyResponse(
            frequency_hz=frequency_hz,
            transfer=np.array([2.0, 3.0], dtype=complex),
            label="test",
            gain_kind="Sdd21_x_Av_port",
            differential_z0_ohm=100.0,
        )
        signal_path = apply_tx_pulse_shaping(tx, channel_afe)
        np.testing.assert_allclose(signal_path.transfer, tx.transfer * channel_afe.transfer)
        self.assertTrue(signal_path.metadata["signal_only_filter"])

    def test_tx_channel_stage_excludes_afe(self) -> None:
        frequency_hz = np.array([0.0, 1e9])
        tx = tx_pulse_shaping_response(frequency_hz, kind="ideal_zoh")
        channel = FrequencyResponse(
            frequency_hz=frequency_hz,
            transfer=np.array([0.9, 0.1], dtype=complex),
            label="channel",
            gain_kind="Sdd21",
            differential_z0_ohm=100.0,
        )
        stage = apply_tx_pulse_shaping_to_channel(tx, channel)
        np.testing.assert_allclose(stage.transfer, channel.transfer)
        self.assertEqual(stage.gain_kind, "Htx_pulse_x_Sdd21")
        self.assertIn("before AFE", stage.metadata["reference_plane"])


if __name__ == "__main__":
    unittest.main()
