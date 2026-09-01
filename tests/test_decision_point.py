from __future__ import annotations

import unittest

import numpy as np

from serdes_model.decision_point import (
    evaluate_decision_point,
    gaussian_pam4_der,
    noise_autocorrelation_at_ui_lags,
    output_noise_psd_one_sided,
    pam4_unit_voltage_from_delivered_power,
    pam4_unit_voltage_from_outer_pp,
    required_pam4_dp_snr_db,
)


class DecisionPointTests(unittest.TestCase):
    def test_pam4_voltage_uses_loaded_outer_pp_swing(self) -> None:
        self.assertAlmostEqual(pam4_unit_voltage_from_outer_pp(0.9), 0.15, places=14)

    def test_pam4_voltage_uses_delivered_differential_power(self) -> None:
        unit_voltage = pam4_unit_voltage_from_delivered_power(0.0, 100.0)
        self.assertAlmostEqual(unit_voltage, np.sqrt(0.02), places=14)
        average_rms_voltage = np.sqrt(5.0) * unit_voltage
        self.assertAlmostEqual(average_rms_voltage**2 / 100.0, 1e-3, places=14)

    def test_required_snr_and_gaussian_der_are_inverse(self) -> None:
        target_der = 1e-3
        required_snr_db = required_pam4_dp_snr_db(target_der)
        error_variance = 5.0 / 10.0 ** (required_snr_db / 10.0)
        self.assertAlmostEqual(required_snr_db, 17.11630103535481, places=12)
        self.assertAlmostEqual(
            gaussian_pam4_der(1.0, error_variance), target_der, places=14
        )

    def test_white_noise_autocorrelation_at_symbol_rate(self) -> None:
        symbol_period_s = 1e-9
        frequency_hz = np.linspace(0.0, 0.5 / symbol_period_s, 10001)
        one_sided_psd = np.full(frequency_hz.shape, 2e-12)
        correlation = noise_autocorrelation_at_ui_lags(
            frequency_hz,
            one_sided_psd,
            symbol_period_s=symbol_period_s,
            lags_ui=[0, 1],
        )
        self.assertAlmostEqual(correlation[0], 1e-3, places=14)
        self.assertAlmostEqual(correlation[1], 0.0, places=14)

    def test_noise_sources_are_referred_through_the_correct_transfer(self) -> None:
        total = np.array([2.0 + 0.0j, 3.0 + 0.0j])
        afe = np.array([4.0 + 0.0j, 5.0 + 0.0j])
        psd = output_noise_psd_one_sided(
            total_transfer=total,
            afe_transfer=afe,
            source_port_density_v_per_sqrt_hz=2.0,
            afe_input_density_v_per_sqrt_hz=3.0,
        )
        np.testing.assert_allclose(psd, 4.0 * np.abs(total) ** 2 + 9.0 * np.abs(afe) ** 2)

    def test_memoryless_channel_matches_closed_form_dpsnr(self) -> None:
        samples_per_ui = 4
        symbol_period_s = 1e-9
        pulse = np.zeros(81)
        pulse_main_index = 40
        pulse[pulse_main_index] = 1.0
        frequency_hz = np.linspace(0.0, 2e9, 4001)
        raw_noise_variance = 1e-4
        noise_psd = np.full(
            frequency_hz.shape, raw_noise_variance / frequency_hz[-1]
        )
        result = evaluate_decision_point(
            pulse,
            pulse_main_index=pulse_main_index,
            samples_per_ui=samples_per_ui,
            symbol_period_s=symbol_period_s,
            pam4_unit_input_v=0.1,
            noise_frequency_hz=frequency_hz,
            one_sided_noise_psd_v2_per_hz=noise_psd,
            channel_pre_ui=0,
            channel_post_ui=0,
            ffe_tap_count=1,
            pattern_tap_count=0,
            optimize_sampling_phase=False,
        )
        expected_snr_db = 10.0 * np.log10(5.0 * 0.1**2 / raw_noise_variance)
        self.assertAlmostEqual(result.dp_snr_db, expected_snr_db, places=9)
        self.assertAlmostEqual(result.residual_isi_variance, 0.0, places=14)
        self.assertAlmostEqual(result.pattern_conditioned_der, result.gaussian_der, places=14)

    def test_pattern_conditioning_retains_bounded_deterministic_isi(self) -> None:
        samples_per_ui = 4
        pulse = np.zeros(81)
        pulse_main_index = 40
        pulse[pulse_main_index] = 1.0
        pulse[pulse_main_index + samples_per_ui] = 0.2
        frequency_hz = np.linspace(0.0, 2e9, 101)
        result = evaluate_decision_point(
            pulse,
            pulse_main_index=pulse_main_index,
            samples_per_ui=samples_per_ui,
            symbol_period_s=1e-9,
            pam4_unit_input_v=1.0,
            noise_frequency_hz=frequency_hz,
            one_sided_noise_psd_v2_per_hz=np.zeros(frequency_hz.shape),
            channel_pre_ui=0,
            channel_post_ui=1,
            ffe_tap_count=1,
            pattern_tap_count=1,
            optimize_sampling_phase=False,
        )
        self.assertEqual(result.pattern_conditioned_der, 0.0)
        self.assertGreater(result.gaussian_der, 0.0)


if __name__ == "__main__":
    unittest.main()
