from __future__ import annotations

import unittest

import numpy as np

from serdes_model.adc_impairments import (
    apply_sampled_adc_impairments,
    synthesize_real_noise_from_one_sided_psd,
)


class AdcImpairmentTests(unittest.TestCase):
    def test_zero_psd_produces_zero_noise(self) -> None:
        frequency_hz = np.linspace(0.0, 50e9, 1025)
        noise = synthesize_real_noise_from_one_sided_psd(
            frequency_hz,
            np.zeros_like(frequency_hz),
            sample_rate_hz=100e9,
            sample_count=4096,
            seed=5,
        )
        np.testing.assert_array_equal(noise, 0.0)

    def test_white_noise_rms_matches_integrated_psd(self) -> None:
        sample_rate_hz = 100e9
        density_v_per_sqrt_hz = 2e-9
        frequency_hz = np.linspace(0.0, sample_rate_hz / 2.0, 2049)
        psd = np.full_like(frequency_hz, density_v_per_sqrt_hz**2)
        noise = synthesize_real_noise_from_one_sided_psd(
            frequency_hz,
            psd,
            sample_rate_hz=sample_rate_hz,
            sample_count=262144,
            seed=7,
        )
        expected_rms = density_v_per_sqrt_hz * np.sqrt(sample_rate_hz / 2.0)
        self.assertAlmostEqual(
            float(np.sqrt(np.mean(noise**2))) / expected_rms,
            1.0,
            delta=0.01,
        )

    def test_aperture_jitter_uses_local_signal_slope(self) -> None:
        sample_rate_hz = 80e9
        waveform = np.arange(4096, dtype=float) / sample_rate_hz * 2e9
        frequency_hz = np.linspace(0.0, sample_rate_hz / 2.0, 1025)
        result = apply_sampled_adc_impairments(
            waveform,
            samples_per_ui=8,
            sample_rate_hz=sample_rate_hz,
            afe_noise_frequency_hz=frequency_hz,
            afe_noise_psd_v2_per_hz=np.zeros_like(frequency_hz),
            additional_adc_noise_rms_v=0.0,
            aperture_jitter_rms_s=100e-15,
            random_seed=11,
        )
        np.testing.assert_allclose(result.signal_slope_v_per_s, 2e9, rtol=1e-12)
        np.testing.assert_allclose(
            result.jitter_error_v,
            result.signal_slope_v_per_s * result.aperture_jitter_s,
        )


if __name__ == "__main__":
    unittest.main()
