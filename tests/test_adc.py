from __future__ import annotations

import unittest

import numpy as np

from serdes_model.adc import EffectiveUniformAdc


class EffectiveUniformAdcTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adc = EffectiveUniformAdc(
            enob_bits=6,
            differential_full_scale_pp_v=0.4,
        )

    def test_six_enob_400_mvpp_has_expected_resolution(self) -> None:
        self.assertEqual(self.adc.code_count, 64)
        self.assertAlmostEqual(self.adc.minimum_input_v, -0.2, places=15)
        self.assertAlmostEqual(self.adc.maximum_input_v, 0.2, places=15)
        self.assertAlmostEqual(self.adc.effective_lsb_v, 6.25e-3, places=15)
        self.assertAlmostEqual(
            self.adc.quantization_noise_rms_v,
            6.25e-3 / np.sqrt(12.0),
            places=15,
        )

    def test_midrise_codes_and_reconstructed_levels(self) -> None:
        result = self.adc.quantize([-0.2, 0.0, 0.2])
        np.testing.assert_array_equal(result.codes, [0, 32, 63])
        np.testing.assert_allclose(
            result.reconstructed_v,
            [-0.196875, 0.003125, 0.196875],
            atol=1e-15,
        )
        self.assertFalse(bool(np.any(result.clipped)))

    def test_overrange_samples_saturate_and_are_flagged(self) -> None:
        result = self.adc.quantize([-0.25, -0.1, 0.25])
        np.testing.assert_array_equal(result.codes, [0, 16, 63])
        np.testing.assert_array_equal(result.clipped, [True, False, True])

    def test_in_range_quantization_error_is_bounded_by_half_lsb(self) -> None:
        input_v = np.linspace(-0.199, 0.199, 1001)
        result = self.adc.quantize(input_v)
        self.assertLessEqual(
            float(np.max(np.abs(result.error_from_input_v))),
            0.5 * self.adc.effective_lsb_v + 1e-15,
        )


if __name__ == "__main__":
    unittest.main()
