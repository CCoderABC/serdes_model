from __future__ import annotations

import unittest

import numpy as np

from serdes_model.code_ffe import train_code_domain_ffe


class CodeDomainFfeTests(unittest.TestCase):
    def test_memoryless_quantized_channel_is_recovered_without_errors(self) -> None:
        generator = np.random.default_rng(31)
        symbols = generator.choice([-3.0, -1.0, 1.0, 3.0], size=4096)
        codes = np.rint(31.5 + 5.0 * symbols).astype(int)
        result = train_code_domain_ffe(
            codes,
            symbols,
            code_midpoint=31.5,
            tap_count=5,
            training_fraction=0.5,
            ridge_fraction=1e-8,
        )
        self.assertEqual(result.empirical_error_count, 0)
        self.assertGreater(result.dp_snr_db, 40.0)
        self.assertGreater(result.coefficients_per_code[2], 0.0)

    def test_ffe_improves_channel_with_symbol_spaced_isi(self) -> None:
        generator = np.random.default_rng(37)
        symbols = generator.choice([-3.0, -1.0, 1.0, 3.0], size=8192)
        analog_codes = 31.5 + 3.5 * (
            symbols
            + 0.65 * np.roll(symbols, 1)
            + 0.25 * np.roll(symbols, 2)
        )
        codes = np.clip(np.rint(analog_codes), 0, 63).astype(int)
        result = train_code_domain_ffe(
            codes,
            symbols,
            code_midpoint=31.5,
            tap_count=9,
            training_fraction=0.5,
            ridge_fraction=1e-5,
        )
        self.assertGreater(result.dp_snr_db, result.raw_dp_snr_db + 6.0)
        self.assertLess(result.empirical_der, result.raw_empirical_der)

    def test_train_and_test_observation_windows_do_not_overlap(self) -> None:
        symbols = np.tile([-3.0, -1.0, 1.0, 3.0], 512)
        codes = np.rint(31.5 + 4.0 * symbols).astype(int)
        result = train_code_domain_ffe(
            codes,
            symbols,
            code_midpoint=31.5,
            tap_count=17,
            training_fraction=0.5,
            ridge_fraction=1e-6,
        )
        half_span = result.tap_offsets_ui.size // 2
        self.assertLess(
            int(result.training_indices[-1]) + half_span,
            int(result.test_indices[0]) - half_span,
        )


if __name__ == "__main__":
    unittest.main()
