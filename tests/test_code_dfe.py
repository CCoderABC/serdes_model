from __future__ import annotations

import unittest

import numpy as np

from serdes_model.code_dfe import train_code_domain_dfe
from serdes_model.code_ffe import train_code_domain_ffe


class CodeDomainDfeTests(unittest.TestCase):
    def test_dfe_cancels_known_postcursor(self) -> None:
        generator = np.random.default_rng(41)
        symbols = generator.choice([-3.0, -1.0, 1.0, 3.0], size=8192)
        analog_codes = 31.5 + 4.0 * (
            symbols + 0.45 * np.roll(symbols, 1) - 0.18 * np.roll(symbols, 2)
        )
        codes = np.clip(np.rint(analog_codes), 0, 63).astype(int)
        ffe = train_code_domain_ffe(
            codes,
            symbols,
            code_midpoint=31.5,
            tap_count=1,
            training_fraction=0.5,
            ridge_fraction=1e-8,
        )
        result = train_code_domain_dfe(
            ffe, feedback_tap_count=2, ridge_fraction=1e-8
        )
        self.assertGreater(
            result.genie_dp_snr_db, result.baseline_dp_snr_db + 8.0
        )
        self.assertLess(
            result.decision_directed_empirical_der,
            result.baseline_empirical_der,
        )
        self.assertEqual(result.test_indices.size, ffe.test_indices.size - 2)

    def test_decision_directed_mode_exposes_error_propagation(self) -> None:
        generator = np.random.default_rng(43)
        symbols = generator.choice([-3.0, -1.0, 1.0, 3.0], size=8192)
        analog_codes = 31.5 + 2.0 * (
            symbols + 0.85 * np.roll(symbols, 1)
        )
        codes = np.clip(np.rint(analog_codes), 0, 63).astype(int)
        ffe = train_code_domain_ffe(
            codes,
            symbols,
            code_midpoint=31.5,
            tap_count=1,
            training_fraction=0.5,
            ridge_fraction=1e-8,
        )
        result = train_code_domain_dfe(
            ffe, feedback_tap_count=1, ridge_fraction=1e-8
        )
        self.assertGreaterEqual(result.error_propagation_symbol_count, 0)
        self.assertGreaterEqual(
            result.decision_directed_empirical_der,
            result.genie_empirical_der,
        )

    def test_rejects_invalid_tap_count(self) -> None:
        symbols = np.tile([-3.0, -1.0, 1.0, 3.0], 256)
        codes = np.rint(31.5 + 4.0 * symbols).astype(int)
        ffe = train_code_domain_ffe(
            codes,
            symbols,
            code_midpoint=31.5,
            tap_count=1,
            training_fraction=0.5,
            ridge_fraction=0.0,
        )
        with self.assertRaises(ValueError):
            train_code_domain_dfe(ffe, feedback_tap_count=0, ridge_fraction=0.0)


if __name__ == "__main__":
    unittest.main()
