from __future__ import annotations

import unittest

import numpy as np

from serdes_model.fec import (
    RS_CORRECTABLE_SYMBOL_ERRORS,
    RS_GENERATOR_COEFFICIENTS,
    RS_K_SYMBOLS,
    RS_N_SYMBOLS,
    evaluate_rs544_fec,
    generate_rs544_pam4_stream,
    gray_bits_to_pam4_symbols,
    pam4_symbols_to_gray_bits,
    required_pre_fec_ber_for_iid_post_ber,
)


class Rs544FecTests(unittest.TestCase):
    def test_gray_pam4_mapping_round_trips(self) -> None:
        bits = np.asarray([0, 0, 0, 1, 1, 1, 1, 0], dtype=bool)
        symbols = gray_bits_to_pam4_symbols(bits)
        np.testing.assert_array_equal(symbols, [-3.0, -1.0, 1.0, 3.0])
        np.testing.assert_array_equal(pam4_symbols_to_gray_bits(symbols), bits)

    def test_ieee_generator_coefficients_match_codec(self) -> None:
        from reedsolo import RSCodec

        codec = RSCodec(
            30, nsize=1023, fcr=0, prim=1033, generator=2, c_exp=10
        )
        np.testing.assert_array_equal(
            np.asarray(codec.gen[30][1:], dtype=np.int64),
            RS_GENERATOR_COEFFICIENTS,
        )

    def test_corrects_fifteen_symbol_errors(self) -> None:
        stream = generate_rs544_pam4_stream(
            RS_N_SYMBOLS * 10 // 2,
            seed=53,
            alignment_offset_pam4_symbols=0,
        )
        detected = stream.pam4_symbols.copy()
        # Flip one bit in each of fifteen different 10-bit RS symbols.
        bits = pam4_symbols_to_gray_bits(detected)
        for symbol_index in range(RS_CORRECTABLE_SYMBOL_ERRORS):
            bits[10 * symbol_index] = ~bits[10 * symbol_index]
        detected = gray_bits_to_pam4_symbols(bits)
        result = evaluate_rs544_fec(
            stream,
            detected,
            np.arange(detected.size, dtype=np.int64),
        )
        self.assertEqual(result.pre_fec_symbol_error_count, 15)
        self.assertEqual(result.post_fec_payload_bit_error_count, 0)
        self.assertEqual(result.corrected_codeword_count, 1)
        self.assertEqual(result.uncorrectable_codeword_count, 0)

    def test_stream_is_systematic_and_has_expected_dimensions(self) -> None:
        stream = generate_rs544_pam4_stream(
            2 * RS_N_SYMBOLS * 10 // 2,
            seed=59,
            alignment_offset_pam4_symbols=0,
        )
        self.assertEqual(stream.payload_gf_symbols.shape, (2, RS_K_SYMBOLS))
        self.assertEqual(stream.encoded_gf_symbols.shape, (2, RS_N_SYMBOLS))
        np.testing.assert_array_equal(
            stream.encoded_gf_symbols[:, :RS_K_SYMBOLS],
            stream.payload_gf_symbols,
        )

    def test_required_prefec_ber_inverts_iid_model(self) -> None:
        from serdes_model.fec import _iid_fec_estimates

        required = required_pre_fec_ber_for_iid_post_ber(1e-12)
        self.assertAlmostEqual(_iid_fec_estimates(required)[2] / 1e-12, 1.0, places=8)


if __name__ == "__main__":
    unittest.main()
