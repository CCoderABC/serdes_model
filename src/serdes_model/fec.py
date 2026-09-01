"""Bit-exact hard-decision RS(544,514) reference FEC for PAM4 links."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from numpy.typing import ArrayLike, NDArray
from reedsolo import RSCodec, ReedSolomonError


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


RS_N_SYMBOLS = 544
RS_K_SYMBOLS = 514
RS_PARITY_SYMBOLS = 30
RS_SYMBOL_BITS = 10
RS_CORRECTABLE_SYMBOL_ERRORS = 15
RS_PRIMITIVE_POLYNOMIAL = 1033
RS_FIELD_GENERATOR = 2
RS_FIRST_CONSECUTIVE_ROOT = 0
RS_GENERATOR_COEFFICIENTS = np.asarray(
    [
        575, 552, 187, 230, 552, 1, 108, 565, 282, 249,
        593, 132, 94, 720, 495, 385, 942, 503, 883, 361,
        788, 610, 193, 392, 127, 185, 158, 128, 834, 523,
    ],
    dtype=np.int64,
)


@dataclass(frozen=True)
class Rs544Pam4Stream:
    """PAM4 source stream containing systematic RS(544,514) codewords."""

    pam4_symbols: FloatArray
    codeword_start_symbol_indices: IntArray
    payload_gf_symbols: IntArray
    encoded_gf_symbols: IntArray
    alignment_offset_pam4_symbols: int


@dataclass(frozen=True)
class Rs544CodewordResult:
    """One received RS codeword and its decoder outcome."""

    codeword_index: int
    start_pam4_symbol_index: int
    pre_fec_bit_errors: int
    pre_fec_symbol_errors: int
    decoder_success: bool
    decoder_reported_corrections: int
    post_fec_payload_bit_errors: int


@dataclass(frozen=True)
class Rs544FecResult:
    """Aggregate measured and independent-error RS-FEC results."""

    codewords: tuple[Rs544CodewordResult, ...]
    evaluated_codeword_count: int
    payload_bit_count: int
    received_codeword_bit_count: int
    pre_fec_bit_error_count: int
    pre_fec_symbol_error_count: int
    pre_fec_ber: float
    pre_fec_symbol_error_rate: float
    corrected_codeword_count: int
    uncorrectable_codeword_count: int
    miscorrected_codeword_count: int
    post_fec_payload_bit_error_count: int
    post_fec_ber: float
    codeword_failure_rate: float
    iid_symbol_error_probability: float
    iid_uncorrectable_codeword_probability: float
    iid_failed_codeword_passthrough_ber_estimate: float


def _codec() -> RSCodec:
    return RSCodec(
        RS_PARITY_SYMBOLS,
        nsize=(1 << RS_SYMBOL_BITS) - 1,
        fcr=RS_FIRST_CONSECUTIVE_ROOT,
        prim=RS_PRIMITIVE_POLYNOMIAL,
        generator=RS_FIELD_GENERATOR,
        c_exp=RS_SYMBOL_BITS,
    )


def _gf_symbols_to_bits(symbols: ArrayLike) -> BoolArray:
    symbols = np.asarray(symbols, dtype=np.int64)
    if np.any((symbols < 0) | (symbols >= (1 << RS_SYMBOL_BITS))):
        raise ValueError("GF symbols must be 10-bit unsigned values")
    shifts = np.arange(RS_SYMBOL_BITS - 1, -1, -1, dtype=np.int64)
    return ((symbols[..., None] >> shifts) & 1).astype(bool).reshape(-1)


def _bits_to_gf_symbols(bits: ArrayLike) -> IntArray:
    bits = np.asarray(bits, dtype=bool)
    if bits.ndim != 1 or bits.size % RS_SYMBOL_BITS:
        raise ValueError("bit count must be a multiple of ten")
    weights = 1 << np.arange(RS_SYMBOL_BITS - 1, -1, -1, dtype=np.int64)
    return np.asarray(bits.reshape(-1, RS_SYMBOL_BITS) @ weights, dtype=np.int64)


def pam4_symbols_to_gray_bits(symbols: ArrayLike) -> BoolArray:
    """Demap PAM4 levels using low-to-high Gray labels 00, 01, 11, 10."""

    symbols = np.asarray(symbols, dtype=float)
    if not np.all(np.isin(symbols, [-3.0, -1.0, 1.0, 3.0])):
        raise ValueError("symbols must contain only normalized PAM4 levels")
    level_indices = np.searchsorted(np.asarray([-3.0, -1.0, 1.0, 3.0]), symbols)
    bit_pairs = np.asarray([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=bool)
    return bit_pairs[level_indices].reshape(-1)


def gray_bits_to_pam4_symbols(bits: ArrayLike) -> FloatArray:
    """Map adjacent bit pairs to normalized Gray-coded PAM4 levels."""

    bits = np.asarray(bits, dtype=bool)
    if bits.ndim != 1 or bits.size % 2:
        raise ValueError("PAM4 bit count must be a positive multiple of two")
    pair_values = 2 * bits[0::2].astype(np.int64) + bits[1::2].astype(np.int64)
    binary_pair_to_level = np.asarray([-3.0, -1.0, 3.0, 1.0])
    return binary_pair_to_level[pair_values]


def generate_rs544_pam4_stream(
    symbol_count: int,
    *,
    seed: int,
    alignment_offset_pam4_symbols: int,
) -> Rs544Pam4Stream:
    """Generate random payloads, encode them, and serialize them as PAM4."""

    codeword_pam4_symbols = RS_N_SYMBOLS * RS_SYMBOL_BITS // 2
    if symbol_count < codeword_pam4_symbols:
        raise ValueError("symbol_count must fit at least one RS codeword")
    if not 0 <= alignment_offset_pam4_symbols < codeword_pam4_symbols:
        raise ValueError("alignment offset must be within one PAM4 codeword")
    generator = np.random.default_rng(seed)
    pam4 = generator.choice(
        np.asarray([-3.0, -1.0, 1.0, 3.0]), size=symbol_count
    ).astype(float)
    codeword_count = (
        symbol_count - alignment_offset_pam4_symbols
    ) // codeword_pam4_symbols
    starts = alignment_offset_pam4_symbols + np.arange(
        codeword_count, dtype=np.int64
    ) * codeword_pam4_symbols
    payloads = generator.integers(
        0,
        1 << RS_SYMBOL_BITS,
        size=(codeword_count, RS_K_SYMBOLS),
        dtype=np.int64,
    )
    encoded = np.empty((codeword_count, RS_N_SYMBOLS), dtype=np.int64)
    codec = _codec()
    for codeword_index, (start, payload) in enumerate(zip(starts, payloads)):
        encoded_symbols = np.asarray(codec.encode(payload.tolist()), dtype=np.int64)
        if encoded_symbols.size != RS_N_SYMBOLS:
            raise RuntimeError("RS encoder returned an unexpected codeword length")
        encoded[codeword_index] = encoded_symbols
        pam4[start : start + codeword_pam4_symbols] = gray_bits_to_pam4_symbols(
            _gf_symbols_to_bits(encoded_symbols)
        )
    return Rs544Pam4Stream(
        pam4_symbols=pam4,
        codeword_start_symbol_indices=starts,
        payload_gf_symbols=payloads,
        encoded_gf_symbols=encoded,
        alignment_offset_pam4_symbols=alignment_offset_pam4_symbols,
    )


def _iid_fec_estimates(pre_fec_ber: float) -> tuple[float, float, float]:
    """Return symbol error, uncorrectable-word, and failed-word BER estimates."""

    if not 0.0 <= pre_fec_ber <= 1.0:
        raise ValueError("pre_fec_ber must be between zero and one")
    symbol_error_probability = 1.0 - (1.0 - pre_fec_ber) ** RS_SYMBOL_BITS
    if symbol_error_probability == 0.0:
        return 0.0, 0.0, 0.0
    probabilities = np.asarray(
        [
            comb(RS_N_SYMBOLS, errors)
            * symbol_error_probability**errors
            * (1.0 - symbol_error_probability) ** (RS_N_SYMBOLS - errors)
            for errors in range(RS_N_SYMBOLS + 1)
        ],
        dtype=float,
    )
    failure_slice = slice(RS_CORRECTABLE_SYMBOL_ERRORS + 1, None)
    failure_probability = float(np.sum(probabilities[failure_slice]))
    error_symbol_indices = np.arange(RS_N_SYMBOLS + 1, dtype=float)
    expected_failed_error_symbols = float(
        np.sum(error_symbol_indices[failure_slice] * probabilities[failure_slice])
    )
    # Assume failed codewords pass through with their original bit errors and
    # successful codewords are corrected perfectly. Miscorrection is excluded.
    failed_word_ber = (
        pre_fec_ber
        / symbol_error_probability
        * expected_failed_error_symbols
        / RS_N_SYMBOLS
    )
    return symbol_error_probability, failure_probability, failed_word_ber


def required_pre_fec_ber_for_iid_post_ber(target_post_fec_ber: float) -> float:
    """Invert the independent-error passthrough model by bisection."""

    if not 0.0 < target_post_fec_ber < 1.0:
        raise ValueError("target post-FEC BER must be between zero and one")
    lower = 0.0
    upper = 0.5
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if _iid_fec_estimates(midpoint)[2] > target_post_fec_ber:
            upper = midpoint
        else:
            lower = midpoint
    return float(0.5 * (lower + upper))


def evaluate_rs544_fec(
    stream: Rs544Pam4Stream,
    detected_pam4_symbols: ArrayLike,
    detected_global_symbol_indices: ArrayLike,
) -> Rs544FecResult:
    """Decode every complete RS codeword covered by the detected symbol span."""

    detected = np.asarray(detected_pam4_symbols, dtype=float)
    global_indices = np.asarray(detected_global_symbol_indices, dtype=np.int64)
    if detected.ndim != 1 or global_indices.shape != detected.shape:
        raise ValueError("detected symbols and indices must be equal 1-D arrays")
    if detected.size < 1 or np.any(np.diff(global_indices) != 1):
        raise ValueError("detected global indices must be non-empty and contiguous")
    codeword_pam4_symbols = RS_N_SYMBOLS * RS_SYMBOL_BITS // 2
    codec = _codec()
    codeword_results: list[Rs544CodewordResult] = []
    for codeword_index, start in enumerate(stream.codeword_start_symbol_indices):
        stop = int(start + codeword_pam4_symbols)
        if start < global_indices[0] or stop - 1 > global_indices[-1]:
            continue
        local_start = int(start - global_indices[0])
        received_pam4 = detected[local_start : local_start + codeword_pam4_symbols]
        received_symbols = _bits_to_gf_symbols(
            pam4_symbols_to_gray_bits(received_pam4)
        )
        transmitted_symbols = stream.encoded_gf_symbols[codeword_index]
        pre_bit_errors = int(
            np.count_nonzero(
                _gf_symbols_to_bits(received_symbols)
                != _gf_symbols_to_bits(transmitted_symbols)
            )
        )
        pre_symbol_errors = int(np.count_nonzero(received_symbols != transmitted_symbols))
        decoder_success = True
        decoder_reported_corrections = 0
        try:
            decoded, _, error_positions = codec.decode(received_symbols.tolist())
            decoded_payload = np.asarray(decoded, dtype=np.int64)
            decoder_reported_corrections = len(error_positions)
        except ReedSolomonError:
            decoder_success = False
            decoded_payload = received_symbols[:RS_K_SYMBOLS]
        payload = stream.payload_gf_symbols[codeword_index]
        post_bit_errors = int(
            np.count_nonzero(
                _gf_symbols_to_bits(decoded_payload) != _gf_symbols_to_bits(payload)
            )
        )
        codeword_results.append(
            Rs544CodewordResult(
                codeword_index=codeword_index,
                start_pam4_symbol_index=int(start),
                pre_fec_bit_errors=pre_bit_errors,
                pre_fec_symbol_errors=pre_symbol_errors,
                decoder_success=decoder_success,
                decoder_reported_corrections=decoder_reported_corrections,
                post_fec_payload_bit_errors=post_bit_errors,
            )
        )
    if not codeword_results:
        raise ValueError("detected span contains no complete RS codeword")

    evaluated_count = len(codeword_results)
    received_bits = evaluated_count * RS_N_SYMBOLS * RS_SYMBOL_BITS
    payload_bits = evaluated_count * RS_K_SYMBOLS * RS_SYMBOL_BITS
    pre_bit_errors = sum(result.pre_fec_bit_errors for result in codeword_results)
    pre_symbol_errors = sum(result.pre_fec_symbol_errors for result in codeword_results)
    post_bit_errors = sum(
        result.post_fec_payload_bit_errors for result in codeword_results
    )
    uncorrectable_count = sum(
        not result.decoder_success for result in codeword_results
    )
    miscorrected_count = sum(
        result.decoder_success and result.post_fec_payload_bit_errors > 0
        for result in codeword_results
    )
    pre_fec_ber = pre_bit_errors / received_bits
    symbol_error_probability, iid_failure, iid_post_ber = _iid_fec_estimates(
        pre_fec_ber
    )
    return Rs544FecResult(
        codewords=tuple(codeword_results),
        evaluated_codeword_count=evaluated_count,
        payload_bit_count=payload_bits,
        received_codeword_bit_count=received_bits,
        pre_fec_bit_error_count=pre_bit_errors,
        pre_fec_symbol_error_count=pre_symbol_errors,
        pre_fec_ber=float(pre_fec_ber),
        pre_fec_symbol_error_rate=float(
            pre_symbol_errors / (evaluated_count * RS_N_SYMBOLS)
        ),
        corrected_codeword_count=evaluated_count - uncorrectable_count,
        uncorrectable_codeword_count=uncorrectable_count,
        miscorrected_codeword_count=miscorrected_count,
        post_fec_payload_bit_error_count=post_bit_errors,
        post_fec_ber=float(post_bit_errors / payload_bits),
        codeword_failure_rate=float(uncorrectable_count / evaluated_count),
        iid_symbol_error_probability=symbol_error_probability,
        iid_uncorrectable_codeword_probability=iid_failure,
        iid_failed_codeword_passthrough_ber_estimate=iid_post_ber,
    )
