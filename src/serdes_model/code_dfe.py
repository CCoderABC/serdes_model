"""Symbol-spaced PAM4 decision-feedback equalizer after the code-domain FFE."""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt

import numpy as np
from numpy.typing import NDArray

from .code_ffe import CodeDomainFfeResult


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class CodeDomainDfeResult:
    """Trained DFE coefficients and held-out decision results."""

    feedback_delays_ui: IntArray
    feedback_coefficients: FloatArray
    ridge_lambda: float
    training_sample_count: int
    test_indices: IntArray
    test_symbols: FloatArray
    ffe_output: FloatArray
    baseline_decisions: FloatArray
    genie_corrected_output: FloatArray
    genie_decisions: FloatArray
    decision_directed_output: FloatArray
    decision_directed_decisions: FloatArray
    decision_gain: float
    decision_offset: float
    slicer_thresholds: FloatArray
    baseline_dp_snr_db: float
    baseline_gaussian_der: float
    baseline_gaussian_ber_approx: float
    baseline_empirical_der: float
    baseline_empirical_ber: float
    baseline_symbol_error_count: int
    baseline_bit_error_count: int
    genie_dp_snr_db: float
    genie_gaussian_der: float
    genie_gaussian_ber_approx: float
    genie_empirical_der: float
    genie_empirical_ber: float
    genie_symbol_error_count: int
    genie_bit_error_count: int
    decision_directed_dp_snr_db: float
    decision_directed_gaussian_der: float
    decision_directed_gaussian_ber_approx: float
    decision_directed_empirical_der: float
    decision_directed_empirical_ber: float
    decision_directed_symbol_error_count: int
    decision_directed_bit_error_count: int
    decision_directed_confusion_matrix: IntArray
    error_propagation_symbol_count: int


def _slice_pam4(observations: FloatArray, gain: float, offset: float) -> FloatArray:
    levels = np.asarray([-3.0, -1.0, 1.0, 3.0])
    normalized = (observations - offset) / gain
    indices = np.argmin(np.abs(normalized[:, None] - levels[None, :]), axis=1)
    return levels[indices]


def _gray_bit_error_count(symbols: FloatArray, decisions: FloatArray) -> int:
    """Count bit errors using the PAM4 Gray map 00, 01, 11, 10."""

    levels = np.asarray([-3.0, -1.0, 1.0, 3.0])
    gray_words = np.asarray([0b00, 0b01, 0b11, 0b10], dtype=np.int64)
    hamming_weight = np.asarray([0, 1, 1, 2], dtype=np.int64)
    true_words = gray_words[np.searchsorted(levels, symbols)]
    decision_words = gray_words[np.searchsorted(levels, decisions)]
    return int(np.sum(hamming_weight[np.bitwise_xor(true_words, decision_words)]))


def _evaluate(
    observations: FloatArray,
    symbols: FloatArray,
    *,
    gain: float,
    offset: float,
) -> tuple[FloatArray, float, float, float, float, int, int, IntArray]:
    decisions = _slice_pam4(observations, gain, offset)
    residual = observations - (offset + gain * symbols)
    error_variance = float(np.mean(np.square(residual)))
    signal_variance = 5.0 * gain**2
    dp_snr_db = (
        np.inf
        if error_variance == 0.0
        else float(10.0 * np.log10(signal_variance / error_variance))
    )
    gaussian_der = (
        0.0
        if error_variance == 0.0
        else float(0.75 * erfc(abs(gain) / sqrt(2.0 * error_variance)))
    )
    symbol_error_count = int(np.count_nonzero(decisions != symbols))
    bit_error_count = _gray_bit_error_count(symbols, decisions)
    confusion = np.zeros((4, 4), dtype=np.int64)
    levels = np.asarray([-3.0, -1.0, 1.0, 3.0])
    np.add.at(
        confusion,
        (np.searchsorted(levels, symbols), np.searchsorted(levels, decisions)),
        1,
    )
    return (
        decisions,
        dp_snr_db,
        gaussian_der,
        float(symbol_error_count / symbols.size),
        float(bit_error_count / (2 * symbols.size)),
        symbol_error_count,
        bit_error_count,
        confusion,
    )


def train_code_domain_dfe(
    ffe: CodeDomainFfeResult,
    *,
    feedback_tap_count: int,
    ridge_fraction: float,
) -> CodeDomainDfeResult:
    """Train postcursor feedback taps and evaluate a held-out PAM4 sequence.

    Training uses known symbols. Held-out results include both a genie-aided
    reference and a causal decision-directed loop. The first ``tap_count`` test
    symbols warm up the decision history and are excluded from all metrics.
    """

    if feedback_tap_count < 1:
        raise ValueError("feedback_tap_count must be positive")
    if ridge_fraction < 0.0:
        raise ValueError("ridge_fraction must be nonnegative")
    if ffe.training_symbols.size <= 4 * feedback_tap_count:
        raise ValueError("insufficient FFE training samples for DFE tap count")
    if ffe.test_symbols.size <= 4 * feedback_tap_count:
        raise ValueError("insufficient held-out samples for DFE tap count")

    delays = np.arange(1, feedback_tap_count + 1, dtype=np.int64)
    training_symbols = np.asarray(ffe.training_symbols, dtype=float)
    training_output = np.asarray(ffe.training_output, dtype=float)
    training_current = training_symbols[feedback_tap_count:]
    training_past = np.column_stack(
        [
            training_symbols[
                feedback_tap_count - delay : training_symbols.size - delay
            ]
            for delay in delays
        ]
    )
    design = np.column_stack(
        (
            training_current,
            training_past,
            np.ones(training_current.size, dtype=float),
        )
    )
    normalized_gram = design.T @ design / training_current.size
    normalized_cross = design.T @ training_output[feedback_tap_count:] / training_current.size
    feedback_scale = float(np.trace(training_past.T @ training_past))
    feedback_scale /= training_past.size
    ridge_lambda = ridge_fraction * feedback_scale
    regularizer = np.zeros_like(normalized_gram)
    regularizer[1 : 1 + feedback_tap_count, 1 : 1 + feedback_tap_count] = (
        ridge_lambda * np.eye(feedback_tap_count)
    )
    coefficients = np.linalg.solve(normalized_gram + regularizer, normalized_cross)
    gain = float(coefficients[0])
    feedback = np.asarray(coefficients[1 : 1 + feedback_tap_count], dtype=float)
    offset = float(coefficients[-1])
    if np.isclose(gain, 0.0):
        raise ValueError("trained DFE input has zero PAM4 decision gain")

    all_test_symbols = np.asarray(ffe.test_symbols, dtype=float)
    all_test_output = np.asarray(ffe.test_output, dtype=float)
    warmup = feedback_tap_count
    evaluation_symbols = all_test_symbols[warmup:]
    evaluation_output = all_test_output[warmup:]
    evaluation_indices = np.asarray(ffe.test_indices[warmup:], dtype=np.int64)
    past_true = np.column_stack(
        [
            all_test_symbols[warmup - delay : all_test_symbols.size - delay]
            for delay in delays
        ]
    )
    genie_output = evaluation_output - past_true @ feedback

    # Prime the causal loop with ordinary FFE decisions, then feed back only
    # decisions made earlier in the held-out sequence.
    decision_history = np.empty_like(all_test_symbols)
    decision_history[:warmup] = _slice_pam4(
        all_test_output[:warmup], gain, offset
    )
    decision_directed_output = np.empty_like(evaluation_output)
    for output_index, sequence_index in enumerate(
        range(warmup, all_test_symbols.size)
    ):
        past_decisions = decision_history[sequence_index - delays]
        corrected = all_test_output[sequence_index] - float(
            feedback @ past_decisions
        )
        decision_directed_output[output_index] = corrected
        decision_history[sequence_index] = _slice_pam4(
            np.asarray([corrected]), gain, offset
        )[0]

    baseline = _evaluate(
        evaluation_output, evaluation_symbols, gain=gain, offset=offset
    )
    genie = _evaluate(genie_output, evaluation_symbols, gain=gain, offset=offset)
    decision_directed = _evaluate(
        decision_directed_output, evaluation_symbols, gain=gain, offset=offset
    )
    error_propagation_count = int(
        np.count_nonzero(decision_directed[0] != genie[0])
    )

    return CodeDomainDfeResult(
        feedback_delays_ui=delays,
        feedback_coefficients=feedback,
        ridge_lambda=float(ridge_lambda),
        training_sample_count=int(training_current.size),
        test_indices=evaluation_indices,
        test_symbols=evaluation_symbols,
        ffe_output=evaluation_output,
        baseline_decisions=baseline[0],
        genie_corrected_output=genie_output,
        genie_decisions=genie[0],
        decision_directed_output=decision_directed_output,
        decision_directed_decisions=decision_directed[0],
        decision_gain=gain,
        decision_offset=offset,
        slicer_thresholds=offset + gain * np.asarray([-2.0, 0.0, 2.0]),
        baseline_dp_snr_db=baseline[1],
        baseline_gaussian_der=baseline[2],
        baseline_gaussian_ber_approx=0.5 * baseline[2],
        baseline_empirical_der=baseline[3],
        baseline_empirical_ber=baseline[4],
        baseline_symbol_error_count=baseline[5],
        baseline_bit_error_count=baseline[6],
        genie_dp_snr_db=genie[1],
        genie_gaussian_der=genie[2],
        genie_gaussian_ber_approx=0.5 * genie[2],
        genie_empirical_der=genie[3],
        genie_empirical_ber=genie[4],
        genie_symbol_error_count=genie[5],
        genie_bit_error_count=genie[6],
        decision_directed_dp_snr_db=decision_directed[1],
        decision_directed_gaussian_der=decision_directed[2],
        decision_directed_gaussian_ber_approx=0.5 * decision_directed[2],
        decision_directed_empirical_der=decision_directed[3],
        decision_directed_empirical_ber=decision_directed[4],
        decision_directed_symbol_error_count=decision_directed[5],
        decision_directed_bit_error_count=decision_directed[6],
        decision_directed_confusion_matrix=decision_directed[7],
        error_propagation_symbol_count=error_propagation_count,
    )
