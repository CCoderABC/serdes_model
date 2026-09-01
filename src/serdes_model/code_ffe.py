"""Data-trained symbol-spaced FFE operating directly on ADC output codes."""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class CodeDomainFfeResult:
    """Training and held-out evaluation results for a code-domain FFE."""

    tap_offsets_ui: IntArray
    coefficients_per_code: FloatArray
    code_midpoint: float
    ridge_lambda: float
    training_indices: IntArray
    test_indices: IntArray
    training_symbols: FloatArray
    test_symbols: FloatArray
    raw_test_codes_centered: FloatArray
    training_output: FloatArray
    test_output: FloatArray
    test_decisions: FloatArray
    decision_gain: float
    decision_offset: float
    slicer_thresholds: FloatArray
    signal_variance: float
    residual_error_variance: float
    dp_snr_db: float
    gaussian_der: float
    empirical_der: float
    empirical_error_count: int
    confusion_matrix: IntArray
    raw_decision_gain: float
    raw_decision_offset: float
    raw_dp_snr_db: float
    raw_gaussian_der: float
    raw_empirical_der: float


def _observation_matrix(
    centered_codes: FloatArray,
    indices: IntArray,
    tap_offsets: IntArray,
) -> FloatArray:
    return centered_codes[indices[:, None] + tap_offsets[None, :]]


def _fit_gain_offset(
    symbols: FloatArray, observations: FloatArray
) -> tuple[float, float]:
    design = np.column_stack((symbols, np.ones(symbols.size, dtype=float)))
    gain, offset = np.linalg.lstsq(design, observations, rcond=None)[0]
    return float(gain), float(offset)


def _decision_metrics(
    *,
    training_symbols: FloatArray,
    training_observations: FloatArray,
    test_symbols: FloatArray,
    test_observations: FloatArray,
) -> tuple[
    float,
    float,
    FloatArray,
    FloatArray,
    float,
    float,
    float,
    float,
    int,
    IntArray,
]:
    gain, offset = _fit_gain_offset(training_symbols, training_observations)
    if np.isclose(gain, 0.0):
        raise ValueError("trained output has zero PAM4 decision gain")
    levels = np.asarray([-3.0, -1.0, 1.0, 3.0])
    normalized_test = (test_observations - offset) / gain
    decision_indices = np.argmin(
        np.abs(normalized_test[:, None] - levels[None, :]), axis=1
    )
    decisions = levels[decision_indices]
    thresholds = offset + gain * np.asarray([-2.0, 0.0, 2.0])
    residual = test_observations - (offset + gain * test_symbols)
    signal_variance = 5.0 * gain**2
    error_variance = float(np.mean(np.square(residual)))
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
    errors = decisions != test_symbols
    error_count = int(np.count_nonzero(errors))
    empirical_der = float(np.mean(errors))
    confusion = np.zeros((4, 4), dtype=np.int64)
    true_indices = np.searchsorted(levels, test_symbols)
    np.add.at(confusion, (true_indices, decision_indices), 1)
    return (
        gain,
        offset,
        thresholds,
        decisions,
        signal_variance,
        error_variance,
        dp_snr_db,
        gaussian_der,
        error_count,
        confusion,
    )


def train_code_domain_ffe(
    adc_codes: ArrayLike,
    target_symbols: ArrayLike,
    *,
    code_midpoint: float,
    tap_count: int,
    training_fraction: float,
    ridge_fraction: float,
) -> CodeDomainFfeResult:
    """Train a centered symbol-spaced FFE and evaluate on a held-out segment.

    Tap offsets are symmetric around the target symbol. Positive offsets are
    realizable with an overall output latency equal to half the tap span.
    A tap-span guard around the train/test boundary prevents either segment's
    observation windows from consuming samples from the other segment.
    """

    adc_codes = np.asarray(adc_codes)
    symbols = np.asarray(target_symbols, dtype=float)
    if adc_codes.ndim != 1 or symbols.ndim != 1 or adc_codes.size != symbols.size:
        raise ValueError("ADC codes and target symbols must be equal one-dimensional arrays")
    if adc_codes.size < 64 or not np.all(np.isfinite(adc_codes)):
        raise ValueError("at least 64 finite ADC codes are required")
    if not np.all(np.isin(symbols, [-3.0, -1.0, 1.0, 3.0])):
        raise ValueError("target symbols must contain only normalized PAM4 levels")
    if tap_count < 1 or tap_count % 2 == 0:
        raise ValueError("tap_count must be a positive odd integer")
    if not 0.1 <= training_fraction <= 0.9:
        raise ValueError("training_fraction must be between 0.1 and 0.9")
    if ridge_fraction < 0.0:
        raise ValueError("ridge_fraction must be nonnegative")

    centered_codes = adc_codes.astype(float) - code_midpoint
    half_span = tap_count // 2
    tap_offsets = np.arange(-half_span, half_span + 1, dtype=np.int64)
    split_index = int(np.floor(training_fraction * adc_codes.size))
    training_indices = np.arange(
        half_span, split_index - half_span, dtype=np.int64
    )
    test_indices = np.arange(
        split_index + half_span,
        adc_codes.size - half_span,
        dtype=np.int64,
    )
    if training_indices.size < 4 * tap_count or test_indices.size < 4 * tap_count:
        raise ValueError("insufficient independent training or test samples for tap count")

    training_matrix = _observation_matrix(
        centered_codes, training_indices, tap_offsets
    )
    test_matrix = _observation_matrix(centered_codes, test_indices, tap_offsets)
    training_symbols = symbols[training_indices]
    test_symbols = symbols[test_indices]
    normalized_gram = training_matrix.T @ training_matrix / training_indices.size
    normalized_cross = training_matrix.T @ training_symbols / training_indices.size
    ridge_lambda = ridge_fraction * float(np.trace(normalized_gram)) / tap_count
    coefficients = np.linalg.solve(
        normalized_gram + ridge_lambda * np.eye(tap_count), normalized_cross
    )
    training_output = training_matrix @ coefficients
    test_output = test_matrix @ coefficients

    (
        gain,
        offset,
        thresholds,
        decisions,
        signal_variance,
        error_variance,
        dp_snr_db,
        gaussian_der,
        error_count,
        confusion,
    ) = _decision_metrics(
        training_symbols=training_symbols,
        training_observations=training_output,
        test_symbols=test_symbols,
        test_observations=test_output,
    )

    raw_training = centered_codes[training_indices]
    raw_test = centered_codes[test_indices]
    (
        raw_gain,
        raw_offset,
        _,
        raw_decisions,
        _,
        raw_error_variance,
        raw_dp_snr_db,
        raw_gaussian_der,
        raw_error_count,
        _,
    ) = _decision_metrics(
        training_symbols=training_symbols,
        training_observations=raw_training,
        test_symbols=test_symbols,
        test_observations=raw_test,
    )

    return CodeDomainFfeResult(
        tap_offsets_ui=tap_offsets,
        coefficients_per_code=np.asarray(coefficients, dtype=float),
        code_midpoint=float(code_midpoint),
        ridge_lambda=float(ridge_lambda),
        training_indices=training_indices,
        test_indices=test_indices,
        training_symbols=training_symbols,
        test_symbols=test_symbols,
        raw_test_codes_centered=raw_test,
        training_output=np.asarray(training_output, dtype=float),
        test_output=np.asarray(test_output, dtype=float),
        test_decisions=decisions,
        decision_gain=gain,
        decision_offset=offset,
        slicer_thresholds=thresholds,
        signal_variance=signal_variance,
        residual_error_variance=error_variance,
        dp_snr_db=dp_snr_db,
        gaussian_der=gaussian_der,
        empirical_der=float(error_count / test_indices.size),
        empirical_error_count=error_count,
        confusion_matrix=confusion,
        raw_decision_gain=raw_gain,
        raw_decision_offset=raw_offset,
        raw_dp_snr_db=raw_dp_snr_db,
        raw_gaussian_der=raw_gaussian_der,
        raw_empirical_der=float(raw_error_count / test_indices.size),
    )
