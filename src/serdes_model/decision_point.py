"""Decision-point SNR and PAM4 detector-error calculations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import erfc, sqrt
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class DecisionPointResult:
    """Result of sampling-phase and symbol-spaced MMSE-FFE optimization."""

    sampling_phase_samples: int
    sampling_phase_ui: float
    raw_cursor_offsets_ui: IntArray
    raw_cursor_v: FloatArray
    ffe_offsets_ui: IntArray
    ffe_coefficients_per_v: FloatArray
    equalized_cursor_offsets_ui: IntArray
    equalized_cursor: FloatArray
    main_cursor: float
    signal_variance: float
    residual_isi_variance: float
    output_noise_variance: float
    dp_snr_db: float
    gaussian_der: float
    pattern_conditioned_der: float
    pattern_tap_offsets_ui: IntArray
    pattern_tap_values: FloatArray
    gaussian_remainder_variance: float
    slicer_thresholds: FloatArray


def pam4_unit_voltage_from_delivered_power(
    average_power_dbm: float,
    differential_resistance_ohm: float,
) -> float:
    """Return the voltage of one normalized PAM4 unit at a matched port.

    Symbols are equiprobable ``[-3, -1, 1, 3]``. ``average_power_dbm`` is
    delivered power calculated from the differential RMS voltage across the
    stated differential resistance; it is not generator or available power.
    """

    if differential_resistance_ohm <= 0.0:
        raise ValueError("differential_resistance_ohm must be positive")
    average_power_w = 1e-3 * 10.0 ** (average_power_dbm / 10.0)
    return sqrt(average_power_w * differential_resistance_ohm / 5.0)


def pam4_unit_voltage_from_outer_pp(outer_pp_v: float) -> float:
    """Return one normalized PAM4 unit from loaded differential outer swing."""

    if outer_pp_v <= 0.0:
        raise ValueError("outer_pp_v must be positive")
    return outer_pp_v / 6.0


def output_noise_psd_one_sided(
    *,
    total_transfer: ArrayLike,
    afe_transfer: ArrayLike,
    source_port_density_v_per_sqrt_hz: float,
    afe_input_density_v_per_sqrt_hz: float,
) -> FloatArray:
    """Refer white source-port and AFE-input voltage noise to the ADC input."""

    total_transfer = np.asarray(total_transfer, dtype=complex)
    afe_transfer = np.asarray(afe_transfer, dtype=complex)
    if total_transfer.shape != afe_transfer.shape or total_transfer.ndim != 1:
        raise ValueError("total_transfer and afe_transfer must be equal one-dimensional arrays")
    if min(source_port_density_v_per_sqrt_hz, afe_input_density_v_per_sqrt_hz) < 0.0:
        raise ValueError("noise voltage densities must be nonnegative")
    return (
        source_port_density_v_per_sqrt_hz**2 * np.abs(total_transfer) ** 2
        + afe_input_density_v_per_sqrt_hz**2 * np.abs(afe_transfer) ** 2
    )


def noise_autocorrelation_at_ui_lags(
    frequency_hz: ArrayLike,
    one_sided_psd_v2_per_hz: ArrayLike,
    *,
    symbol_period_s: float,
    lags_ui: ArrayLike,
) -> FloatArray:
    """Integrate a one-sided real-voltage PSD into sampled autocorrelation."""

    frequency_hz = np.asarray(frequency_hz, dtype=float)
    psd = np.asarray(one_sided_psd_v2_per_hz, dtype=float)
    lags_ui = np.asarray(lags_ui, dtype=float)
    if frequency_hz.ndim != 1 or psd.shape != frequency_hz.shape:
        raise ValueError("frequency_hz and PSD must be equal one-dimensional arrays")
    if frequency_hz.size < 2 or np.any(np.diff(frequency_hz) <= 0.0):
        raise ValueError("frequency_hz must be strictly increasing")
    if symbol_period_s <= 0.0 or np.any(psd < 0.0) or not np.all(np.isfinite(psd)):
        raise ValueError("symbol period and PSD must be finite and nonnegative")
    phase = 2.0 * np.pi * np.outer(lags_ui * symbol_period_s, frequency_hz)
    integrand = psd[None, :] * np.cos(phase)
    return np.sum(
        0.5 * (integrand[:, 1:] + integrand[:, :-1]) * np.diff(frequency_hz),
        axis=1,
    )


def required_pam4_dp_snr_db(target_der: float) -> float:
    """Return variance-based PAM4 SNR required by the symmetric AWGN model."""

    if not 0.0 < target_der < 0.75:
        raise ValueError("target_der must be between zero and 0.75")
    q_argument = NormalDist().inv_cdf(1.0 - target_der / 1.5)
    return 10.0 * np.log10(5.0 * q_argument**2)


def gaussian_pam4_der(main_cursor: float, error_variance: float) -> float:
    """Return symmetric PAM4 DER for Gaussian error and midpoint thresholds."""

    if error_variance < 0.0:
        raise ValueError("error_variance must be nonnegative")
    if error_variance == 0.0:
        return 0.0 if main_cursor != 0.0 else 0.75
    q_eye = abs(main_cursor) / sqrt(error_variance)
    return 0.75 * erfc(q_eye / sqrt(2.0))


def _sample_raw_cursors(
    pulse: FloatArray,
    *,
    main_index: int,
    phase_samples: int,
    samples_per_ui: int,
    offsets_ui: IntArray,
    pam4_unit_input_v: float,
) -> FloatArray:
    sample_indices = main_index + phase_samples + offsets_ui * samples_per_ui
    cursors = np.zeros(offsets_ui.shape, dtype=float)
    valid = (sample_indices >= 0) & (sample_indices < pulse.size)
    cursors[valid] = pulse[sample_indices[valid]] * pam4_unit_input_v
    return cursors


def _noise_covariance(
    ffe_offsets_ui: IntArray,
    *,
    frequency_hz: FloatArray,
    one_sided_noise_psd_v2_per_hz: FloatArray,
    symbol_period_s: float,
) -> FloatArray:
    lag_matrix = ffe_offsets_ui[:, None] - ffe_offsets_ui[None, :]
    unique_lags, inverse = np.unique(lag_matrix, return_inverse=True)
    correlations = noise_autocorrelation_at_ui_lags(
        frequency_hz,
        one_sided_noise_psd_v2_per_hz,
        symbol_period_s=symbol_period_s,
        lags_ui=unique_lags,
    )
    return correlations[inverse].reshape(lag_matrix.shape)


def _design_mmse_ffe(
    cursor_offsets_ui: IntArray,
    cursor_v: FloatArray,
    ffe_offsets_ui: IntArray,
    noise_covariance_v2: FloatArray,
    *,
    symbol_variance: float,
) -> FloatArray:
    """Solve the finite observation-vector LMMSE problem for symbol ``a[n]``."""

    symbol_index_min = int(cursor_offsets_ui[0] + ffe_offsets_ui[0])
    symbol_index_max = int(cursor_offsets_ui[-1] + ffe_offsets_ui[-1])
    symbol_indices = np.arange(symbol_index_min, symbol_index_max + 1, dtype=np.int64)
    channel_matrix = np.zeros((ffe_offsets_ui.size, symbol_indices.size), dtype=float)
    cursor_lookup = {int(offset): value for offset, value in zip(cursor_offsets_ui, cursor_v)}
    for row, ffe_offset in enumerate(ffe_offsets_ui):
        for column, symbol_index in enumerate(symbol_indices):
            channel_matrix[row, column] = cursor_lookup.get(
                int(symbol_index - ffe_offset), 0.0
            )
    target_columns = np.flatnonzero(symbol_indices == 0)
    if target_columns.size != 1:
        raise ValueError("FFE and cursor spans do not contain the target symbol")
    cross_correlation = symbol_variance * channel_matrix[:, target_columns[0]]
    observation_covariance = (
        symbol_variance * channel_matrix @ channel_matrix.T + noise_covariance_v2
    )
    # A tiny diagonal term protects the solve when the input is nearly noiseless.
    regularization = max(
        np.finfo(float).eps,
        1e-12 * float(np.trace(observation_covariance)) / ffe_offsets_ui.size,
    )
    return np.linalg.solve(
        observation_covariance + regularization * np.eye(ffe_offsets_ui.size),
        cross_correlation,
    )


def _q_array(values: FloatArray) -> FloatArray:
    return np.fromiter(
        (0.5 * erfc(float(value) / sqrt(2.0)) for value in values),
        dtype=float,
        count=values.size,
    )


def _pattern_conditioned_der(
    equalized_offsets_ui: IntArray,
    equalized_cursor: FloatArray,
    *,
    output_noise_variance: float,
    pattern_tap_count: int,
    symbol_variance: float,
) -> tuple[float, IntArray, FloatArray, float, FloatArray]:
    main_matches = np.flatnonzero(equalized_offsets_ui == 0)
    if main_matches.size != 1:
        raise ValueError("equalized response must contain exactly one zero-offset cursor")
    main_index = int(main_matches[0])
    main_cursor = float(equalized_cursor[main_index])
    isi_indices = np.flatnonzero(equalized_offsets_ui != 0)
    if pattern_tap_count < 0 or pattern_tap_count > 10:
        raise ValueError("pattern_tap_count must be between zero and ten")
    selected_count = min(pattern_tap_count, isi_indices.size)
    strength_order = isi_indices[np.argsort(np.abs(equalized_cursor[isi_indices]))[::-1]]
    selected_indices = np.sort(strength_order[:selected_count])
    remaining_indices = np.setdiff1d(isi_indices, selected_indices, assume_unique=True)
    selected_offsets = equalized_offsets_ui[selected_indices]
    selected_taps = equalized_cursor[selected_indices]
    remainder_variance = output_noise_variance + symbol_variance * float(
        np.sum(equalized_cursor[remaining_indices] ** 2)
    )

    pam4_levels = np.array([-3.0, -1.0, 1.0, 3.0])
    if selected_count:
        patterns = np.asarray(tuple(product(pam4_levels, repeat=selected_count)), dtype=float)
        interference = patterns @ selected_taps
    else:
        interference = np.zeros(1, dtype=float)
    thresholds = np.array([-2.0, 0.0, 2.0], dtype=float) * main_cursor

    if remainder_variance == 0.0:
        errors = []
        for level_index, level in enumerate(pam4_levels):
            samples = main_cursor * level + interference
            decisions = np.digitize(samples, thresholds)
            errors.append(np.mean(decisions != level_index))
        der = float(np.mean(errors))
    else:
        sigma = sqrt(remainder_variance)
        errors = []
        for level_index, level in enumerate(pam4_levels):
            means = main_cursor * level + interference
            if level_index == 0:
                error_probability = _q_array((thresholds[0] - means) / sigma)
            elif level_index == 3:
                error_probability = _q_array((means - thresholds[2]) / sigma)
            else:
                lower_error = _q_array((means - thresholds[level_index - 1]) / sigma)
                upper_error = _q_array((thresholds[level_index] - means) / sigma)
                error_probability = lower_error + upper_error
            errors.append(float(np.mean(error_probability)))
        der = float(np.mean(errors))
    return der, selected_offsets, selected_taps, remainder_variance, thresholds


def evaluate_decision_point(
    pulse: ArrayLike,
    *,
    pulse_main_index: int,
    samples_per_ui: int,
    symbol_period_s: float,
    pam4_unit_input_v: float,
    noise_frequency_hz: ArrayLike,
    one_sided_noise_psd_v2_per_hz: ArrayLike,
    channel_pre_ui: int,
    channel_post_ui: int,
    ffe_tap_count: int,
    pattern_tap_count: int,
    optimize_sampling_phase: bool = True,
) -> DecisionPointResult:
    """Optimize sampling phase and return MMSE-FFE dpSNR and DER metrics."""

    pulse = np.asarray(pulse, dtype=float)
    frequency_hz = np.asarray(noise_frequency_hz, dtype=float)
    noise_psd = np.asarray(one_sided_noise_psd_v2_per_hz, dtype=float)
    if pulse.ndim != 1 or pulse.size < 2 or not 0 <= pulse_main_index < pulse.size:
        raise ValueError("pulse and pulse_main_index are inconsistent")
    if samples_per_ui < 1 or symbol_period_s <= 0.0 or pam4_unit_input_v <= 0.0:
        raise ValueError("sampling and signal-amplitude parameters must be positive")
    if min(channel_pre_ui, channel_post_ui) < 0:
        raise ValueError("channel cursor spans must be nonnegative")
    if ffe_tap_count < 1 or ffe_tap_count % 2 == 0:
        raise ValueError("ffe_tap_count must be a positive odd number")

    cursor_offsets = np.arange(-channel_pre_ui, channel_post_ui + 1, dtype=np.int64)
    half_ffe = ffe_tap_count // 2
    ffe_offsets = np.arange(-half_ffe, half_ffe + 1, dtype=np.int64)
    noise_covariance = _noise_covariance(
        ffe_offsets,
        frequency_hz=frequency_hz,
        one_sided_noise_psd_v2_per_hz=noise_psd,
        symbol_period_s=symbol_period_s,
    )
    if optimize_sampling_phase:
        phase_start = -(samples_per_ui // 2)
        phase_candidates = range(phase_start, phase_start + samples_per_ui)
    else:
        phase_candidates = (0,)

    best_result: DecisionPointResult | None = None
    symbol_variance = 5.0
    for phase_samples in phase_candidates:
        raw_cursor_v = _sample_raw_cursors(
            pulse,
            main_index=pulse_main_index,
            phase_samples=phase_samples,
            samples_per_ui=samples_per_ui,
            offsets_ui=cursor_offsets,
            pam4_unit_input_v=pam4_unit_input_v,
        )
        ffe = _design_mmse_ffe(
            cursor_offsets,
            raw_cursor_v,
            ffe_offsets,
            noise_covariance,
            symbol_variance=symbol_variance,
        )
        equalized_cursor = np.convolve(ffe, raw_cursor_v)
        equalized_offsets = np.arange(
            int(ffe_offsets[0] + cursor_offsets[0]),
            int(ffe_offsets[-1] + cursor_offsets[-1]) + 1,
            dtype=np.int64,
        )
        main_index = int(np.flatnonzero(equalized_offsets == 0)[0])
        if equalized_cursor[main_index] < 0.0:
            ffe = -ffe
            equalized_cursor = -equalized_cursor
        main_cursor = float(equalized_cursor[main_index])
        isi_mask = equalized_offsets != 0
        signal_variance = symbol_variance * main_cursor**2
        isi_variance = symbol_variance * float(np.sum(equalized_cursor[isi_mask] ** 2))
        output_noise_variance = float(ffe @ noise_covariance @ ffe)
        error_variance = isi_variance + output_noise_variance
        if signal_variance == 0.0:
            dp_snr_db = -np.inf
        elif error_variance == 0.0:
            dp_snr_db = np.inf
        else:
            dp_snr_db = 10.0 * np.log10(signal_variance / error_variance)
        pattern_der, pattern_offsets, pattern_taps, remainder_variance, thresholds = (
            _pattern_conditioned_der(
                equalized_offsets,
                equalized_cursor,
                output_noise_variance=output_noise_variance,
                pattern_tap_count=pattern_tap_count,
                symbol_variance=symbol_variance,
            )
        )
        result = DecisionPointResult(
            sampling_phase_samples=int(phase_samples),
            sampling_phase_ui=float(phase_samples / samples_per_ui),
            raw_cursor_offsets_ui=cursor_offsets.copy(),
            raw_cursor_v=raw_cursor_v,
            ffe_offsets_ui=ffe_offsets.copy(),
            ffe_coefficients_per_v=ffe,
            equalized_cursor_offsets_ui=equalized_offsets,
            equalized_cursor=equalized_cursor,
            main_cursor=main_cursor,
            signal_variance=signal_variance,
            residual_isi_variance=isi_variance,
            output_noise_variance=output_noise_variance,
            dp_snr_db=float(dp_snr_db),
            gaussian_der=gaussian_pam4_der(main_cursor, error_variance),
            pattern_conditioned_der=pattern_der,
            pattern_tap_offsets_ui=pattern_offsets,
            pattern_tap_values=pattern_taps,
            gaussian_remainder_variance=remainder_variance,
            slicer_thresholds=thresholds,
        )
        if best_result is None or result.dp_snr_db > best_result.dp_snr_db:
            best_result = result
    if best_result is None:  # pragma: no cover - phase_candidates is always nonempty.
        raise RuntimeError("no sampling phase was evaluated")
    return best_result
