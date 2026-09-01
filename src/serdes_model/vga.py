"""Frequency-flat VGA and peak-target AGC helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .frequency_response import FrequencyResponse


def worst_case_pam_waveform_peak_v(
    symbol_pulse_v_per_unit: ArrayLike,
    *,
    samples_per_ui: int,
    maximum_symbol_magnitude: float,
) -> float:
    """Return the bounded worst-case waveform peak over every sample phase.

    At a fixed sample phase, each symbol can independently select its sign, so
    the worst-case absolute value is the maximum symbol magnitude times the
    sum of the absolute symbol-spaced pulse coefficients at that phase.
    """

    pulse = np.asarray(symbol_pulse_v_per_unit, dtype=float)
    if pulse.ndim != 1 or pulse.size < 1 or not np.all(np.isfinite(pulse)):
        raise ValueError("symbol pulse must be finite, non-empty, and one-dimensional")
    if samples_per_ui < 1:
        raise ValueError("samples_per_ui must be positive")
    if maximum_symbol_magnitude <= 0.0:
        raise ValueError("maximum_symbol_magnitude must be positive")
    phase_bounds_v = [
        maximum_symbol_magnitude * float(np.sum(np.abs(pulse[phase::samples_per_ui])))
        for phase in range(samples_per_ui)
    ]
    return max(phase_bounds_v)


def peak_target_gain_db(
    input_worst_case_peak_v: float,
    *,
    output_half_scale_v: float,
    target_fraction: float,
    minimum_gain_db: float,
    maximum_gain_db: float,
) -> tuple[float, float, bool]:
    """Choose a bounded scalar gain that maps worst-case peak to a target range."""

    if input_worst_case_peak_v <= 0.0 or output_half_scale_v <= 0.0:
        raise ValueError("input peak and output half scale must be positive")
    if not 0.0 < target_fraction <= 1.0:
        raise ValueError("target_fraction must be in (0, 1]")
    if minimum_gain_db > maximum_gain_db:
        raise ValueError("minimum_gain_db must not exceed maximum_gain_db")
    requested_gain_linear = (
        target_fraction * output_half_scale_v / input_worst_case_peak_v
    )
    requested_gain_db = 20.0 * np.log10(requested_gain_linear)
    applied_gain_db = float(
        np.clip(requested_gain_db, minimum_gain_db, maximum_gain_db)
    )
    gain_limited = not np.isclose(applied_gain_db, requested_gain_db)
    return applied_gain_db, float(requested_gain_db), gain_limited


def flat_vga_response(
    frequency_hz: ArrayLike,
    *,
    gain_db: float,
    differential_z0_ohm: float,
) -> FrequencyResponse:
    """Return an ideal broadband differential port-voltage VGA response."""

    frequency_hz = np.asarray(frequency_hz, dtype=float)
    gain_linear = 10.0 ** (gain_db / 20.0)
    return FrequencyResponse(
        frequency_hz=frequency_hz,
        transfer=np.full(frequency_hz.shape, gain_linear, dtype=complex),
        label="Flat VGA Av_port",
        gain_kind="Av_port",
        differential_z0_ohm=differential_z0_ohm,
        metadata={
            "model": "ideal_frequency_flat_vga",
            "gain_db": gain_db,
            "phase_deg": 0.0,
            "group_delay_s": 0.0,
            "voltage_reference": "Vout_diff / Vin_diff at VGA ports",
        },
    )
