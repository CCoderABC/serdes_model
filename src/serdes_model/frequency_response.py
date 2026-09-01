"""Linear differential channel and AFE frequency-response models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class FrequencyResponse:
    """A one-sided complex frequency response with explicit gain semantics."""

    frequency_hz: FloatArray
    transfer: ComplexArray
    label: str
    gain_kind: str
    differential_z0_ohm: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frequency_hz = np.asarray(self.frequency_hz, dtype=float)
        transfer = np.asarray(self.transfer, dtype=complex)
        if frequency_hz.ndim != 1 or transfer.ndim != 1:
            raise ValueError("frequency_hz and transfer must be one-dimensional")
        if frequency_hz.size < 2 or frequency_hz.size != transfer.size:
            raise ValueError("frequency_hz and transfer must have equal length >= 2")
        if not np.all(np.isfinite(frequency_hz)) or not np.all(np.isfinite(transfer)):
            raise ValueError("frequency response contains non-finite values")
        if np.any(frequency_hz < 0.0) or np.any(np.diff(frequency_hz) <= 0.0):
            raise ValueError("frequency_hz must be nonnegative and strictly increasing")
        if self.differential_z0_ohm is not None and self.differential_z0_ohm <= 0.0:
            raise ValueError("differential_z0_ohm must be positive")
        object.__setattr__(self, "frequency_hz", frequency_hz)
        object.__setattr__(self, "transfer", transfer)

    @property
    def magnitude_db(self) -> FloatArray:
        return 20.0 * np.log10(np.maximum(np.abs(self.transfer), np.finfo(float).tiny))

    @property
    def phase_rad(self) -> FloatArray:
        return np.unwrap(np.angle(self.transfer))

    @property
    def group_delay_s(self) -> FloatArray:
        # Group delay is -d(phi)/d(omega), retaining the response's absolute delay.
        return -np.gradient(self.phase_rad, self.frequency_hz) / (2.0 * np.pi)


def uniform_frequency_grid(max_frequency_hz: float, frequency_points: int) -> FloatArray:
    """Return a DC-to-Nyquist uniform grid suitable for a real IFFT."""

    if max_frequency_hz <= 0.0:
        raise ValueError("max_frequency_hz must be positive")
    if frequency_points < 3:
        raise ValueError("frequency_points must be at least 3")
    return np.linspace(0.0, max_frequency_hz, frequency_points, dtype=float)


def analytic_channel_response(
    frequency_hz: ArrayLike,
    *,
    delay_s: float,
    loss_reference_hz: float,
    loss_dc_db: float,
    loss_sqrt_db_at_reference: float,
    loss_linear_db_at_reference: float,
    differential_z0_ohm: float = 100.0,
) -> FrequencyResponse:
    """Construct a smooth lossy matched-channel ``Sdd21`` demonstration model."""

    frequency_hz = np.asarray(frequency_hz, dtype=float)
    if delay_s < 0.0:
        raise ValueError("delay_s must be nonnegative")
    if loss_reference_hz <= 0.0:
        raise ValueError("loss_reference_hz must be positive")
    if min(loss_dc_db, loss_sqrt_db_at_reference, loss_linear_db_at_reference) < 0.0:
        raise ValueError("analytic loss terms must be nonnegative")
    normalized_frequency = frequency_hz / loss_reference_hz
    insertion_loss_db = (
        loss_dc_db
        + loss_sqrt_db_at_reference * np.sqrt(normalized_frequency)
        + loss_linear_db_at_reference * normalized_frequency
    )
    magnitude = 10.0 ** (-insertion_loss_db / 20.0)
    if not np.isclose(frequency_hz[0], 0.0, atol=1e-12) or not np.allclose(
        np.diff(frequency_hz), np.diff(frequency_hz)[0], rtol=1e-10, atol=0.0
    ):
        raise ValueError("analytic channel requires a uniform frequency grid starting at DC")

    # Reconstruct a causal minimum-phase loss response from its sampled magnitude.
    time_samples = 2 * (frequency_hz.size - 1)
    real_cepstrum = np.fft.irfft(np.log(np.maximum(magnitude, np.finfo(float).tiny)), n=time_samples)
    minimum_phase_cepstrum = np.zeros_like(real_cepstrum)
    minimum_phase_cepstrum[0] = real_cepstrum[0]
    minimum_phase_cepstrum[1 : time_samples // 2] = 2.0 * real_cepstrum[1 : time_samples // 2]
    minimum_phase_cepstrum[time_samples // 2] = real_cepstrum[time_samples // 2]
    minimum_phase_loss = np.exp(np.fft.rfft(minimum_phase_cepstrum, n=time_samples))
    propagation_phase = np.exp(-1j * 2.0 * np.pi * frequency_hz * delay_s)
    return FrequencyResponse(
        frequency_hz=frequency_hz,
        transfer=minimum_phase_loss * propagation_phase,
        label="Channel Sdd21",
        gain_kind="Sdd21",
        differential_z0_ohm=differential_z0_ohm,
        metadata={
            "model": "analytic_sqrt_plus_linear_loss",
            "phase_model": "minimum_phase_loss_plus_explicit_delay",
            "delay_s": delay_s,
            "loss_reference_hz": loss_reference_hz,
            "loss_dc_db": loss_dc_db,
            "loss_sqrt_db_at_reference": loss_sqrt_db_at_reference,
            "loss_linear_db_at_reference": loss_linear_db_at_reference,
        },
    )


def analytic_high_loss_channel_response(
    frequency_hz: ArrayLike,
    *,
    delay_s: float,
    reference_frequency_hz: float,
    insertion_loss_db_at_reference: float,
    loss_dc_db: float,
    skin_effect_fraction: float,
    differential_z0_ohm: float = 100.0,
) -> FrequencyResponse:
    """Construct a high-loss channel targeting a specified loss at reference.

    Loss above ``loss_dc_db`` is divided between square-root skin-effect loss
    and linear dielectric loss. The reference is normally the Nyquist
    frequency. Phase uses the same minimum-phase reconstruction as the base
    analytic channel plus an explicit propagation delay.
    """

    if insertion_loss_db_at_reference < loss_dc_db:
        raise ValueError("reference insertion loss must be at least the DC loss")
    if not 0.0 <= skin_effect_fraction <= 1.0:
        raise ValueError("skin_effect_fraction must be between zero and one")
    frequency_dependent_loss_db = insertion_loss_db_at_reference - loss_dc_db
    response = analytic_channel_response(
        frequency_hz,
        delay_s=delay_s,
        loss_reference_hz=reference_frequency_hz,
        loss_dc_db=loss_dc_db,
        loss_sqrt_db_at_reference=skin_effect_fraction * frequency_dependent_loss_db,
        loss_linear_db_at_reference=(1.0 - skin_effect_fraction)
        * frequency_dependent_loss_db,
        differential_z0_ohm=differential_z0_ohm,
    )
    return FrequencyResponse(
        frequency_hz=response.frequency_hz,
        transfer=response.transfer,
        label="High-loss channel Sdd21",
        gain_kind=response.gain_kind,
        differential_z0_ohm=response.differential_z0_ohm,
        metadata={
            **response.metadata,
            "model": "analytic_high_loss_target",
            "reference_frequency_hz": reference_frequency_hz,
            "insertion_loss_db_at_reference": insertion_loss_db_at_reference,
            "skin_effect_fraction": skin_effect_fraction,
        },
    )


def ctle_afe_response(
    frequency_hz: ArrayLike,
    *,
    dc_gain_db: float,
    zero_frequencies_hz: Iterable[float],
    pole_frequencies_hz: Iterable[float],
    differential_input_z0_ohm: float = 100.0,
) -> FrequencyResponse:
    """Construct a differential port-voltage AFE gain from real pole/zero factors."""

    frequency_hz = np.asarray(frequency_hz, dtype=float)
    zeros_hz = np.asarray(tuple(zero_frequencies_hz), dtype=float)
    poles_hz = np.asarray(tuple(pole_frequencies_hz), dtype=float)
    if np.any(zeros_hz <= 0.0) or np.any(poles_hz <= 0.0):
        raise ValueError("all AFE pole and zero frequencies must be positive")
    s = 1j * 2.0 * np.pi * frequency_hz
    transfer = np.full(frequency_hz.shape, 10.0 ** (dc_gain_db / 20.0), dtype=complex)
    for zero_hz in zeros_hz:
        transfer *= 1.0 + s / (2.0 * np.pi * zero_hz)
    for pole_hz in poles_hz:
        transfer /= 1.0 + s / (2.0 * np.pi * pole_hz)
    return FrequencyResponse(
        frequency_hz=frequency_hz,
        transfer=transfer,
        label="AFE Av_port",
        gain_kind="Av_port",
        differential_z0_ohm=differential_input_z0_ohm,
        metadata={
            "model": "real_pole_zero_ctle",
            "dc_gain_db": dc_gain_db,
            "zero_frequencies_hz": zeros_hz.tolist(),
            "pole_frequencies_hz": poles_hz.tolist(),
            "voltage_reference": "Vout_diff / Vin_diff at the AFE port",
        },
    )


def _interpolate_complex_response(
    source_frequency_hz: FloatArray,
    source_transfer: ComplexArray,
    target_frequency_hz: FloatArray,
) -> tuple[ComplexArray, bool]:
    """Interpolate log magnitude and unwrapped phase, with low-frequency extrapolation."""

    tolerance = max(1.0, source_frequency_hz[-1]) * 1e-12
    if target_frequency_hz[-1] > source_frequency_hz[-1] + tolerance:
        raise ValueError("target grid exceeds the measured Touchstone bandwidth")
    log_magnitude = np.log(np.maximum(np.abs(source_transfer), np.finfo(float).tiny))
    phase = np.unwrap(np.angle(source_transfer))
    target_log_magnitude = np.interp(
        target_frequency_hz,
        source_frequency_hz,
        log_magnitude,
        left=log_magnitude[0],
        right=log_magnitude[-1],
    )
    target_phase = np.interp(
        target_frequency_hz,
        source_frequency_hz,
        phase,
        left=phase[0],
        right=phase[-1],
    )
    dc_extrapolated = bool(target_frequency_hz[0] < source_frequency_hz[0])
    if dc_extrapolated:
        fit_points = min(8, source_frequency_hz.size)
        phase_slope, phase_intercept = np.polyfit(
            source_frequency_hz[:fit_points], phase[:fit_points], 1
        )
        low_mask = target_frequency_hz < source_frequency_hz[0]
        target_phase[low_mask] = phase_intercept + phase_slope * target_frequency_hz[low_mask]
    return np.exp(target_log_magnitude + 1j * target_phase), dc_extrapolated


def load_touchstone_sdd21(
    path: str | Path,
    *,
    target_frequency_hz: ArrayLike | None = None,
    port_order: Sequence[int] = (0, 1, 2, 3),
    expected_differential_z0_ohm: float | None = 100.0,
    z0_relative_tolerance: float = 0.02,
) -> FrequencyResponse:
    """Load a four-port channel and extract ``Sdd21``.

    ``port_order`` identifies ``[in+, in-, out+, out-]`` in the source file.
    scikit-rf then produces mixed-mode order ``[d0, d1, c0, c1]``.
    """

    import skrf as rf

    path = Path(path)
    network = rf.Network(str(path))
    if network.nports != 4:
        raise ValueError(f"expected a four-port channel, found {network.nports} ports")
    if sorted(port_order) != [0, 1, 2, 3]:
        raise ValueError("port_order must be a permutation of [0, 1, 2, 3]")

    ordered_s = network.s[:, port_order, :][:, :, port_order]
    ordered_z0 = network.z0[:, port_order]
    ordered_network = rf.Network(
        frequency=network.frequency.copy(),
        s=ordered_s,
        z0=ordered_z0,
        s_def=network.s_def,
        name=network.name,
    )
    ordered_network.se2gmm(p=2)
    differential_z0 = float(np.median(np.real(ordered_network.z0[:, 0])))
    if expected_differential_z0_ohm is not None and not np.isclose(
        differential_z0,
        expected_differential_z0_ohm,
        rtol=z0_relative_tolerance,
        atol=0.0,
    ):
        raise ValueError(
            f"Touchstone differential reference is {differential_z0:.3f} ohm, "
            f"expected {expected_differential_z0_ohm:.3f} ohm"
        )

    source_frequency_hz = np.asarray(ordered_network.f, dtype=float)
    source_sdd21 = np.asarray(ordered_network.s[:, 1, 0], dtype=complex)
    dc_extrapolated = False
    if target_frequency_hz is None:
        frequency_hz = source_frequency_hz
        transfer = source_sdd21
    else:
        frequency_hz = np.asarray(target_frequency_hz, dtype=float)
        transfer, dc_extrapolated = _interpolate_complex_response(
            source_frequency_hz, source_sdd21, frequency_hz
        )

    return FrequencyResponse(
        frequency_hz=frequency_hz,
        transfer=transfer,
        label="Channel Sdd21",
        gain_kind="Sdd21",
        differential_z0_ohm=differential_z0,
        metadata={
            "model": "touchstone_mixed_mode",
            "source_path": str(path.resolve()),
            "source_port_order": list(port_order),
            "mixed_mode_order": ["d0", "d1", "c0", "c1"],
            "dc_extrapolated": dc_extrapolated,
        },
    )


def cascade_responses(
    channel: FrequencyResponse,
    afe: FrequencyResponse,
) -> FrequencyResponse:
    """Multiply matched channel ``Sdd21`` by differential AFE ``Av_port``."""

    if channel.gain_kind != "Sdd21":
        raise ValueError("channel gain_kind must be Sdd21")
    if afe.gain_kind != "Av_port":
        raise ValueError("AFE gain_kind must be Av_port")
    if not np.array_equal(channel.frequency_hz, afe.frequency_hz):
        raise ValueError("channel and AFE must use exactly the same frequency grid")
    if (
        channel.differential_z0_ohm is not None
        and afe.differential_z0_ohm is not None
        and not np.isclose(
            channel.differential_z0_ohm,
            afe.differential_z0_ohm,
            rtol=0.02,
            atol=0.0,
        )
    ):
        raise ValueError("channel and AFE differential impedance references do not match")
    return FrequencyResponse(
        frequency_hz=channel.frequency_hz,
        transfer=channel.transfer * afe.transfer,
        label="Channel Sdd21 x AFE Av_port",
        gain_kind="Sdd21_x_Av_port",
        differential_z0_ohm=channel.differential_z0_ohm,
        metadata={
            "assumption": "matched channel termination; AFE loading included in Av_port",
            "channel": channel.metadata,
            "afe": afe.metadata,
        },
    )


def frequency_to_impulse(response: FrequencyResponse) -> tuple[FloatArray, FloatArray]:
    """Convert a uniform DC-starting one-sided response to a real discrete impulse."""

    frequency_hz = response.frequency_hz
    if not np.isclose(frequency_hz[0], 0.0, atol=1e-12):
        raise ValueError("frequency response must start at DC for real-IFFT conversion")
    spacing_hz = np.diff(frequency_hz)
    if not np.allclose(spacing_hz, spacing_hz[0], rtol=1e-10, atol=0.0):
        raise ValueError("frequency grid must be uniform for real-IFFT conversion")
    time_samples = 2 * (frequency_hz.size - 1)
    sample_period_s = 1.0 / (time_samples * spacing_hz[0])
    impulse = np.fft.irfft(response.transfer, n=time_samples)
    time_s = np.arange(time_samples, dtype=float) * sample_period_s
    return time_s, impulse


def rectangular_pulse_response(
    impulse_time_s: ArrayLike,
    impulse: ArrayLike,
    *,
    symbol_period_s: float,
) -> tuple[FloatArray, FloatArray, int]:
    """Convolve the discrete impulse with one unit-amplitude NRZ symbol."""

    impulse_time_s = np.asarray(impulse_time_s, dtype=float)
    impulse = np.asarray(impulse, dtype=float)
    if impulse_time_s.size != impulse.size or impulse_time_s.size < 2:
        raise ValueError("impulse_time_s and impulse must have equal length >= 2")
    sample_period_s = impulse_time_s[1] - impulse_time_s[0]
    samples_per_ui_float = symbol_period_s / sample_period_s
    samples_per_ui = int(round(samples_per_ui_float))
    if samples_per_ui < 1 or not np.isclose(samples_per_ui_float, samples_per_ui, rtol=1e-9):
        raise ValueError("symbol period must be an integer multiple of the impulse sample period")
    pulse = np.convolve(impulse, np.ones(samples_per_ui, dtype=float), mode="full")
    pulse_time_s = np.arange(pulse.size, dtype=float) * sample_period_s
    return pulse_time_s, pulse, samples_per_ui


def sample_cursor_taps(
    pulse: ArrayLike,
    *,
    samples_per_ui: int,
    precursor_ui: int,
    postcursor_ui: int,
) -> tuple[NDArray[np.int64], FloatArray, int]:
    """Sample symbol-spaced cursor values around the maximum pulse magnitude."""

    pulse = np.asarray(pulse, dtype=float)
    if samples_per_ui < 1 or min(precursor_ui, postcursor_ui) < 0:
        raise ValueError("cursor sampling arguments must be nonnegative")
    main_index = int(np.argmax(np.abs(pulse)))
    offsets_ui = np.arange(-precursor_ui, postcursor_ui + 1, dtype=np.int64)
    indices = main_index + offsets_ui * samples_per_ui
    taps = np.zeros(offsets_ui.shape, dtype=float)
    valid = (indices >= 0) & (indices < pulse.size)
    taps[valid] = pulse[indices[valid]]
    return offsets_ui, taps, main_index
