"""Linear transmitter pulse-shaping models for PAM SerDes studies."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .frequency_response import FrequencyResponse


def tx_pulse_shaping_response(
    frequency_hz: ArrayLike,
    *,
    kind: str,
    rise_time_20_80_s: float | None = None,
) -> FrequencyResponse:
    """Return the TX response applied in addition to the symbol ZOH pulse.

    ``ideal_zoh`` contributes unity here because the rectangular one-UI hold is
    applied later in the pulse-response construction. ``gaussian_rise_time``
    implements the IEEE COM Gaussian source-rise-time response.
    """

    frequency_hz = np.asarray(frequency_hz, dtype=float)
    if frequency_hz.ndim != 1 or frequency_hz.size < 2:
        raise ValueError("frequency_hz must be a one-dimensional array of length >= 2")
    normalized_kind = kind.lower()
    if normalized_kind == "ideal_zoh":
        transfer = np.ones(frequency_hz.shape, dtype=complex)
        metadata = {
            "model": "ideal_zoh",
            "zoh_application": "one-UI rectangular pulse applied in time domain",
            "rise_time_20_80_s": 0.0,
        }
    elif normalized_kind == "gaussian_rise_time":
        if rise_time_20_80_s is None or rise_time_20_80_s <= 0.0:
            raise ValueError("gaussian_rise_time requires positive rise_time_20_80_s")
        transfer = np.exp(
            -(np.pi * frequency_hz * rise_time_20_80_s / 1.6832) ** 2
        ).astype(complex)
        metadata = {
            "model": "gaussian_rise_time",
            "zoh_application": "one-UI rectangular pulse applied in time domain",
            "rise_time_definition": "20_to_80_percent",
            "rise_time_20_80_s": rise_time_20_80_s,
            "dc_gain": 1.0,
        }
    else:
        raise ValueError(
            "unsupported TX pulse shaping kind; use ideal_zoh or gaussian_rise_time"
        )
    return FrequencyResponse(
        frequency_hz=frequency_hz,
        transfer=transfer,
        label="TX pulse shaping",
        gain_kind="Htx_pulse",
        metadata=metadata,
    )


def apply_tx_pulse_shaping(
    tx_response: FrequencyResponse,
    channel_afe_response: FrequencyResponse,
) -> FrequencyResponse:
    """Apply TX shaping to the channel-plus-AFE signal path only."""

    if tx_response.gain_kind != "Htx_pulse":
        raise ValueError("tx_response gain_kind must be Htx_pulse")
    if channel_afe_response.gain_kind != "Sdd21_x_Av_port":
        raise ValueError("channel_afe_response gain_kind must be Sdd21_x_Av_port")
    if not np.array_equal(tx_response.frequency_hz, channel_afe_response.frequency_hz):
        raise ValueError("TX and channel-plus-AFE responses must share a frequency grid")
    return FrequencyResponse(
        frequency_hz=channel_afe_response.frequency_hz,
        transfer=tx_response.transfer * channel_afe_response.transfer,
        label="TX pulse x Channel Sdd21 x AFE Av_port",
        gain_kind="Htx_pulse_x_Sdd21_x_Av_port",
        differential_z0_ohm=channel_afe_response.differential_z0_ohm,
        metadata={
            "signal_only_filter": True,
            "tx": tx_response.metadata,
            "channel_afe": channel_afe_response.metadata,
        },
    )


def apply_tx_pulse_shaping_to_channel(
    tx_response: FrequencyResponse,
    channel_response: FrequencyResponse,
) -> FrequencyResponse:
    """Return the signal transfer at the channel output, before the AFE."""

    if tx_response.gain_kind != "Htx_pulse":
        raise ValueError("tx_response gain_kind must be Htx_pulse")
    if channel_response.gain_kind != "Sdd21":
        raise ValueError("channel_response gain_kind must be Sdd21")
    if not np.array_equal(tx_response.frequency_hz, channel_response.frequency_hz):
        raise ValueError("TX and channel responses must share a frequency grid")
    return FrequencyResponse(
        frequency_hz=channel_response.frequency_hz,
        transfer=tx_response.transfer * channel_response.transfer,
        label="TX pulse x Channel Sdd21",
        gain_kind="Htx_pulse_x_Sdd21",
        differential_z0_ohm=channel_response.differential_z0_ohm,
        metadata={
            "reference_plane": "matched differential channel output before AFE",
            "tx": tx_response.metadata,
            "channel": channel_response.metadata,
        },
    )


def zoh_frequency_response(
    frequency_hz: ArrayLike,
    *,
    symbol_period_s: float,
) -> np.ndarray:
    """Return the continuous one-UI ZOH spectrum normalized to unity at DC."""

    frequency_hz = np.asarray(frequency_hz, dtype=float)
    if frequency_hz.ndim != 1 or symbol_period_s <= 0.0:
        raise ValueError("frequency_hz must be one-dimensional and symbol period positive")
    return np.sinc(frequency_hz * symbol_period_s) * np.exp(
        -1j * np.pi * frequency_hz * symbol_period_s
    )


def centered_symbol_pulse_response(
    response: FrequencyResponse,
    *,
    symbol_period_s: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return a centered one-symbol ZOH pulse after a complex response."""

    frequency_hz = response.frequency_hz
    if not np.isclose(frequency_hz[0], 0.0, atol=1e-12):
        raise ValueError("frequency response must start at DC")
    spacing_hz = np.diff(frequency_hz)
    if not np.allclose(spacing_hz, spacing_hz[0], rtol=1e-10, atol=0.0):
        raise ValueError("frequency grid must be uniform")
    time_samples = 2 * (frequency_hz.size - 1)
    sample_period_s = 1.0 / (time_samples * spacing_hz[0])
    samples_per_ui_float = symbol_period_s / sample_period_s
    samples_per_ui = int(round(samples_per_ui_float))
    if samples_per_ui < 1 or not np.isclose(
        samples_per_ui_float, samples_per_ui, rtol=1e-9
    ):
        raise ValueError("symbol period must be an integer number of samples")
    impulse = np.fft.fftshift(np.fft.irfft(response.transfer, n=time_samples))
    impulse_time_s = (
        np.arange(time_samples, dtype=float) - time_samples // 2
    ) * sample_period_s
    pulse = np.convolve(impulse, np.ones(samples_per_ui, dtype=float), mode="full")
    pulse_time_s = impulse_time_s[0] + np.arange(pulse.size, dtype=float) * sample_period_s
    return pulse_time_s, pulse, samples_per_ui
