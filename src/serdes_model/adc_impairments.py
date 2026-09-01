"""Stochastic analog-noise and aperture-jitter models at the ADC input."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SampledAdcImpairmentResult:
    """ADC decision samples and their separately observable impairments."""

    noiseless_samples_v: FloatArray
    afe_noise_samples_v: FloatArray
    additional_adc_noise_samples_v: FloatArray
    aperture_jitter_s: FloatArray
    signal_slope_v_per_s: FloatArray
    jitter_error_v: FloatArray
    impaired_samples_v: FloatArray
    expected_afe_noise_rms_v: float
    observed_afe_noise_rms_v: float
    observed_additional_adc_noise_rms_v: float
    observed_jitter_error_rms_v: float
    observed_total_impairment_rms_v: float


def synthesize_real_noise_from_one_sided_psd(
    frequency_hz: ArrayLike,
    one_sided_psd_v2_per_hz: ArrayLike,
    *,
    sample_rate_hz: float,
    sample_count: int,
    seed: int,
) -> FloatArray:
    """Synthesize zero-mean real Gaussian noise from a one-sided voltage PSD."""

    source_frequency_hz = np.asarray(frequency_hz, dtype=float)
    source_psd = np.asarray(one_sided_psd_v2_per_hz, dtype=float)
    if source_frequency_hz.ndim != 1 or source_psd.shape != source_frequency_hz.shape:
        raise ValueError("frequency_hz and PSD must be equal one-dimensional arrays")
    if source_frequency_hz.size < 2 or np.any(np.diff(source_frequency_hz) <= 0.0):
        raise ValueError("frequency_hz must be strictly increasing")
    if source_frequency_hz[0] != 0.0 or np.any(source_psd < 0.0):
        raise ValueError("PSD grid must start at DC and be nonnegative")
    if sample_rate_hz <= 0.0 or sample_count < 2:
        raise ValueError("sample rate and sample count must be positive")
    nyquist_hz = 0.5 * sample_rate_hz
    if source_frequency_hz[-1] < nyquist_hz * (1.0 - 1e-12):
        raise ValueError("PSD grid does not cover the requested Nyquist frequency")

    target_frequency_hz = np.fft.rfftfreq(sample_count, d=1.0 / sample_rate_hz)
    target_psd = np.interp(
        target_frequency_hz, source_frequency_hz, source_psd
    )
    generator = np.random.default_rng(seed)
    spectrum = np.zeros(target_frequency_hz.size, dtype=complex)

    # For interior positive-frequency bins, E[2|X[k]|^2/(Fs*N)] = S1(f).
    interior_stop = -1 if sample_count % 2 == 0 else None
    interior_psd = target_psd[1:interior_stop]
    scale = np.sqrt(interior_psd * sample_rate_hz * sample_count / 4.0)
    spectrum[1:interior_stop] = scale * (
        generator.standard_normal(interior_psd.size)
        + 1j * generator.standard_normal(interior_psd.size)
    )
    spectrum[0] = 0.0  # Enforce zero sample mean rather than random DC offset.
    if sample_count % 2 == 0:
        spectrum[-1] = np.sqrt(
            target_psd[-1] * sample_rate_hz * sample_count
        ) * generator.standard_normal()
    return np.asarray(np.fft.irfft(spectrum, n=sample_count), dtype=float)


def apply_sampled_adc_impairments(
    noiseless_waveform_v: ArrayLike,
    *,
    samples_per_ui: int,
    sample_rate_hz: float,
    afe_noise_frequency_hz: ArrayLike,
    afe_noise_psd_v2_per_hz: ArrayLike,
    additional_adc_noise_rms_v: float,
    aperture_jitter_rms_s: float,
    random_seed: int,
) -> SampledAdcImpairmentResult:
    """Add AFE noise, ADC noise, and first-order aperture-jitter error.

    Aperture jitter is independent Gaussian random jitter. Its voltage error is
    the local noiseless-signal slope multiplied by the timing displacement.
    """

    waveform = np.asarray(noiseless_waveform_v, dtype=float)
    frequency_hz = np.asarray(afe_noise_frequency_hz, dtype=float)
    psd = np.asarray(afe_noise_psd_v2_per_hz, dtype=float)
    if waveform.ndim != 1 or waveform.size < 2 or not np.all(np.isfinite(waveform)):
        raise ValueError("noiseless waveform must be finite and one-dimensional")
    if samples_per_ui < 1 or sample_rate_hz <= 0.0:
        raise ValueError("sampling parameters must be positive")
    if min(additional_adc_noise_rms_v, aperture_jitter_rms_s) < 0.0:
        raise ValueError("noise RMS and jitter RMS must be nonnegative")
    sample_indices = np.arange(0, waveform.size, samples_per_ui, dtype=np.int64)
    noiseless_samples = waveform[sample_indices]

    afe_noise_waveform = synthesize_real_noise_from_one_sided_psd(
        frequency_hz,
        psd,
        sample_rate_hz=sample_rate_hz,
        sample_count=waveform.size,
        seed=random_seed,
    )
    afe_noise_samples = afe_noise_waveform[sample_indices]
    generator = np.random.default_rng(random_seed + 1)
    additional_adc_noise = generator.normal(
        0.0, additional_adc_noise_rms_v, noiseless_samples.size
    )
    aperture_jitter = generator.normal(
        0.0, aperture_jitter_rms_s, noiseless_samples.size
    )
    sample_period_s = 1.0 / sample_rate_hz
    signal_slope = np.gradient(waveform, sample_period_s)[sample_indices]
    jitter_error = signal_slope * aperture_jitter
    total_impairment = afe_noise_samples + additional_adc_noise + jitter_error
    expected_afe_noise_variance = float(
        np.sum(0.5 * (psd[1:] + psd[:-1]) * np.diff(frequency_hz))
    )

    return SampledAdcImpairmentResult(
        noiseless_samples_v=noiseless_samples,
        afe_noise_samples_v=afe_noise_samples,
        additional_adc_noise_samples_v=additional_adc_noise,
        aperture_jitter_s=aperture_jitter,
        signal_slope_v_per_s=signal_slope,
        jitter_error_v=jitter_error,
        impaired_samples_v=noiseless_samples + total_impairment,
        expected_afe_noise_rms_v=float(np.sqrt(expected_afe_noise_variance)),
        observed_afe_noise_rms_v=float(np.sqrt(np.mean(afe_noise_samples**2))),
        observed_additional_adc_noise_rms_v=float(
            np.sqrt(np.mean(additional_adc_noise**2))
        ),
        observed_jitter_error_rms_v=float(np.sqrt(np.mean(jitter_error**2))),
        observed_total_impairment_rms_v=float(
            np.sqrt(np.mean(total_impairment**2))
        ),
    )
