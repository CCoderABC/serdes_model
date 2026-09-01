"""Effective uniform-ADC model for sampled SerDes waveforms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class AdcQuantizationResult:
    """Codes, reconstructed levels, and saturation flags for ADC samples."""

    input_v: FloatArray
    codes: IntArray
    reconstructed_v: FloatArray
    clipped_low: BoolArray
    clipped_high: BoolArray

    @property
    def clipped(self) -> BoolArray:
        return self.clipped_low | self.clipped_high

    @property
    def error_from_input_v(self) -> FloatArray:
        """Total conversion error, including any overload error."""

        return self.reconstructed_v - self.input_v


@dataclass(frozen=True)
class EffectiveUniformAdc:
    """Ideal-equivalent mid-rise ADC specified by ENOB and full-scale range.

    ENOB is interpreted as an effective uniform resolution. This is appropriate
    for a first-order model when nominal converter bits, DNL/INL, aperture
    jitter, and the separate ADC noise spectrum are not yet available.
    """

    enob_bits: int
    differential_full_scale_pp_v: float

    def __post_init__(self) -> None:
        if not isinstance(self.enob_bits, (int, np.integer)) or self.enob_bits < 1:
            raise ValueError("enob_bits must be a positive integer")
        if self.differential_full_scale_pp_v <= 0.0:
            raise ValueError("differential_full_scale_pp_v must be positive")

    @property
    def code_count(self) -> int:
        return 1 << int(self.enob_bits)

    @property
    def minimum_input_v(self) -> float:
        return -0.5 * self.differential_full_scale_pp_v

    @property
    def maximum_input_v(self) -> float:
        return 0.5 * self.differential_full_scale_pp_v

    @property
    def effective_lsb_v(self) -> float:
        return self.differential_full_scale_pp_v / self.code_count

    @property
    def quantization_noise_rms_v(self) -> float:
        return self.effective_lsb_v / np.sqrt(12.0)

    @property
    def full_scale_sine_snr_db(self) -> float:
        return 20.0 * np.log10(self.code_count * np.sqrt(3.0 / 2.0))

    def quantize(self, input_v: ArrayLike) -> AdcQuantizationResult:
        """Clip and convert samples to unsigned mid-rise ADC codes."""

        input_v = np.asarray(input_v, dtype=float)
        if not np.all(np.isfinite(input_v)):
            raise ValueError("ADC input contains non-finite values")

        clipped_low = input_v < self.minimum_input_v
        clipped_high = input_v > self.maximum_input_v
        # The upper endpoint belongs to the highest code, not a fictitious code N.
        upper_inside_v = np.nextafter(self.maximum_input_v, self.minimum_input_v)
        bounded_input_v = np.clip(
            input_v, self.minimum_input_v, upper_inside_v
        )
        codes = np.floor(
            (bounded_input_v - self.minimum_input_v) / self.effective_lsb_v
        ).astype(np.int64)
        codes = np.clip(codes, 0, self.code_count - 1)
        reconstructed_v = self.minimum_input_v + (
            codes.astype(float) + 0.5
        ) * self.effective_lsb_v
        return AdcQuantizationResult(
            input_v=input_v,
            codes=codes,
            reconstructed_v=reconstructed_v,
            clipped_low=clipped_low,
            clipped_high=clipped_high,
        )
