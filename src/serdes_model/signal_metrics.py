"""Signal statistics for SerDes waveform reference planes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class PaprResult:
    """Voltage-domain statistics and peak-to-average power ratio."""

    sample_count: int
    mean_v: float
    rms_v: float
    peak_abs_v: float
    crest_factor: float
    papr_linear: float
    papr_db: float


def voltage_papr(samples_v: ArrayLike) -> PaprResult:
    """Calculate PAPR from voltage samples without removing their mean.

    For a fixed resistance, instantaneous and average powers are proportional
    to voltage squared, so the resistance cancels from the PAPR ratio.
    """

    samples_v = np.asarray(samples_v, dtype=float)
    if samples_v.ndim != 1 or samples_v.size < 1:
        raise ValueError("samples_v must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(samples_v)):
        raise ValueError("samples_v contains non-finite values")
    mean_square_v2 = float(np.mean(np.square(samples_v)))
    if mean_square_v2 <= 0.0:
        raise ValueError("PAPR is undefined for an all-zero waveform")
    peak_abs_v = float(np.max(np.abs(samples_v)))
    rms_v = float(np.sqrt(mean_square_v2))
    crest_factor = peak_abs_v / rms_v
    papr_linear = crest_factor**2
    return PaprResult(
        sample_count=int(samples_v.size),
        mean_v=float(np.mean(samples_v)),
        rms_v=rms_v,
        peak_abs_v=peak_abs_v,
        crest_factor=crest_factor,
        papr_linear=papr_linear,
        papr_db=float(10.0 * np.log10(papr_linear)),
    )
