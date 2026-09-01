#!/usr/bin/env python3
"""Run the channel plus AFE linear frequency-response demonstration."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "runs" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / "runs" / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from serdes_model import (
    EffectiveUniformAdc,
    RS_CORRECTABLE_SYMBOL_ERRORS,
    RS_GENERATOR_COEFFICIENTS,
    RS_K_SYMBOLS,
    RS_N_SYMBOLS,
    RS_PARITY_SYMBOLS,
    RS_PRIMITIVE_POLYNOMIAL,
    RS_SYMBOL_BITS,
    apply_sampled_adc_impairments,
    analytic_channel_response,
    analytic_high_loss_channel_response,
    apply_tx_pulse_shaping,
    apply_tx_pulse_shaping_to_channel,
    cascade_responses,
    centered_symbol_pulse_response,
    ctle_afe_response,
    eye_diagram_from_symbol_pulse,
    evaluate_decision_point,
    evaluate_rs544_fec,
    flat_vga_response,
    frequency_to_impulse,
    generate_rs544_pam4_stream,
    load_touchstone_sdd21,
    main_cursor_index,
    output_noise_psd_one_sided,
    pam4_unit_voltage_from_delivered_power,
    pam4_unit_voltage_from_outer_pp,
    pam4_symbol_sequence,
    peak_target_gain_db,
    rectangular_pulse_response,
    required_pam4_dp_snr_db,
    required_pre_fec_ber_for_iid_post_ber,
    sample_cursor_taps,
    synthesize_symbol_waveform,
    tx_pulse_shaping_response,
    train_code_domain_dfe,
    train_code_domain_ffe,
    uniform_frequency_grid,
    voltage_papr,
    worst_case_pam_waveform_peak_v,
    zoh_frequency_response,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "channel_afe_demo.yaml",
        help="YAML model configuration",
    )
    parser.add_argument(
        "--tx-pulse-shaping",
        choices=("ideal_zoh", "gaussian_rise_time"),
        help="override transmitter.pulse_shaping.kind",
    )
    parser.add_argument(
        "--tx-rise-time-ps",
        type=float,
        help="override the Gaussian TX 20%%-80%% rise time",
    )
    parser.add_argument(
        "--channel-loss-db-at-nyquist",
        type=float,
        help="override analytic_high_loss insertion loss at Nyquist",
    )
    return parser.parse_args()


def _load_channel(config: dict, frequency_hz: np.ndarray, nyquist_hz: float):
    kind = config["kind"].lower()
    if kind == "analytic":
        return analytic_channel_response(
            frequency_hz,
            delay_s=float(config["delay_ps"]) * 1e-12,
            loss_reference_hz=float(config["loss_reference_ghz"]) * 1e9,
            loss_dc_db=float(config["loss_dc_db"]),
            loss_sqrt_db_at_reference=float(config["loss_sqrt_db_at_reference"]),
            loss_linear_db_at_reference=float(config["loss_linear_db_at_reference"]),
            differential_z0_ohm=float(config["differential_z0_ohm"]),
        )
    if kind == "analytic_high_loss":
        return analytic_high_loss_channel_response(
            frequency_hz,
            delay_s=float(config["delay_ps"]) * 1e-12,
            reference_frequency_hz=nyquist_hz,
            insertion_loss_db_at_reference=float(
                config["insertion_loss_db_at_nyquist"]
            ),
            loss_dc_db=float(config["loss_dc_db"]),
            skin_effect_fraction=float(config["skin_effect_fraction"]),
            differential_z0_ohm=float(config["differential_z0_ohm"]),
        )
    if kind == "touchstone":
        source_path = Path(config["path"])
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
        return load_touchstone_sdd21(
            source_path,
            target_frequency_hz=frequency_hz,
            port_order=tuple(config.get("port_order", [0, 1, 2, 3])),
            expected_differential_z0_ohm=float(
                config.get("expected_differential_z0_ohm", 100.0)
            ),
        )
    raise ValueError(f"unsupported channel kind: {kind}")


def _write_frequency_csv(path: Path, channel, afe, channel_afe, tx, signal_path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frequency_hz",
                "channel_sdd21_real",
                "channel_sdd21_imag",
                "channel_sdd21_db",
                "afe_av_port_real",
                "afe_av_port_imag",
                "afe_av_port_db",
                "channel_afe_real",
                "channel_afe_imag",
                "channel_afe_db",
                "tx_pulse_real",
                "tx_pulse_imag",
                "tx_pulse_db",
                "signal_path_real",
                "signal_path_imag",
                "signal_path_db",
                "signal_path_group_delay_s",
            ]
        )
        for index, frequency in enumerate(channel.frequency_hz):
            writer.writerow(
                [
                    frequency,
                    channel.transfer[index].real,
                    channel.transfer[index].imag,
                    channel.magnitude_db[index],
                    afe.transfer[index].real,
                    afe.transfer[index].imag,
                    afe.magnitude_db[index],
                    channel_afe.transfer[index].real,
                    channel_afe.transfer[index].imag,
                    channel_afe.magnitude_db[index],
                    tx.transfer[index].real,
                    tx.transfer[index].imag,
                    tx.magnitude_db[index],
                    signal_path.transfer[index].real,
                    signal_path.transfer[index].imag,
                    signal_path.magnitude_db[index],
                    signal_path.group_delay_s[index],
                ]
            )


def _write_stage_frequency_csv(
    path: Path,
    *,
    frequency_hz: np.ndarray,
    tx_symbol_spectrum: np.ndarray,
    channel_symbol_spectrum: np.ndarray,
    channel,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frequency_hz",
                "tx_symbol_spectrum_real",
                "tx_symbol_spectrum_imag",
                "tx_symbol_spectrum_db_relative_to_dc",
                "channel_sdd21_db",
                "channel_output_symbol_spectrum_real",
                "channel_output_symbol_spectrum_imag",
                "channel_output_symbol_spectrum_db_relative_to_tx_dc",
            ]
        )
        for index, frequency in enumerate(frequency_hz):
            writer.writerow(
                [
                    frequency,
                    tx_symbol_spectrum[index].real,
                    tx_symbol_spectrum[index].imag,
                    20.0
                    * np.log10(
                        max(abs(tx_symbol_spectrum[index]), np.finfo(float).tiny)
                    ),
                    channel.magnitude_db[index],
                    channel_symbol_spectrum[index].real,
                    channel_symbol_spectrum[index].imag,
                    20.0
                    * np.log10(
                        max(
                            abs(channel_symbol_spectrum[index]),
                            np.finfo(float).tiny,
                        )
                    ),
                ]
            )


def _write_stage_pulse_csv(
    path: Path,
    *,
    tx_time_ui: np.ndarray,
    tx_pulse_v: np.ndarray,
    channel_time_ui: np.ndarray,
    channel_pulse_v: np.ndarray,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "tx_relative_time_ui",
                "tx_pulse_v_per_pam_unit",
                "channel_relative_time_ui",
                "channel_pulse_v_per_pam_unit",
            ]
        )
        for values in zip(tx_time_ui, tx_pulse_v, channel_time_ui, channel_pulse_v):
            writer.writerow(values)


def _write_ctle_afe_frequency_csv(
    path: Path,
    *,
    frequency_hz: np.ndarray,
    afe,
    channel_symbol_spectrum: np.ndarray,
    afe_output_symbol_spectrum: np.ndarray,
) -> None:
    """Export the complex CTLE response and its effect on the signal spectrum."""

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frequency_hz",
                "afe_av_port_real",
                "afe_av_port_imag",
                "afe_av_port_magnitude_db",
                "afe_av_port_phase_deg_unwrapped",
                "afe_av_port_group_delay_s",
                "channel_output_symbol_spectrum_real",
                "channel_output_symbol_spectrum_imag",
                "channel_output_symbol_spectrum_db_relative_to_tx_dc",
                "afe_output_symbol_spectrum_real",
                "afe_output_symbol_spectrum_imag",
                "afe_output_symbol_spectrum_db_relative_to_tx_dc",
            ]
        )
        for index, frequency in enumerate(frequency_hz):
            writer.writerow(
                [
                    frequency,
                    afe.transfer[index].real,
                    afe.transfer[index].imag,
                    afe.magnitude_db[index],
                    np.rad2deg(afe.phase_rad[index]),
                    afe.group_delay_s[index],
                    channel_symbol_spectrum[index].real,
                    channel_symbol_spectrum[index].imag,
                    20.0
                    * np.log10(
                        max(
                            abs(channel_symbol_spectrum[index]),
                            np.finfo(float).tiny,
                        )
                    ),
                    afe_output_symbol_spectrum[index].real,
                    afe_output_symbol_spectrum[index].imag,
                    20.0
                    * np.log10(
                        max(
                            abs(afe_output_symbol_spectrum[index]),
                            np.finfo(float).tiny,
                        )
                    ),
                ]
            )


def _write_ctle_afe_pulse_csv(
    path: Path,
    *,
    time_ui: np.ndarray,
    pulse_v: np.ndarray,
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "afe_output_relative_time_ui",
                "afe_output_pulse_v_per_pam_unit",
            ]
        )
        writer.writerows(zip(time_ui, pulse_v))


def _plot_tx_channel_stages(
    path: Path,
    *,
    frequency_hz: np.ndarray,
    tx_edge_response: np.ndarray,
    tx_symbol_spectrum: np.ndarray,
    channel_symbol_spectrum: np.ndarray,
    channel,
    tx_time_ui: np.ndarray,
    tx_pulse_v: np.ndarray,
    channel_time_ui: np.ndarray,
    channel_pulse_v: np.ndarray,
    differential_outer_pp_v: float,
    nyquist_hz: float,
    plot_max_hz: float,
    pulse_pre_ui: float,
    pulse_post_ui: float,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    frequency_mask = frequency_hz <= plot_max_hz
    frequency_ghz = frequency_hz / 1e9
    tx_spectrum_db = 20.0 * np.log10(
        np.maximum(np.abs(tx_symbol_spectrum), 1e-7)
    )
    channel_spectrum_db = 20.0 * np.log10(
        np.maximum(np.abs(channel_symbol_spectrum), 1e-7)
    )

    tx_frequency_axis = axes[0, 0]
    tx_edge_db = 20.0 * np.log10(np.maximum(np.abs(tx_edge_response), 1e-7))
    tx_frequency_axis.plot(
        frequency_ghz[frequency_mask],
        tx_edge_db[frequency_mask],
        linestyle="--",
        label="TX edge filter",
    )
    tx_frequency_axis.plot(
        frequency_ghz[frequency_mask],
        tx_spectrum_db[frequency_mask],
        label="TX edge × ZOH",
    )
    tx_frequency_axis.axvline(
        nyquist_hz / 1e9, color="black", linewidth=0.8, linestyle="--", label="Nyquist"
    )
    tx_frequency_axis.set_title("Fig. 3a — TX-output frequency response")
    tx_frequency_axis.set_xlabel("Frequency (GHz)")
    tx_frequency_axis.set_ylabel("Magnitude relative to TX DC (dB)")
    tx_frequency_axis.set_ylim(-100, 5)
    tx_frequency_axis.grid(True, alpha=0.3)
    tx_frequency_axis.legend()

    tx_time_axis = axes[0, 1]
    tx_time_mask = (tx_time_ui >= -1.5) & (tx_time_ui <= 2.0)
    tx_time_axis.plot(tx_time_ui[tx_time_mask], tx_pulse_v[tx_time_mask] * 1e3)
    tx_time_axis.set_title("Fig. 3b — TX-output one-symbol pulse")
    tx_time_axis.set_xlabel("Relative time (UI)")
    tx_time_axis.set_ylabel("Differential voltage (mV/PAM unit)")
    tx_time_axis.grid(True, alpha=0.3)

    channel_frequency_axis = axes[1, 0]
    channel_frequency_axis.plot(
        frequency_ghz[frequency_mask],
        channel.magnitude_db[frequency_mask],
        linestyle="--",
        label="Channel Sdd21",
    )
    channel_frequency_axis.plot(
        frequency_ghz[frequency_mask],
        channel_spectrum_db[frequency_mask],
        label="TX pulse × channel",
    )
    channel_frequency_axis.axvline(
        nyquist_hz / 1e9, color="black", linewidth=0.8, linestyle=":"
    )
    channel_frequency_axis.set_title("Fig. 3c — Channel-output symbol-pulse spectrum")
    channel_frequency_axis.set_xlabel("Frequency (GHz)")
    channel_frequency_axis.set_ylabel("Magnitude relative to TX DC (dB)")
    channel_frequency_axis.set_ylim(-120, 5)
    channel_frequency_axis.grid(True, alpha=0.3)
    channel_frequency_axis.legend()

    channel_time_axis = axes[1, 1]
    channel_time_mask = (channel_time_ui >= -pulse_pre_ui) & (
        channel_time_ui <= pulse_post_ui
    )
    channel_time_axis.plot(
        channel_time_ui[channel_time_mask], channel_pulse_v[channel_time_mask] * 1e3
    )
    channel_time_axis.set_title("Fig. 3d — Channel-output one-symbol pulse")
    channel_time_axis.set_xlabel("Relative time (UI)")
    channel_time_axis.set_ylabel("Differential voltage (mV/PAM unit)")
    channel_time_axis.grid(True, alpha=0.3)

    figure.suptitle(
        f"Fig. 3 — Loaded differential TX ({differential_outer_pp_v * 1e3:.0f} mVpp) "
        "and high-loss channel output; AFE/RX EQ excluded"
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_ctle_afe_stage(
    path: Path,
    *,
    frequency_hz: np.ndarray,
    afe,
    channel_symbol_spectrum: np.ndarray,
    afe_output_symbol_spectrum: np.ndarray,
    channel_time_ui: np.ndarray,
    channel_pulse_v: np.ndarray,
    afe_time_ui: np.ndarray,
    afe_pulse_v: np.ndarray,
    nyquist_hz: float,
    plot_max_hz: float,
    pulse_pre_ui: float,
    pulse_post_ui: float,
) -> None:
    """Plot complex CTLE behavior and the resulting AFE-output pulse."""

    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    frequency_mask = frequency_hz <= plot_max_hz
    frequency_ghz = frequency_hz / 1e9

    magnitude_axis = axes[0, 0]
    magnitude_axis.plot(
        frequency_ghz[frequency_mask], afe.magnitude_db[frequency_mask]
    )
    magnitude_axis.axvline(
        nyquist_hz / 1e9,
        color="black",
        linewidth=0.8,
        linestyle="--",
        label="Nyquist",
    )
    magnitude_axis.set_title("Fig. 5a — CTLE/AFE magnitude, Av_port")
    magnitude_axis.set_xlabel("Frequency (GHz)")
    magnitude_axis.set_ylabel("Differential voltage gain (dB)")
    magnitude_axis.grid(True, alpha=0.3)
    magnitude_axis.legend()

    delay_axis = axes[0, 1]
    delay_axis.plot(
        frequency_ghz[frequency_mask],
        afe.group_delay_s[frequency_mask] * 1e12,
        color="tab:purple",
    )
    delay_axis.axhline(0.0, color="0.45", linewidth=0.7, linestyle=":")
    delay_axis.axvline(
        nyquist_hz / 1e9, color="black", linewidth=0.8, linestyle="--"
    )
    delay_axis.set_title("Fig. 5b — CTLE/AFE group delay")
    delay_axis.set_xlabel("Frequency (GHz)")
    delay_axis.set_ylabel("Group delay of Av_port (ps)")
    delay_axis.grid(True, alpha=0.3)

    spectrum_axis = axes[1, 0]
    channel_spectrum_db = 20.0 * np.log10(
        np.maximum(np.abs(channel_symbol_spectrum), 1e-7)
    )
    afe_spectrum_db = 20.0 * np.log10(
        np.maximum(np.abs(afe_output_symbol_spectrum), 1e-7)
    )
    spectrum_axis.plot(
        frequency_ghz[frequency_mask],
        channel_spectrum_db[frequency_mask],
        label="Channel output",
    )
    spectrum_axis.plot(
        frequency_ghz[frequency_mask],
        afe_spectrum_db[frequency_mask],
        label="CTLE/AFE output",
    )
    spectrum_axis.axvline(
        nyquist_hz / 1e9, color="black", linewidth=0.8, linestyle="--"
    )
    spectrum_axis.set_title("Fig. 5c — Symbol-pulse spectra before/after CTLE")
    spectrum_axis.set_xlabel("Frequency (GHz)")
    spectrum_axis.set_ylabel("Magnitude relative to TX DC (dB)")
    spectrum_axis.set_ylim(-120, 35)
    spectrum_axis.grid(True, alpha=0.3)
    spectrum_axis.legend()

    pulse_axis = axes[1, 1]
    channel_mask = (channel_time_ui >= -pulse_pre_ui) & (
        channel_time_ui <= pulse_post_ui
    )
    afe_mask = (afe_time_ui >= -pulse_pre_ui) & (afe_time_ui <= pulse_post_ui)
    pulse_axis.plot(
        channel_time_ui[channel_mask],
        channel_pulse_v[channel_mask] * 1e3,
        linestyle="--",
        label="Channel output",
    )
    pulse_axis.plot(
        afe_time_ui[afe_mask],
        afe_pulse_v[afe_mask] * 1e3,
        label="CTLE/AFE output",
    )
    pulse_axis.axvline(0.0, color="black", linewidth=0.8, linestyle=":")
    pulse_axis.set_title("Fig. 5d — One-symbol pulse before/after CTLE")
    pulse_axis.set_xlabel("Time relative to each main cursor (UI)")
    pulse_axis.set_ylabel("Differential voltage (mV/PAM unit)")
    pulse_axis.grid(True, alpha=0.3)
    pulse_axis.legend()

    figure.suptitle(
        "Fig. 5 — Complex CTLE/AFE response; Av_port = Vout,diff / Vin,diff\n"
        "Matched 100-ohm differential input; noise, ADC, and digital RX EQ excluded"
    )
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _write_eye_center_samples_csv(
    path: Path, tx_eye, channel_eye, afe_eye=None
) -> None:
    """Write the actual (non-interpolated) center samples used in the eye plot."""

    if not np.array_equal(tx_eye.symbol_indices, channel_eye.symbol_indices):
        raise ValueError("TX and channel eyes must use the same symbol indices")
    if afe_eye is not None and not np.array_equal(
        tx_eye.symbol_indices, afe_eye.symbol_indices
    ):
        raise ValueError("TX and AFE eyes must use the same symbol indices")
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = [
            "symbol_index",
            "normalized_pam4_symbol",
            "tx_center_sample_v",
            "channel_center_sample_v",
        ]
        if afe_eye is not None:
            header.append("afe_center_sample_v")
        writer.writerow(header)
        rows = zip(
            tx_eye.symbol_indices,
            tx_eye.transmitted_symbols,
            tx_eye.center_samples_v,
            channel_eye.center_samples_v,
        )
        for row_index, values in enumerate(rows):
            row = list(values)
            if afe_eye is not None:
                row.append(afe_eye.center_samples_v[row_index])
            writer.writerow(row)


def _plot_tx_channel_eyes(
    path: Path,
    *,
    tx_eye,
    channel_eye,
    pam4_unit_voltage_v: float,
    channel_dc_gain: float,
    differential_outer_pp_v: float,
    symbol_rate_hz: float,
    channel_loss_db_at_nyquist: float,
) -> None:
    """Plot noiseless two-UI PAM4 eyes at the TX and channel output planes."""

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    stages = (
        (
            axes[0],
            tx_eye,
            "tab:blue",
            "Fig. 4a — Loaded differential TX-output eye",
            pam4_unit_voltage_v,
        ),
        (
            axes[1],
            channel_eye,
            "tab:red",
            "Fig. 4b — Matched channel-output eye",
            pam4_unit_voltage_v * channel_dc_gain,
        ),
    )
    for axis, eye, color, title, settled_unit_v in stages:
        for trace in eye.traces_v:
            axis.plot(
                eye.time_ui,
                trace * 1e3,
                color=color,
                alpha=0.035,
                linewidth=0.55,
            )
        axis.scatter(
            np.zeros(eye.center_samples_v.size),
            eye.center_samples_v * 1e3,
            s=4,
            color="black",
            alpha=0.12,
            linewidths=0,
            zorder=3,
        )
        for level in (-3.0, -1.0, 1.0, 3.0):
            axis.axhline(
                level * settled_unit_v * 1e3,
                color="0.45",
                linestyle=":",
                linewidth=0.65,
                alpha=0.45,
            )
        axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_xlim(-1.0, 1.0)
        axis.set_xlabel("Time relative to main-cursor sample (UI)")
        axis.set_ylabel("Loaded differential voltage (mV)")
        axis.set_title(title)
        axis.grid(True, alpha=0.22)

        voltage_limit_mv = max(
            1.1 * float(np.max(np.abs(eye.traces_v))) * 1e3,
            3.4 * settled_unit_v * 1e3,
        )
        axis.set_ylim(-voltage_limit_mv, voltage_limit_mv)

    axes[0].text(
        0.02,
        0.03,
        "Dotted: nominal PAM4 levels",
        transform=axes[0].transAxes,
        fontsize=8,
        color="0.35",
    )
    axes[1].text(
        0.02,
        0.03,
        "Dotted: DC-settled PAM4 levels\nVertical scales are independent",
        transform=axes[1].transAxes,
        fontsize=8,
        color="0.35",
    )
    figure.suptitle(
        f"Fig. 4 — Deterministic {symbol_rate_hz / 1e9:.0f} GBd PAM4 eyes, "
        f"{differential_outer_pp_v * 1e3:.0f} mVpp TX, "
        f"{abs(channel_loss_db_at_nyquist):.0f} dB channel loss at Nyquist\n"
        "Same symbol sequence; AFE, noise, jitter, ADC, and RX equalization excluded"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_tx_channel_afe_eyes(
    path: Path,
    *,
    tx_eye,
    channel_eye,
    afe_eye,
    tx_unit_voltage_v: float,
    channel_dc_unit_voltage_v: float,
    afe_dc_unit_voltage_v: float,
    differential_outer_pp_v: float,
    symbol_rate_hz: float,
    channel_loss_db_at_nyquist: float,
) -> None:
    """Plot deterministic PAM4 eyes at all three analog reference planes."""

    figure, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    stages = (
        (
            axes[0],
            tx_eye,
            "tab:blue",
            "Fig. 6a — TX output",
            tx_unit_voltage_v,
            "Nominal TX levels",
        ),
        (
            axes[1],
            channel_eye,
            "tab:red",
            "Fig. 6b — Channel output",
            channel_dc_unit_voltage_v,
            "DC-settled levels",
        ),
        (
            axes[2],
            afe_eye,
            "tab:green",
            "Fig. 6c — CTLE/AFE output",
            afe_dc_unit_voltage_v,
            "DC-settled levels",
        ),
    )
    for axis, eye, color, title, settled_unit_v, guide_label in stages:
        for trace in eye.traces_v:
            axis.plot(
                eye.time_ui,
                trace * 1e3,
                color=color,
                alpha=0.032,
                linewidth=0.5,
            )
        axis.scatter(
            np.zeros(eye.center_samples_v.size),
            eye.center_samples_v * 1e3,
            s=3.5,
            color="black",
            alpha=0.11,
            linewidths=0,
            zorder=3,
        )
        for level in (-3.0, -1.0, 1.0, 3.0):
            axis.axhline(
                level * settled_unit_v * 1e3,
                color="0.45",
                linestyle=":",
                linewidth=0.6,
                alpha=0.42,
            )
        axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
        axis.set_xlim(-1.0, 1.0)
        axis.set_xlabel("Time relative to main cursor (UI)")
        axis.set_ylabel("Loaded differential voltage (mV)")
        axis.set_title(title)
        axis.grid(True, alpha=0.2)
        axis.text(
            0.02,
            0.03,
            f"Dotted: {guide_label}",
            transform=axis.transAxes,
            fontsize=8,
            color="0.35",
        )
        voltage_limit_mv = max(
            1.08 * float(np.max(np.abs(eye.traces_v))) * 1e3,
            3.35 * settled_unit_v * 1e3,
        )
        axis.set_ylim(-voltage_limit_mv, voltage_limit_mv)

    figure.suptitle(
        f"Fig. 6 — Deterministic {symbol_rate_hz / 1e9:.0f} GBd PAM4 eyes, "
        f"{differential_outer_pp_v * 1e3:.0f} mVpp TX, "
        f"{abs(channel_loss_db_at_nyquist):.0f} dB channel loss at Nyquist\n"
        "Same symbols; complex CTLE amplitude and phase included at AFE output; "
        "noise, jitter, ADC, and digital RX EQ excluded; vertical scales independent"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _interior_waveform_segment(
    symbols: np.ndarray,
    symbol_pulse_v: np.ndarray,
    *,
    sampling_index: int,
    samples_per_ui: int,
    guard_symbols: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an edge-free continuous segment and its symbol-center samples."""

    waveform_v = synthesize_symbol_waveform(
        symbols, symbol_pulse_v, samples_per_ui=samples_per_ui
    )
    first_symbol = guard_symbols
    stop_symbol = symbols.size - guard_symbols
    segment_start = sampling_index + first_symbol * samples_per_ui
    segment_stop = sampling_index + stop_symbol * samples_per_ui
    segment_v = waveform_v[segment_start:segment_stop]
    center_indices = sampling_index + np.arange(
        first_symbol, stop_symbol, dtype=int
    ) * samples_per_ui
    return segment_v, waveform_v[center_indices]


def _write_vga_frequency_csv(path: Path, *, vga) -> None:
    """Export the ideal broadband VGA response with explicit gain semantics."""

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frequency_hz",
                "vga_av_port_real",
                "vga_av_port_imag",
                "vga_av_port_magnitude_db",
                "vga_av_port_phase_deg",
                "vga_av_port_group_delay_s",
            ]
        )
        for index, frequency in enumerate(vga.frequency_hz):
            writer.writerow(
                [
                    frequency,
                    vga.transfer[index].real,
                    vga.transfer[index].imag,
                    vga.magnitude_db[index],
                    np.rad2deg(vga.phase_rad[index]),
                    vga.group_delay_s[index],
                ]
            )


def _write_adc_center_samples_csv(
    path: Path, *, afe_eye, adc_input_eye, adc_result
) -> None:
    """Export symbol-center ADC inputs, output codes, and overload flags."""

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "symbol_index",
                "normalized_pam4_symbol",
                "pre_vga_ctle_output_v",
                "adc_analog_input_v",
                "adc_output_code_unsigned",
                "adc_reconstructed_voltage_v",
                "adc_total_conversion_error_v",
                "clipped_low",
                "clipped_high",
                "clipped",
            ]
        )
        for values in zip(
            afe_eye.symbol_indices,
            afe_eye.transmitted_symbols,
            afe_eye.center_samples_v,
            adc_result.input_v,
            adc_result.codes,
            adc_result.reconstructed_v,
            adc_result.error_from_input_v,
            adc_result.clipped_low,
            adc_result.clipped_high,
            adc_result.clipped,
        ):
            writer.writerow(values)


def _write_adc_impairment_samples_csv(path: Path, *, impairments, adc_result) -> None:
    """Export the separated stochastic contributions at each ADC decision."""

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "relative_symbol_index",
                "noiseless_adc_input_v",
                "afe_noise_v",
                "additional_adc_noise_v",
                "aperture_jitter_s",
                "signal_slope_v_per_s",
                "jitter_voltage_error_v",
                "impaired_adc_input_v",
                "adc_output_code_unsigned",
                "adc_clipped",
            ]
        )
        for values in zip(
            np.arange(impairments.impaired_samples_v.size),
            impairments.noiseless_samples_v,
            impairments.afe_noise_samples_v,
            impairments.additional_adc_noise_samples_v,
            impairments.aperture_jitter_s,
            impairments.signal_slope_v_per_s,
            impairments.jitter_error_v,
            impairments.impaired_samples_v,
            adc_result.codes,
            adc_result.clipped,
        ):
            writer.writerow(values)


def _plot_adc_impairment_summary(
    path: Path,
    *,
    frequency_hz: np.ndarray,
    adc_input_afe_noise_psd: np.ndarray,
    impairments,
    additional_adc_noise_rms_v: float,
    aperture_jitter_rms_s: float,
) -> None:
    """Plot ADC-input noise spectrum and sampled impairment contributions."""

    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)

    density_axis = axes[0, 0]
    density_axis.plot(
        frequency_hz / 1e9,
        np.sqrt(adc_input_afe_noise_psd) * 1e9,
    )
    density_axis.set_title("Fig. 11a — AFE noise density at ADC input")
    density_axis.set_xlabel("Frequency (GHz)")
    density_axis.set_ylabel("Differential noise density (nV/√Hz)")
    density_axis.grid(True, alpha=0.25)

    sequence_axis = axes[0, 1]
    point_count = min(300, impairments.impaired_samples_v.size)
    symbol_index = np.arange(point_count)
    sequence_axis.plot(
        symbol_index,
        impairments.afe_noise_samples_v[:point_count] * 1e3,
        linewidth=0.8,
        label="AFE noise",
    )
    sequence_axis.plot(
        symbol_index,
        impairments.additional_adc_noise_samples_v[:point_count] * 1e3,
        linewidth=0.8,
        label="Additional ADC noise",
    )
    sequence_axis.plot(
        symbol_index,
        impairments.jitter_error_v[:point_count] * 1e3,
        linewidth=0.8,
        label="Aperture-jitter error",
    )
    sequence_axis.set_title("Fig. 11b — Sampled stochastic contributions")
    sequence_axis.set_xlabel("Relative symbol index")
    sequence_axis.set_ylabel("Differential error (mV)")
    sequence_axis.grid(True, alpha=0.25)
    sequence_axis.legend(fontsize=8)

    histogram_axis = axes[1, 0]
    components = (
        (impairments.afe_noise_samples_v, "AFE noise"),
        (impairments.additional_adc_noise_samples_v, "ADC noise"),
        (impairments.jitter_error_v, "Jitter error"),
    )
    for values, label in components:
        histogram_axis.hist(
            values * 1e3,
            bins=70,
            density=True,
            histtype="step",
            linewidth=1.2,
            label=label,
        )
    histogram_axis.set_title("Fig. 11c — Sampled impairment distributions")
    histogram_axis.set_xlabel("Differential error (mV)")
    histogram_axis.set_ylabel("Probability density (1/mV)")
    histogram_axis.grid(True, alpha=0.25)
    histogram_axis.legend(fontsize=8)

    slope_axis = axes[1, 1]
    slope_axis.scatter(
        impairments.signal_slope_v_per_s / 1e9,
        impairments.jitter_error_v * 1e3,
        s=5,
        alpha=0.18,
        linewidths=0,
    )
    slope_axis.set_title("Fig. 11d — Jitter error versus local signal slope")
    slope_axis.set_xlabel("Local differential slope (V/ns)")
    slope_axis.set_ylabel("Aperture-jitter error (mV)")
    slope_axis.grid(True, alpha=0.25)

    figure.suptitle(
        "Fig. 11 — Stochastic ADC-input impairments used by the code-domain FFE/DFE\n"
        f"AFE noise RMS {impairments.observed_afe_noise_rms_v * 1e3:.2f} mV; "
        f"additional ADC noise {additional_adc_noise_rms_v * 1e3:.2f} mV RMS; "
        f"aperture jitter {aperture_jitter_rms_s * 1e15:.0f} fs RMS → "
        f"{impairments.observed_jitter_error_rms_v * 1e3:.2f} mV RMS"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_adc_stage(
    path: Path,
    *,
    adc_input_eye,
    adc,
    adc_eye_result,
    adc_center_result,
    vga,
    vga_worst_case_input_peak_v: float,
    vga_target_peak_v: float,
    frequency_hz: np.ndarray,
    plot_max_hz: float,
    symbol_rate_hz: float,
) -> None:
    """Plot flat VGA response, ADC transfer, input eye, and output codes."""

    figure, axes = plt.subplots(2, 3, figsize=(17, 8.5), constrained_layout=True)
    half_scale_v = 0.5 * adc.differential_full_scale_pp_v
    center_clip_fraction = float(np.mean(adc_center_result.clipped))
    frequency_mask = frequency_hz <= plot_max_hz
    frequency_ghz = frequency_hz / 1e9

    vga_magnitude_axis = axes[0, 0]
    vga_magnitude_axis.plot(
        frequency_ghz[frequency_mask], vga.magnitude_db[frequency_mask]
    )
    vga_magnitude_axis.set_title("Fig. 7a — Broadband VGA magnitude")
    vga_magnitude_axis.set_xlabel("Frequency (GHz)")
    vga_magnitude_axis.set_ylabel("Av_port (dB)")
    vga_magnitude_axis.grid(True, alpha=0.25)

    vga_delay_axis = axes[0, 1]
    vga_delay_axis.plot(
        frequency_ghz[frequency_mask],
        vga.group_delay_s[frequency_mask] * 1e12,
        color="tab:purple",
    )
    vga_delay_axis.set_ylim(-1.0, 1.0)
    vga_delay_axis.set_title("Fig. 7b — Broadband VGA group delay")
    vga_delay_axis.set_xlabel("Frequency (GHz)")
    vga_delay_axis.set_ylabel("Group delay of Av_port (ps)")
    vga_delay_axis.grid(True, alpha=0.25)

    transfer_axis = axes[0, 2]
    transfer_input_v = np.linspace(-1.3 * half_scale_v, 1.3 * half_scale_v, 4097)
    transfer_result = adc.quantize(transfer_input_v)
    transfer_axis.plot(transfer_input_v * 1e3, transfer_result.codes, linewidth=1.2)
    transfer_axis.axvline(
        adc.minimum_input_v * 1e3, color="tab:red", linestyle="--", linewidth=0.9
    )
    transfer_axis.axvline(
        adc.maximum_input_v * 1e3, color="tab:red", linestyle="--", linewidth=0.9
    )
    transfer_axis.set_title("Fig. 7c — Effective ADC transfer characteristic")
    transfer_axis.set_xlabel("Differential ADC input (mV)")
    transfer_axis.set_ylabel("Unsigned output code")
    transfer_axis.set_ylim(-2, adc.code_count + 1)
    transfer_axis.grid(True, alpha=0.25)

    analog_eye_axis = axes[1, 0]
    for trace in adc_input_eye.traces_v:
        analog_eye_axis.plot(
            adc_input_eye.time_ui,
            trace * 1e3,
            color="tab:green",
            alpha=0.025,
            linewidth=0.5,
        )
    analog_limit_v = max(
        1.1 * half_scale_v,
        1.1 * float(np.max(np.abs(adc_input_eye.traces_v))),
    )
    analog_eye_axis.set_ylim(-analog_limit_v * 1e3, analog_limit_v * 1e3)
    analog_eye_axis.axhspan(
        half_scale_v * 1e3,
        analog_limit_v * 1e3,
        color="tab:red",
        alpha=0.08,
    )
    analog_eye_axis.axhspan(
        -analog_limit_v * 1e3,
        -half_scale_v * 1e3,
        color="tab:red",
        alpha=0.08,
    )
    for rail_v in (adc.minimum_input_v, adc.maximum_input_v):
        analog_eye_axis.axhline(
            rail_v * 1e3, color="tab:red", linestyle="--", linewidth=0.9
        )
    analog_eye_axis.axvline(0.0, color="black", linestyle=":", linewidth=0.8)
    analog_eye_axis.set_xlim(-1.0, 1.0)
    analog_eye_axis.set_title("Fig. 7d — VGA output at ADC input")
    analog_eye_axis.set_xlabel("Time relative to main cursor (UI)")
    analog_eye_axis.set_ylabel("Differential voltage (mV)")
    analog_eye_axis.grid(True, alpha=0.2)

    analog_eye_axis.text(
        0.02,
        0.03,
        f"CTLE worst-case bound: {vga_worst_case_input_peak_v * 1e3:.1f} mV\n"
        f"AGC target: ±{vga_target_peak_v * 1e3:.1f} mV",
        transform=analog_eye_axis.transAxes,
        fontsize=8,
        color="0.35",
    )

    code_eye_axis = axes[1, 1]
    for code_trace in adc_eye_result.codes:
        code_eye_axis.plot(
            adc_input_eye.time_ui,
            code_trace,
            color="tab:orange",
            alpha=0.025,
            linewidth=0.5,
        )
    code_eye_axis.scatter(
        np.zeros(adc_center_result.codes.size),
        adc_center_result.codes,
        s=4,
        color="black",
        alpha=0.13,
        linewidths=0,
        zorder=3,
    )
    code_eye_axis.axvline(0.0, color="black", linestyle=":", linewidth=0.8)
    code_eye_axis.set_xlim(-1.0, 1.0)
    code_eye_axis.set_ylim(-2, adc.code_count + 1)
    code_eye_axis.set_title("Fig. 7e — Effective-code eye visualization")
    code_eye_axis.set_xlabel("Time relative to main cursor (UI)")
    code_eye_axis.set_ylabel("Unsigned output code")
    code_eye_axis.grid(True, alpha=0.2)
    code_eye_axis.text(
        0.02,
        0.03,
        "Lines use the 8-sample/UI simulation grid;\nADC decisions use the center sample only",
        transform=code_eye_axis.transAxes,
        fontsize=8,
        color="0.35",
    )

    conditional_axis = axes[1, 2]
    pam4_levels = np.asarray([-3.0, -1.0, 1.0, 3.0])
    counts = np.zeros((pam4_levels.size, adc.code_count), dtype=int)
    for level_index, level in enumerate(pam4_levels):
        level_codes = adc_center_result.codes[
            adc_input_eye.transmitted_symbols == level
        ]
        counts[level_index] = np.bincount(
            level_codes, minlength=adc.code_count
        )
    image = conditional_axis.imshow(
        counts,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        extent=(-0.5, adc.code_count - 0.5, -0.5, 3.5),
        cmap="viridis",
    )
    conditional_axis.set_yticks(np.arange(4), ["−3", "−1", "+1", "+3"])
    conditional_axis.set_title("Fig. 7f — Center-code distribution by TX symbol")
    conditional_axis.set_xlabel("Unsigned ADC output code")
    conditional_axis.set_ylabel("Normalized transmitted PAM4 symbol")
    figure.colorbar(image, ax=conditional_axis, label="Sample count")

    figure.suptitle(
        f"Fig. 7 — {adc.enob_bits}-ENOB effective ADC, "
        f"{adc.differential_full_scale_pp_v * 1e3:.0f} mVpp differential full scale, "
        f"{symbol_rate_hz / 1e9:.0f} GSa/s (1 sample/UI)\n"
        f"Ideal flat VGA Av_port = {vga.metadata['gain_db']:.2f} dB, "
        f"95% worst-case full-scale target; "
        f"Effective LSB = {adc.effective_lsb_v * 1e3:.2f} mV; "
        f"main-cursor samples clipped = {100.0 * center_clip_fraction:.1f}%; "
        "no digital equalization"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_papr_csv(path: Path, records: list[dict]) -> None:
    """Export consistent voltage and PAPR statistics at every signal node."""

    fieldnames = [
        "node",
        "reference_plane",
        "sampling",
        "sample_count",
        "mean_v",
        "rms_v",
        "peak_abs_v",
        "crest_factor",
        "papr_linear",
        "papr_db",
        "mean_removed",
        "noise_included",
        "jitter_included",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _plot_papr_summary(path: Path, *, records: list[dict]) -> None:
    """Plot PAPR and absolute voltage utilization at all modeled nodes."""

    figure, axes = plt.subplots(2, 1, figsize=(13, 8.5), constrained_layout=True)
    display_labels = {
        "ADC input decision samples": "ADC samples\n(noiseless)",
        "ADC reconstructed output": "ADC output\n(noiseless)",
        "Impaired ADC input samples": "ADC samples\n(impaired)",
        "Impaired ADC output": "ADC output\n(impaired)",
    }
    labels = [
        display_labels.get(record["node"], record["node"]) for record in records
    ]
    positions = np.arange(len(records))
    colors = [
        "tab:blue"
        if record["sampling"].endswith("continuous grid")
        else "tab:orange"
        for record in records
    ]

    papr_axis = axes[0]
    papr_values_db = np.asarray([record["papr_db"] for record in records])
    bars = papr_axis.bar(positions, papr_values_db, color=colors, alpha=0.82)
    papr_axis.axhline(
        10.0 * np.log10(9.0 / 5.0),
        color="0.35",
        linestyle="--",
        linewidth=0.9,
        label="Ideal equiprobable PAM4 levels: 2.55 dB",
    )
    papr_axis.bar_label(bars, fmt="%.2f dB", padding=3, fontsize=8)
    papr_axis.set_ylabel("PAPR (dB)")
    papr_axis.set_title("Fig. 8a — Peak-to-average power ratio by reference plane")
    papr_axis.set_xticks(positions, labels)
    papr_axis.grid(True, axis="y", alpha=0.25)
    papr_axis.legend()

    voltage_axis = axes[1]
    width = 0.36
    rms_mv = np.asarray([record["rms_v"] for record in records]) * 1e3
    peak_mv = np.asarray([record["peak_abs_v"] for record in records]) * 1e3
    voltage_axis.bar(
        positions - width / 2,
        rms_mv,
        width,
        label="RMS voltage",
        color="tab:green",
        alpha=0.82,
    )
    voltage_axis.bar(
        positions + width / 2,
        peak_mv,
        width,
        label="Observed absolute peak",
        color="tab:red",
        alpha=0.72,
    )
    voltage_axis.set_ylabel("Differential voltage (mV)")
    voltage_axis.set_title("Fig. 8b — RMS and observed peak voltage")
    voltage_axis.set_xticks(positions, labels)
    voltage_axis.grid(True, axis="y", alpha=0.25)
    voltage_axis.legend()

    figure.suptitle(
        "Fig. 8 — PAPR by reference plane; final two ADC nodes include noise and jitter"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_code_ffe_taps_csv(path: Path, result) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tap_offset_ui", "coefficient_per_centered_adc_code"])
        writer.writerows(
            zip(result.tap_offsets_ui, result.coefficients_per_code)
        )


def _write_code_ffe_test_csv(path: Path, *, result, adc_codes: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sequence_index",
                "transmitted_pam4_symbol",
                "raw_adc_code_unsigned",
                "raw_adc_code_centered",
                "ffe_output",
                "ffe_decision",
                "decision_error",
            ]
        )
        for values in zip(
            result.test_indices,
            result.test_symbols,
            adc_codes[result.test_indices],
            result.raw_test_codes_centered,
            result.test_output,
            result.test_decisions,
            result.test_decisions != result.test_symbols,
        ):
            writer.writerow(values)


def _zero_error_upper_bound_95(error_count: int, opportunity_count: int):
    """Return the exact one-sided binomial upper bound when zero errors occur."""

    if error_count != 0:
        return None
    return 1.0 - 0.05 ** (1.0 / opportunity_count)


def _wilson_interval_95(error_count: int, opportunity_count: int) -> tuple[float, float]:
    """Return a two-sided 95% Wilson score interval for a binomial rate."""

    if not 0 <= error_count <= opportunity_count or opportunity_count < 1:
        raise ValueError("invalid binomial counts")
    z_value = 1.959963984540054
    estimate = error_count / opportunity_count
    denominator = 1.0 + z_value**2 / opportunity_count
    center = (
        estimate + z_value**2 / (2.0 * opportunity_count)
    ) / denominator
    half_width = z_value / denominator * np.sqrt(
        estimate * (1.0 - estimate) / opportunity_count
        + z_value**2 / (4.0 * opportunity_count**2)
    )
    return float(center - half_width), float(center + half_width)


def _write_code_dfe_taps_csv(path: Path, result) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feedback_delay_ui", "feedback_coefficient"])
        writer.writerows(
            zip(result.feedback_delays_ui, result.feedback_coefficients)
        )


def _write_code_dfe_test_csv(path: Path, *, result) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sequence_index",
                "transmitted_pam4_symbol",
                "ffe_output",
                "ffe_baseline_decision",
                "genie_dfe_corrected_output",
                "genie_dfe_decision",
                "decision_directed_dfe_corrected_output",
                "decision_directed_dfe_decision",
                "decision_directed_symbol_error",
            ]
        )
        for values in zip(
            result.test_indices,
            result.test_symbols,
            result.ffe_output,
            result.baseline_decisions,
            result.genie_corrected_output,
            result.genie_decisions,
            result.decision_directed_output,
            result.decision_directed_decisions,
            result.decision_directed_decisions != result.test_symbols,
        ):
            writer.writerow(values)


def _plot_code_ffe_summary(
    path: Path,
    *,
    result,
    symbol_rate_hz: float,
    impairment_note: str,
) -> None:
    """Plot code-domain FFE taps, response, held-out levels, and decisions."""

    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)

    tap_axis = axes[0, 0]
    tap_axis.stem(
        result.tap_offsets_ui,
        result.coefficients_per_code,
        basefmt=" ",
    )
    tap_axis.set_title("Fig. 9a — Code-domain FFE coefficients")
    tap_axis.set_xlabel("Tap offset (UI)")
    tap_axis.set_ylabel("Coefficient (normalized PAM/code)")
    tap_axis.grid(True, alpha=0.25)

    frequency_axis = axes[0, 1]
    normalized_frequency = np.linspace(0.0, 0.5, 1001)
    response = np.exp(
        1j
        * 2.0
        * np.pi
        * normalized_frequency[:, None]
        * result.tap_offsets_ui[None, :]
    ) @ result.coefficients_per_code
    response_db = 20.0 * np.log10(
        np.maximum(np.abs(response), np.finfo(float).tiny)
    )
    frequency_axis.plot(normalized_frequency * symbol_rate_hz / 1e9, response_db)
    frequency_axis.set_title("Fig. 9b — Symbol-spaced FFE magnitude")
    frequency_axis.set_xlabel("Frequency (GHz)")
    frequency_axis.set_ylabel("Magnitude (dB normalized-PAM/code)")
    frequency_axis.grid(True, alpha=0.25)

    distribution_axis = axes[1, 0]
    levels = np.asarray([-3.0, -1.0, 1.0, 3.0])
    level_outputs = [
        result.test_output[result.test_symbols == level] for level in levels
    ]
    distribution_axis.boxplot(
        level_outputs,
        tick_labels=["−3", "−1", "+1", "+3"],
        showfliers=False,
        whis=(1.0, 99.0),
    )
    for threshold in np.sort(result.slicer_thresholds):
        distribution_axis.axhline(
            threshold, color="tab:red", linestyle="--", linewidth=0.8
        )
    distribution_axis.set_title("Fig. 9c — Held-out FFE output by PAM4 symbol")
    distribution_axis.set_xlabel("Transmitted PAM4 symbol")
    distribution_axis.set_ylabel("FFE output")
    distribution_axis.grid(True, axis="y", alpha=0.25)

    confusion_axis = axes[1, 1]
    image = confusion_axis.imshow(
        result.confusion_matrix,
        origin="upper",
        interpolation="nearest",
        cmap="Blues",
    )
    confusion_axis.set_xticks(np.arange(4), ["−3", "−1", "+1", "+3"])
    confusion_axis.set_yticks(np.arange(4), ["−3", "−1", "+1", "+3"])
    confusion_axis.set_xlabel("Detected PAM4 symbol")
    confusion_axis.set_ylabel("Transmitted PAM4 symbol")
    confusion_axis.set_title("Fig. 9d — Held-out confusion matrix")
    for row in range(4):
        for column in range(4):
            count = int(result.confusion_matrix[row, column])
            confusion_axis.text(
                column,
                row,
                str(count),
                ha="center",
                va="center",
                color="white"
                if count > 0.5 * np.max(result.confusion_matrix)
                else "black",
                fontsize=9,
            )
    figure.colorbar(image, ax=confusion_axis, label="Held-out sample count")

    zero_error_upper_95 = (
        1.0 - 0.05 ** (1.0 / result.test_indices.size)
        if result.empirical_error_count == 0
        else None
    )
    confidence_text = (
        f"0/{result.test_indices.size} errors; 95% zero-error upper bound "
        f"{zero_error_upper_95:.3e}"
        if zero_error_upper_95 is not None
        else f"{result.empirical_error_count}/{result.test_indices.size} errors"
    )
    figure.suptitle(
        f"Fig. 9 — {result.tap_offsets_ui.size}-tap symbol-spaced FFE trained on "
        "centered 6-ENOB ADC codes\n"
        f"Held-out raw dpSNR {result.raw_dp_snr_db:.2f} dB → FFE dpSNR "
        f"{result.dp_snr_db:.2f} dB; empirical DER "
        f"{result.raw_empirical_der:.3e} → {result.empirical_der:.3e}; "
        f"{confidence_text}; {impairment_note}"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_code_dfe_summary(path: Path, *, result, impairment_note: str) -> None:
    """Plot DFE taps, residuals, held-out levels, and honest BER evidence."""

    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)

    tap_axis = axes[0, 0]
    tap_axis.stem(
        result.feedback_delays_ui,
        result.feedback_coefficients,
        basefmt=" ",
    )
    tap_axis.set_title("Fig. 10a — Postcursor DFE feedback coefficients")
    tap_axis.set_xlabel("Feedback delay (UI)")
    tap_axis.set_ylabel("Coefficient (FFE-output/PAM unit)")
    tap_axis.grid(True, alpha=0.25)

    residual_axis = axes[0, 1]
    point_count = min(300, result.test_symbols.size)
    residual_axis.plot(
        result.test_indices[:point_count],
        (
            result.ffe_output[:point_count]
            - result.decision_offset
            - result.decision_gain * result.test_symbols[:point_count]
        )
        / result.decision_gain,
        linewidth=0.9,
        alpha=0.75,
        label="FFE residual",
    )
    residual_axis.plot(
        result.test_indices[:point_count],
        (
            result.decision_directed_output[:point_count]
            - result.decision_offset
            - result.decision_gain * result.test_symbols[:point_count]
        )
        / result.decision_gain,
        linewidth=0.9,
        alpha=0.8,
        label="Decision-directed DFE residual",
    )
    residual_axis.axhline(0.0, color="black", linewidth=0.7, linestyle=":")
    residual_axis.set_title("Fig. 10b — Held-out decision residual")
    residual_axis.set_xlabel("Symbol index")
    residual_axis.set_ylabel("Error (normalized PAM units)")
    residual_axis.grid(True, alpha=0.25)
    residual_axis.legend()

    distribution_axis = axes[1, 0]
    levels = np.asarray([-3.0, -1.0, 1.0, 3.0])
    level_outputs = [
        result.decision_directed_output[result.test_symbols == level]
        for level in levels
    ]
    distribution_axis.boxplot(
        level_outputs,
        tick_labels=["−3", "−1", "+1", "+3"],
        showfliers=False,
        whis=(1.0, 99.0),
    )
    for threshold in np.sort(result.slicer_thresholds):
        distribution_axis.axhline(
            threshold, color="tab:red", linestyle="--", linewidth=0.8
        )
    distribution_axis.set_title("Fig. 10c — Decision-directed DFE output")
    distribution_axis.set_xlabel("Transmitted PAM4 symbol")
    distribution_axis.set_ylabel("Corrected FFE output")
    distribution_axis.grid(True, axis="y", alpha=0.25)

    rate_axis = axes[1, 1]
    bit_count = 2 * result.test_symbols.size
    ffe_upper = _zero_error_upper_bound_95(
        result.baseline_bit_error_count, bit_count
    )
    dfe_upper = _zero_error_upper_bound_95(
        result.decision_directed_bit_error_count, bit_count
    )
    empirical_values = [
        ffe_upper
        if ffe_upper is not None
        else result.baseline_empirical_ber,
        dfe_upper
        if dfe_upper is not None
        else result.decision_directed_empirical_ber,
    ]
    estimated_values = [
        result.baseline_gaussian_ber_approx,
        result.decision_directed_gaussian_ber_approx,
    ]
    x_positions = np.arange(2)
    rate_axis.scatter(
        x_positions - 0.10,
        empirical_values,
        marker="v" if ffe_upper is not None and dfe_upper is not None else "o",
        s=65,
        label="Measured BER or 95% upper bound",
    )
    for x_position, value, error_count in zip(
        x_positions - 0.10,
        empirical_values,
        [result.baseline_bit_error_count, result.decision_directed_bit_error_count],
    ):
        if error_count:
            lower, upper = _wilson_interval_95(error_count, bit_count)
            rate_axis.vlines(x_position, lower, upper, linewidth=1.2)
            rate_axis.hlines(
                [lower, upper], x_position - 0.025, x_position + 0.025, linewidth=1.2
            )
    rate_axis.scatter(
        x_positions + 0.10,
        estimated_values,
        marker="o",
        s=50,
        label="Gaussian-tail BER estimate",
    )
    for x_position, value, upper in zip(
        x_positions - 0.10, empirical_values, [ffe_upper, dfe_upper]
    ):
        prefix = "< " if upper is not None else ""
        rate_axis.annotate(
            f"{prefix}{value:.2e}",
            (x_position, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    for x_position, value in zip(x_positions + 0.10, estimated_values):
        rate_axis.annotate(
            f"{value:.2e}",
            (x_position, value),
            xytext=(0, -13),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    rate_axis.set_yscale("log")
    rate_axis.set_xticks(x_positions, ["FFE", "FFE + DFE"])
    rate_axis.set_ylabel("Gray-coded BER")
    rate_axis.set_title("Fig. 10d — Measured BER vs Gaussian estimate")
    rate_axis.grid(True, axis="y", which="both", alpha=0.25)
    rate_axis.legend(fontsize=8)

    dfe_observation_text = (
        f"0/{result.test_symbols.size} errors; 95% DER upper bound "
        f"{_zero_error_upper_bound_95(result.decision_directed_symbol_error_count, result.test_symbols.size):.3e}"
        if result.decision_directed_symbol_error_count == 0
        else f"{result.decision_directed_symbol_error_count}/{result.test_symbols.size} "
        f"errors; empirical DER {result.decision_directed_empirical_der:.3e}"
    )
    figure.suptitle(
        f"Fig. 10 — {result.feedback_delays_ui.size}-tap PAM4 DFE after code-domain FFE\n"
        f"Held-out dpSNR {result.baseline_dp_snr_db:.2f} → "
        f"{result.decision_directed_dp_snr_db:.2f} dB; {dfe_observation_text}; "
        f"{impairment_note}"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_fec_codeword_csv(path: Path, *, result) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "codeword_index",
                "start_pam4_symbol_index",
                "pre_fec_bit_errors",
                "pre_fec_rs_symbol_errors",
                "decoder_success",
                "decoder_reported_corrections",
                "post_fec_payload_bit_errors",
            ]
        )
        for codeword in result.codewords:
            writer.writerow(
                [
                    codeword.codeword_index,
                    codeword.start_pam4_symbol_index,
                    codeword.pre_fec_bit_errors,
                    codeword.pre_fec_symbol_errors,
                    codeword.decoder_success,
                    codeword.decoder_reported_corrections,
                    codeword.post_fec_payload_bit_errors,
                ]
            )


def _plot_fec_summary(path: Path, *, result, symbol_rate_hz: float) -> None:
    """Plot the RS codeword structure and measured/estimated correction result."""

    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)

    structure_axis = axes[0, 0]
    structure_axis.barh(
        [0], [RS_K_SYMBOLS], color="tab:blue", label="Payload"
    )
    structure_axis.barh(
        [0],
        [RS_PARITY_SYMBOLS],
        left=[RS_K_SYMBOLS],
        color="tab:orange",
        label="Parity",
    )
    structure_axis.set_xlim(0, RS_N_SYMBOLS)
    structure_axis.set_yticks([])
    structure_axis.set_xlabel("10-bit RS symbols per codeword")
    structure_axis.set_title("Fig. 12a — Systematic RS(544,514) codeword")
    structure_axis.legend()
    structure_axis.grid(True, axis="x", alpha=0.25)

    errors_axis = axes[0, 1]
    codeword_labels = [f"CW {word.codeword_index}" for word in result.codewords]
    codeword_positions = np.arange(result.evaluated_codeword_count)
    symbol_errors = [word.pre_fec_symbol_errors for word in result.codewords]
    bars = errors_axis.bar(codeword_positions, symbol_errors, color="tab:blue")
    errors_axis.axhline(
        RS_CORRECTABLE_SYMBOL_ERRORS,
        color="tab:red",
        linestyle="--",
        linewidth=1.0,
        label="Correction limit: 15 symbols",
    )
    errors_axis.bar_label(bars, padding=3)
    errors_axis.set_xticks(codeword_positions, codeword_labels)
    errors_axis.set_ylabel("Erroneous 10-bit RS symbols")
    errors_axis.set_title("Fig. 12b — Held-out errors by complete codeword")
    errors_axis.grid(True, axis="y", alpha=0.25)
    errors_axis.legend()

    ber_axis = axes[1, 0]
    post_upper = _zero_error_upper_bound_95(
        result.post_fec_payload_bit_error_count, result.payload_bit_count
    )
    post_measured_value = (
        post_upper if post_upper is not None else result.post_fec_ber
    )
    ber_values = [
        result.pre_fec_ber,
        post_measured_value,
        result.iid_failed_codeword_passthrough_ber_estimate,
    ]
    ber_labels = ["Pre-FEC\nmeasured", "Post-FEC\nmeasured/bound", "Post-FEC\nIID estimate"]
    ber_markers = ["o", "v" if post_upper is not None else "o", "s"]
    for x_position, value, marker in zip(range(3), ber_values, ber_markers):
        ber_axis.scatter(x_position, value, marker=marker, s=70)
        prefix = "< " if x_position == 1 and post_upper is not None else ""
        ber_axis.annotate(
            f"{prefix}{value:.2e}",
            (x_position, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ber_axis.set_yscale("log")
    ber_axis.set_xticks(range(3), ber_labels)
    ber_axis.set_ylabel("Gray-coded BER")
    ber_axis.set_title("Fig. 12c — Pre-FEC and post-FEC BER evidence")
    ber_axis.grid(True, axis="y", which="both", alpha=0.25)

    failure_axis = axes[1, 1]
    measured_failure_upper = _zero_error_upper_bound_95(
        result.uncorrectable_codeword_count, result.evaluated_codeword_count
    )
    measured_failure_value = (
        measured_failure_upper
        if measured_failure_upper is not None
        else result.codeword_failure_rate
    )
    failure_values = [
        measured_failure_value,
        result.iid_uncorrectable_codeword_probability,
    ]
    failure_labels = ["Measured/bound", "IID estimate"]
    for x_position, value in enumerate(failure_values):
        marker = "v" if x_position == 0 and measured_failure_upper is not None else "o"
        failure_axis.scatter(x_position, value, marker=marker, s=70)
        prefix = "< " if x_position == 0 and measured_failure_upper is not None else ""
        failure_axis.annotate(
            f"{prefix}{value:.2e}",
            (x_position, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    failure_axis.set_yscale("log")
    failure_axis.set_xticks(range(2), failure_labels)
    failure_axis.set_ylabel("Uncorrectable codeword probability")
    failure_axis.set_title("Fig. 12d — Codeword-failure evidence")
    failure_axis.grid(True, axis="y", which="both", alpha=0.25)

    codeword_span_ns = (
        RS_N_SYMBOLS * RS_SYMBOL_BITS / 2.0 / symbol_rate_hz * 1e9
    )
    figure.suptitle(
        "Fig. 12 — Bit-exact hard-decision RS(544,514) after the DFE\n"
        f"{result.evaluated_codeword_count} complete held-out codewords; "
        f"{result.pre_fec_bit_error_count} pre-FEC bit errors → "
        f"{result.post_fec_payload_bit_error_count} payload bit errors; "
        f"codeword span {codeword_span_ns:.2f} ns"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_decision_point_csv(path: Path, decision_point) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["series", "offset_ui", "value", "unit"])
        for offset, value in zip(
            decision_point.raw_cursor_offsets_ui, decision_point.raw_cursor_v
        ):
            writer.writerow(["raw_channel_cursor", int(offset), value, "V_per_PAM_unit"])
        for offset, value in zip(
            decision_point.ffe_offsets_ui, decision_point.ffe_coefficients_per_v
        ):
            writer.writerow(["mmse_ffe", int(offset), value, "1_per_V"])
        for offset, value in zip(
            decision_point.equalized_cursor_offsets_ui, decision_point.equalized_cursor
        ):
            writer.writerow(["equalized_cursor", int(offset), value, "normalized_symbol"])


def _plot_decision_point(path: Path, decision_point, metrics: dict) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)

    raw_axis = axes[0, 0]
    raw_axis.stem(
        decision_point.raw_cursor_offsets_ui,
        decision_point.raw_cursor_v * 1e3,
        basefmt=" ",
    )
    raw_axis.set_title("Fig. 2a — Sampled signal-path cursors")
    raw_axis.set_xlabel("Cursor offset (UI)")
    raw_axis.set_ylabel("ADC-input response (mV/PAM unit)")
    raw_axis.grid(True, alpha=0.3)

    ffe_axis = axes[0, 1]
    ffe_axis.stem(
        decision_point.ffe_offsets_ui,
        decision_point.ffe_coefficients_per_v,
        basefmt=" ",
    )
    ffe_axis.set_title("Fig. 2b — Symbol-spaced MMSE FFE")
    ffe_axis.set_xlabel("FFE tap offset (UI)")
    ffe_axis.set_ylabel("Coefficient (1/V)")
    ffe_axis.grid(True, alpha=0.3)

    equalized_axis = axes[1, 0]
    normalized_equalized = decision_point.equalized_cursor / decision_point.main_cursor
    equalized_axis.stem(
        decision_point.equalized_cursor_offsets_ui,
        normalized_equalized,
        basefmt=" ",
    )
    equalized_axis.set_xlim(-12, 24)
    equalized_axis.set_title("Fig. 2c — Equalized cursor response")
    equalized_axis.set_xlabel("Cursor offset (UI)")
    equalized_axis.set_ylabel("Normalized cursor")
    equalized_axis.grid(True, alpha=0.3)

    summary_axis = axes[1, 1]
    summary_axis.axis("off")
    decision_metrics = metrics["decision_point"]
    summary_lines = [
        f"Optimized sampling phase: {decision_metrics['sampling_phase_ui']:+.3f} UI",
        f"dpSNR: {decision_metrics['dp_snr_db']:.2f} dB",
        f"Required dpSNR: {decision_metrics['required_dp_snr_db']:.2f} dB",
        f"dpSNR margin: {decision_metrics['dp_snr_margin_db']:.2f} dB",
        f"Gaussian DER: {decision_metrics['gaussian_der']:.3e}",
        f"Pattern-conditioned DER: {decision_metrics['pattern_conditioned_der']:.3e}",
        "",
        "Model scope:",
        f"• {metrics['transmitter']['pulse_shaping']['model']} TX shaping",
        f"• {abs(metrics['nyquist']['channel_sdd21_db']):.1f} dB channel loss at Nyquist",
        "• Channel + AFE complex frequency response",
        "• AFE input-referred Gaussian noise",
        "• Ideal PAM4 levels and reference MMSE FFE",
        "• No jitter, ADC quantization, or DFE propagation",
    ]
    summary_axis.text(0.02, 0.98, "\n".join(summary_lines), va="top", family="monospace")
    summary_axis.set_title("Fig. 2d — Decision-point metrics")

    figure.suptitle("Fig. 2 — PAM4 decision-point evaluation")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_summary(
    path: Path,
    *,
    channel,
    afe,
    channel_afe,
    tx,
    signal_path,
    impulse_time_s: np.ndarray,
    impulse: np.ndarray,
    pulse_time_s: np.ndarray,
    pulse: np.ndarray,
    main_index: int,
    symbol_period_s: float,
    plot_max_hz: float,
    pulse_pre_ui: int,
    pulse_post_ui: int,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)
    frequency_ghz = channel.frequency_hz / 1e9
    frequency_mask = channel.frequency_hz <= plot_max_hz

    magnitude_axis = axes[0, 0]
    magnitude_axis.plot(frequency_ghz[frequency_mask], channel.magnitude_db[frequency_mask], label="Channel Sdd21")
    magnitude_axis.plot(frequency_ghz[frequency_mask], afe.magnitude_db[frequency_mask], label="AFE Av_port")
    magnitude_axis.plot(
        frequency_ghz[frequency_mask],
        channel_afe.magnitude_db[frequency_mask],
        label="Channel + AFE",
    )
    magnitude_axis.plot(
        frequency_ghz[frequency_mask],
        signal_path.magnitude_db[frequency_mask],
        label=f"Signal path ({tx.metadata['model']})",
    )
    magnitude_axis.set_title("Fig. 1a — Frequency magnitude")
    magnitude_axis.set_xlabel("Frequency (GHz)")
    magnitude_axis.set_ylabel("Magnitude (dB)")
    magnitude_axis.grid(True, alpha=0.3)
    magnitude_axis.legend()

    delay_axis = axes[0, 1]
    delay_mask = frequency_mask & (channel.frequency_hz >= 1e9)
    delay_axis.plot(
        frequency_ghz[delay_mask], signal_path.group_delay_s[delay_mask] * 1e12
    )
    delay_axis.set_title("Fig. 1b — Signal-path group delay")
    delay_axis.set_xlabel("Frequency (GHz)")
    delay_axis.set_ylabel("Group delay (ps)")
    delay_axis.grid(True, alpha=0.3)

    impulse_axis = axes[1, 0]
    impulse_main_index = int(np.argmax(np.abs(impulse)))
    impulse_center_s = impulse_time_s[impulse_main_index]
    impulse_window = np.abs(impulse_time_s - impulse_center_s) <= 8.0 * symbol_period_s
    impulse_axis.plot((impulse_time_s[impulse_window] - impulse_center_s) / symbol_period_s, impulse[impulse_window])
    impulse_axis.set_title("Fig. 1c — Cascade impulse response")
    impulse_axis.set_xlabel("Relative time (UI)")
    impulse_axis.set_ylabel("Discrete impulse amplitude")
    impulse_axis.grid(True, alpha=0.3)

    pulse_axis = axes[1, 1]
    pulse_main_time_s = pulse_time_s[main_index]
    pulse_ui = (pulse_time_s - pulse_main_time_s) / symbol_period_s
    pulse_window = (pulse_ui >= -pulse_pre_ui) & (pulse_ui <= pulse_post_ui)
    pulse_axis.plot(pulse_ui[pulse_window], pulse[pulse_window])
    pulse_axis.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    pulse_axis.set_title("Fig. 1d — One-symbol pulse response")
    pulse_axis.set_xlabel("Relative time (UI)")
    pulse_axis.set_ylabel("Differential voltage transfer")
    pulse_axis.grid(True, alpha=0.3)

    figure.suptitle("Fig. 1 — High-loss channel, TX shaping, and AFE response")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = _parse_args()
    with args.config.open() as handle:
        config = yaml.safe_load(handle)

    link = config["link"]
    symbol_rate_hz = float(link["symbol_rate_gbaud"]) * 1e9
    symbol_period_s = 1.0 / symbol_rate_hz
    samples_per_ui = int(link["samples_per_ui"])
    max_frequency_hz = 0.5 * samples_per_ui * symbol_rate_hz
    frequency_hz = uniform_frequency_grid(max_frequency_hz, int(link["frequency_points"]))
    nyquist_hz = symbol_rate_hz / 2.0

    channel_config = dict(config["channel"])
    if args.channel_loss_db_at_nyquist is not None:
        if channel_config["kind"].lower() != "analytic_high_loss":
            raise ValueError("channel-loss override requires channel kind analytic_high_loss")
        channel_config["insertion_loss_db_at_nyquist"] = (
            args.channel_loss_db_at_nyquist
        )
    channel = _load_channel(channel_config, frequency_hz, nyquist_hz)
    afe_config = config["afe"]
    if afe_config.get("gain_kind", "Av_port") != "Av_port":
        raise ValueError("this baseline requires AFE gain_kind: Av_port")
    afe = ctle_afe_response(
        frequency_hz,
        dc_gain_db=float(afe_config["dc_gain_db"]),
        zero_frequencies_hz=np.asarray(afe_config["zeros_ghz"], dtype=float) * 1e9,
        pole_frequencies_hz=np.asarray(afe_config["poles_ghz"], dtype=float) * 1e9,
        differential_input_z0_ohm=float(afe_config["differential_input_z0_ohm"]),
    )
    channel_afe = cascade_responses(channel, afe)

    transmitter_config = config["transmitter"]
    if transmitter_config.get("modulation", "PAM4").upper() != "PAM4":
        raise ValueError("this decision-point evaluator requires PAM4 modulation")
    pulse_shaping_config = dict(transmitter_config["pulse_shaping"])
    if args.tx_pulse_shaping is not None:
        pulse_shaping_config["kind"] = args.tx_pulse_shaping
    if args.tx_rise_time_ps is not None:
        pulse_shaping_config["rise_time_20_80_ps"] = args.tx_rise_time_ps
    rise_time_ps = pulse_shaping_config.get("rise_time_20_80_ps")
    tx_response = tx_pulse_shaping_response(
        frequency_hz,
        kind=str(pulse_shaping_config["kind"]),
        rise_time_20_80_s=None
        if rise_time_ps is None
        else float(rise_time_ps) * 1e-12,
    )
    tx_channel_response = apply_tx_pulse_shaping_to_channel(tx_response, channel)
    signal_path = apply_tx_pulse_shaping(tx_response, channel_afe)
    zoh_response = zoh_frequency_response(
        frequency_hz, symbol_period_s=symbol_period_s
    )
    tx_symbol_spectrum = tx_response.transfer * zoh_response
    channel_symbol_spectrum = tx_channel_response.transfer * zoh_response
    afe_output_symbol_spectrum = signal_path.transfer * zoh_response
    tx_stage_time_s, tx_stage_pulse, tx_stage_samples_per_ui = (
        centered_symbol_pulse_response(
            tx_response,
            symbol_period_s=symbol_period_s,
        )
    )
    channel_stage_time_s, channel_stage_pulse, channel_stage_samples_per_ui = (
        centered_symbol_pulse_response(
            tx_channel_response,
            symbol_period_s=symbol_period_s,
        )
    )
    afe_stage_time_s, afe_stage_pulse, afe_stage_samples_per_ui = (
        centered_symbol_pulse_response(
            signal_path,
            symbol_period_s=symbol_period_s,
        )
    )
    if min(
        tx_stage_samples_per_ui,
        channel_stage_samples_per_ui,
        afe_stage_samples_per_ui,
    ) != samples_per_ui:
        raise RuntimeError("staged-response sample rates are inconsistent")
    impulse_time_s, impulse = frequency_to_impulse(signal_path)
    pulse_time_s, pulse, derived_samples_per_ui = rectangular_pulse_response(
        impulse_time_s,
        impulse,
        symbol_period_s=symbol_period_s,
    )
    if derived_samples_per_ui != samples_per_ui:
        raise RuntimeError("frequency grid and requested samples/UI are inconsistent")

    if channel.differential_z0_ohm is None:
        raise ValueError("absolute PAM4 power requires a differential channel impedance")
    if "differential_outer_pp_v" in transmitter_config:
        differential_outer_pp_v = float(transmitter_config["differential_outer_pp_v"])
        pam4_unit_input_v = pam4_unit_voltage_from_outer_pp(differential_outer_pp_v)
        differential_port_rms_voltage_v = np.sqrt(5.0) * pam4_unit_input_v
        average_delivered_power_w = (
            differential_port_rms_voltage_v**2 / channel.differential_z0_ohm
        )
        average_delivered_power_dbm = 10.0 * np.log10(
            average_delivered_power_w / 1e-3
        )
    else:
        average_delivered_power_dbm = float(
            transmitter_config["average_delivered_power_dbm"]
        )
        pam4_unit_input_v = pam4_unit_voltage_from_delivered_power(
            average_delivered_power_dbm,
            channel.differential_z0_ohm,
        )
        differential_outer_pp_v = 6.0 * pam4_unit_input_v
        differential_port_rms_voltage_v = np.sqrt(5.0) * pam4_unit_input_v

    tx_stage_pulse_v = tx_stage_pulse * pam4_unit_input_v
    channel_stage_pulse_v = channel_stage_pulse * pam4_unit_input_v
    afe_stage_pulse_v = afe_stage_pulse * pam4_unit_input_v
    tx_stage_peak_index = main_cursor_index(tx_stage_pulse_v)
    channel_stage_peak_index = main_cursor_index(channel_stage_pulse_v)
    afe_stage_peak_index = main_cursor_index(afe_stage_pulse_v)
    tx_stage_time_ui = (
        tx_stage_time_s - tx_stage_time_s[tx_stage_peak_index]
    ) / symbol_period_s
    channel_stage_time_ui = (
        channel_stage_time_s - channel_stage_time_s[channel_stage_peak_index]
    ) / symbol_period_s
    afe_stage_time_ui = (
        afe_stage_time_s - afe_stage_time_s[afe_stage_peak_index]
    ) / symbol_period_s
    staged_peak_delay_s = (
        channel_stage_time_s[channel_stage_peak_index]
        - tx_stage_time_s[tx_stage_peak_index]
    )
    afe_incremental_peak_delay_s = (
        afe_stage_time_s[afe_stage_peak_index]
        - channel_stage_time_s[channel_stage_peak_index]
    )

    eye_config = config.get("eye_diagram", {})
    eye_symbol_count = int(eye_config.get("symbol_count", 8192))
    eye_trace_count = int(eye_config.get("trace_count", 600))
    eye_guard_symbols = int(eye_config.get("guard_symbols", 512))
    eye_interpolation_samples_per_ui = int(
        eye_config.get("interpolation_samples_per_ui", 64)
    )
    eye_random_seed = int(eye_config.get("random_seed", 2026))
    fec_config = config.get("fec", {})
    fec_enabled = bool(fec_config.get("enabled", False))
    if fec_enabled:
        if fec_config.get("kind") != "rs_544_514":
            raise ValueError("unsupported FEC kind")
        fec_stream = generate_rs544_pam4_stream(
            eye_symbol_count,
            seed=eye_random_seed,
            alignment_offset_pam4_symbols=int(
                fec_config["alignment_offset_pam4_symbols"]
            ),
        )
        eye_symbols = fec_stream.pam4_symbols
    else:
        fec_stream = None
        eye_symbols = pam4_symbol_sequence(eye_symbol_count, seed=eye_random_seed)
    tx_eye = eye_diagram_from_symbol_pulse(
        eye_symbols,
        tx_stage_pulse_v,
        sampling_index=tx_stage_peak_index,
        samples_per_ui=samples_per_ui,
        trace_count=eye_trace_count,
        guard_symbols=eye_guard_symbols,
        interpolation_samples_per_ui=eye_interpolation_samples_per_ui,
    )
    channel_eye = eye_diagram_from_symbol_pulse(
        eye_symbols,
        channel_stage_pulse_v,
        sampling_index=channel_stage_peak_index,
        samples_per_ui=samples_per_ui,
        trace_count=eye_trace_count,
        guard_symbols=eye_guard_symbols,
        interpolation_samples_per_ui=eye_interpolation_samples_per_ui,
    )
    afe_eye = eye_diagram_from_symbol_pulse(
        eye_symbols,
        afe_stage_pulse_v,
        sampling_index=afe_stage_peak_index,
        samples_per_ui=samples_per_ui,
        trace_count=eye_trace_count,
        guard_symbols=eye_guard_symbols,
        interpolation_samples_per_ui=eye_interpolation_samples_per_ui,
    )

    adc_config = config["adc"]
    if adc_config.get("model", "effective_uniform_midrise") != "effective_uniform_midrise":
        raise ValueError("this baseline requires ADC model: effective_uniform_midrise")
    if int(adc_config.get("samples_per_ui", 1)) != 1:
        raise ValueError("this ADC decision-sample baseline requires samples_per_ui: 1")
    if adc_config.get("sampling_phase", "main_cursor") != "main_cursor":
        raise ValueError("this ADC baseline requires sampling_phase: main_cursor")
    adc = EffectiveUniformAdc(
        enob_bits=int(adc_config["enob_bits"]),
        differential_full_scale_pp_v=float(
            adc_config["differential_full_scale_pp_v"]
        ),
    )

    vga_config = config["agc_vga"]
    if vga_config.get("model", "ideal_frequency_flat_vga") != "ideal_frequency_flat_vga":
        raise ValueError("this baseline requires AGC/VGA model: ideal_frequency_flat_vga")
    if vga_config.get("control_mode", "worst_case_pulse_bound") != "worst_case_pulse_bound":
        raise ValueError("this baseline requires AGC control_mode: worst_case_pulse_bound")
    vga_target_fraction = float(vga_config["target_adc_full_scale_fraction"])
    vga_worst_case_input_peak_v = worst_case_pam_waveform_peak_v(
        afe_stage_pulse_v,
        samples_per_ui=samples_per_ui,
        maximum_symbol_magnitude=3.0,
    )
    vga_gain_db, vga_requested_gain_db, vga_gain_limited = peak_target_gain_db(
        vga_worst_case_input_peak_v,
        output_half_scale_v=adc.maximum_input_v,
        target_fraction=vga_target_fraction,
        minimum_gain_db=float(vga_config["minimum_gain_db"]),
        maximum_gain_db=float(vga_config["maximum_gain_db"]),
    )
    vga = flat_vga_response(
        frequency_hz,
        gain_db=vga_gain_db,
        differential_z0_ohm=float(afe.differential_z0_ohm),
    )
    vga_gain_linear = float(abs(vga.transfer[0]))
    vga_target_peak_v = vga_target_fraction * adc.maximum_input_v
    vga_stage_pulse_v = afe_stage_pulse_v * vga_gain_linear
    vga_eye = eye_diagram_from_symbol_pulse(
        eye_symbols,
        vga_stage_pulse_v,
        sampling_index=afe_stage_peak_index,
        samples_per_ui=samples_per_ui,
        trace_count=eye_trace_count,
        guard_symbols=eye_guard_symbols,
        interpolation_samples_per_ui=eye_interpolation_samples_per_ui,
    )

    adc_eye_result = adc.quantize(vga_eye.traces_v)
    adc_center_result = adc.quantize(vga_eye.center_samples_v)
    adc_center_clipping_fraction = float(np.mean(adc_center_result.clipped))
    adc_eye_grid_clipping_fraction = float(np.mean(adc_eye_result.clipped))
    adc_input_peak_v = float(np.max(np.abs(vga_eye.traces_v)))
    maximum_no_clip_gain = adc.maximum_input_v / adc_input_peak_v

    tx_waveform_v, tx_center_samples_v = _interior_waveform_segment(
        eye_symbols,
        tx_stage_pulse_v,
        sampling_index=tx_stage_peak_index,
        samples_per_ui=samples_per_ui,
        guard_symbols=eye_guard_symbols,
    )
    channel_waveform_v, channel_center_samples_v = _interior_waveform_segment(
        eye_symbols,
        channel_stage_pulse_v,
        sampling_index=channel_stage_peak_index,
        samples_per_ui=samples_per_ui,
        guard_symbols=eye_guard_symbols,
    )
    afe_waveform_v, afe_center_samples_v = _interior_waveform_segment(
        eye_symbols,
        afe_stage_pulse_v,
        sampling_index=afe_stage_peak_index,
        samples_per_ui=samples_per_ui,
        guard_symbols=eye_guard_symbols,
    )
    vga_waveform_v = afe_waveform_v * vga_gain_linear
    vga_center_samples_v = afe_center_samples_v * vga_gain_linear
    noiseless_full_adc_result = adc.quantize(vga_center_samples_v)

    noise_config = config["noise"]
    if noise_config.get("psd_sidedness", "one_sided") != "one_sided":
        raise ValueError("noise PSD must be configured as one_sided")
    afe_noise_density_v = (
        float(noise_config["afe_input_referred_density_nv_per_sqrt_hz"]) * 1e-9
    )
    source_noise_density_v = (
        float(noise_config["source_port_density_nv_per_sqrt_hz"]) * 1e-9
    )
    afe_output_noise_psd = output_noise_psd_one_sided(
        total_transfer=channel_afe.transfer,
        afe_transfer=afe.transfer,
        source_port_density_v_per_sqrt_hz=source_noise_density_v,
        afe_input_density_v_per_sqrt_hz=afe_noise_density_v,
    )
    adc_input_afe_noise_psd = afe_output_noise_psd * vga_gain_linear**2
    jitter_config = config["sampling_jitter"]
    if jitter_config.get("model") != "independent_gaussian_aperture_jitter":
        raise ValueError("unsupported sampling-jitter model")
    if jitter_config.get("voltage_error_model") != "first_order_local_signal_slope":
        raise ValueError("unsupported aperture-jitter voltage-error model")
    code_noise_enabled = bool(noise_config["code_path_injection_enabled"])
    code_impairment_note = (
        "AFE noise + ADC noise + "
        f"{float(jitter_config['rms_fs']):g} fs RJ + quantization"
        if code_noise_enabled
        else "signal + quantization only"
    )
    impairments = apply_sampled_adc_impairments(
        vga_waveform_v,
        samples_per_ui=samples_per_ui,
        sample_rate_hz=samples_per_ui * symbol_rate_hz,
        afe_noise_frequency_hz=frequency_hz,
        afe_noise_psd_v2_per_hz=(
            adc_input_afe_noise_psd
            if code_noise_enabled
            else np.zeros_like(adc_input_afe_noise_psd)
        ),
        additional_adc_noise_rms_v=(
            float(noise_config["additional_adc_input_referred_noise_rms_mv"])
            * 1e-3
            if code_noise_enabled
            else 0.0
        ),
        aperture_jitter_rms_s=(
            float(jitter_config["rms_fs"]) * 1e-15
            if code_noise_enabled
            else 0.0
        ),
        random_seed=int(noise_config["random_seed"]),
    )
    if not np.allclose(impairments.noiseless_samples_v, vga_center_samples_v):
        raise RuntimeError("ADC impairment samples are not aligned to symbol centers")
    full_adc_result = adc.quantize(impairments.impaired_samples_v)

    def papr_record(
        node: str,
        reference_plane: str,
        sampling: str,
        samples_v: np.ndarray,
        *,
        noise_included: bool = False,
        jitter_included: bool = False,
    ) -> dict:
        result = voltage_papr(samples_v)
        return {
            "node": node,
            "reference_plane": reference_plane,
            "sampling": sampling,
            "sample_count": result.sample_count,
            "mean_v": result.mean_v,
            "rms_v": result.rms_v,
            "peak_abs_v": result.peak_abs_v,
            "crest_factor": result.crest_factor,
            "papr_linear": result.papr_linear,
            "papr_db": result.papr_db,
            "mean_removed": False,
            "noise_included": noise_included,
            "jitter_included": jitter_included,
        }

    continuous_sampling = f"{samples_per_ui} samples/UI continuous grid"
    papr_records = [
        papr_record(
            "TX output",
            "loaded differential matched channel input",
            continuous_sampling,
            tx_waveform_v,
        ),
        papr_record(
            "Channel output",
            "matched differential channel output before AFE",
            continuous_sampling,
            channel_waveform_v,
        ),
        papr_record(
            "CTLE/AFE output",
            "differential CTLE/AFE voltage output before VGA",
            continuous_sampling,
            afe_waveform_v,
        ),
        papr_record(
            "VGA / ADC input",
            "differential VGA output at ADC input",
            continuous_sampling,
            vga_waveform_v,
        ),
        papr_record(
            "ADC input decision samples",
            "differential VGA output sampled at one sample/UI",
            "1 sample/UI at main cursor",
            vga_center_samples_v,
        ),
        papr_record(
            "ADC reconstructed output",
            "signed voltage-equivalent reconstruction of ADC output codes",
            "1 sample/UI at main cursor",
            noiseless_full_adc_result.reconstructed_v,
        ),
        papr_record(
            "Impaired ADC input samples",
            "differential ADC input including stochastic impairments",
            "1 sample/UI at jittered decision point",
            impairments.impaired_samples_v,
            noise_included=code_noise_enabled,
            jitter_included=code_noise_enabled,
        ),
        papr_record(
            "Impaired ADC output",
            "voltage-equivalent reconstruction of impaired ADC codes",
            "1 sample/UI at decision point",
            full_adc_result.reconstructed_v,
            noise_included=code_noise_enabled,
            jitter_included=code_noise_enabled,
        ),
    ]

    code_ffe_config = config["code_domain_ffe"]
    if code_ffe_config.get("input_representation") != (
        "unsigned ADC codes with midscale removal"
    ):
        raise ValueError("code-domain FFE requires unsigned ADC codes with midscale removal")
    code_ffe_symbols = eye_symbols[eye_guard_symbols:-eye_guard_symbols]
    if code_ffe_symbols.size != full_adc_result.codes.size:
        raise RuntimeError("ADC codes and code-domain FFE symbols are misaligned")
    code_ffe = train_code_domain_ffe(
        full_adc_result.codes,
        code_ffe_symbols,
        code_midpoint=0.5 * (adc.code_count - 1),
        tap_count=int(code_ffe_config["tap_count"]),
        training_fraction=float(code_ffe_config["training_fraction"]),
        ridge_fraction=float(code_ffe_config["ridge_fraction"]),
    )
    code_ffe_zero_error_upper_95 = _zero_error_upper_bound_95(
        code_ffe.empirical_error_count, code_ffe.test_indices.size
    )

    code_dfe_config = config["code_domain_dfe"]
    code_dfe = train_code_domain_dfe(
        code_ffe,
        feedback_tap_count=int(code_dfe_config["feedback_tap_count"]),
        ridge_fraction=float(code_dfe_config["ridge_fraction"]),
    )
    code_dfe_zero_symbol_error_upper_95 = _zero_error_upper_bound_95(
        code_dfe.decision_directed_symbol_error_count,
        code_dfe.test_symbols.size,
    )
    code_dfe_zero_bit_error_upper_95 = _zero_error_upper_bound_95(
        code_dfe.decision_directed_bit_error_count,
        2 * code_dfe.test_symbols.size,
    )
    if fec_enabled:
        if fec_stream is None:
            raise RuntimeError("enabled FEC stream was not generated")
        fec_result = evaluate_rs544_fec(
            fec_stream,
            code_dfe.decision_directed_decisions,
            code_dfe.test_indices + eye_guard_symbols,
        )
    else:
        fec_result = None

    decision_config = config["decision_point"]
    if decision_config.get("equalizer", "symbol_spaced_mmse_ffe") != "symbol_spaced_mmse_ffe":
        raise ValueError("this baseline requires equalizer: symbol_spaced_mmse_ffe")
    pulse_main_index = int(np.argmax(np.abs(pulse)))
    decision_point = evaluate_decision_point(
        pulse,
        pulse_main_index=pulse_main_index,
        samples_per_ui=samples_per_ui,
        symbol_period_s=symbol_period_s,
        pam4_unit_input_v=pam4_unit_input_v,
        noise_frequency_hz=frequency_hz,
        one_sided_noise_psd_v2_per_hz=afe_output_noise_psd,
        channel_pre_ui=int(decision_config["channel_pre_ui"]),
        channel_post_ui=int(decision_config["channel_post_ui"]),
        ffe_tap_count=int(decision_config["ffe_tap_count"]),
        pattern_tap_count=int(decision_config["pattern_tap_count"]),
        optimize_sampling_phase=bool(decision_config["optimize_sampling_phase"]),
    )
    target_der = float(decision_config["target_der"])
    required_dp_snr_db = required_pam4_dp_snr_db(target_der)
    implementation_loss_db = float(decision_config["implementation_loss_db"])
    estimated_dp_snr_db = decision_point.dp_snr_db - implementation_loss_db
    dp_snr_margin_db = estimated_dp_snr_db - required_dp_snr_db

    output_config = config["output"]
    output_directory = Path(output_config["directory"])
    if not output_directory.is_absolute():
        output_directory = PROJECT_ROOT / output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    offsets_ui, cursor_taps, main_index = sample_cursor_taps(
        pulse,
        samples_per_ui=samples_per_ui,
        precursor_ui=int(output_config["pulse_pre_ui"]),
        postcursor_ui=int(output_config["pulse_post_ui"]),
    )
    nyquist_index = int(np.argmin(np.abs(frequency_hz - nyquist_hz)))
    main_cursor = float(cursor_taps[offsets_ui == 0][0])
    normalized_taps = cursor_taps / main_cursor
    metrics = {
        "configuration": str(args.config.resolve()),
        "symbol_rate_hz": symbol_rate_hz,
        "nyquist_frequency_hz": nyquist_hz,
        "samples_per_ui": samples_per_ui,
        "frequency_max_hz": max_frequency_hz,
        "differential_reference_ohm": channel.differential_z0_ohm,
        "gain_conventions": {
            "channel": channel.gain_kind,
            "afe": afe.gain_kind,
            "channel_afe": channel_afe.gain_kind,
            "tx_pulse": tx_response.gain_kind,
            "signal_path": signal_path.gain_kind,
        },
        "nyquist": {
            "channel_sdd21_db": float(channel.magnitude_db[nyquist_index]),
            "afe_av_port_db": float(afe.magnitude_db[nyquist_index]),
            "channel_afe_db": float(channel_afe.magnitude_db[nyquist_index]),
            "tx_pulse_shaping_db": float(tx_response.magnitude_db[nyquist_index]),
            "signal_path_db": float(signal_path.magnitude_db[nyquist_index]),
            "signal_path_group_delay_ps": float(
                signal_path.group_delay_s[nyquist_index] * 1e12
            ),
        },
        "pulse_response": {
            "main_cursor": main_cursor,
            "cursor_offsets_ui": offsets_ui.tolist(),
            "cursor_values": cursor_taps.tolist(),
            "cursor_values_normalized": normalized_taps.tolist(),
        },
        "tx_channel_stages": {
            "afe_included": False,
            "rx_equalizer_included": False,
            "pam4_unit_voltage_v": pam4_unit_input_v,
            "tx_reference_plane": "loaded differential channel input",
            "channel_output_reference_plane": "matched differential channel output before AFE",
            "zoh_magnitude_db_at_nyquist": float(
                20.0
                * np.log10(
                    max(abs(zoh_response[nyquist_index]), np.finfo(float).tiny)
                )
            ),
            "tx_edge_filter_db_at_nyquist": float(
                tx_response.magnitude_db[nyquist_index]
            ),
            "tx_symbol_spectrum_db_at_nyquist": float(
                20.0
                * np.log10(
                    max(
                        abs(tx_symbol_spectrum[nyquist_index]),
                        np.finfo(float).tiny,
                    )
                )
            ),
            "channel_output_symbol_spectrum_db_at_nyquist": float(
                20.0
                * np.log10(
                    max(
                        abs(channel_symbol_spectrum[nyquist_index]),
                        np.finfo(float).tiny,
                    )
                )
            ),
            "tx_pulse_peak_v_per_pam_unit": float(
                tx_stage_pulse_v[tx_stage_peak_index]
            ),
            "channel_pulse_peak_v_per_pam_unit": float(
                channel_stage_pulse_v[channel_stage_peak_index]
            ),
            "channel_peak_delay_ps": float(staged_peak_delay_s * 1e12),
        },
        "ctle_afe_stage": {
            "gain_kind": afe.gain_kind,
            "voltage_reference": afe.metadata["voltage_reference"],
            "differential_input_reference_ohm": afe.differential_z0_ohm,
            "matched_channel_termination_assumed": True,
            "complex_response_used_for_pulse_and_eye": True,
            "afe_noise_included": False,
            "adc_included": False,
            "digital_rx_equalizer_included": False,
            "dc_gain_db": float(afe.magnitude_db[0]),
            "magnitude_db_at_nyquist": float(afe.magnitude_db[nyquist_index]),
            "phase_deg_at_nyquist": float(
                np.rad2deg(afe.phase_rad[nyquist_index])
            ),
            "group_delay_ps_at_nyquist": float(
                afe.group_delay_s[nyquist_index] * 1e12
            ),
            "afe_output_symbol_spectrum_db_at_nyquist": float(
                20.0
                * np.log10(
                    max(
                        abs(afe_output_symbol_spectrum[nyquist_index]),
                        np.finfo(float).tiny,
                    )
                )
            ),
            "afe_output_pulse_peak_v_per_pam_unit": float(
                afe_stage_pulse_v[afe_stage_peak_index]
            ),
            "incremental_peak_delay_from_channel_output_ps": float(
                afe_incremental_peak_delay_s * 1e12
            ),
        },
        "eye_diagram": {
            "reference_planes": [
                "loaded differential channel input",
                "matched differential channel output before AFE",
                "differential CTLE/AFE output",
            ],
            "symbol_count": eye_symbol_count,
            "trace_count": int(tx_eye.traces_v.shape[0]),
            "random_seed": eye_random_seed,
            "guard_symbols": eye_guard_symbols,
            "duration_ui": 2.0,
            "interpolation_samples_per_ui": eye_interpolation_samples_per_ui,
            "sampling_alignment": "main symbol-pulse cursor independently at each plane",
            "same_symbol_sequence_at_both_planes": True,
            "same_symbol_sequence_at_all_planes": True,
            "afe_included_at_afe_output_plane": True,
            "afe_complex_amplitude_and_phase_included": True,
            "noise_included": False,
            "jitter_included": False,
            "adc_included": False,
            "rx_equalizer_included": False,
            "tx_center_sample_min_v": float(np.min(tx_eye.center_samples_v)),
            "tx_center_sample_max_v": float(np.max(tx_eye.center_samples_v)),
            "channel_center_sample_min_v": float(
                np.min(channel_eye.center_samples_v)
            ),
            "channel_center_sample_max_v": float(
                np.max(channel_eye.center_samples_v)
            ),
            "afe_center_sample_min_v": float(np.min(afe_eye.center_samples_v)),
            "afe_center_sample_max_v": float(np.max(afe_eye.center_samples_v)),
        },
        "vga_agc": {
            "model": "ideal_frequency_flat_vga",
            "gain_kind": vga.gain_kind,
            "voltage_reference": vga.metadata["voltage_reference"],
            "control_mode": "worst_case_pulse_bound",
            "frequency_emphasis_included": False,
            "phase_shift_included": False,
            "group_delay_s": 0.0,
            "target_adc_full_scale_fraction": vga_target_fraction,
            "adc_target_peak_v": vga_target_peak_v,
            "pre_vga_worst_case_peak_bound_v": vga_worst_case_input_peak_v,
            "requested_gain_db": vga_requested_gain_db,
            "applied_gain_db": vga_gain_db,
            "applied_gain_linear": vga_gain_linear,
            "minimum_gain_db": float(vga_config["minimum_gain_db"]),
            "maximum_gain_db": float(vga_config["maximum_gain_db"]),
            "gain_limited": vga_gain_limited,
            "noise_added": False,
            "bandwidth_limit_included": False,
        },
        "adc": {
            "model": "effective_uniform_midrise",
            "enob_bits": adc.enob_bits,
            "enob_interpretation": "ideal-equivalent uniform resolution; configured additional ADC noise is treated as an independent term beyond quantization",
            "input_reference": adc_config["input_reference"],
            "differential_full_scale_pp_v": adc.differential_full_scale_pp_v,
            "minimum_input_v": adc.minimum_input_v,
            "maximum_input_v": adc.maximum_input_v,
            "code_count": adc.code_count,
            "unsigned_code_range": [0, adc.code_count - 1],
            "effective_lsb_v": adc.effective_lsb_v,
            "effective_quantization_noise_rms_v": adc.quantization_noise_rms_v,
            "ideal_full_scale_sine_snr_db": adc.full_scale_sine_snr_db,
            "sample_rate_hz": symbol_rate_hz,
            "samples_per_ui": 1,
            "sampling_phase": "VGA output at AFE main-cursor timing",
            "automatic_gain_control_included": bool(
                adc_config["automatic_gain_control_included"]
            ),
            "analog_noise_included_in_adc_stage_plot": False,
            "code_path_analog_noise_included": code_noise_enabled,
            "code_path_aperture_jitter_included": code_noise_enabled,
            "dnl_inl_included": False,
            "main_cursor_sample_count": int(adc_center_result.codes.size),
            "main_cursor_clipped_sample_count": int(
                np.count_nonzero(adc_center_result.clipped)
            ),
            "main_cursor_clipping_fraction": adc_center_clipping_fraction,
            "eye_grid_clipping_fraction": adc_eye_grid_clipping_fraction,
            "pre_vga_minimum_observed_v": float(np.min(afe_eye.traces_v)),
            "pre_vga_maximum_observed_v": float(np.max(afe_eye.traces_v)),
            "minimum_observed_input_v": float(np.min(vga_eye.traces_v)),
            "maximum_observed_input_v": float(np.max(vga_eye.traces_v)),
            "minimum_observed_code": int(np.min(adc_center_result.codes)),
            "maximum_observed_code": int(np.max(adc_center_result.codes)),
            "maximum_scalar_gain_for_observed_no_clip": maximum_no_clip_gain,
            "maximum_scalar_gain_for_observed_no_clip_db": float(
                20.0 * np.log10(maximum_no_clip_gain)
            ),
            "impaired_code_path_sample_count": int(full_adc_result.codes.size),
            "impaired_code_path_clipped_sample_count": int(
                np.count_nonzero(full_adc_result.clipped)
            ),
            "impaired_code_path_clipping_fraction": float(
                np.mean(full_adc_result.clipped)
            ),
            "impaired_code_path_minimum_code": int(np.min(full_adc_result.codes)),
            "impaired_code_path_maximum_code": int(np.max(full_adc_result.codes)),
            "digital_equalizer_included": False,
            "current_dpsnr_der_metrics_include_adc": False,
        },
        "papr": {
            "definition": "max(v^2) / mean(v^2)",
            "mean_removed": False,
            "per_record_impairment_flags": True,
            "symbol_count": eye_symbol_count,
            "guard_symbols_excluded_at_each_end": eye_guard_symbols,
            "analog_samples_per_ui": samples_per_ui,
            "records": papr_records,
        },
        "code_domain_ffe": {
            "input_representation": code_ffe_config["input_representation"],
            "adc_quantization_included": True,
            "adc_clipping_included": True,
            "afe_analog_noise_included": code_noise_enabled,
            "additional_adc_noise_included": code_noise_enabled,
            "jitter_included": code_noise_enabled,
            "tap_count": int(code_ffe.tap_offsets_ui.size),
            "tap_offsets_ui": code_ffe.tap_offsets_ui.tolist(),
            "coefficients_per_centered_adc_code": (
                code_ffe.coefficients_per_code.tolist()
            ),
            "hardware_interpretation": "symmetric analysis offsets realizable with output latency of half the tap span",
            "code_midpoint_removed": code_ffe.code_midpoint,
            "training_fraction_requested": float(
                code_ffe_config["training_fraction"]
            ),
            "training_sample_count": int(code_ffe.training_indices.size),
            "held_out_test_sample_count": int(code_ffe.test_indices.size),
            "train_test_boundary_guard": "one half-tap span on each side",
            "ridge_fraction": float(code_ffe_config["ridge_fraction"]),
            "ridge_lambda": code_ffe.ridge_lambda,
            "decision_gain": code_ffe.decision_gain,
            "decision_offset": code_ffe.decision_offset,
            "slicer_thresholds": code_ffe.slicer_thresholds.tolist(),
            "signal_variance": code_ffe.signal_variance,
            "residual_error_variance": code_ffe.residual_error_variance,
            "dp_snr_definition": "5*g^2 / mean((FFE_output - offset - g*target_symbol)^2) on held-out data",
            "dp_snr_db": code_ffe.dp_snr_db,
            "gaussian_der": code_ffe.gaussian_der,
            "empirical_der": code_ffe.empirical_der,
            "empirical_error_count": code_ffe.empirical_error_count,
            "zero_error_der_upper_bound_95": code_ffe_zero_error_upper_95,
            "zero_error_bound_assumption": "independent binomial decisions",
            "confusion_matrix_rows_true_columns_detected": (
                code_ffe.confusion_matrix.tolist()
            ),
            "raw_adc_dp_snr_db": code_ffe.raw_dp_snr_db,
            "raw_adc_gaussian_der": code_ffe.raw_gaussian_der,
            "raw_adc_empirical_der": code_ffe.raw_empirical_der,
            "linear_reference_dp_snr_db": decision_point.dp_snr_db,
            "linear_reference_comparison_note": "not apples-to-apples: linear reference includes AFE noise but excludes ADC quantization, added ADC noise, jitter, and DFE",
        },
        "code_domain_dfe": {
            "location": "after code-domain FFE and before PAM4 slicer decision feedback",
            "feedback_tap_count": int(code_dfe.feedback_delays_ui.size),
            "feedback_delays_ui": code_dfe.feedback_delays_ui.tolist(),
            "feedback_coefficients": code_dfe.feedback_coefficients.tolist(),
            "precursor_cancellation": False,
            "postcursor_cancellation": True,
            "training_mode": code_dfe_config["training_mode"],
            "training_sample_count": code_dfe.training_sample_count,
            "ridge_fraction": float(code_dfe_config["ridge_fraction"]),
            "ridge_lambda": code_dfe.ridge_lambda,
            "decision_gain": code_dfe.decision_gain,
            "decision_offset": code_dfe.decision_offset,
            "slicer_thresholds": code_dfe.slicer_thresholds.tolist(),
            "held_out_test_sample_count_after_warmup": int(
                code_dfe.test_symbols.size
            ),
            "warmup_sample_count_excluded": int(
                code_dfe.feedback_delays_ui.size
            ),
            "gray_mapping_low_to_high": ["00", "01", "11", "10"],
            "baseline_ffe": {
                "dp_snr_db": code_dfe.baseline_dp_snr_db,
                "gaussian_der": code_dfe.baseline_gaussian_der,
                "gaussian_ber_nearest_neighbor_approx": (
                    code_dfe.baseline_gaussian_ber_approx
                ),
                "empirical_der": code_dfe.baseline_empirical_der,
                "empirical_ber": code_dfe.baseline_empirical_ber,
                "symbol_error_count": code_dfe.baseline_symbol_error_count,
                "bit_error_count": code_dfe.baseline_bit_error_count,
                "empirical_ber_wilson_interval_95": list(
                    _wilson_interval_95(
                        code_dfe.baseline_bit_error_count,
                        2 * code_dfe.test_symbols.size,
                    )
                ),
            },
            "genie_aided_reference": {
                "uses_true_past_symbols": True,
                "dp_snr_db": code_dfe.genie_dp_snr_db,
                "gaussian_der": code_dfe.genie_gaussian_der,
                "gaussian_ber_nearest_neighbor_approx": (
                    code_dfe.genie_gaussian_ber_approx
                ),
                "empirical_der": code_dfe.genie_empirical_der,
                "empirical_ber": code_dfe.genie_empirical_ber,
                "symbol_error_count": code_dfe.genie_symbol_error_count,
                "bit_error_count": code_dfe.genie_bit_error_count,
                "empirical_ber_wilson_interval_95": list(
                    _wilson_interval_95(
                        code_dfe.genie_bit_error_count,
                        2 * code_dfe.test_symbols.size,
                    )
                ),
            },
            "decision_directed": {
                "uses_only_previous_decisions": True,
                "dp_snr_db": code_dfe.decision_directed_dp_snr_db,
                "gaussian_der": code_dfe.decision_directed_gaussian_der,
                "gaussian_ber_nearest_neighbor_approx": (
                    code_dfe.decision_directed_gaussian_ber_approx
                ),
                "empirical_der": code_dfe.decision_directed_empirical_der,
                "empirical_ber": code_dfe.decision_directed_empirical_ber,
                "symbol_error_count": (
                    code_dfe.decision_directed_symbol_error_count
                ),
                "bit_error_count": code_dfe.decision_directed_bit_error_count,
                "empirical_ber_wilson_interval_95": list(
                    _wilson_interval_95(
                        code_dfe.decision_directed_bit_error_count,
                        2 * code_dfe.test_symbols.size,
                    )
                ),
                "zero_symbol_error_der_upper_bound_95": (
                    code_dfe_zero_symbol_error_upper_95
                ),
                "zero_bit_error_ber_upper_bound_95": (
                    code_dfe_zero_bit_error_upper_95
                ),
                "zero_error_bound_assumption": "independent binomial opportunities",
                "confusion_matrix_rows_true_columns_detected": (
                    code_dfe.decision_directed_confusion_matrix.tolist()
                ),
                "decisions_different_from_genie_reference": (
                    code_dfe.error_propagation_symbol_count
                ),
            },
            "ber_interpretation": "empirical rate is measured on the held-out stochastic realization; Gaussian BER remains an extrapolation from residual variance",
            "afe_analog_noise_included": code_noise_enabled,
            "additional_adc_noise_included": code_noise_enabled,
            "jitter_included": code_noise_enabled,
        },
        "transmitter": {
            "modulation": "PAM4",
            "normalized_levels": [-3, -1, 1, 3],
            "differential_outer_pp_voltage_v": differential_outer_pp_v,
            "average_delivered_power_dbm": average_delivered_power_dbm,
            "differential_port_rms_voltage_v": float(differential_port_rms_voltage_v),
            "pam4_unit_input_voltage_v": pam4_unit_input_v,
            "outer_level_peak_voltage_v": 3.0 * pam4_unit_input_v,
            "voltage_reference": transmitter_config["voltage_reference"],
            "pulse_shaping": tx_response.metadata,
        },
        "noise": {
            "psd_sidedness": "one_sided",
            "code_path_injection_enabled": code_noise_enabled,
            "random_seed": int(noise_config["random_seed"]),
            "afe_input_referred_density_v_per_sqrt_hz": afe_noise_density_v,
            "source_port_density_v_per_sqrt_hz": source_noise_density_v,
            "afe_noise_reference": noise_config["afe_noise_reference"],
            "receiver_termination_noise_included_in_afe_density": bool(
                noise_config["receiver_termination_noise_included_in_afe_density"]
            ),
            "adc_input_load_noise_included": bool(
                noise_config["adc_input_load_noise_included"]
            ),
            "additional_adc_input_referred_noise_rms_v": float(
                noise_config["additional_adc_input_referred_noise_rms_mv"]
            )
            * 1e-3,
            "additional_adc_noise_reference": noise_config[
                "additional_adc_noise_reference"
            ],
            "reference_temperature_k": float(noise_config["reference_temperature_k"]),
            "integrated_afe_output_noise_before_vga_rms_v": float(
                np.sqrt(
                    np.sum(
                        0.5
                        * (afe_output_noise_psd[1:] + afe_output_noise_psd[:-1])
                        * np.diff(frequency_hz)
                    )
                )
            ),
            "expected_afe_noise_at_adc_input_after_vga_rms_v": (
                impairments.expected_afe_noise_rms_v
            ),
            "observed_afe_noise_at_symbol_samples_rms_v": (
                impairments.observed_afe_noise_rms_v
            ),
            "observed_additional_adc_noise_rms_v": (
                impairments.observed_additional_adc_noise_rms_v
            ),
            "observed_total_stochastic_impairment_rms_v": (
                impairments.observed_total_impairment_rms_v
            ),
        },
        "sampling_jitter": {
            "model": jitter_config["model"],
            "rms_s": float(jitter_config["rms_fs"]) * 1e-15,
            "rms_ui": float(jitter_config["rms_fs"])
            * 1e-15
            / symbol_period_s,
            "voltage_error_model": jitter_config["voltage_error_model"],
            "slope_reference": "noiseless differential signal at the VGA output",
            "observed_aperture_jitter_rms_s": float(
                np.sqrt(np.mean(impairments.aperture_jitter_s**2))
            ),
            "observed_jitter_error_rms_v": (
                impairments.observed_jitter_error_rms_v
            ),
        },
        "decision_point": {
            "equalizer": "symbol_spaced_mmse_ffe",
            "sampling_phase_samples": decision_point.sampling_phase_samples,
            "sampling_phase_ui": decision_point.sampling_phase_ui,
            "ffe_tap_count": int(decision_point.ffe_offsets_ui.size),
            "ffe_offsets_ui": decision_point.ffe_offsets_ui.tolist(),
            "ffe_coefficients_per_v": decision_point.ffe_coefficients_per_v.tolist(),
            "equalized_cursor_offsets_ui": decision_point.equalized_cursor_offsets_ui.tolist(),
            "equalized_cursor_values": decision_point.equalized_cursor.tolist(),
            "main_cursor": decision_point.main_cursor,
            "signal_variance": decision_point.signal_variance,
            "residual_isi_variance": decision_point.residual_isi_variance,
            "output_noise_variance": decision_point.output_noise_variance,
            "dp_snr_db": decision_point.dp_snr_db,
            "target_der": target_der,
            "required_dp_snr_db": required_dp_snr_db,
            "implementation_loss_db": implementation_loss_db,
            "estimated_dp_snr_db": estimated_dp_snr_db,
            "dp_snr_margin_db": dp_snr_margin_db,
            "gaussian_der": decision_point.gaussian_der,
            "pattern_conditioned_der": decision_point.pattern_conditioned_der,
            "pattern_tap_offsets_ui": decision_point.pattern_tap_offsets_ui.tolist(),
            "pattern_tap_values": decision_point.pattern_tap_values.tolist(),
            "gaussian_remainder_variance": decision_point.gaussian_remainder_variance,
            "slicer_thresholds": decision_point.slicer_thresholds.tolist(),
            "slicer_threshold_policy": "symmetric_midpoints",
            "der_scope": "ideal PAM4 detector; no DFE error propagation or FEC",
            "adc_quantization_and_clipping_included": False,
        },
        "channel_metadata": channel.metadata,
        "afe_metadata": afe.metadata,
        "channel_afe_metadata": channel_afe.metadata,
        "signal_path_metadata": signal_path.metadata,
    }
    if fec_result is not None:
        target_post_fec_ber = float(fec_config["target_post_fec_ber"])
        required_pre_fec_ber = required_pre_fec_ber_for_iid_post_ber(
            target_post_fec_ber
        )
        metrics["fec"] = {
            "enabled": True,
            "kind": "RS(544,514)",
            "location": "after decision-directed DFE hard decisions",
            "evaluated_codewords_were_encoded_before_channel": True,
            "finite_stream_prefix_suffix_are_random_alignment_filler": True,
            "systematic_encoding": True,
            "field": "GF(2^10)",
            "field_symbol_bits": RS_SYMBOL_BITS,
            "codeword_symbols": RS_N_SYMBOLS,
            "payload_symbols": RS_K_SYMBOLS,
            "parity_symbols": RS_PARITY_SYMBOLS,
            "correctable_symbol_errors": RS_CORRECTABLE_SYMBOL_ERRORS,
            "primitive_polynomial_decimal": RS_PRIMITIVE_POLYNOMIAL,
            "first_consecutive_root": 0,
            "generator_polynomial_coefficients_after_leading_one": (
                RS_GENERATOR_COEFFICIENTS.tolist()
            ),
            "code_rate": RS_K_SYMBOLS / RS_N_SYMBOLS,
            "parity_overhead_relative_to_payload": (
                RS_PARITY_SYMBOLS / RS_K_SYMBOLS
            ),
            "pam4_symbols_per_codeword": RS_N_SYMBOLS * RS_SYMBOL_BITS // 2,
            "codeword_time_s": (
                RS_N_SYMBOLS * RS_SYMBOL_BITS / 2.0 / symbol_rate_hz
            ),
            "net_payload_rate_bit_per_s_excluding_other_pcs_overhead": (
                2.0 * symbol_rate_hz * RS_K_SYMBOLS / RS_N_SYMBOLS
            ),
            "alignment_offset_pam4_symbols": int(
                fec_config["alignment_offset_pam4_symbols"]
            ),
            "bit_order_within_fec_symbol": fec_config[
                "bit_order_within_fec_symbol"
            ],
            "pam4_gray_mapping_low_to_high": fec_config[
                "pam4_gray_mapping_low_to_high"
            ],
            "evaluated_complete_codeword_count": (
                fec_result.evaluated_codeword_count
            ),
            "received_codeword_bit_count": (
                fec_result.received_codeword_bit_count
            ),
            "payload_bit_count": fec_result.payload_bit_count,
            "pre_fec_bit_error_count": fec_result.pre_fec_bit_error_count,
            "pre_fec_rs_symbol_error_count": (
                fec_result.pre_fec_symbol_error_count
            ),
            "pre_fec_ber": fec_result.pre_fec_ber,
            "pre_fec_rs_symbol_error_rate": (
                fec_result.pre_fec_symbol_error_rate
            ),
            "corrected_codeword_count": fec_result.corrected_codeword_count,
            "uncorrectable_codeword_count": (
                fec_result.uncorrectable_codeword_count
            ),
            "miscorrected_codeword_count": (
                fec_result.miscorrected_codeword_count
            ),
            "post_fec_payload_bit_error_count": (
                fec_result.post_fec_payload_bit_error_count
            ),
            "post_fec_ber": fec_result.post_fec_ber,
            "target_post_fec_ber": target_post_fec_ber,
            "iid_required_pre_fec_ber_for_target": required_pre_fec_ber,
            "measured_pre_fec_ber_divided_by_required": (
                fec_result.pre_fec_ber / required_pre_fec_ber
            ),
            "post_fec_zero_error_ber_upper_bound_95": (
                _zero_error_upper_bound_95(
                    fec_result.post_fec_payload_bit_error_count,
                    fec_result.payload_bit_count,
                )
            ),
            "measured_codeword_failure_rate": (
                fec_result.codeword_failure_rate
            ),
            "zero_uncorrectable_codeword_rate_upper_bound_95": (
                _zero_error_upper_bound_95(
                    fec_result.uncorrectable_codeword_count,
                    fec_result.evaluated_codeword_count,
                )
            ),
            "iid_random_error_estimate": {
                "assumption": "independent bit errors grouped into 10-bit RS symbols; successful words correct perfectly; failed words pass through; miscorrection excluded",
                "rs_symbol_error_probability": (
                    fec_result.iid_symbol_error_probability
                ),
                "uncorrectable_codeword_probability": (
                    fec_result.iid_uncorrectable_codeword_probability
                ),
                "failed_codeword_passthrough_ber": (
                    fec_result.iid_failed_codeword_passthrough_ber_estimate
                ),
            },
            "codewords": [
                {
                    "codeword_index": word.codeword_index,
                    "start_pam4_symbol_index": word.start_pam4_symbol_index,
                    "pre_fec_bit_errors": word.pre_fec_bit_errors,
                    "pre_fec_rs_symbol_errors": word.pre_fec_symbol_errors,
                    "decoder_success": word.decoder_success,
                    "decoder_reported_corrections": (
                        word.decoder_reported_corrections
                    ),
                    "post_fec_payload_bit_errors": (
                        word.post_fec_payload_bit_errors
                    ),
                }
                for word in fec_result.codewords
            ],
        }

    frequency_csv = output_directory / "frequency_response.csv"
    stage_frequency_csv = output_directory / "tx_channel_frequency_response.csv"
    stage_pulse_csv = output_directory / "tx_channel_pulse_response.csv"
    eye_center_samples_csv = output_directory / "tx_channel_eye_center_samples.csv"
    ctle_afe_frequency_csv = output_directory / "ctle_afe_frequency_response.csv"
    ctle_afe_pulse_csv = output_directory / "ctle_afe_pulse_response.csv"
    three_stage_eye_samples_csv = (
        output_directory / "tx_channel_afe_eye_center_samples.csv"
    )
    vga_frequency_csv = output_directory / "vga_frequency_response.csv"
    adc_center_samples_csv = output_directory / "adc_center_samples.csv"
    adc_impairment_samples_csv = output_directory / "adc_impairment_samples.csv"
    papr_csv = output_directory / "papr_summary.csv"
    code_ffe_taps_csv = output_directory / "code_domain_ffe_taps.csv"
    code_ffe_test_csv = output_directory / "code_domain_ffe_test_samples.csv"
    code_dfe_taps_csv = output_directory / "code_domain_dfe_taps.csv"
    code_dfe_test_csv = output_directory / "code_domain_dfe_test_samples.csv"
    fec_codeword_csv = output_directory / "fec_codeword_summary.csv"
    decision_point_csv = output_directory / "decision_point.csv"
    metrics_json = output_directory / "metrics.json"
    summary_png = output_directory / "channel_afe_summary.png"
    decision_point_png = output_directory / "decision_point_summary.png"
    stage_summary_png = output_directory / "tx_channel_stage_summary.png"
    eye_summary_png = output_directory / "tx_channel_eye_summary.png"
    ctle_afe_summary_png = output_directory / "ctle_afe_summary.png"
    three_stage_eye_png = output_directory / "tx_channel_afe_eye_summary.png"
    adc_summary_png = output_directory / "adc_summary.png"
    adc_impairment_summary_png = output_directory / "adc_impairment_summary.png"
    papr_summary_png = output_directory / "papr_summary.png"
    code_ffe_summary_png = output_directory / "code_domain_ffe_summary.png"
    code_dfe_summary_png = output_directory / "code_domain_dfe_summary.png"
    fec_summary_png = output_directory / "fec_summary.png"
    _write_frequency_csv(
        frequency_csv, channel, afe, channel_afe, tx_response, signal_path
    )
    _write_decision_point_csv(decision_point_csv, decision_point)
    _write_stage_frequency_csv(
        stage_frequency_csv,
        frequency_hz=frequency_hz,
        tx_symbol_spectrum=tx_symbol_spectrum,
        channel_symbol_spectrum=channel_symbol_spectrum,
        channel=channel,
    )
    _write_stage_pulse_csv(
        stage_pulse_csv,
        tx_time_ui=tx_stage_time_ui,
        tx_pulse_v=tx_stage_pulse_v,
        channel_time_ui=channel_stage_time_ui,
        channel_pulse_v=channel_stage_pulse_v,
    )
    _write_eye_center_samples_csv(eye_center_samples_csv, tx_eye, channel_eye)
    _write_ctle_afe_frequency_csv(
        ctle_afe_frequency_csv,
        frequency_hz=frequency_hz,
        afe=afe,
        channel_symbol_spectrum=channel_symbol_spectrum,
        afe_output_symbol_spectrum=afe_output_symbol_spectrum,
    )
    _write_ctle_afe_pulse_csv(
        ctle_afe_pulse_csv,
        time_ui=afe_stage_time_ui,
        pulse_v=afe_stage_pulse_v,
    )
    _write_eye_center_samples_csv(
        three_stage_eye_samples_csv, tx_eye, channel_eye, afe_eye
    )
    _write_vga_frequency_csv(vga_frequency_csv, vga=vga)
    _write_adc_center_samples_csv(
        adc_center_samples_csv,
        afe_eye=afe_eye,
        adc_input_eye=vga_eye,
        adc_result=adc_center_result,
    )
    _write_adc_impairment_samples_csv(
        adc_impairment_samples_csv,
        impairments=impairments,
        adc_result=full_adc_result,
    )
    _write_papr_csv(papr_csv, papr_records)
    _write_code_ffe_taps_csv(code_ffe_taps_csv, code_ffe)
    _write_code_ffe_test_csv(
        code_ffe_test_csv,
        result=code_ffe,
        adc_codes=full_adc_result.codes,
    )
    _write_code_dfe_taps_csv(code_dfe_taps_csv, code_dfe)
    _write_code_dfe_test_csv(code_dfe_test_csv, result=code_dfe)
    if fec_result is not None:
        _write_fec_codeword_csv(fec_codeword_csv, result=fec_result)
    with metrics_json.open("w") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    _plot_summary(
        summary_png,
        channel=channel,
        afe=afe,
        channel_afe=channel_afe,
        tx=tx_response,
        signal_path=signal_path,
        impulse_time_s=impulse_time_s,
        impulse=impulse,
        pulse_time_s=pulse_time_s,
        pulse=pulse,
        main_index=main_index,
        symbol_period_s=symbol_period_s,
        plot_max_hz=float(output_config["plot_max_ghz"]) * 1e9,
        pulse_pre_ui=int(output_config["pulse_pre_ui"]),
        pulse_post_ui=int(output_config["pulse_post_ui"]),
    )
    _plot_decision_point(decision_point_png, decision_point, metrics)
    _plot_tx_channel_stages(
        stage_summary_png,
        frequency_hz=frequency_hz,
        tx_edge_response=tx_response.transfer,
        tx_symbol_spectrum=tx_symbol_spectrum,
        channel_symbol_spectrum=channel_symbol_spectrum,
        channel=channel,
        tx_time_ui=tx_stage_time_ui,
        tx_pulse_v=tx_stage_pulse_v,
        channel_time_ui=channel_stage_time_ui,
        channel_pulse_v=channel_stage_pulse_v,
        differential_outer_pp_v=differential_outer_pp_v,
        nyquist_hz=nyquist_hz,
        plot_max_hz=float(output_config["plot_max_ghz"]) * 1e9,
        pulse_pre_ui=float(output_config["pulse_pre_ui"]),
        pulse_post_ui=float(output_config["pulse_post_ui"]),
    )
    _plot_tx_channel_eyes(
        eye_summary_png,
        tx_eye=tx_eye,
        channel_eye=channel_eye,
        pam4_unit_voltage_v=pam4_unit_input_v,
        channel_dc_gain=float(abs(channel.transfer[0])),
        differential_outer_pp_v=differential_outer_pp_v,
        symbol_rate_hz=symbol_rate_hz,
        channel_loss_db_at_nyquist=float(channel.magnitude_db[nyquist_index]),
    )
    _plot_ctle_afe_stage(
        ctle_afe_summary_png,
        frequency_hz=frequency_hz,
        afe=afe,
        channel_symbol_spectrum=channel_symbol_spectrum,
        afe_output_symbol_spectrum=afe_output_symbol_spectrum,
        channel_time_ui=channel_stage_time_ui,
        channel_pulse_v=channel_stage_pulse_v,
        afe_time_ui=afe_stage_time_ui,
        afe_pulse_v=afe_stage_pulse_v,
        nyquist_hz=nyquist_hz,
        plot_max_hz=float(output_config["plot_max_ghz"]) * 1e9,
        pulse_pre_ui=float(output_config["pulse_pre_ui"]),
        pulse_post_ui=float(output_config["pulse_post_ui"]),
    )
    _plot_tx_channel_afe_eyes(
        three_stage_eye_png,
        tx_eye=tx_eye,
        channel_eye=channel_eye,
        afe_eye=afe_eye,
        tx_unit_voltage_v=pam4_unit_input_v,
        channel_dc_unit_voltage_v=pam4_unit_input_v * abs(channel.transfer[0]),
        afe_dc_unit_voltage_v=pam4_unit_input_v * abs(signal_path.transfer[0]),
        differential_outer_pp_v=differential_outer_pp_v,
        symbol_rate_hz=symbol_rate_hz,
        channel_loss_db_at_nyquist=float(channel.magnitude_db[nyquist_index]),
    )
    _plot_adc_stage(
        adc_summary_png,
        adc_input_eye=vga_eye,
        adc=adc,
        adc_eye_result=adc_eye_result,
        adc_center_result=adc_center_result,
        vga=vga,
        vga_worst_case_input_peak_v=vga_worst_case_input_peak_v,
        vga_target_peak_v=vga_target_peak_v,
        frequency_hz=frequency_hz,
        plot_max_hz=float(output_config["plot_max_ghz"]) * 1e9,
        symbol_rate_hz=symbol_rate_hz,
    )
    _plot_adc_impairment_summary(
        adc_impairment_summary_png,
        frequency_hz=frequency_hz,
        adc_input_afe_noise_psd=adc_input_afe_noise_psd,
        impairments=impairments,
        additional_adc_noise_rms_v=float(
            noise_config["additional_adc_input_referred_noise_rms_mv"]
        )
        * 1e-3,
        aperture_jitter_rms_s=float(jitter_config["rms_fs"]) * 1e-15,
    )
    _plot_papr_summary(papr_summary_png, records=papr_records)
    _plot_code_ffe_summary(
        code_ffe_summary_png,
        result=code_ffe,
        symbol_rate_hz=symbol_rate_hz,
        impairment_note=code_impairment_note,
    )
    _plot_code_dfe_summary(
        code_dfe_summary_png,
        result=code_dfe,
        impairment_note=code_impairment_note,
    )
    if fec_result is not None:
        _plot_fec_summary(
            fec_summary_png,
            result=fec_result,
            symbol_rate_hz=symbol_rate_hz,
        )

    print(f"Channel Sdd21 at Nyquist: {metrics['nyquist']['channel_sdd21_db']:.3f} dB")
    print(f"AFE Av_port at Nyquist:    {metrics['nyquist']['afe_av_port_db']:.3f} dB")
    print(f"TX shaping at Nyquist:     {metrics['nyquist']['tx_pulse_shaping_db']:.3f} dB")
    print(f"Signal path at Nyquist:    {metrics['nyquist']['signal_path_db']:.3f} dB")
    print(f"Signal-path group delay:   {metrics['nyquist']['signal_path_group_delay_ps']:.3f} ps")
    print(f"dpSNR:                    {decision_point.dp_snr_db:.3f} dB")
    print(f"Required dpSNR:           {required_dp_snr_db:.3f} dB")
    print(f"dpSNR margin:             {dp_snr_margin_db:.3f} dB")
    print(f"Gaussian DER:             {decision_point.gaussian_der:.3e}")
    print(f"Pattern-conditioned DER: {decision_point.pattern_conditioned_der:.3e}")
    print(
        "TX pulse peak / PAM unit: "
        f"{metrics['tx_channel_stages']['tx_pulse_peak_v_per_pam_unit'] * 1e3:.3f} mV"
    )
    print(
        "Channel pulse peak:       "
        f"{metrics['tx_channel_stages']['channel_pulse_peak_v_per_pam_unit'] * 1e3:.3f} mV"
    )
    print(
        "CTLE group delay @ Nyq:   "
        f"{metrics['ctle_afe_stage']['group_delay_ps_at_nyquist']:.3f} ps"
    )
    print(
        "AFE pulse peak:           "
        f"{metrics['ctle_afe_stage']['afe_output_pulse_peak_v_per_pam_unit'] * 1e3:.3f} mV"
    )
    print(f"Flat VGA gain:            {vga_gain_db:.3f} dB")
    print(
        "VGA worst-case target:    "
        f"{vga_worst_case_input_peak_v * 1e3:.3f} -> "
        f"{vga_target_peak_v * 1e3:.3f} mV peak"
    )
    print(f"ADC effective LSB:        {adc.effective_lsb_v * 1e3:.3f} mV")
    print(
        "ADC center clipping:      "
        f"{100.0 * adc_center_clipping_fraction:.2f}%"
    )
    for record in papr_records:
        print(f"PAPR {record['node']:<28} {record['papr_db']:.3f} dB")
    print(f"Raw ADC-code dpSNR:       {code_ffe.raw_dp_snr_db:.3f} dB")
    print(f"Code-domain FFE dpSNR:    {code_ffe.dp_snr_db:.3f} dB")
    print(f"Raw ADC empirical DER:    {code_ffe.raw_empirical_der:.3e}")
    print(f"Code-domain FFE DER:      {code_ffe.empirical_der:.3e}")
    print(
        "DFE decision-directed dpSNR: "
        f"{code_dfe.decision_directed_dp_snr_db:.3f} dB"
    )
    print(
        "DFE Gaussian BER estimate:   "
        f"{code_dfe.decision_directed_gaussian_ber_approx:.3e}"
    )
    print(
        "DFE measured bit errors:     "
        f"{code_dfe.decision_directed_bit_error_count}/"
        f"{2 * code_dfe.test_symbols.size}"
    )
    if fec_result is not None:
        print(
            "RS-FEC measured result:      "
            f"{fec_result.pre_fec_bit_error_count} pre-FEC errors -> "
            f"{fec_result.post_fec_payload_bit_error_count} post-FEC errors "
            f"across {fec_result.evaluated_codeword_count} codewords"
        )
        print(
            "RS-FEC IID post-BER estimate: "
            f"{fec_result.iid_failed_codeword_passthrough_ber_estimate:.3e}"
        )
        print(
            "Pre-FEC BER needed for 1e-12: "
            f"{required_pre_fec_ber_for_iid_post_ber(float(fec_config['target_post_fec_ber'])):.3e}"
        )
    print(f"Wrote {frequency_csv}")
    print(f"Wrote {decision_point_csv}")
    print(f"Wrote {stage_frequency_csv}")
    print(f"Wrote {stage_pulse_csv}")
    print(f"Wrote {eye_center_samples_csv}")
    print(f"Wrote {ctle_afe_frequency_csv}")
    print(f"Wrote {ctle_afe_pulse_csv}")
    print(f"Wrote {three_stage_eye_samples_csv}")
    print(f"Wrote {vga_frequency_csv}")
    print(f"Wrote {adc_center_samples_csv}")
    print(f"Wrote {adc_impairment_samples_csv}")
    print(f"Wrote {papr_csv}")
    print(f"Wrote {code_ffe_taps_csv}")
    print(f"Wrote {code_ffe_test_csv}")
    print(f"Wrote {code_dfe_taps_csv}")
    print(f"Wrote {code_dfe_test_csv}")
    if fec_result is not None:
        print(f"Wrote {fec_codeword_csv}")
    print(f"Wrote {metrics_json}")
    print(f"Wrote {summary_png}")
    print(f"Wrote {decision_point_png}")
    print(f"Wrote {stage_summary_png}")
    print(f"Wrote {eye_summary_png}")
    print(f"Wrote {ctle_afe_summary_png}")
    print(f"Wrote {three_stage_eye_png}")
    print(f"Wrote {adc_summary_png}")
    print(f"Wrote {adc_impairment_summary_png}")
    print(f"Wrote {papr_summary_png}")
    print(f"Wrote {code_ffe_summary_png}")
    print(f"Wrote {code_dfe_summary_png}")
    if fec_result is not None:
        print(f"Wrote {fec_summary_png}")


if __name__ == "__main__":
    main()
