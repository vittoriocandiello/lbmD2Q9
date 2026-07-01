#!/usr/bin/env python3
"""Dissipation and dispersion validation for the Gaussian-pulse solver."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = Path("output/gaussianpulse_probes.csv")
NPPW_MIN = 20.0
NPPW_MAX = 1000.0
SPECTRUM_CUTOFF = 1.0e-8


def load_probe_data(path):
    """Load a CSV whose header follows optional comment lines."""
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream):
            if line.strip() and not line.startswith("#"):
                names = [name.strip() for name in line.split(",")]
                break
        else:
            raise ValueError(f"{path} has no CSV header")

    data = np.genfromtxt(
        path,
        delimiter=",",
        names=names,
        skip_header=line_number + 1,
    )
    data = np.atleast_1d(data)
    if data.size < 2:
        raise ValueError(f"{path} does not contain enough samples")
    return data


def pressure_matrix(data, suffix, count):
    columns = []
    available = set(data.dtype.names or ())
    for probe in range(count):
        name = f"p_probe_{probe:02d}{suffix}"
        if name not in available:
            raise ValueError(f"missing column {name} in probe file")
        columns.append(np.asarray(data[name], dtype=float))
    return np.column_stack(columns)


def remove_reflections(data, metadata):
    """Discard samples that can contain a pulse reflected by either x wall."""
    time = np.asarray(data["t_phys"], dtype=float)
    probes = np.asarray(metadata["probe_x_positions_m"], dtype=float)
    dx = float(metadata["dx_m"])
    offset = int(metadata["aux_probe_offset_cells"]) * dx
    positions = np.concatenate((probes, probes - offset, probes + offset))
    x_min, x_max = map(float, metadata["domain_x_m"])
    x0 = float(metadata["x0_m"])
    c0 = float(metadata["sound_speed_m_per_s"])

    left_path = (x0 - x_min) + (positions - x_min)
    right_path = (x_max - x0) + (x_max - positions)
    reflection_time = min(np.min(left_path), np.min(right_path)) / c0

    # Stop six Gaussian half-widths before the reflected pulse centre arrives.
    safe_time = reflection_time - 6.0 * float(metadata["sigma_m"]) / c0
    keep = time < safe_time
    if np.count_nonzero(keep) < 2:
        raise ValueError("not enough samples before the first wall reflection")
    return data[keep], safe_time


def theoretical_coefficients(omega, metadata):
    """Exact viscous plane-wave attenuation and phase speed."""
    c0 = float(metadata["sound_speed_m_per_s"])
    nu = float(metadata["nu_physical"])

    # The validation model uses 4/3*nu_shear + nu_bulk, and D2Q9 BGK has
    # nu_bulk = nu_shear. The longitudinal viscosity is therefore 7/3*nu.
    relaxation = ((7.0 / 3.0) * nu) / (c0 * c0)
    omega_tau = omega * relaxation
    root = np.sqrt(1.0 + omega_tau * omega_tau)
    denominator = 1.0 + omega_tau * omega_tau

    alpha = (
        omega
        / (np.sqrt(2.0) * c0)
        * np.sqrt(
            (omega_tau * omega_tau)
            / ((root + 1.0) * denominator)
        )
    )
    sound_speed = (
        np.sqrt(2.0)
        * c0
        * np.sqrt(denominator / (root + 1.0))
    )
    return alpha, sound_speed


def compute_errors(data, metadata):
    probes = np.asarray(metadata["probe_x_positions_m"], dtype=float)
    count = probes.size
    dx = float(metadata["dx_m"])
    offset = int(metadata["aux_probe_offset_cells"])
    c0 = float(metadata["sound_speed_m_per_s"])

    main = pressure_matrix(data, "", count)
    minus = pressure_matrix(data, "_minus", count)
    plus = pressure_matrix(data, "_plus", count)
    main -= np.mean(main, axis=0)
    minus -= np.mean(minus, axis=0)
    plus -= np.mean(plus, axis=0)

    time = np.asarray(data["t_phys"], dtype=float)
    dt = float(np.mean(np.diff(time)))
    frequency = np.fft.rfftfreq(time.size, dt)
    omega = 2.0 * np.pi * frequency

    # Conjugation gives positive spatial phase for a right-travelling wave.
    main_hat = np.conj(np.fft.rfft(main, axis=0)).T
    minus_hat = np.conj(np.fft.rfft(minus, axis=0)).T
    plus_hat = np.conj(np.fft.rfft(plus, axis=0)).T

    alpha_theory, c_theory = theoretical_coefficients(omega, metadata)
    alpha_numerical = np.full(frequency.size, np.nan)
    c_numerical = np.full(frequency.size, np.nan)
    amplitude_floor = SPECTRUM_CUTOFF * np.max(np.abs(main_hat[0]))
    relative_x = probes - probes[0]
    derivative_distance = offset * dx

    for index in range(1, frequency.size):
        spectrum = main_hat[:, index]
        amplitude = np.abs(spectrum)
        valid = amplitude > amplitude_floor
        if np.count_nonzero(valid) < 3:
            continue

        slope = np.polyfit(
            relative_x[valid],
            np.log(amplitude[valid] / amplitude[valid][0]),
            1,
        )[0]
        alpha_numerical[index] = -slope

        derivative = (
            plus_hat[:, index] - minus_hat[:, index]
        ) / (2.0 * derivative_distance)
        wave_number = np.imag(derivative[valid] / spectrum[valid])
        wave_number = wave_number[np.isfinite(wave_number)]
        if wave_number.size and np.mean(wave_number) > 0.0:
            c_numerical[index] = omega[index] / np.mean(wave_number)

    with np.errstate(divide="ignore", invalid="ignore"):
        nppw = c0 / (frequency * dx)

    valid = (
        (nppw >= NPPW_MIN)
        & (nppw <= NPPW_MAX)
        & np.isfinite(alpha_numerical)
        & np.isfinite(c_numerical)
        & (alpha_theory > 0.0)
        & (c_theory > 0.0)
    )
    if not np.any(valid):
        raise ValueError("no valid frequencies remain for plotting")

    alpha_scale = np.max(alpha_theory[valid])
    c_scale = np.max(c_theory[valid])
    alpha_error = (
        100.0 * np.abs(alpha_numerical[valid] - alpha_theory[valid])
        / alpha_scale
    )
    c_error = (
        100.0 * np.abs(c_numerical[valid] - c_theory[valid])
        / c_scale
    )
    return nppw[valid], alpha_error, c_error


def plot_error(nppw, error, ylabel, title, path):
    valid = np.isfinite(error) & (error > 0.0)
    if np.count_nonzero(valid) < 2:
        raise ValueError(f"not enough positive values for {title}")

    x = nppw[valid]
    y = error[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    middle = len(x) // 2
    reference = y[middle] * (x / x[middle]) ** -2.0

    fig, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    axis.loglog(x, y, color="#0F4C81", linewidth=1.8, label="D2Q9 BGK")
    axis.loglog(
        x,
        reference,
        color="#6B7280",
        linestyle="--",
        linewidth=1.2,
        label=r"$NPPW^{-2}$",
    )
    axis.set_xlabel("Nodes per wavelength (NPPW)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def validate(input_path):
    metadata_path = input_path.with_name("gaussianpulse_metadata.json")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    data = load_probe_data(input_path)
    original_size = data.size
    data, safe_time = remove_reflections(data, metadata)
    nppw, alpha_error, c_error = compute_errors(data, metadata)

    alpha_path = input_path.parent / "alpha_error.png"
    c_path = input_path.parent / "c_error.png"
    plot_error(
        nppw,
        alpha_error,
        r"$\epsilon_\alpha$ [%]",
        "Dissipation Error",
        alpha_path,
    )
    plot_error(
        nppw,
        c_error,
        r"$\epsilon_c$ [%]",
        "Dispersion Error",
        c_path,
    )

    if data.size < original_size:
        print(
            f"Ignored wall reflections after t={safe_time:.6e} s "
            f"({original_size - data.size} samples)."
        )
    print(f"Saved: {alpha_path}")
    print(f"Saved: {c_path}")
    print(
        f"Retained {nppw.size} frequencies, "
        f"NPPW={np.min(nppw):.3g}..{np.max(nppw):.3g}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot Gaussian-pulse dissipation and dispersion errors."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"probe CSV (default: {DEFAULT_INPUT})",
    )
    arguments = parser.parse_args()

    try:
        validate(arguments.input)
    except (OSError, KeyError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
