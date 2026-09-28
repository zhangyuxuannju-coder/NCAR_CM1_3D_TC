#!/usr/bin/env python3
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path("output/jet_outflow_mechanism_validation/periodicity")
INFILE = BASE / "detrended_timeseries.csv"
FIGDIR = BASE / "figures"

METRICS = [
    ("outflow_flux_300km_kg_s", "M300", "300 km outflow mass flux"),
    ("outflow_flux_600km_kg_s", "M600", "600 km outflow mass flux"),
    ("outflow_max_ur_200_1200_ms", "ur_max", "max radial wind, 200–1200 km"),
]
COLORS = {"CTRL": "#303030", "JET": "#b2182b"}


def fft_spectrum(x):
    x = np.asarray(x, float)
    x = x - np.nanmean(x)
    n = x.size
    # Hann taper reduces the finite-window leakage in the short record.
    taper = np.hanning(n)
    z = np.fft.rfft(x * taper)
    f = np.fft.rfftfreq(n, d=1.0)  # cycles per hour
    p = np.abs(z) ** 2
    return f, p


def read_data():
    with INFILE.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def main():
    rows = read_data()
    times = np.array([float(r["time_h"]) for r in rows])
    out = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), sharex=True, sharey=True,
                             constrained_layout=True)
    for i, (metric, label, title) in enumerate(METRICS):
        for j, case in enumerate(("CTRL", "JET")):
            ax = axes[j, i]
            prefix = f"{case}_{metric}"
            series = {
                "observed": np.array([float(r[f"{prefix}_observed"]) for r in rows]),
                "intensity/RMW fit": np.array([float(r[f"{prefix}_trend"]) for r in rows]),
                "residual": np.array([float(r[f"{prefix}_residual"]) for r in rows]),
            }
            for name, x in series.items():
                f, p = fft_spectrum(x)
                valid = (f > 0) & (1.0 / f >= 4.0) & (1.0 / f <= 80.0)
                period = 1.0 / f[valid]
                power = p[valid]
                order = np.argsort(period)
                period, power = period[order], power[order]
                # Normalize within plotted band so curves compare by spectral shape.
                power = power / np.sum(power)
                style = {"observed": ("#8c8c8c", "-", 1.2),
                         "intensity/RMW fit": ("#f28e2b", "--", 1.2),
                         "residual": (COLORS[case], "-", 2.0)}[name]
                ax.plot(period, power, color=style[0], ls=style[1], lw=style[2],
                        label=name)
                if name == "residual":
                    band = (period >= 4) & (period <= 32)
                    if np.any(band):
                        k = np.where(band)[0][np.argmax(power[band])]
                        peak = float(period[k])
                        ax.axvline(peak, color=COLORS[case], lw=1.0, alpha=.75)
                        ax.text(peak, ax.get_ylim()[1] * .82, f"{peak:.1f} h",
                                color=COLORS[case], rotation=90, ha="right", va="top",
                                fontsize=9)
                        out.append({"case": case, "metric": label,
                                    "residual_fft_peak_period_h": peak,
                                    "residual_fft_peak_frequency_cpd": 24.0 / peak})
            ax.set_title(f"{case}: {title}")
            ax.set_xlim(4, 80)
            ax.set_yscale("log")
            ax.grid(alpha=.25, which="both")
            if j == 1:
                ax.set_xlabel("Period (h)")
            if i == 0:
                ax.set_ylabel("Normalized Fourier power")
            if i == 0 and j == 0:
                ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle("Fourier spectral decomposition of outflow signals\n"
                 "gray: observed; orange dashed: intensity/RMW fit; colored: residual",
                 fontsize=15)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "outflow_fourier_decomposition.png", dpi=220)
    fig.savefig(FIGDIR / "outflow_fourier_decomposition.pdf")
    with (BASE / "fourier_peaks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print("wrote", FIGDIR / "outflow_fourier_decomposition.png")


if __name__ == "__main__":
    main()
