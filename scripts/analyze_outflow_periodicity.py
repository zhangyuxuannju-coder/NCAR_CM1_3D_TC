#!/usr/bin/env python3
"""Estimate outflow periodicity after removing intensity/RMW low-frequency state."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import coherence, csd, lombscargle, savgol_filter, welch


METRICS = {
    "outflow_flux_300km_kg_s": "M300",
    "outflow_flux_600km_kg_s": "M600",
    "outflow_max_ur_200_1200_ms": "ur_max",
}


def _odd_window(value: int, n: int) -> int:
    value = min(int(value), n if n % 2 else n - 1)
    value = max(value, 5)
    return value if value % 2 else value - 1


def _fill_nan(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    good = np.isfinite(values)
    if good.all():
        return values
    if good.sum() < 3:
        raise ValueError("fewer than three finite values")
    idx = np.arange(values.size)
    return np.interp(idx, idx[good], values[good])


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    values = _fill_nan(values)
    return savgol_filter(values, _odd_window(window, values.size), 2, mode="interp")


def _huber_fit(y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Iteratively reweighted regression with standardized predictors."""
    scale = np.nanstd(x[:, 1:], axis=0)
    scale[scale == 0] = 1.0
    xx = x.copy()
    xx[:, 1:] = (xx[:, 1:] - np.nanmean(xx[:, 1:], axis=0)) / scale
    beta = np.linalg.lstsq(xx, y, rcond=None)[0]
    for _ in range(10):
        residual = y - xx @ beta
        mad = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        mad = max(float(mad), 1.0e-8)
        weights = np.minimum(1.0, 1.5 * mad / np.maximum(np.abs(residual), 1.0e-12))
        lhs = xx.T @ (weights[:, None] * xx)
        rhs = xx.T @ (weights * y)
        beta = np.linalg.solve(lhs + 1.0e-10 * np.eye(lhs.shape[0]), rhs)
    fitted = xx @ beta
    return fitted, beta


def _residualize(times: np.ndarray, rows: Dict[str, np.ndarray], metric: str, window: int):
    y = np.log(np.maximum(_fill_nan(rows[metric]), 1.0e-8)) if metric != "outflow_max_ur_200_1200_ms" else _fill_nan(rows[metric])
    v = _smooth(rows["vmax_ms"], window)
    rmw = _smooth(rows["rmw_km"], window)
    x = np.column_stack([np.ones(times.size), v, v * v, rmw])
    fitted, _ = _huber_fit(y, x)
    return y, fitted, y - fitted


def _ar1_threshold(residual: np.ndarray, freq: np.ndarray, args, rng):
    x = residual - np.mean(residual)
    phi = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if x.size > 3 else 0.0
    phi = float(np.clip(phi, -0.95, 0.95))
    innovation = float(np.std(x[1:] - phi * x[:-1]))
    powers = []
    for _ in range(args.surrogates):
        surrogate = np.empty_like(x)
        surrogate[0] = rng.normal(0.0, np.std(x))
        surrogate[1:] = phi * surrogate[:-1] + rng.normal(0.0, innovation, x.size - 1)
        nseg = min(args.segment_hours, x.size)
        sf, sp = welch(surrogate, fs=1.0, window="hann", nperseg=nseg, nfft=max(256, 8 * nseg),
                       noverlap=nseg // 2, detrend="linear")
        powers.append(np.interp(freq, sf, sp, left=np.nan, right=np.nan))
    return phi, np.nanpercentile(np.asarray(powers), 95, axis=0), np.nanmax(powers, axis=1)


def _spectrum(residual: np.ndarray, args, rng):
    nperseg = min(args.segment_hours, residual.size)
    freq, power = welch(residual, fs=1.0, window="hann", nperseg=nperseg, nfft=max(256, 8 * nperseg),
                        noverlap=nperseg // 2, detrend="linear")
    period = np.divide(1.0, freq, out=np.full_like(freq, np.inf), where=freq > 0)
    band = (freq > 0) & (period >= args.period_min) & (period <= args.period_max)
    if not np.any(band):
        raise ValueError("no Welch bins in requested period band")
    bf, bp = freq[band], power[band]
    phi, red95, surrogate_max = _ar1_threshold(residual, bf, args, rng)
    ip = int(np.nanargmax(bp))
    peak_period = float(1.0 / bf[ip])
    peak_power = float(bp[ip])
    red_at_peak = float(red95[ip])
    return {
        "frequency_cph": bf.tolist(), "period_h": (1.0 / bf).tolist(), "power": bp.tolist(),
        "peak_period_h": peak_period, "peak_frequency_cpd": 24.0 / peak_period,
        "peak_power": peak_power, "red_noise_95_at_peak": red_at_peak,
        "pointwise_significant": bool(peak_power > red_at_peak),
        "global_significant": bool(peak_power > np.nanpercentile(surrogate_max, 95)),
        "ar1_phi": phi,
    }


def _lomb_peak(times, residual, args, mask):
    t = times[mask]
    x = residual[mask] - np.mean(residual[mask])
    freq = np.linspace(1.0 / args.period_max, 1.0 / args.period_min, 800)
    power = lombscargle(t - np.mean(t), x, 2.0 * np.pi * freq, normalize=True)
    i = int(np.nanargmax(power))
    return {"period_h": float(1.0 / freq[i]), "frequency_cpd": float(24.0 * freq[i]), "power": float(power[i])}


def _cross_diagnostics(x, y, args, peak_period):
    nperseg = min(args.segment_hours, x.size)
    freq, coh = coherence(x, y, fs=1.0, window="hann", nperseg=nperseg, nfft=max(256, 8 * nperseg), noverlap=nperseg // 2)
    valid = (freq > 0) & (1.0 / freq >= args.period_min) & (1.0 / freq <= args.period_max)
    if np.any(valid):
        i = np.where(valid)[0][int(np.argmin(np.abs(1.0 / freq[valid] - peak_period)))]
        coherence_at_peak = float(coh[i])
    else:
        coherence_at_peak = np.nan
    lags = np.arange(-24, 25)
    correlations = []
    for lag in lags:
        if lag < 0:
            a, b = x[-lag:], y[:lag]
        elif lag > 0:
            a, b = x[:-lag], y[lag:]
        else:
            a, b = x, y
        correlations.append(float(np.corrcoef(a, b)[0, 1]) if a.size > 3 else np.nan)
    i = int(np.nanargmax(correlations))
    return {"coherence_at_peak": coherence_at_peak, "lag_h_600_minus_300": int(lags[i]),
            "lag_correlation": float(correlations[i]), "lags_h": lags.tolist(),
            "lag_correlation_values": correlations}


def _read_rows(intensity_path: Path, outflow_path: Path, start: float, end: float):
    with intensity_path.open(encoding="utf-8") as handle:
        intensity_records = list(csv.DictReader(handle))
    with outflow_path.open(encoding="utf-8") as handle:
        outflow_records = list(csv.DictReader(handle))
    intensity = {(r["case"], float(r["time_h"])): r for r in intensity_records}
    outflow = {float(r["time_h"]): r for r in outflow_records}
    times = np.array(sorted(t for t in outflow if start <= t <= end))
    result = {}
    for case in ("CTRL", "JET"):
        if not all((case, t) in intensity for t in times):
            raise ValueError(f"{case} intensity series does not cover the requested time window")
        result[case] = {
            "vmax_ms": np.array([float(intensity[(case, t)]["vmax_ms"]) for t in times]),
            "rmw_km": np.array([float(intensity[(case, t)]["rmw_km"]) for t in times]),
        }
        for key in METRICS:
            result[case][key] = np.array([float(outflow[t][f"{case}_{key}"]) for t in times])
    return times, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intensity-timeseries", type=Path, default=Path("output/jet_outflow_mechanism_validation/timeseries.csv"))
    parser.add_argument("--outflow-timeseries", type=Path, default=Path("output/jet_outflow_mechanism_validation/outflow_intensity_timeseries.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/jet_outflow_mechanism_validation/periodicity"))
    parser.add_argument("--start-hour", type=float, default=40.0)
    parser.add_argument("--end-hour", type=float, default=150.0)
    parser.add_argument("--trend-windows", default="19,25,37")
    parser.add_argument("--segment-hours", type=int, default=48)
    parser.add_argument("--period-min", type=float, default=4.0)
    parser.add_argument("--period-max", type=float, default=32.0)
    parser.add_argument("--surrogates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figdir = args.output_dir / "figures"; figdir.mkdir(exist_ok=True)
    times, cases = _read_rows(args.intensity_timeseries, args.outflow_timeseries, args.start_hour, args.end_hour)
    windows = [int(x) for x in args.trend_windows.split(",")]
    rng = np.random.default_rng(args.seed)
    all_metrics = []; stored = {}; main_window = 25
    for case in ("CTRL", "JET"):
        for metric in METRICS:
            stored[(case, metric)] = {}
            for window in windows:
                observed, fitted, residual = _residualize(times, cases[case], metric, window)
                spec = _spectrum(residual, args, rng)
                mask = ~((times >= 72.0) & (times <= 82.0))
                sensitivity = _lomb_peak(times, residual, args, mask)
                stored[(case, metric)][window] = {"observed": observed, "fitted": fitted, "residual": residual, "spectrum": spec, "gap_sensitivity": sensitivity}
                all_metrics.append({"case": case, "metric": metric, "metric_label": METRICS[metric], "trend_window_h": window, **{k: v for k, v in spec.items() if not isinstance(v, list)}, "gap_sensitivity_period_h": sensitivity["period_h"]})
    summary = []
    for case in ("CTRL", "JET"):
        for metric in METRICS:
            peaks = np.array([stored[(case, metric)][w]["spectrum"]["peak_period_h"] for w in windows])
            sig = np.array([stored[(case, metric)][w]["spectrum"]["pointwise_significant"] for w in windows])
            summary.append({"case": case, "metric": metric, "metric_label": METRICS[metric], "robust_peak_period_h": float(np.median(peaks)), "peak_period_q25_h": float(np.percentile(peaks, 25)), "peak_period_q75_h": float(np.percentile(peaks, 75)), "significant_window_count": int(sig.sum()), "n_windows": len(windows), "robust": bool(sig.sum() >= 2 and (np.max(peaks) - np.min(peaks)) / max(np.median(peaks), 1.0) <= 0.25)})
    with (args.output_dir / "periodicity_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_metrics[0])); writer.writeheader(); writer.writerows(all_metrics)
    with (args.output_dir / "periodicity_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    with (args.output_dir / "detrended_timeseries.csv").open("w", newline="", encoding="utf-8") as handle:
        names = ["time_h"] + [f"{case}_{metric}_observed" for case in ("CTRL", "JET") for metric in METRICS] + [f"{case}_{metric}_trend" for case in ("CTRL", "JET") for metric in METRICS] + [f"{case}_{metric}_residual" for case in ("CTRL", "JET") for metric in METRICS]
        writer = csv.DictWriter(handle, fieldnames=names); writer.writeheader()
        for i, time in enumerate(times):
            row = {"time_h": time}
            for case in ("CTRL", "JET"):
                for metric in METRICS:
                    q = stored[(case, metric)][main_window]; row[f"{case}_{metric}_observed"] = q["observed"][i]; row[f"{case}_{metric}_trend"] = q["fitted"][i]; row[f"{case}_{metric}_residual"] = q["residual"][i]
            writer.writerow(row)
    cross_results = {}
    for case in ("CTRL", "JET"):
        x = stored[(case, "outflow_flux_300km_kg_s")][main_window]["residual"]
        y = stored[(case, "outflow_flux_600km_kg_s")][main_window]["residual"]
        peak = stored[(case, "outflow_flux_300km_kg_s")][main_window]["spectrum"]["peak_period_h"]
        cross_results[case] = _cross_diagnostics(x, y, args, peak)
    with (args.output_dir / "cross_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(cross_results, handle, indent=2, ensure_ascii=False)
    with (args.output_dir / "periodicity_report.json").open("w", encoding="utf-8") as handle:
        json.dump({"window_h": [args.start_hour, args.end_hour], "trend_windows_h": windows, "period_band_h": [args.period_min, args.period_max], "segment_hours": args.segment_hours, "summary": summary, "cross_metrics": cross_results, "method": "log mass-flux residualized against smoothed Vmax, Vmax squared and RMW; robust Huber regression; Welch spectrum; AR1 surrogates; Lomb-Scargle sensitivity excluding 72-82 h"}, handle, indent=2, ensure_ascii=False)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for case, color in (("CTRL", "#222222"), ("JET", "#b2182b")):
        q = stored[(case, "outflow_flux_300km_kg_s")][main_window]
        axes[0, 0].plot(times, np.exp(q["observed"]) / 1e9, color=color, alpha=.45, label=f"{case} observed")
        axes[0, 0].plot(times, np.exp(q["fitted"]) / 1e9, color=color, lw=2, label=f"{case} intensity/RMW fit")
        axes[0, 1].plot(times, q["residual"], color=color, lw=1.5, label=case)
        spec = stored[(case, "outflow_flux_300km_kg_s")][main_window]["spectrum"]
        axes[1, 0].plot(spec["period_h"], spec["power"], color=color, lw=2, label=case)
    for ax in axes.flat:
        ax.axvspan(50, 75, color="#fdae61", alpha=.10); ax.axvspan(80, 150, color="#2c7bb6", alpha=.07); ax.grid(alpha=.25)
    axes[0, 0].set(xlabel="Time (h)", ylabel="M300 (10^9 kg s$^{-1}$)", title="Observed outflow and fitted intensity/RMW background")
    axes[0, 1].set(xlabel="Time (h)", ylabel="log residual", title="Outflow residual after removing low-frequency state")
    axes[1, 0].set(xlabel="Period (h)", ylabel="Welch power", title="M300 residual spectrum (25 h trend window)"); axes[1, 0].set_xlim(args.period_min, args.period_max); axes[1, 0].legend(frameon=False)
    for case, color in (("CTRL", "#222222"), ("JET", "#b2182b")):
        q = stored[(case, "outflow_flux_300km_kg_s")][main_window]["residual"]; ac = np.correlate(q - q.mean(), q - q.mean(), mode="full")[q.size - 1:]; ac = ac / ac[0]; lag = np.arange(ac.size)
        axes[1, 1].plot(lag, ac, color=color, label=case)
    axes[1, 1].set(xlabel="Lag (h)", ylabel="ACF", title="Residual autocorrelation"); axes[1, 1].set_xlim(0, 48); axes[1, 1].legend(frameon=False)
    fig.savefig(figdir / "outflow_periodicity_summary.png", dpi=220); fig.savefig(figdir / "outflow_periodicity_summary.pdf"); plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
