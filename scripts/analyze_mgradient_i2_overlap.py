#!/usr/bin/env python3
"""Test whether Bui inertial-instability differences follow dM/dr.

The calculation uses the same instantaneous, storm-centred, azimuthally
averaged basic state for all quantities.  This avoids comparing instantaneous
Bui I2 with a separately time-averaged angular-momentum gradient.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
import numpy as np

from analyze_inertial_stability_attribution import analyze_case, read_pmin, stack


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nojet", required=True)
    p.add_argument("--jet", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--hours", type=float, nargs="+", default=[45, 55, 65, 72, 75])
    p.add_argument("--max-r-km", type=float, default=1200.0)
    p.add_argument("--dr-km", type=float, default=12.0)
    p.add_argument("--max-z-km", type=float, default=20.0)
    p.add_argument("--f", type=float, default=5.464e-5)
    p.add_argument("--eddy-average", choices=["reynolds", "favre"], default="reynolds")
    p.add_argument("--center-window", type=int, default=21)
    p.add_argument("--center-method", choices=["min", "mean"], default="min")
    p.add_argument("--target-hour", type=float, default=72.0)
    return p.parse_args()


def nature_diverging():
    return LinearSegmentedColormap.from_list(
        "nature_diverging", ["#3B6FB6", "#B9CCE3", "#F7F7F5", "#F0B6A6", "#B9363E"], N=256
    )


def cell_weights(r_km, z_km):
    r = np.asarray(r_km) * 1000.0
    z = np.asarray(z_km) * 1000.0
    dr = np.abs(np.gradient(r))
    dz = np.abs(np.gradient(z))
    return np.maximum(r, 1.0)[None, :] * dr[None, :] * dz[:, None]


def domain_mask(r_km, z_km):
    rr, zz = np.meshgrid(r_km, z_km)
    return (rr >= 50.0) & (rr <= 350.0) & (zz >= 10.0) & (zz <= 16.0)


def wsum(x, w, mask):
    use = mask & np.isfinite(x)
    return float(np.sum(w[use] * x[use]))


def wmean(x, w, mask):
    use = mask & np.isfinite(x)
    return float(np.sum(w[use] * x[use]) / np.sum(w[use]))


def wrms(x, w, mask):
    return float(np.sqrt(wmean(np.asarray(x) ** 2, w, mask)))


def wcorr(x, y, w, mask):
    use = mask & np.isfinite(x) & np.isfinite(y)
    ww = w[use]
    xx = x[use]
    yy = y[use]
    mx = np.sum(ww * xx) / np.sum(ww)
    my = np.sum(ww * yy) / np.sum(ww)
    num = np.sum(ww * (xx - mx) * (yy - my))
    den = np.sqrt(np.sum(ww * (xx - mx) ** 2) * np.sum(ww * (yy - my) ** 2))
    return float(num / den) if den > 0 else np.nan


def mask_metrics(i2, eta, xi, i2_vort, i2_baro, w, mask):
    a = (i2 < 0) & mask
    b = (eta < 0) & mask
    inter = a & b
    union = a | b
    total = wsum(np.ones_like(i2), w, mask)
    wa = wsum(np.ones_like(i2), w, a)
    wb = wsum(np.ones_like(i2), w, b)
    wi = wsum(np.ones_like(i2), w, inter)
    wu = wsum(np.ones_like(i2), w, union)
    agree = (((i2 < 0) == (eta < 0)) & mask)
    return {
        "i2_negative_fraction": wa / total,
        "m_gradient_negative_fraction": wb / total,
        "negative_jaccard": wi / wu if wu > 0 else np.nan,
        "i2_negative_captured_by_m_gradient": wi / wa if wa > 0 else np.nan,
        "m_gradient_negative_captured_by_i2": wi / wb if wb > 0 else np.nan,
        "sign_agreement": wsum(np.ones_like(i2), w, agree) / total,
        "xi_negative_fraction": wsum(np.ones_like(i2), w, (xi < 0) & mask) / total,
        "corr_i2_vorticity_component": wcorr(i2, i2_vort, w, mask),
        "baroclinic_rms_fraction": wrms(i2_baro, w, mask) / wrms(i2, w, mask),
    }


def overlap_category(i2, eta):
    # 0 both stable, 1 both unstable, 2 dM/dr-only, 3 Bui-I2-only
    out = np.zeros(i2.shape, dtype=np.int8)
    a = i2 < 0
    b = eta < 0
    out[a & b] = 1
    out[(~a) & b] = 2
    out[a & (~b)] = 3
    return out


def add_outflow(ax, r, z, ur):
    levels = [2.0, 5.0, 10.0]
    if np.nanmax(ur) >= levels[0]:
        cs = ax.contour(r, z, ur, levels=[x for x in levels if np.nanmax(ur) >= x],
                        colors="#159D73", linewidths=[0.8, 1.1, 1.4][:sum(np.nanmax(ur) >= x for x in levels)])
        ax.clabel(cs, fmt="%g", fontsize=7, inline=True)


def make_cross_section_figure(records, r, z, output, target_hour):
    k = int(np.argmin(np.abs(np.asarray([x["hour"] for x in records["noJET"]]) - target_hour)))
    actual_hour = records["noJET"][k]["hour"]
    fig, axes = plt.subplots(2, 3, figsize=(15.2, 7.4), sharex=True, sharey=True, constrained_layout=True)
    cmap = nature_diverging()
    cat_cmap = ListedColormap(["#ECECE8", "#B9363E", "#3B6FB6", "#E69F00"])
    cat_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cat_cmap.N)
    mgrad_fields = [x["eta"] * (r[None, :] * 1000.0) for x in (records["noJET"][k], records["JET"][k])]
    vmax_m = np.nanpercentile(np.abs(np.concatenate([x[:, (r >= 50) & (r <= 350)].ravel() for x in mgrad_fields])), 98)
    i2_fields = [records[c][k]["I2"] for c in ("noJET", "JET")]
    vmax_i = np.nanpercentile(np.abs(np.concatenate([x[:, (r >= 50) & (r <= 350)].ravel() for x in i2_fields])), 98)

    for row, case in enumerate(("noJET", "JET")):
        rec = records[case][k]
        mgrad = rec["eta"] * (r[None, :] * 1000.0)
        norm_m = TwoSlopeNorm(vmin=-vmax_m, vcenter=0.0, vmax=vmax_m)
        p0 = axes[row, 0].pcolormesh(r, z, mgrad, shading="auto", cmap=cmap, norm=norm_m)
        axes[row, 0].contour(r, z, rec["I2"], levels=[0], colors="k", linewidths=1.5)
        add_outflow(axes[row, 0], r, z, rec["ur"])
        axes[row, 0].set_title(f"{case}: $\\partial M/\\partial r$\nblack: Bui $I^2=0$")

        norm_i = TwoSlopeNorm(vmin=-vmax_i, vcenter=0.0, vmax=vmax_i)
        p1 = axes[row, 1].pcolormesh(r, z, rec["I2"], shading="auto", cmap=cmap, norm=norm_i)
        axes[row, 1].contour(r, z, mgrad, levels=[0], colors="#8B1A8B", linewidths=1.5)
        add_outflow(axes[row, 1], r, z, rec["ur"])
        axes[row, 1].set_title(f"{case}: raw Bui $I^2$\nmagenta: $\\partial M/\\partial r=0$")

        cat = overlap_category(rec["I2"], rec["eta"])
        p2 = axes[row, 2].pcolormesh(r, z, cat, shading="auto", cmap=cat_cmap, norm=cat_norm)
        add_outflow(axes[row, 2], r, z, rec["ur"])
        axes[row, 2].set_title(f"{case}: sign-overlap categories")

        for col in range(3):
            axes[row, col].set_xlim(50, 350)
            axes[row, col].set_ylim(10, 16)
            axes[row, col].set_xlabel("Radius (km)")
            axes[row, col].grid(color="0.75", linewidth=0.35, alpha=0.35)
        axes[row, 0].set_ylabel("Height (km)")

    cb0 = fig.colorbar(p0, ax=axes[:, 0], shrink=0.92, pad=0.01)
    cb0.set_label(r"$\partial M/\partial r$ (m s$^{-1}$)")
    cb1 = fig.colorbar(p1, ax=axes[:, 1], shrink=0.92, pad=0.01)
    cb1.set_label(r"Bui $I^2$ (s$^{-2}$)")
    cb2 = fig.colorbar(p2, ax=axes[:, 2], ticks=[0, 1, 2, 3], shrink=0.92, pad=0.01)
    cb2.ax.set_yticklabels(["both stable", "both unstable", "$M_r<0$ only", "$I^2<0$ only"])
    fig.suptitle(f"Angular-momentum-gradient control of inertial instability at {actual_hour:g} h", fontsize=15, fontweight="bold")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_robustness_figure(records, metrics, delta_metrics, r, z, w, mask, output, target_hour):
    hours = np.asarray([x["hour"] for x in records["noJET"]])
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), constrained_layout=True)
    colors = {"noJET": "#3B6FB6", "JET": "#B9363E"}
    for case in ("noJET", "JET"):
        rows = [x for x in metrics if x["case"] == case]
        axes[0, 0].plot(hours, [x["i2_negative_fraction"] for x in rows], "o-", color=colors[case], label=f"{case}: Bui $I^2<0$")
        axes[0, 0].plot(hours, [x["m_gradient_negative_fraction"] for x in rows], "s--", color=colors[case], alpha=0.75, label=f"{case}: $M_r<0$")

    axes[0, 0].set_title("Negative-area fractions")
    axes[0, 0].set_ylabel("Cylindrical area fraction")
    max_fraction_mismatch = max(
        abs(x["i2_negative_fraction"] - x["m_gradient_negative_fraction"])
        for x in metrics
    )
    axes[0, 0].text(0.98, 0.06,
                    f"maximum area-fraction mismatch = {100 * max_fraction_mismatch:.2f}%",
                    transform=axes[0, 0].transAxes, va="bottom", ha="right", fontsize=9)
    axes[0, 0].set_xlabel("Simulation time (h)")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
    axes[0, 0].grid(alpha=0.22)

    corr_delta = np.asarray([x["corr_total_vorticity"] for x in delta_metrics])
    baro_fraction = np.asarray([x["baroclinic_rms_fraction"] for x in delta_metrics]) * 100.0
    axes[0, 1].plot(hours, corr_delta, "o-", color="#6A3D9A", label=r"corr($\Delta I^2$, $\Delta I^2_{M_r}$)")
    axes[0, 1].set_ylim(min(0.9997, float(np.nanmin(corr_delta)) - 0.00002), 1.00002)
    axes[0, 1].set_ylabel("Spatial correlation", color="#6A3D9A")
    axes[0, 1].tick_params(axis="y", labelcolor="#6A3D9A")
    axb = axes[0, 1].twinx()
    axb.plot(hours, baro_fraction, "s--", color="#E69F00", label="baroclinic RMS / total")
    axb.set_ylim(0, max(2.0, float(np.nanmax(baro_fraction)) * 1.25))
    axb.set_ylabel("Baroclinic residual (%)", color="#B56B00")
    axb.tick_params(axis="y", labelcolor="#B56B00")
    axes[0, 1].set_title("JET-noJET difference reconstruction")
    axes[0, 1].set_xlabel("Simulation time (h)")
    lines = axes[0, 1].get_lines() + axb.get_lines()
    axes[0, 1].legend(lines, [x.get_label() for x in lines], frameon=False, fontsize=8, loc="center left")
    axes[0, 1].grid(alpha=0.22)

    k = int(np.argmin(np.abs(hours - target_hour)))
    dt = records["JET"][k]["I2"] - records["noJET"][k]["I2"]
    dv = records["JET"][k]["I2_vort"] - records["noJET"][k]["I2_vort"]
    db = records["JET"][k]["I2_baroc"] - records["noJET"][k]["I2_baroc"]
    vmax = np.nanpercentile(np.abs(dt[mask]), 98)
    p = axes[1, 0].pcolormesh(r, z, dt, shading="auto", cmap=nature_diverging(), norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax))
    axes[1, 0].contour(r, z, dv, levels=[0], colors="k", linewidths=1.3)
    axes[1, 0].set_xlim(50, 350); axes[1, 0].set_ylim(10, 16)
    axes[1, 0].set_title(r"JET-noJET $\Delta I^2$; black: $\Delta I^2_{M_r}=0$")
    axes[1, 0].set_xlabel("Radius (km)"); axes[1, 0].set_ylabel("Height (km)")
    cb = fig.colorbar(p, ax=axes[1, 0], pad=0.01); cb.set_label(r"$\Delta I^2$ (s$^{-2}$)")

    names = [r"total $\Delta I^2$", r"$M_r$/vorticity", "baroclinic"]
    rms = [wrms(dt, w, mask), wrms(dv, w, mask), wrms(db, w, mask)]
    means = [wmean(dt, w, mask), wmean(dv, w, mask), wmean(db, w, mask)]
    x = np.arange(3)
    axes[1, 1].bar(x - 0.18, np.asarray(rms) * 1e12, width=0.36, color="#4477AA", label="RMS")
    axes[1, 1].bar(x + 0.18, np.asarray(means) * 1e12, width=0.36, color="#CC6677", label="signed mean")
    axes[1, 1].axhline(0, color="0.4", lw=0.7)
    axes[1, 1].set_xticks(x, names)
    axes[1, 1].set_ylabel(r"Contribution ($10^{-12}$ s$^{-2}$)")
    axes[1, 1].set_title(f"{hours[k]:g} h component reconstruction")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(axis="y", alpha=0.22)
    axes[1, 1].text(0.02, 0.98,
        f"corr(total, $M_r$ term) = {delta_metrics[k]['corr_total_vorticity']:.4f}\n"
        f"baroclinic RMS / total = {delta_metrics[k]['baroclinic_rms_fraction']:.3f}",
        transform=axes[1, 1].transAxes, va="top", ha="left", fontsize=9)
    fig.suptitle("Robustness of the angular-momentum-gradient explanation", fontsize=15, fontweight="bold")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    a = parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hours = np.unique(np.asarray(a.hours, dtype=float))
    cfg = SimpleNamespace(
        output_dir=str(out), max_r_km=a.max_r_km, dr_km=a.dr_km,
        max_z_km=a.max_z_km, center_window=a.center_window,
        center_method=a.center_method, f=a.f, eddy_average=a.eddy_average,
    )
    nojet = analyze_case(a.nojet, hours, read_pmin(a.nojet, hours), cfg, "noJET")
    jet = analyze_case(a.jet, hours, read_pmin(a.jet, hours), cfg, "JET")
    records = {"noJET": nojet, "JET": jet}
    r = np.asarray(nojet[0]["r_km"])
    z = np.asarray(nojet[0]["z_km"])
    w = cell_weights(r, z)
    mask = domain_mask(r, z)

    metrics = []
    for case in ("noJET", "JET"):
        for rec in records[case]:
            row = {"case": case, "hour": rec["hour"]}
            row.update(mask_metrics(rec["I2"], rec["eta"], rec["xi"], rec["I2_vort"], rec["I2_baroc"], w, mask))
            metrics.append(row)

    delta_metrics = []
    for i, hour in enumerate(hours):
        dt = jet[i]["I2"] - nojet[i]["I2"]
        dv = jet[i]["I2_vort"] - nojet[i]["I2_vort"]
        db = jet[i]["I2_baroc"] - nojet[i]["I2_baroc"]
        xi_bar = 0.5 * (jet[i]["xi"] + nojet[i]["xi"])
        eta_bar = 0.5 * (jet[i]["eta"] + nojet[i]["eta"])
        wind_factor = eta_bar * (jet[i]["xi"] - nojet[i]["xi"])
        mgrad_factor = xi_bar * (jet[i]["eta"] - nojet[i]["eta"])
        delta_metrics.append({
            "hour": float(hour),
            "delta_i2_mean": wmean(dt, w, mask),
            "delta_vorticity_mean": wmean(dv, w, mask),
            "delta_baroclinic_mean": wmean(db, w, mask),
            "delta_i2_rms": wrms(dt, w, mask),
            "delta_vorticity_rms": wrms(dv, w, mask),
            "delta_baroclinic_rms": wrms(db, w, mask),
            "corr_total_vorticity": wcorr(dt, dv, w, mask),
            "baroclinic_rms_fraction": wrms(db, w, mask) / wrms(dt, w, mask),
            "reconstruction_max_abs_error": float(np.nanmax(np.abs(dt - dv - db))),
            "classic_wind_factor_mean": wmean(wind_factor, w, mask),
            "classic_mgradient_factor_mean": wmean(mgrad_factor, w, mask),
        })

    make_cross_section_figure(records, r, z, out / "figure12_mgradient_i2_overlap_72h.png", a.target_hour)
    make_robustness_figure(records, metrics, delta_metrics, r, z, w, mask,
                           out / "figure13_mgradient_i2_robustness.png", a.target_hour)

    with (out / "mgradient_i2_overlap_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader(); writer.writerows(metrics)
    with (out / "mgradient_i2_delta_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(delta_metrics[0]))
        writer.writeheader(); writer.writerows(delta_metrics)

    np.savez_compressed(
        out / "mgradient_i2_overlap_fields.npz", hours=hours, r_km=r, z_km=z,
        nojet_I2=stack(nojet, "I2"), jet_I2=stack(jet, "I2"),
        nojet_I2_vort=stack(nojet, "I2_vort"), jet_I2_vort=stack(jet, "I2_vort"),
        nojet_I2_baroc=stack(nojet, "I2_baroc"), jet_I2_baroc=stack(jet, "I2_baroc"),
        nojet_eta=stack(nojet, "eta"), jet_eta=stack(jet, "eta"),
        nojet_xi=stack(nojet, "xi"), jet_xi=stack(jet, "xi"),
        nojet_ur=stack(nojet, "ur"), jet_ur=stack(jet, "ur"),
    )
    k = int(np.argmin(np.abs(hours - a.target_hour)))
    summary = {
        "verification_status": "ANALYZED_MECHANISTIC_IDENTITY_NOT_EXCLUSIVE_CAUSATION",
        "domain_km": {"radius": [50, 350], "height": [10, 16]},
        "definitions": {
            "m_gradient": "dM/dr = r*(f+zeta)",
            "bui_i2": "chi*xi*(1/r*dM/dr) + C*dchi/dr",
            "weights": "cylindrical r*dr*dz",
        },
        "target_hour": float(hours[k]),
        "case_metrics": {case: next(x for x in metrics if x["case"] == case and abs(x["hour"] - hours[k]) < 1e-6) for case in ("noJET", "JET")},
        "difference_metrics": delta_metrics[k],
        "interpretation_boundary": "Regional coincidence and exact algebra identify the immediate state-variable pathway; two simulations do not identify the exclusive upstream cause of dM/dr changes.",
    }
    (out / "mgradient_i2_overlap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
