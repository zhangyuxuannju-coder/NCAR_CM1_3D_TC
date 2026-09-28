#!/usr/bin/env python3
"""Plot 200-hPa horizontal eddy-flux convergence and storm-centred VWS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import gaussian_filter, gaussian_filter1d

SECONDS_PER_DAY = 86400.0


def pressure_interp(field, pressure, target_pa):
    field = np.asarray(field, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    crossing = (pressure[:-1] - target_pa) * (pressure[1:] - target_pa) <= 0.0
    valid = crossing & np.isfinite(field[:-1]) & np.isfinite(field[1:])
    any_valid = np.any(valid, axis=0)
    k = np.argmax(valid, axis=0)
    jj, ii = np.indices(k.shape)
    p0, p1 = pressure[k, jj, ii], pressure[k + 1, jj, ii]
    f0, f1 = field[k, jj, ii], field[k + 1, jj, ii]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = f0 + (target_pa - p0) / (p1 - p0) * (f1 - f0)
    out[~any_valid] = np.nan
    return out


def k_slice(zh_km, lower_km, upper_km):
    idx = np.where((zh_km >= lower_km) & (zh_km <= upper_km))[0]
    if idx.size < 2:
        raise ValueError(f"No usable levels in {lower_km}-{upper_km} km")
    return slice(max(0, int(idx[0]) - 1), min(zh_km.size, int(idx[-1]) + 2))


def read_wind_pressure(ds, it, ks, js, is_, read_rho=False):
    j0, j1 = int(js.start or 0), int(js.stop)
    i0, i1 = int(is_.start or 0), int(is_.stop)
    pressure = np.asarray(ds["prs"][it, ks, j0:j1, i0:i1], dtype=np.float64)
    us = np.asarray(ds["u"][it, ks, j0:j1, i0 : i1 + 1], dtype=np.float64)
    vs = np.asarray(ds["v"][it, ks, j0 : j1 + 1, i0:i1], dtype=np.float64)
    u = 0.5 * (us[..., :-1] + us[..., 1:])
    v = 0.5 * (vs[..., :-1, :] + vs[..., 1:, :])
    rho = np.asarray(ds["rho"][it, ks, j0:j1, i0:i1], dtype=np.float64) if read_rho else None
    return pressure, u, v, rho


def radial_mean(field, bins, use, nr):
    good = use & np.isfinite(field) & (bins >= 0) & (bins < nr)
    count = np.bincount(bins[good], minlength=nr).astype(float)
    total = np.bincount(bins[good], weights=field[good], minlength=nr)
    out = np.full(nr, np.nan)
    np.divide(total, count, out=out, where=count > 0)
    return out, count


def fill_radial_gaps(a):
    out = np.asarray(a, float).copy()
    finite = np.isfinite(out)
    if finite.sum() >= 2:
        x = np.arange(out.size)
        out[~finite] = np.interp(x[~finite], x[finite], out[finite])
    return out


def diagnose_efc(u, v, rho, x_km, y_km, cx, cy, r_km, dr_km, min_count, smooth_bins):
    xx, yy = np.meshgrid(x_km - cx, y_km - cy)
    radius = np.hypot(xx, yy)
    angle = np.arctan2(yy, xx)
    ur = u * np.cos(angle) + v * np.sin(angle)
    vt = -u * np.sin(angle) + v * np.cos(angle)
    nr = r_km.size
    bins = np.floor(radius / dr_km).astype(np.int32)
    use = radius < nr * dr_km
    ur_bar, count = radial_mean(ur, bins, use, nr)
    vt_bar, _ = radial_mean(vt, bins, use, nr)
    rho_bar, _ = radial_mean(rho, bins, use, nr)
    valid_bin = (bins >= 0) & (bins < nr)
    urp = np.full_like(ur, np.nan)
    vtp = np.full_like(vt, np.nan)
    urp[valid_bin] = ur[valid_bin] - ur_bar[bins[valid_bin]]
    vtp[valid_bin] = vt[valid_bin] - vt_bar[bins[valid_bin]]
    flux, _ = radial_mean(rho * urp * vtp, bins, use, nr)
    work = fill_radial_gaps(flux)
    if smooth_bins > 0:
        work = gaussian_filter1d(work, smooth_bins, mode="nearest")
    r_m = r_km * 1000.0
    with np.errstate(invalid="ignore", divide="ignore"):
        efc = -np.gradient(work * r_m**2, r_m, edge_order=2) / (rho_bar * r_m**2)
    efc[(count < min_count) | ~np.isfinite(rho_bar) | (rho_bar <= 0)] = np.nan
    return efc * SECONDS_PER_DAY, flux, count


def cell_widths(coord):
    edges = np.empty(coord.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (coord[:-1] + coord[1:])
    edges[0] = coord[0] - 0.5 * (coord[1] - coord[0])
    edges[-1] = coord[-1] + 0.5 * (coord[-1] - coord[-2])
    return np.diff(edges)


def area_mean_uv(u, v, weights):
    valid = np.isfinite(u) & np.isfinite(v)
    if not valid.any():
        return np.nan, np.nan
    ww = np.broadcast_to(weights, u.shape)[valid]
    return float(np.sum(u[valid] * ww) / np.sum(ww)), float(np.sum(v[valid] * ww) / np.sum(ww))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--label", default="25N JET30 9deg")
    p.add_argument("--radius-max-km", type=float, default=1500.0)
    p.add_argument("--dr-km", type=float, default=25.0)
    p.add_argument("--box-half-width-km", type=float, default=555.0)
    p.add_argument("--flux-smooth-bins", type=float, default=1.0)
    p.add_argument("--min-bin-count", type=int, default=12)
    args = p.parse_args()

    outdir = Path(args.output_dir)
    figdir, proddir, logdir = outdir / "figures", outdir / "products", outdir / "logs"
    for d in (figdir, proddir, logdir):
        d.mkdir(parents=True, exist_ok=True)

    with Dataset(args.input) as ds:
        time_h = np.asarray(ds["time"][:], float) / 3600.0
        x_km, y_km = np.asarray(ds["xh"][:], float), np.asarray(ds["yh"][:], float)
        dx_km, dy_km = cell_widths(x_km), cell_widths(y_km)
        zh_km = np.asarray(ds["zh"][:], float)
        nt = time_h.size
        r_km = np.arange(args.dr_km / 2.0, args.radius_max_km, args.dr_km)
        nr = r_km.size
        efc, flux = np.full((nt, nr), np.nan), np.full((nt, nr), np.nan)
        counts = np.zeros((nt, nr), dtype=np.int32)
        centre_x, centre_y = np.full(nt, np.nan), np.full(nt, np.nan)
        s2085, s2050, s5085 = np.full(nt, np.nan), np.full(nt, np.nan), np.full(nt, np.nan)
        ks200, ks500, ks850 = k_slice(zh_km, 8, 17), k_slice(zh_km, 2, 10), k_slice(zh_km, 0, 5)
        full_y, full_x = slice(0, y_km.size), slice(0, x_km.size)

        for it in range(nt):
            ps = gaussian_filter(np.asarray(ds["psfc"][it], float), 2.0, mode="nearest")
            jc, ic = np.unravel_index(np.nanargmin(ps), ps.shape)
            cx, cy = float(x_km[ic]), float(y_km[jc])
            centre_x[it], centre_y[it] = cx, cy
            p200, u3, v3, r3 = read_wind_pressure(ds, it, ks200, full_y, full_x, True)
            u200, v200 = pressure_interp(u3, p200, 20000.0), pressure_interp(v3, p200, 20000.0)
            rho200 = pressure_interp(r3, p200, 20000.0)
            efc[it], flux[it], counts[it] = diagnose_efc(
                u200, v200, rho200, x_km, y_km, cx, cy, r_km, args.dr_km,
                args.min_bin_count, args.flux_smooth_bins
            )
            ix = np.where(np.abs(x_km - cx) <= args.box_half_width_km)[0]
            iy = np.where(np.abs(y_km - cy) <= args.box_half_width_km)[0]
            ib, jb = slice(int(ix[0]), int(ix[-1]) + 1), slice(int(iy[0]), int(iy[-1]) + 1)
            weights = dy_km[jb, None] * dx_km[None, ib]
            means = {200: area_mean_uv(u200[jb, ib], v200[jb, ib], weights)}
            for hpa, ks in ((500, ks500), (850, ks850)):
                pp, uu3, vv3, _ = read_wind_pressure(ds, it, ks, jb, ib)
                means[hpa] = area_mean_uv(
                    pressure_interp(uu3, pp, hpa * 100.0),
                    pressure_interp(vv3, pp, hpa * 100.0),
                    weights,
                )
            shear = lambda a, b: float(np.hypot(means[a][0] - means[b][0], means[a][1] - means[b][1]))
            s2085[it], s2050[it], s5085[it] = shear(200, 850), shear(200, 500), shear(500, 850)
            if it % 10 == 0 or it == nt - 1:
                print(f"processed {it + 1}/{nt}: t={time_h[it]:.1f} h", flush=True)

    np.savez_compressed(proddir / "jet_efc200_vws_timeseries.npz", time_h=time_h, radius_km=r_km,
        efc_m_s_day=efc, radial_mass_flux=flux, bin_count=counts, centre_x_km=centre_x,
        centre_y_km=centre_y, shear_200_850_m_s=s2085, shear_200_500_m_s=s2050,
        shear_500_850_m_s=s5085)
    np.savetxt(proddir / "jet_vws_timeseries.csv", np.column_stack([time_h, centre_x, centre_y, s2085, s2050, s5085]),
        delimiter=",", header="time_h,centre_x_km,centre_y_km,VWS_200_850_m_s,VWS_200_500_m_s,VWS_500_850_m_s",
        comments="", fmt="%.6g")

    finite = np.isfinite(efc)
    smooth = gaussian_filter(np.where(finite, efc, 0.0), (0.7, 0.8))
    weight = gaussian_filter(finite.astype(float), (0.7, 0.8))
    display = np.divide(smooth, weight, out=np.full_like(smooth, np.nan), where=weight > 0.25)
    robust = float(np.nanpercentile(np.abs(efc[:, r_km >= 100]), 99))
    clim = max(5.0, min(100.0, float(np.ceil(robust / 5) * 5)))
    fig, (a, b) = plt.subplots(2, 1, figsize=(11.2, 10.2), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]})
    m = a.pcolormesh(time_h, r_km, display.T, cmap="RdBu_r", vmin=-clim, vmax=clim, shading="auto", rasterized=True)
    cb = fig.colorbar(m, ax=a, pad=0.015)
    cb.set_label(r"Horizontal EFC at 200 hPa (m s$^{-1}$ day$^{-1}$)")
    a.set(ylabel="Storm-relative radius (km)", title=f"(a) {args.label}: azimuthal eddy angular-momentum flux convergence")
    a.grid(alpha=0.16)
    b.plot(time_h, s5085, "--", color="#2455ff", lw=2, label="500–850 hPa")
    b.plot(time_h, s2050, ":", color="#ef3b2c", lw=2.3, label="200–500 hPa")
    b.plot(time_h, s2085, color="black", lw=2, label="200–850 hPa")
    b.set(xlabel="Model hour", ylabel=r"Vector vertical wind shear (m s$^{-1}$)",
          title=f"(b) Area-mean shear in storm-centred ~10°×10° box (±{args.box_half_width_km:.0f} km)")
    b.grid(alpha=0.22); b.legend(frameon=False, ncol=3, loc="upper center"); b.set_ylim(bottom=0); b.set_xlim(time_h[0], time_h[-1])
    fig.suptitle("JET experiment: upper-level EFC and environmental vertical wind shear", fontsize=15)
    fig.tight_layout(); fig.savefig(figdir / "jet_EFC200_and_VWS.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    vals = efc[np.isfinite(efc)]
    manifest = {"input": str(Path(args.input)), "label": args.label, "time_coverage_h": [float(time_h[0]), float(time_h[-1])],
        "n_times": int(nt), "efc_definition": "-1/(rho_bar*r^2) d_r[r^2 <rho*u_r_prime*v_t_prime>]",
        "efc_includes_vertical_flux": False, "eddy_average": "Reynolds azimuthal departures; density retained inside covariance",
        "pressure_surface_hpa": 200, "radius_km": [float(r_km[0]), float(r_km[-1])], "radial_bin_width_km": args.dr_km,
        "flux_smoothing_bins_before_derivative": args.flux_smooth_bins, "display_gaussian_sigma_time_radius": [0.7, 0.8],
        "display_color_limit_m_s_day": clim, "raw_efc_min_max_m_s_day": [float(vals.min()), float(vals.max())],
        "vws_box": f"storm-centred square +/-{args.box_half_width_km:g} km", "vws_definition": "magnitude of difference of cell-area-weighted mean wind vectors",
        "outputs": {"figure": "figures/jet_EFC200_and_VWS.png", "arrays": "products/jet_efc200_vws_timeseries.npz", "vws_csv": "products/jet_vws_timeseries.csv"}}
    text = json.dumps(manifest, indent=2)
    (outdir / "manifest.json").write_text(text, encoding="utf-8")
    (logdir / "run_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
