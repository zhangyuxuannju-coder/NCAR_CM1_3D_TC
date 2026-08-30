#!/usr/bin/env python3
"""Compare local three-dimensional inertial-stability proxies in a CM1 outflow layer.

The plotted quantities are deliberately labelled as local proxies, not as the
axisymmetric Bui SE coefficient:

  I_M^2   = (f + 2 v_t/r) (1/r) dM/dr
  I_eta^2 = (f + 2 v_t/r) (f + zeta_z)

Their difference is caused by the non-axisymmetric radial-wind derivative,
because (1/r)dM/dr = f + zeta_z + (1/r)d(u_r)/d(lambda).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import netCDF4
import numpy as np
from scipy.ndimage import gaussian_filter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--nojet", required=True)
    p.add_argument("--jet", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary-json", default=None)
    p.add_argument("--time-hours", type=float, default=72.0)
    p.add_argument("--height-km", type=float, default=13.75)
    p.add_argument("--x-half-width-km", type=float, default=1100.0)
    p.add_argument("--y-south-km", type=float, default=500.0)
    p.add_argument("--y-north-km", type=float, default=1200.0)
    p.add_argument("--circular-radius-km", type=float, default=None)
    p.add_argument("--configured-jet-offset-km", type=float, default=888.0)
    p.add_argument("--jet-search-half-width-km", type=float, default=350.0)
    p.add_argument("--center-smooth-sigma", type=float, default=2.0)
    p.add_argument("--field-smooth-sigma", type=float, default=1.0)
    p.add_argument("--inner-mask-km", type=float, default=30.0)
    p.add_argument("--outflow-threshold-m-s", type=float, default=2.0)
    p.add_argument("--color-percentile", type=float, default=98.5)
    p.add_argument("--color-reference-min-radius-km", type=float, default=100.0)
    return p.parse_args()


def destagger_uv(u_stag: np.ndarray, v_stag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = 0.5 * (u_stag[:, :-1] + u_stag[:, 1:])
    v = 0.5 * (v_stag[:-1, :] + v_stag[1:, :])
    return u, v


def coordinate_widths(coord_m: np.ndarray) -> np.ndarray:
    edges = np.empty(coord_m.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (coord_m[:-1] + coord_m[1:])
    edges[0] = coord_m[0] - 0.5 * (coord_m[1] - coord_m[0])
    edges[-1] = coord_m[-1] + 0.5 * (coord_m[-1] - coord_m[-2])
    return np.diff(edges)


def read_case(path: str, target_hour: float, target_height_km: float,
              center_sigma: float, field_sigma: float,
              inner_mask_km: float) -> dict[str, np.ndarray | float]:
    with netCDF4.Dataset(path) as ds:
        time = np.asarray(ds.variables["time"][:], dtype=float)
        x = np.asarray(ds.variables["xh"][:], dtype=float)
        y = np.asarray(ds.variables["yh"][:], dtype=float)
        z = np.asarray(ds.variables["zh"][:], dtype=float)
        it = int(np.nanargmin(np.abs(time - target_hour * 3600.0)))
        iz = int(np.nanargmin(np.abs(z - target_height_km)))
        if "f_cor" in ds.variables:
            f = float(np.asarray(ds.variables["f_cor"][:]).ravel()[0])
        else:
            f = float(getattr(ds, "fcor", 5.464e-5))
        psfc = np.asarray(ds.variables["psfc"][it], dtype=float)
        psfc_initial = np.asarray(ds.variables["psfc"][0], dtype=float)
        u_stag = np.asarray(ds.variables["u"][it, iz], dtype=float)
        v_stag = np.asarray(ds.variables["v"][it, iz], dtype=float)

    ps = gaussian_filter(psfc, sigma=center_sigma, mode="nearest")
    iyc, ixc = np.unravel_index(np.nanargmin(ps), ps.shape)
    xc, yc = float(x[ixc]), float(y[iyc])
    ps_initial = gaussian_filter(psfc_initial, sigma=center_sigma, mode="nearest")
    iy0, ix0 = np.unravel_index(np.nanargmin(ps_initial), ps_initial.shape)
    u, v = destagger_uv(u_stag, v_stag)
    if field_sigma > 0:
        u = gaussian_filter(u, sigma=field_sigma, mode="nearest")
        v = gaussian_filter(v, sigma=field_sigma, mode="nearest")

    xr = x - xc
    yr = y - yc
    xx_km, yy_km = np.meshgrid(xr, yr)
    radius_km = np.hypot(xx_km, yy_km)
    radius_m = radius_km * 1000.0
    safe_r = np.where(radius_km >= inner_mask_km, radius_m, np.nan)
    coslam = np.divide(xx_km, radius_km, out=np.zeros_like(xx_km), where=radius_km > 0)
    sinlam = np.divide(yy_km, radius_km, out=np.zeros_like(yy_km), where=radius_km > 0)

    ur = u * coslam + v * sinlam
    vt = -u * sinlam + v * coslam
    xm, ym = x * 1000.0, y * 1000.0
    du_dy, du_dx = np.gradient(u, ym, xm, edge_order=2)
    dv_dy, dv_dx = np.gradient(v, ym, xm, edge_order=2)
    zeta = dv_dx - du_dy
    eta = f + zeta

    M = safe_r * vt + 0.5 * f * safe_r**2
    dM_dy, dM_dx = np.gradient(M, ym, xm, edge_order=2)
    dM_dr = dM_dx * coslam + dM_dy * sinlam
    xi = f + 2.0 * vt / safe_r
    im2 = xi * dM_dr / safe_r
    ieta2 = xi * eta
    nonaxis_term = im2 - ieta2

    area = np.outer(coordinate_widths(ym), coordinate_widths(xm))
    return {
        "x_rel_km": xr, "y_rel_km": yr,
        "u": u, "v": v, "ur": ur, "vt": vt,
        "eta": eta, "xi": xi, "M": M,
        "im2": im2, "ieta2": ieta2, "nonaxis_term": nonaxis_term,
        "area_m2": area, "radius_km": radius_km,
        "selected_hour": float(time[it] / 3600.0),
        "selected_height_km": float(z[iz]),
        "center_x_km": xc, "center_y_km": yc,
        "initial_center_x_km": float(x[ix0]), "initial_center_y_km": float(y[iy0]),
        "min_psfc_hpa": float(ps[iyc, ixc] / 100.0), "f": f,
    }


def subset(rec: dict, a: argparse.Namespace) -> dict:
    x = np.asarray(rec["x_rel_km"])
    y = np.asarray(rec["y_rel_km"])
    ix = np.flatnonzero(np.abs(x) <= a.x_half_width_km)
    iy = np.flatnonzero((y >= -a.y_south_km) & (y <= a.y_north_km))
    out = {"x": x[ix], "y": y[iy]}
    for name in ("u", "v", "ur", "im2", "ieta2", "nonaxis_term", "area_m2", "radius_km"):
        out[name] = np.asarray(rec[name])[np.ix_(iy, ix)]
    out["plot_mask"] = np.ones_like(out["radius_km"], dtype=bool)
    if a.circular_radius_km is not None:
        out["plot_mask"] &= out["radius_km"] <= a.circular_radius_km
    return out


def weighted_fraction(mask: np.ndarray, weights: np.ndarray, domain: np.ndarray) -> float:
    valid = domain & np.isfinite(weights)
    den = float(np.sum(weights[valid]))
    return float(np.sum(weights[valid & mask]) / den) if den > 0 else float("nan")


def diagnose_jet_axis(jet: dict, a: argparse.Namespace,
                      configured_axis_relative_km: float) -> tuple[float, float]:
    x = np.asarray(jet["x_rel_km"])
    y = np.asarray(jet["y_rel_km"])
    u = np.asarray(jet["u"])
    ix = np.abs(x) <= a.x_half_width_km
    target = configured_axis_relative_km
    iy = (y >= target - a.jet_search_half_width_km) & (y <= target + a.jet_search_half_width_km)
    profile = np.nanmean(u[np.ix_(iy, ix)], axis=1)
    jj = int(np.nanargmax(profile))
    return float(y[iy][jj]), float(profile[jj])


def main() -> None:
    a = parse_args()
    nojet = read_case(a.nojet, a.time_hours, a.height_km,
                      a.center_smooth_sigma, a.field_smooth_sigma, a.inner_mask_km)
    jet = read_case(a.jet, a.time_hours, a.height_km,
                    a.center_smooth_sigma, a.field_smooth_sigma, a.inner_mask_km)
    ns, js = subset(nojet, a), subset(jet, a)
    configured_axis_relative = (
        float(jet["initial_center_y_km"])
        + a.configured_jet_offset_km
        - float(jet["center_y_km"])
    )
    jet_axis, jet_umax = diagnose_jet_axis(jet, a, configured_axis_relative)

    # The two simulations share a grid, but their storm centres can differ by a
    # few grid points.  Interpolate JET fields to the noJET storm-relative grid
    # before constructing differences.
    from scipy.interpolate import RegularGridInterpolator
    yy, xx = np.meshgrid(ns["y"], ns["x"], indexing="ij")
    points = np.column_stack((yy.ravel(), xx.ravel()))
    j_on_n = {}
    for name in ("im2", "ieta2"):
        j_on_n[name] = RegularGridInterpolator(
            (js["y"], js["x"]), js[name], bounds_error=False, fill_value=np.nan
        )(points).reshape(yy.shape)
    jur_on_n = RegularGridInterpolator(
        (js["y"], js["x"]), js["ur"], bounds_error=False, fill_value=np.nan
    )(points).reshape(yy.shape)

    rows = [
        ("im2", r"Storm-centred $I_M^2=\xi r^{-1}\partial_r M$"),
        ("ieta2", r"3-D vorticity proxy $I_\eta^2=\xi(f+\zeta_z)$"),
    ]
    cmap = LinearSegmentedColormap.from_list(
        "nature_stability",
        ["#24486E", "#5F88B4", "#B9D0E2", "#F7F7F4",
         "#F2C2A8", "#D96B4C", "#8E2C3A"], N=256,
    )
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.3,
        "axes.linewidth": 0.75, "xtick.direction": "out", "ytick.direction": "out",
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.0), sharex=True, sharey=True,
                             constrained_layout=True)
    letters = "abcdef"
    row_mappables = []
    for i, (name, row_title) in enumerate(rows):
        fields = [ns[name], j_on_n[name], j_on_n[name] - ns[name]]
        color_mask = (
            (ns["radius_km"] >= a.color_reference_min_radius_km)
            & ns["plot_mask"]
        )
        finite = np.concatenate([
            np.abs(q[color_mask & np.isfinite(q)]) for q in fields
        ])
        lim = max(0.1e-8, float(np.nanpercentile(finite, a.color_percentile)))
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        for j, (field, title) in enumerate(zip(fields, ("noJET", "JET", "JET − noJET"))):
            ax = axes[i, j]
            field = np.where(ns["plot_mask"], field, np.nan)
            p = ax.pcolormesh(ns["x"], ns["y"], field * 1.0e8,
                              cmap=cmap,
                              norm=TwoSlopeNorm(vmin=-lim*1e8, vcenter=0.0, vmax=lim*1e8),
                              shading="auto", rasterized=True)
            if j == 0:
                ur = np.where(ns["plot_mask"], ns["ur"], np.nan)
            elif j == 1:
                ur = np.where(ns["plot_mask"], jur_on_n, np.nan)
            else:
                ur = np.where(ns["plot_mask"], jur_on_n - ns["ur"], np.nan)
            levels = [-10.0, -5.0, -2.0, 2.0, 5.0, 10.0] if j < 2 else [-5.0, -2.0, 2.0, 5.0]
            colors = (["#355C9A"] * 3 + ["#177E62"] * 3) if j < 2 else (["#355C9A"] * 2 + ["#177E62"] * 2)
            linestyles = (["--"] * 3 + ["-"] * 3) if j < 2 else (["--"] * 2 + ["-"] * 2)
            use = [q for q in levels if np.nanmin(ur) <= q <= np.nanmax(ur)]
            if use:
                use_colors = [colors[levels.index(q)] for q in use]
                use_ls = [linestyles[levels.index(q)] for q in use]
                cs = ax.contour(ns["x"], ns["y"], ur, levels=use,
                                colors=use_colors, linestyles=use_ls, linewidths=0.85)
                ax.clabel(cs, fmt="%g", fontsize=6.8)
            zero = ax.contour(ns["x"], ns["y"], field, levels=[0.0],
                              colors="#242424", linewidths=0.55)
            ax.axhline(configured_axis_relative, color="#8C6BB1", lw=1.0, ls="--")
            ax.axhline(jet_axis, color="#7A1FA2", lw=1.25, ls="-.")
            ax.scatter(0.0, 0.0, marker="x", s=30, c="#A1262D", linewidths=1.3, zorder=8)
            if a.circular_radius_km is not None:
                ax.add_patch(plt.Circle(
                    (0.0, 0.0), a.circular_radius_km, fill=False,
                    color="#5A5A5A", lw=0.75, ls=":"
                ))
            ax.set_title(f"{title}\n{row_title}", fontsize=10.0)
            ax.text(-0.10, 1.02, f"({letters[i*3+j]})", transform=ax.transAxes,
                    fontweight="bold", va="bottom")
            ax.set_xlim(-a.x_half_width_km, a.x_half_width_km)
            ax.set_ylim(-a.y_south_km, a.y_north_km)
            ax.set_aspect("equal", adjustable="box")
        row_mappables.append(p)

    for ax in axes[-1]:
        ax.set_xlabel("Storm-relative x (km)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Storm-relative y (km)")
    for i, p in enumerate(row_mappables):
        cb = fig.colorbar(p, ax=axes[i, :], orientation="horizontal", shrink=0.72,
                          pad=0.035, aspect=42, extend="both")
        cb.set_label(r"Local stability proxy ($10^{-8}$ s$^{-2}$); blue < 0")
    axes[0, 2].plot([], [], color="#177E62", label=r"$u_r=+2$ m s$^{-1}$")
    axes[0, 2].plot([], [], color="#355C9A", ls="--", label=r"$u_r=-2$ m s$^{-1}$")
    jet_outside = (
        a.circular_radius_km is not None
        and configured_axis_relative > a.circular_radius_km
    )
    if not jet_outside:
        axes[0, 2].plot([], [], color="#8C6BB1", ls="--", label="configured jet latitude")
        axes[0, 2].plot([], [], color="#7A1FA2", ls="-.", label="diagnosed JET axis")
    axes[0, 2].legend(loc="lower left", fontsize=7.3, frameon=True)
    jet_note = ""
    if jet_outside:
        jet_note = (
            f"\nJet lies north/outside: configured +{configured_axis_relative:.0f} km; "
            f"diagnosed x-mean-u maximum +{jet_axis:.0f} km"
        )
        axes[0, 1].annotate(
            "jet to north\n(outside panel)",
            xy=(0.0, a.y_north_km * 0.98),
            xytext=(0.0, a.y_north_km * 0.72),
            ha="center", va="top", color="#6F4C8B", fontsize=7.4,
            arrowprops=dict(arrowstyle="-|>", color="#6F4C8B", lw=1.0),
        )
    fig.suptitle(
        f"Horizontal local inertial-stability proxies in the outflow layer · "
        f"t={nojet['selected_hour']:.0f} h, z={nojet['selected_height_km']:.2f} km\n"
        "Black: zero stability; green/blue: 3-D radial outflow/inflow"
        f"{jet_note}",
        fontsize=13, fontweight="bold",
    )
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=280, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "definition": {
            "I_M2": "(f+2 vt/r) * (1/r) dM/dr; storm-centred local proxy",
            "I_eta2": "(f+2 vt/r) * (f+zeta_z); full 3-D vertical-vorticity proxy",
            "difference_identity": "I_M2-I_eta2=(f+2vt/r)*(1/r)*d(ur)/d(lambda)",
        },
        "requested_time_hours": a.time_hours,
        "selected_time_hours": float(nojet["selected_hour"]),
        "requested_height_km": a.height_km,
        "selected_height_km": float(nojet["selected_height_km"]),
        "configured_initial_jet_offset_km": a.configured_jet_offset_km,
        "configured_jet_axis_relative_y_km_at_selected_time": configured_axis_relative,
        "diagnosed_jet_axis_relative_y_km": jet_axis,
        "diagnosed_jet_axis_xmean_u_m_s": jet_umax,
        "cases": {},
    }
    for label, q in (("noJET", ns), ("JET", js)):
        domain = (q["radius_km"] >= a.inner_mask_km) & q["plot_mask"]
        outflow = domain & (q["ur"] >= a.outflow_threshold_m_s)
        area = q["area_m2"]
        summary["cases"][label] = {
            "center_x_km": float((nojet if label == "noJET" else jet)["center_x_km"]),
            "center_y_km": float((nojet if label == "noJET" else jet)["center_y_km"]),
            "min_psfc_hpa": float((nojet if label == "noJET" else jet)["min_psfc_hpa"]),
            "I_M2_negative_area_fraction": weighted_fraction(q["im2"] < 0, area, domain),
            "I_eta2_negative_area_fraction": weighted_fraction(q["ieta2"] < 0, area, domain),
            "outflow_area_fraction": weighted_fraction(outflow, area, domain),
            "outflow_I_M2_negative_fraction": weighted_fraction(q["im2"] < 0, area, outflow),
            "outflow_I_eta2_negative_fraction": weighted_fraction(q["ieta2"] < 0, area, outflow),
        }
    common_domain = (
        (ns["radius_km"] >= a.inner_mask_km)
        & ns["plot_mask"]
        & np.isfinite(jur_on_n)
    )
    nojet_outflow = common_domain & (ns["ur"] >= a.outflow_threshold_m_s)
    jet_outflow = common_domain & (jur_on_n >= a.outflow_threshold_m_s)
    area = ns["area_m2"]
    summary["same_mask_comparison"] = {}
    for mask_name, mask in (
        ("outflow_union", nojet_outflow | jet_outflow),
        ("outflow_intersection", nojet_outflow & jet_outflow),
    ):
        summary["same_mask_comparison"][mask_name] = {
            "area_fraction_of_common_domain": weighted_fraction(mask, area, common_domain),
            "noJET_I_M2_negative_fraction": weighted_fraction(ns["im2"] < 0, area, mask),
            "JET_I_M2_negative_fraction": weighted_fraction(j_on_n["im2"] < 0, area, mask),
            "noJET_I_eta2_negative_fraction": weighted_fraction(ns["ieta2"] < 0, area, mask),
            "JET_I_eta2_negative_fraction": weighted_fraction(j_on_n["ieta2"] < 0, area, mask),
        }
    summary_path = Path(a.summary_json) if a.summary_json else output.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
