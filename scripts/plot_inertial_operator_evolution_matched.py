#!/usr/bin/env python3
"""Matched-time inertial-stability operator forcing and SE response evolution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.se_bui import build_basic_state, invert_balanced_theta
from scripts.solve_matched_operator_forcing_outer100 import (
    build_ctrl_operator, grad, read_case, smooth, solve,
)


CMAP = LinearSegmentedColormap.from_list(
    "nature_div", ["#2F4B7C", "#7895B7", "#D6E0EA", "#F7F7F4",
                   "#F2C1AE", "#DD7055", "#9D2933"], N=256,
)

MATCHES = [
    # jet_h, ctrl_h, dP=JET-CTRL hPa, dV m/s, dR km
    (60.0, 60.0, 0.11, 2.12, 0.0),
    (65.0, 65.0, 2.28, -1.77, 0.0),
    (70.0, 65.0, 6.89, -3.51, 12.0),
    (75.0, 70.0, 0.41, 0.74, -12.0),
    (80.0, 90.0, -9.43, 6.98, -12.0),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ctrl", required=True)
    p.add_argument("--jet", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-r-km", type=float, default=1200.0)
    p.add_argument("--dr-km", type=float, default=12.0)
    p.add_argument("--max-z-km", type=float, default=20.0)
    p.add_argument("--plot-max-z-km", type=float, default=18.0)
    p.add_argument("--mask-radius-km", type=float, default=100.0)
    p.add_argument("--jet-plot-r0-km", type=float, default=600.0)
    p.add_argument("--jet-plot-r1-km", type=float, default=1150.0)
    p.add_argument("--jet-plot-z0-km", type=float, default=8.0)
    p.add_argument("--jet-plot-z1-km", type=float, default=18.0)
    p.add_argument("--jet-axis-r-km", type=float, default=888.0)
    p.add_argument("--jet-axis-z-km", type=float, default=12.0)
    p.add_argument("--f", type=float, default=5.464e-5)
    p.add_argument("--smooth-sigma-z", type=float, default=0.75)
    p.add_argument("--smooth-sigma-r", type=float, default=1.0)
    p.add_argument("--eps-ratio", type=float, default=1.0e-5)
    p.add_argument(
        "--matches-json",
        help=("Optional JSON file containing a list of [jet_h, ctrl_h, dP, dV, dR] "
              "records. When omitted, the legacy built-in matches are used."),
    )
    return p.parse_args()


def load_matches(a):
    if not a.matches_json:
        return MATCHES
    records = json.loads(Path(a.matches_json).read_text(encoding="utf-8"))
    matches = []
    for record in records:
        if isinstance(record, dict):
            values = (
                record["jet_h"], record["ctrl_h"], record["dP"],
                record["dV"], record.get("dR", 0.0),
            )
        else:
            values = record
        if len(values) != 5:
            raise ValueError("Each match must contain jet_h, ctrl_h, dP, dV, dR")
        matches.append(tuple(float(value) for value in values))
    if not matches:
        raise ValueError("No matches found in --matches-json")
    return matches


def robust_abs(fields, masks=None, percentile=99.0, floor=1e-30):
    chunks = []
    for i, field in enumerate(fields):
        mask = np.isfinite(field) if masks is None else (masks[i] & np.isfinite(field))
        if np.any(mask):
            chunks.append(np.abs(field[mask]))
    values = np.concatenate(chunks) if chunks else np.array([floor])
    return max(float(np.nanpercentile(values, percentile)), floor)


def rms(field, mask):
    good = mask & np.isfinite(field)
    return float(np.sqrt(np.nanmean(field[good]**2))) if np.any(good) else np.nan


def compute_one(jet_h, ctrl_h, dp, dv, drmw, a):
    print(f"[INERTIAL] JET {jet_h:g} h / CTRL {ctrl_h:g} h", flush=True)
    ctrl = read_case(a.ctrl, ctrl_h, a)
    jet = read_case(a.jet, jet_h, a)
    r = np.asarray(ctrl["r_km"], float)
    z = np.asarray(ctrl["z_km"], float)
    r_m, z_m = r*1000.0, z*1000.0
    theta_c, tw_c = invert_balanced_theta(
        ctrl["ut"], ctrl["theta"], r_m, z_m, a.f, outer_smooth_window=1,
    )
    theta_j, tw_j = invert_balanced_theta(
        jet["ut"], jet["theta"], r_m, z_m, a.f, outer_smooth_window=1,
    )
    bc = build_basic_state(ctrl["ut"], theta_c, ctrl["rho"], r_m, z_m, a.f)
    bj = build_basic_state(jet["ut"], theta_j, jet["rho"], r_m, z_m, a.f)
    delta_i2 = smooth(np.asarray(bj["K3_raw"])-np.asarray(bc["K3_raw"]), a)
    ur_ctrl = smooth(np.asarray(ctrl["ur"], float), a)
    s_i_raw = grad(delta_i2*ur_ctrl, z_m, axis=0)
    s_i = np.where(r[None, :] >= a.mask_radius_km, s_i_raw, 0.0)
    basic_ctrl, operator, reg = build_ctrl_operator(bc, r_m, z_m, a)
    response = solve(operator, s_i, basic_ctrl["rho"], r_m, z_m)
    return {
        "jet_h": jet_h, "ctrl_h": ctrl_h, "dP": dp, "dV": dv, "dR": drmw,
        "r": r, "z": z, "delta_i2": delta_i2, "forcing": s_i,
        "psi": response["psi"], "u_se": response["u"], "w_se": response["w"],
        "ur_jet": np.asarray(jet["ur"], float), "ur_ctrl": np.asarray(ctrl["ur"], float),
        "reg": reg, "tw_ctrl": tw_c, "tw_jet": tw_j,
    }


def plot_forcing(results, a, output):
    fields, masks = [], []
    for q in results:
        r, z = q["r"], q["z"]
        mask = ((r[None, :] >= a.jet_plot_r0_km) & (r[None, :] <= a.jet_plot_r1_km) &
                (z[:, None] >= a.jet_plot_z0_km) & (z[:, None] <= a.jet_plot_z1_km))
        fields.append(q["forcing"]); masks.append(mask)
    vmax = robust_abs(fields, masks, percentile=99.3)
    linthresh = max(vmax*0.025, 1e-30)
    exp = int(np.floor(np.log10(vmax)))
    scale = 10.0**exp
    norm = SymLogNorm(linthresh=linthresh/scale, linscale=0.9,
                      vmin=-vmax/scale, vmax=vmax/scale, base=10)
    fig, axes = plt.subplots(1, len(results), figsize=(20, 4.8), sharey=True,
                             constrained_layout=True)
    maps = []
    for ax, q, mask in zip(axes, results, masks):
        fld = q["forcing"]/scale
        im = ax.pcolormesh(q["r"], q["z"], fld, cmap=CMAP, norm=norm,
                           shading="auto", rasterized=True)
        maps.append(im)
        ax.contour(q["r"], q["z"], q["forcing"], levels=[0], colors="0.25", linewidths=0.45)
        ax.plot(a.jet_axis_r_km, a.jet_axis_z_km, "*", ms=10,
                color="#7A1FA2", mec="white", mew=0.6)
        jrms = rms(q["forcing"], mask)
        ax.set_xlim(a.jet_plot_r0_km, a.jet_plot_r1_km)
        ax.set_ylim(a.jet_plot_z0_km, a.jet_plot_z1_km)
        ax.set_title(f"JET {q['jet_h']:g} h / CTRL {q['ctrl_h']:g} h\n"
                     f"jet-box RMS={jrms:.2e}", fontsize=9.5)
        ax.set_xlabel("Radius (km)")
        ax.grid(alpha=0.17, lw=0.4)
    axes[0].set_ylabel("Height (km)")
    cb = fig.colorbar(maps[-1], ax=axes, orientation="horizontal", shrink=0.7, pad=0.12)
    cb.set_label(rf"$S_I=\partial_z(u_{{C,match}}\,\delta I^2)$ "
                 rf"($\times10^{{{exp}}}$ K$^{{-1}}$ s$^{{-3}}$; symmetric-log scale)")
    fig.suptitle("Inertial-stability equivalent operator forcing in the imposed-jet region\n"
                 "Inner-core values are excluded from both view and colour scaling",
                 fontsize=14, fontweight="bold")
    fig.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(fig)


def contour_radial_wind(ax, r, z, ur):
    levels = [-10, -5, -2, 2, 5, 10]
    cs = ax.contour(r, z, ur, levels=levels, colors="black", linewidths=0.55)
    ax.clabel(cs, fmt="%g", fontsize=6.2, inline=True)


def plot_u_response(results, a, output):
    fields = [q["u_se"] for q in results]
    plot_masks = [np.broadcast_to(q["z"][:,None] <= a.plot_max_z_km, q["u_se"].shape) for q in results]
    vmax = robust_abs(fields, plot_masks, percentile=98.8, floor=1e-5)
    fig, axes = plt.subplots(1, len(results), figsize=(20, 5.2), sharey=True,
                             constrained_layout=True)
    maps=[]
    for ax, q in zip(axes, results):
        im=ax.pcolormesh(q["r"], q["z"], q["u_se"], cmap=CMAP,
                         norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax),
                         shading="auto", rasterized=True)
        maps.append(im)
        contour_radial_wind(ax, q["r"], q["z"], q["ur_jet"])
        ax.axvline(a.mask_radius_km, color="#7A1FA2", ls="--", lw=0.8)
        ax.set_xlim(0,a.max_r_km); ax.set_ylim(0,a.plot_max_z_km)
        ax.set_title(f"JET {q['jet_h']:g} h / CTRL {q['ctrl_h']:g} h\n"
                     f"ΔP={q['dP']:+.1f} hPa, ΔV={q['dV']:+.1f} m s$^{{-1}}$",
                     fontsize=9.2)
        ax.set_xlabel("Radius (km)")
    axes[0].set_ylabel("Height (km)")
    cb=fig.colorbar(maps[-1], ax=axes, orientation="horizontal", shrink=0.7, pad=0.12)
    cb.set_label(r"Inertial-operator SE radial-wind response $u_I^{SE}$ (m s$^{-1}$; outward +)")
    handles=[Line2D([0],[0],color="black",lw=0.8,label="JET actual azimuthal-mean radial wind contours"),
             Line2D([0],[0],color="#7A1FA2",ls="--",lw=0.9,label="100-km forcing cutoff")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.5,0.91),ncol=2,frameon=False,fontsize=8.5)
    fig.suptitle("Inertial-stability operator contribution to radial circulation\n"
                 "Colours: operator-only SE response; black contours: actual JET radial wind",
                 fontsize=14,fontweight="bold")
    fig.savefig(output,dpi=260,bbox_inches="tight"); plt.close(fig)


def plot_w_response(results, a, output):
    fields=[q["w_se"] for q in results]
    masks=[np.broadcast_to(q["z"][:,None] <= a.plot_max_z_km,q["w_se"].shape) for q in results]
    vmax=robust_abs(fields,masks,percentile=98.8,floor=1e-6)
    psi_lim=robust_abs([q["psi"] for q in results],masks,percentile=96.0)
    psi_levels=np.linspace(-psi_lim,psi_lim,13)
    psi_levels=psi_levels[np.abs(psi_levels)>1e-12*psi_lim]
    fig,axes=plt.subplots(1,len(results),figsize=(20,5.2),sharey=True,constrained_layout=True)
    maps=[]
    for ax,q in zip(axes,results):
        im=ax.pcolormesh(q["r"],q["z"],q["w_se"],cmap=CMAP,
                         norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax),
                         shading="auto",rasterized=True)
        maps.append(im)
        ax.contour(q["r"],q["z"],q["psi"],levels=psi_levels,colors="0.35",linewidths=0.48)
        contour_radial_wind(ax,q["r"],q["z"],q["ur_jet"])
        ax.axvline(a.mask_radius_km,color="#7A1FA2",ls="--",lw=0.8)
        ax.set_xlim(0,a.max_r_km); ax.set_ylim(0,a.plot_max_z_km)
        ax.set_title(f"JET {q['jet_h']:g} h / CTRL {q['ctrl_h']:g} h\n"
                     f"regularized cells={100*q['reg']['changed_coefficient_fraction']:.1f}%",
                     fontsize=9.2)
        ax.set_xlabel("Radius (km)")
    axes[0].set_ylabel("Height (km)")
    cb=fig.colorbar(maps[-1],ax=axes,orientation="horizontal",shrink=0.7,pad=0.12)
    cb.set_label(r"Inertial-operator SE vertical response $w_I^{SE}$ (m s$^{-1}$; upward +)")
    handles=[Line2D([0],[0],color="black",lw=.8,label="JET actual radial-wind contours"),
             Line2D([0],[0],color="0.35",lw=.8,label="operator-response streamfunction contours")]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(0.5,0.91),ncol=2,frameon=False,fontsize=8.5)
    fig.suptitle("Inertial-stability operator contribution to vertical circulation\n"
                 "Colours: vertical response; grey: response streamfunction; black: actual JET radial wind",
                 fontsize=14,fontweight="bold")
    fig.savefig(output,dpi=260,bbox_inches="tight"); plt.close(fig)


def main():
    a=parse_args(); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    matches = load_matches(a)
    results=[compute_one(*match,a) for match in matches]
    plot_forcing(results,a,out/"figure1_inertial_forcing_jet_region.png")
    plot_u_response(results,a,out/"figure2_inertial_SE_radial_response_with_jet_ur.png")
    plot_w_response(results,a,out/"figure3_inertial_SE_vertical_response_with_jet_ur.png")
    metrics=[]
    for q in results:
        r,z=q["r"],q["z"]
        jetbox=((r[None,:]>=650)&(r[None,:]<=1150)&(z[:,None]>=10)&(z[:,None]<=16))
        innerout=((r[None,:]<=300)&(z[:,None]>=10)&(z[:,None]<=18))
        lowcore=((r[None,:]<=100)&(z[:,None]<=2))
        metrics.append({
            "jet_hour":q["jet_h"],"ctrl_match_hour":q["ctrl_h"],
            "matching_residual":{"dP_hPa":q["dP"],"dV_m_s":q["dV"],"dRMW_km":q["dR"]},
            "forcing_jetbox_rms":rms(q["forcing"],jetbox),
            "u_SE_inner_outflow_rms":rms(q["u_se"],innerout),
            "w_SE_inner_outflow_rms":rms(q["w_se"],innerout),
            "u_SE_lowcore_rms":rms(q["u_se"],lowcore),
            "w_SE_lowcore_rms":rms(q["w_se"],lowcore),
            "regularization":q["reg"],
        })
        np.savez_compressed(out/f"inertial_operator_J{q['jet_h']:03.0f}_C{q['ctrl_h']:03.0f}.npz",
                            r_km=r,z_km=z,delta_I2=q["delta_i2"],S_inertial=q["forcing"],
                            psi_I=q["psi"],u_I=q["u_se"],w_I=q["w_se"],ur_jet=q["ur_jet"])
    weak_matches = [
        {"jet_hour": q[0], "ctrl_hour": q[1], "dP_hPa": q[2],
         "dV_m_s": q[3], "dRMW_km": q[4]}
        for q in matches
        if abs(q[2]) > 2.0 or abs(q[3]) > 2.0 or abs(q[4]) > 18.0
    ]
    summary={
        "definition":"S_I=d_z[u_CTRL_matched*(I2_JET-I2_CTRL_matched)]",
        "forcing_mask":f"S_I=0 for r<{a.mask_radius_km:g} km",
        "coefficient_state":"separately thermal-wind-balanced JET and matched CTRL states",
        "solver":"full-radius sparse SE inversion with homogeneous Dirichlet streamfunction boundaries",
        "eps_ratio":a.eps_ratio,
        "coriolis_f_s-1":a.f,
        "panels":metrics,
        "weak_matches": weak_matches,
        "warning": ("Some matches exceed the residual thresholds; interpret those panels "
                    "structurally, not as pure intensity-controlled jet effects")
                   if weak_matches else "All matches satisfy the configured residual thresholds",
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
