#!/usr/bin/env python3
"""Connect CM1 angular-momentum budgets, raw Bui I2, and intensification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


CASES = ("noJET", "JET")
ATTR_CASE = {"noJET": "ctrl", "JET": "jet"}
COLORS = {"noJET": "#0072B2", "JET": "#D55E00"}
M_TERMS = ("mean_r", "mean_z", "eddy_r", "eddy_z", "pgrad", "diffusion", "rdamp")
RI_THRESHOLD = 15.4333  # 30 kt in m s-1 over 24 h


def nature_balance():
    return LinearSegmentedColormap.from_list(
        "nature_balance", ["#214478", "#f7f7f3", "#b2182b"]
    )


def weighted_mean(a, w, mask):
    use = mask & np.isfinite(a) & np.isfinite(w)
    if not np.any(use):
        return np.nan
    return float(np.sum(a[use] * w[use]) / np.sum(w[use]))


def weighted_fraction(condition, w, mask):
    use = mask & np.isfinite(w)
    return float(np.sum(w[use] * condition[use]) / np.sum(w[use]))


def weighted_jaccard(a, b, w, mask):
    use = mask & np.isfinite(w)
    union = (a | b) & use
    if not np.any(union):
        return 1.0
    return float(np.sum(w[union & a & b]) / np.sum(w[union]))


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    use = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(use) < 3 or np.nanstd(x[use]) == 0 or np.nanstd(y[use]) == 0:
        return np.nan
    return float(np.corrcoef(x[use], y[use])[0, 1])


def rank_corr(x, y):
    x, y = pd.Series(x), pd.Series(y)
    use = x.notna() & y.notna()
    return corr(x[use].rank().to_numpy(), y[use].rank().to_numpy())


def residualize(y, controls):
    y = np.asarray(y, float)
    x = np.asarray(controls, float)
    x = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


def partial_corr(x, y, controls):
    x, y, controls = np.asarray(x, float), np.asarray(y, float), np.asarray(controls, float)
    use = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(controls), axis=1)
    if np.count_nonzero(use) < controls.shape[1] + 4:
        return np.nan
    return corr(residualize(x[use], controls[use]), residualize(y[use], controls[use]))


def area_masks(r, z):
    rr, zz = np.meshgrid(r, z)
    w = np.maximum(rr, 0.5 * np.nanmedian(np.diff(r)))
    return rr, zz, w, {
        "core": (rr >= 50) & (rr <= 200) & (zz >= 10) & (zz <= 16),
        "inner": (rr >= 50) & (rr <= 350) & (zz >= 10) & (zz <= 16),
    }


def budget_key(case, hour, field):
    return f"{case}_{hour:g}h_{field}"


def boundary_contrast(field, r, z):
    inner = (r >= 50) & (r <= 75)
    outer = (r >= 325) & (r <= 350)
    lev = (z >= 10) & (z <= 16)
    a = np.nanmean(field[np.ix_(lev, inner)])
    b = np.nanmean(field[np.ix_(lev, outer)])
    return float(b - a)


def build_coupling_rows(budget, metrics, attr, ats):
    r, z = budget["r_km"], budget["z_km"]
    ar, az = attr["r_km"], attr["z_km"]
    if not (np.allclose(r, ar) and np.allclose(z, az)):
        raise ValueError("Budget and inertial-stability grids do not match")
    _, _, w, masks = area_masks(r, z)
    ah = attr["hours"]
    rows = []
    for case in CASES:
        acase = ATTR_CASE[case]
        unstable_series = {
            domain: ats[f"{domain if domain == 'core' else 'inner'}_outflow_unstable_{acase}"].to_numpy(float)
            for domain in masks
        }
        derivatives = {
            domain: np.gradient(values, ats["hour"].to_numpy(float) * 3600.0)
            for domain, values in unstable_series.items()
        }
        for hour in budget["hours"]:
            ai = int(np.argmin(np.abs(ah - hour)))
            if abs(ah[ai] - hour) > 1e-6:
                continue
            bui = attr[f"I2_{acase}"][ai]
            classic = budget[budget_key(case, hour, "I")]
            eta = budget[budget_key(case, hour, "eta")]
            xi = budget[budget_key(case, hour, "xi")]
            metric = metrics[
                (metrics["case"] == case) &
                (np.isclose(metrics["hour"], hour)) &
                (metrics["form"] == "reynolds") &
                (metrics["domain"] == "inner_outflow")
            ].iloc[0]
            base = {
                "case": case, "hour": float(hour),
                "M_closure_corr": float(metric["M_closure_corr"]),
                "M_normalized_residual": float(metric["M_normalized_residual"]),
                "I_closure_corr": float(metric["I_closure_corr"]),
                "I_normalized_residual": float(metric["I_normalized_residual"]),
                "B_observed": boundary_contrast(budget[budget_key(case, hour, "local_m")], r, z),
                "B_sum": boundary_contrast(budget[budget_key(case, hour, "model_m")], r, z),
                "B_residual": boundary_contrast(budget[budget_key(case, hour, "residual_m")], r, z),
            }
            for group, names in {
                "mean": ("mean_r", "mean_z"),
                "eddy": ("eddy_r", "eddy_z"),
                "pgrad": ("pgrad",),
                "diffusion": ("diffusion",),
                "rdamp": ("rdamp",),
            }.items():
                field = sum(budget[budget_key(case, hour, f"M_{name}")] for name in names)
                base[f"B_{group}"] = boundary_contrast(field, r, z)
            for domain, mask in masks.items():
                bui_neg = bui <= 0
                classic_neg = classic <= 0
                base[f"{domain}_bui_unstable_fraction"] = weighted_fraction(bui_neg, w, mask)
                base[f"{domain}_classic_unstable_fraction"] = weighted_fraction(classic_neg, w, mask)
                base[f"{domain}_eta_negative_fraction"] = weighted_fraction(eta <= 0, w, mask)
                base[f"{domain}_sign_agreement"] = weighted_fraction(bui_neg == classic_neg, w, mask)
                base[f"{domain}_negative_jaccard"] = weighted_jaccard(bui_neg, classic_neg, w, mask)
                base[f"{domain}_bui_negative_strength"] = weighted_mean(np.maximum(-bui, 0), w, mask)
                base[f"{domain}_classic_negative_strength"] = weighted_mean(np.maximum(-classic, 0), w, mask)
                base[f"{domain}_unstable_fraction_tendency"] = float(np.interp(
                    hour, ats["hour"], derivatives[domain]
                ))
            base["xi_negative_fraction_inner"] = weighted_fraction(xi <= 0, w, masks["inner"])
            rows.append(base)
    return pd.DataFrame(rows)


def ri_metrics(ats):
    rows = []
    h = ats["hour"].to_numpy(float)
    for case in CASES:
        acase = ATTR_CASE[case]
        vmax = ats[f"vtmax_{acase}"].to_numpy(float)
        pmin = ats[f"pmin_{acase}"].to_numpy(float)
        for domain in ("core", "inner"):
            unstable = ats[f"{domain}_outflow_unstable_{acase}"].to_numpy(float)
            for lead in (6.0, 12.0, 24.0):
                vf = np.interp(h + lead, h, vmax, left=np.nan, right=np.nan)
                pf = np.interp(h + lead, h, pmin, left=np.nan, right=np.nan)
                dv = vf - vmax
                dpfall = pmin - pf
                controls_v = np.column_stack([(h - np.nanmean(h)) / np.nanstd(h), vmax])
                controls_p = np.column_stack([(h - np.nanmean(h)) / np.nanstd(h), pmin])
                rows.append({
                    "case": case, "domain": domain, "lead_hours": lead,
                    "n": int(np.count_nonzero(np.isfinite(unstable) & np.isfinite(dv))),
                    "pearson_unstable_future_dv": corr(unstable, dv),
                    "spearman_unstable_future_dv": rank_corr(unstable, dv),
                    "partial_time_vmax_future_dv": partial_corr(unstable, dv, controls_v),
                    "pearson_unstable_future_pressure_fall": corr(unstable, dpfall),
                    "partial_time_pmin_future_pressure_fall": partial_corr(unstable, dpfall, controls_p),
                })
    return pd.DataFrame(rows)


def add_future_intensity(ats):
    out = ats.copy()
    h = out["hour"].to_numpy(float)
    for case in CASES:
        acase = ATTR_CASE[case]
        vmax = out[f"vtmax_{acase}"].to_numpy(float)
        pmin = out[f"pmin_{acase}"].to_numpy(float)
        for lead in (6.0, 12.0, 24.0):
            out[f"{case}_future_{lead:g}h_dv"] = np.interp(
                h + lead, h, vmax, left=np.nan, right=np.nan
            ) - vmax
            out[f"{case}_future_{lead:g}h_pressure_fall"] = pmin - np.interp(
                h + lead, h, pmin, left=np.nan, right=np.nan
            )
        out[f"{case}_RI_reference"] = out[f"{case}_future_24h_dv"] >= RI_THRESHOLD
    return out


def add_zero_contour(ax, r, z, field, color="k", ls="--"):
    if np.nanmin(field) <= 0 <= np.nanmax(field):
        ax.contour(r, z, field, levels=[0], colors=color, linewidths=1.2, linestyles=ls)


def overlay_radial_outflow(ax, r, z, ur, label_contours=False):
    """Overlay axisymmetric radial wind; positive ur denotes outflow."""
    finite = ur[np.isfinite(ur)]
    if finite.size == 0:
        return
    upper = max(2.01, float(np.nanmax(finite)) + 0.01)
    if upper > 2.0:
        ax.contourf(r, z, ur, levels=[2.0, upper], colors=["#009E73"], alpha=0.10)
    levels = [level for level in (2.0, 5.0, 10.0) if np.nanmin(finite) <= level <= np.nanmax(finite)]
    if levels:
        cs = ax.contour(r, z, ur, levels=levels, colors="#007F5F", linewidths=[1.3] * len(levels))
        if label_contours:
            ax.clabel(cs, inline=True, fmt=lambda value: f"{value:g}", fontsize=7)
    if np.nanmin(finite) <= 0 <= np.nanmax(finite):
        ax.contour(r, z, ur, levels=[0], colors="#555555", linewidths=0.9, linestyles="-.")


def figure_state_relation(budget, attr, coupling, output, hour=72.0):
    r, z = budget["r_km"], budget["z_km"]
    ai = int(np.argmin(np.abs(attr["hours"] - hour)))
    view = (z[:, None] >= 10) & (z[:, None] <= 16) & (r[None, :] >= 50) & (r[None, :] <= 350)
    bui_fields, classic_fields, m_fields = [], [], []
    for case in CASES:
        bui_fields.append(attr[f"I2_{ATTR_CASE[case]}"][ai])
        classic_fields.append(budget[budget_key(case, hour, "I")])
        m_fields.extend([
            budget[budget_key(case, hour, "local_m")],
            budget[budget_key(case, hour, "model_m")],
        ])
    lim_bui = np.nanpercentile(np.abs(np.concatenate([x[view] for x in bui_fields])), 98)
    lim_classic = np.nanpercentile(np.abs(np.concatenate([x[view] for x in classic_fields])), 98)
    lim_m = np.nanpercentile(np.abs(np.concatenate([x[view] for x in m_fields])), 98)
    fig, ax = plt.subplots(2, 4, figsize=(18, 8.8), sharex=True, sharey=True, constrained_layout=True)
    ims = [None] * 4
    for i, case in enumerate(CASES):
        bui = bui_fields[i]
        classic = classic_fields[i]
        obs = budget[budget_key(case, hour, "local_m")]
        model = budget[budget_key(case, hour, "model_m")]
        ur = budget[budget_key(case, hour, "ur")]
        ims[0] = ax[i, 0].pcolormesh(r, z, bui * 1e11, shading="auto", cmap=nature_balance(),
                                     vmin=-lim_bui * 1e11, vmax=lim_bui * 1e11)
        add_zero_contour(ax[i, 0], r, z, classic, color="#009E73")
        ims[1] = ax[i, 1].pcolormesh(r, z, classic * 1e9, shading="auto", cmap=nature_balance(),
                                     vmin=-lim_classic * 1e9, vmax=lim_classic * 1e9)
        add_zero_contour(ax[i, 1], r, z, bui, color="k")
        for j, field in enumerate((obs, model), start=2):
            ims[j] = ax[i, j].pcolormesh(r, z, field, shading="auto", cmap=nature_balance(),
                                         vmin=-lim_m, vmax=lim_m)
            add_zero_contour(ax[i, j], r, z, bui, color="k")
        for j in range(4):
            overlay_radial_outflow(ax[i, j], r, z, ur, label_contours=(j == 0))
        row = coupling[(coupling.case == case) & np.isclose(coupling.hour, hour)].iloc[0]
        ax[i, 0].set_title(f"{case}: raw Bui $I^2$\nclassic/Bui negative Jaccard={row.inner_negative_jaccard:.2f}")
        ax[i, 1].set_title(f"{case}: classic I = xi M_r/r\nblack: Bui I² = 0")
        ax[i, 2].set_title(f"{case}: observed ΔM/Δt")
        ax[i, 3].set_title(f"{case}: sum of CM1 M-budget terms")
        for j in range(4):
            ax[i, j].set_xlim(50, 350); ax[i, j].set_ylim(10, 16)
            ax[i, j].set_xlabel("Radius (km)")
        ax[i, 0].set_ylabel("Height (km)")
    fig.colorbar(ims[0], ax=ax[:, 0], orientation="horizontal", pad=0.08,
                 label=r"Raw Bui $I^2$ ($10^{-11}$ scaled units)")
    fig.colorbar(ims[1], ax=ax[:, 1], orientation="horizontal", pad=0.08,
                 label=r"Classic inertial stability ($10^{-9}$ s$^{-2}$)")
    fig.colorbar(ims[3], ax=ax[:, 2:], orientation="horizontal", pad=0.08,
                 label=r"M tendency (m$^2$ s$^{-2}$)")
    fig.suptitle("State and tendency connection between angular momentum and inertial stability at 72 h",
                 fontsize=16, fontweight="bold", y=1.09)
    fig.legend(
        handles=[
            Patch(facecolor="#009E73", alpha=0.16, edgecolor="none",
                  label=r"Radial outflow layer: $u_r\geq2$ m s$^{-1}$"),
            Line2D([0], [0], color="#007F5F", lw=1.4,
                   label=r"Radial-wind contours: 2, 5, 10 m s$^{-1}$"),
            Line2D([0], [0], color="#555555", lw=0.9, ls="-.",
                   label=r"$u_r=0$"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 1.055), ncol=3, frameon=False,
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_budget_time(coupling, output):
    fig, ax = plt.subplots(2, 3, figsize=(17, 9), sharex="col", constrained_layout=True)
    for i, case in enumerate(CASES):
        q = coupling[coupling.case == case].sort_values("hour")
        ax[i, 0].plot(q.hour, q.inner_bui_unstable_fraction, "o-", color=COLORS[case], label="Bui $I^2<0$")
        ax[i, 0].plot(q.hour, q.inner_classic_unstable_fraction, "s--", color="#009E73", label="Classic $I<0$")
        ax[i, 0].set_ylabel("Area-weighted unstable fraction")
        ax[i, 0].set_ylim(0, max(0.5, 1.08 * q[["inner_bui_unstable_fraction", "inner_classic_unstable_fraction"]].max().max()))
        ax[i, 0].legend(frameon=False)
        for name, color, marker in (("B_observed", "k", "o"), ("B_sum", "#666666", "s"),
                                    ("B_mean", "#0072B2", "^"), ("B_eddy", "#D55E00", "D"),
                                    ("B_pgrad", "#CC79A7", "v")):
            ax[i, 1].plot(q.hour, q[name], marker=marker, label=name.replace("B_", ""), color=color)
        ax[i, 1].axhline(0, color="0.6", lw=0.8)
        ax[i, 1].set_ylabel(r"Outer − inner M tendency (m$^2$ s$^{-2}$)")
        ax[i, 1].set_title("Negative values erode radial M increase")
        ax[i, 1].legend(frameon=False, ncol=2, fontsize=8)
        ax[i, 2].plot(q.hour, q.M_closure_corr, "o-", color=COLORS[case], label="M spatial correlation")
        ax[i, 2].plot(q.hour, q.M_normalized_residual, "s--", color="#7A5195", label="M residual ratio")
        ax[i, 2].plot(q.hour, q.inner_sign_agreement, "D-.", color="#009E73", label="Bui/classic sign agreement")
        ax[i, 2].axhline(0.85, color="0.5", ls=":", lw=1)
        ax[i, 2].axhline(0.5, color="0.7", ls="--", lw=1)
        ax[i, 2].set_ylim(0, max(1.1, 1.05 * q.M_normalized_residual.max()))
        ax[i, 2].set_ylabel("Dimensionless")
        ax[i, 2].legend(frameon=False, fontsize=8)
        for j in range(3):
            ax[i, j].set_title(f"{case}: {ax[i, j].get_title()}" if ax[i, j].get_title() else case)
            ax[i, j].grid(alpha=0.2)
            ax[i, j].set_xlabel("Time (h)")
    fig.suptitle("Time-dependent connection between the M budget and inertial-instability area",
                 fontsize=16, fontweight="bold")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def figure_ri(ats, ri, output):
    h = ats.hour.to_numpy(float)
    fig, ax = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    for i, case in enumerate(CASES):
        acase = ATTR_CASE[case]
        unstable = ats[f"inner_outflow_unstable_{acase}"].to_numpy(float)
        vmax = ats[f"vtmax_{acase}"].to_numpy(float)
        future24 = ats[f"{case}_future_24h_dv"].to_numpy(float)
        ax[i, 0].plot(h, unstable, "o-", color=COLORS[case], label="Inner-outflow $I^2<0$ fraction")
        twin = ax[i, 0].twinx()
        twin.plot(h, vmax, "s--", color="k", label="Vt max")
        ri_mask = np.isfinite(future24) & (future24 >= RI_THRESHOLD)
        for hh in h[ri_mask]:
            ax[i, 0].axvspan(hh, hh + 5, color="#F0E442", alpha=0.18, lw=0)
        ax[i, 0].set_ylabel("Unstable fraction")
        twin.set_ylabel(r"Max azimuthal Vt (m s$^{-1}$)")
        ax[i, 0].set_title(f"{case}: yellow marks 24-h RI-reference start times")
        ax[i, 0].grid(alpha=0.2)
        use = np.isfinite(unstable) & np.isfinite(future24)
        ax[i, 1].scatter(unstable[use], future24[use], c=h[use], cmap="viridis", s=55, edgecolor="white")
        ax[i, 1].axhline(RI_THRESHOLD, color="#D55E00", ls="--", label="30 kt / 24 h")
        ax[i, 1].set_xlabel("Unstable fraction at initial time")
        ax[i, 1].set_ylabel(r"Future 24-h $\Delta V_t$ (m s$^{-1}$)")
        ax[i, 1].set_title(f"{case}: temporal association, color=time")
        ax[i, 1].legend(frameon=False)
        q = ri[(ri.case == case) & (ri.domain == "inner")]
        x = np.arange(len(q)); width = 0.36
        ax[i, 2].bar(x - width/2, q.pearson_unstable_future_dv, width, color=COLORS[case], label="Raw Pearson")
        ax[i, 2].bar(x + width/2, q.partial_time_vmax_future_dv, width, color="#7A5195", label="Partial: time + current V")
        ax[i, 2].axhline(0, color="0.5", lw=0.8)
        ax[i, 2].set_xticks(x, [f"{v:g} h" for v in q.lead_hours])
        ax[i, 2].set_ylim(-1, 1)
        ax[i, 2].set_ylabel("Correlation")
        ax[i, 2].set_title(f"{case}: instability vs future strengthening")
        ax[i, 2].legend(frameon=False, fontsize=8)
        for j in range(3): ax[i, j].set_xlabel(ax[i, j].get_xlabel() or "Time (h)")
    fig.suptitle("Does upper-level inertial instability precede rapid intensification?",
                 fontsize=16, fontweight="bold")
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize(coupling, ri, ats):
    out = {"verification_status": "ANALYZED_NOT_CAUSALLY_IDENTIFIED", "RI_threshold_m_s_per_24h": RI_THRESHOLD}
    out["at_72h"] = {}
    for case in CASES:
        q = coupling[(coupling.case == case) & np.isclose(coupling.hour, 72)].iloc[0]
        out["at_72h"][case] = {
            "bui_unstable_fraction": float(q.inner_bui_unstable_fraction),
            "classic_unstable_fraction": float(q.inner_classic_unstable_fraction),
            "negative_region_jaccard": float(q.inner_negative_jaccard),
            "sign_agreement": float(q.inner_sign_agreement),
            "boundary_observed": float(q.B_observed),
            "boundary_sum": float(q.B_sum),
            "boundary_mean": float(q.B_mean),
            "boundary_eddy": float(q.B_eddy),
            "boundary_pgrad": float(q.B_pgrad),
        }
    out["RI_reference"] = {}
    for case in CASES:
        flag = ats[f"{case}_RI_reference"].fillna(False).to_numpy(bool)
        future = ats[f"{case}_future_24h_dv"].to_numpy(float)
        out["RI_reference"][case] = {
            "number_of_sampled_start_times": int(np.sum(flag)),
            "maximum_future_24h_dv": float(np.nanmax(future)),
            "first_sampled_start_hour": float(ats.hour.to_numpy()[np.flatnonzero(flag)[0]]) if np.any(flag) else None,
        }
    out["inner_outflow_lag_metrics"] = ri[ri.domain == "inner"].to_dict("records")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget-npz", required=True)
    p.add_argument("--budget-metrics", required=True)
    p.add_argument("--attribution-npz", required=True)
    p.add_argument("--attribution-timeseries", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    budget = np.load(args.budget_npz)
    metrics = pd.read_csv(args.budget_metrics)
    attr = np.load(args.attribution_npz)
    ats = pd.read_csv(args.attribution_timeseries)
    ats = add_future_intensity(ats)
    coupling = build_coupling_rows(budget, metrics, attr, ats)
    ri = ri_metrics(ats)
    coupling.to_csv(out / "stability_budget_coupling_metrics.csv", index=False)
    ri.to_csv(out / "inertial_instability_ri_lag_metrics.csv", index=False)
    ats.to_csv(out / "intensity_instability_timeseries.csv", index=False)
    figure_state_relation(budget, attr, coupling, out / "figure9_state_relation_72h.png")
    figure_budget_time(coupling, out / "figure10_budget_stability_time.png")
    figure_ri(ats, ri, out / "figure11_inertial_instability_ri.png")
    summary = summarize(coupling, ri, ats)
    with open(out / "stability_budget_ri_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
