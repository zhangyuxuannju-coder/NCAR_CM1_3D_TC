#!/usr/bin/env python3
"""Compare resolved absolute-angular-momentum transport in R2 JET and CTRL.

This uses hourly storm-centred reduced fields already diagnosed from the two
R2 Thompson/OML100 integrations.  It deliberately does not combine CM1 ub_*/
vb_* snapshots with hourly state differences: those source terms are output-
time RK snapshots, not interval accumulations.  The displayed terms are the
four transport contributions that are well defined from a single model state.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


ROOT = Path("output/jet_outflow_mechanism_validation")
DATA = ROOT / "data"
OUT = ROOT / "angular_momentum_transport_budget"
FIG = OUT / "figures"

TERMS = (
    ("mean_r", r"Mean radial: $-\bar u_r\,\partial_r\bar M_a$"),
    ("mean_z", r"Mean vertical: $-\bar w\,\partial_z\bar M_a$"),
    ("eddy_r", r"Eddy radial: $rF'_{\lambda,r}$"),
    ("eddy_z", r"Eddy vertical: $rF'_{\lambda,z}$"),
    ("resolved", "Sum of four resolved transports"),
)
COLORS = {
    "mean_r": "#0072B2", "mean_z": "#56B4E9",
    "eddy_r": "#D55E00", "eddy_z": "#E69F00", "resolved": "#000000",
}


def smooth_r(a: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return gaussian_filter1d(np.asarray(a, float), sigma=sigma, axis=-1, mode="nearest")


def load_state(case: str, hour: float) -> dict[str, np.ndarray]:
    path = DATA / f"{case}_{hour:06.1f}h.npz"
    with np.load(path) as d:
        q = {k: np.asarray(d[k], float) for k in d.files}
    r_m = q["r_km"] * 1000.0
    z_m = q["z_km"] * 1000.0
    m = smooth_r(q["M"])
    dm_dr = np.gradient(m, r_m, axis=1, edge_order=2)
    dm_dz = np.gradient(m, z_m, axis=0, edge_order=2)
    mean_r = -smooth_r(q["ur_zr"]) * dm_dr
    mean_z = -smooth_r(q["w_zr"]) * dm_dz
    eddy_r = r_m[None, :] * smooth_r(q["F_lambda_eddy_radial"])
    eddy_z = r_m[None, :] * smooth_r(q["F_lambda_eddy_vertical"])
    resolved = mean_r + mean_z + eddy_r + eddy_z
    return {
        "r_km": q["r_km"], "z_km": q["z_km"], "M": m,
        "mean_r": mean_r, "mean_z": mean_z,
        "eddy_r": eddy_r, "eddy_z": eddy_z, "resolved": resolved,
    }


def read_pairs(phase: str) -> list[dict[str, str]]:
    with (ROOT / f"matched_{phase}_pairs.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def composite(case: str, pairs: list[dict[str, str]]) -> dict[str, np.ndarray]:
    states = [load_state(case, float(p["jet_h"] if case == "JET" else p["ctrl_h"])) for p in pairs]
    return {
        "r_km": states[0]["r_km"], "z_km": states[0]["z_km"],
        **{name: np.nanmean([q[name] for q in states], axis=0) for name, _ in TERMS},
    }


def robust_limit(fields: list[np.ndarray], r: np.ndarray, z: np.ndarray, percentile=98.5) -> float:
    region = ((r[None, :] <= 900.0) & (z[:, None] <= 20.0))
    vals = np.concatenate([np.abs(a[region & np.isfinite(a)]) for a in fields])
    return max(float(np.nanpercentile(vals, percentile)), 1e-12)


def plot_matched(phase: str) -> Path:
    pairs = read_pairs(phase)
    jet = composite("JET", pairs)
    ctrl = composite("CTRL", pairs)
    r, z = jet["r_km"], jet["z_km"]
    rr, zz = np.meshgrid(r, z)
    fig, axes = plt.subplots(3, len(TERMS), figsize=(21, 10.5), sharex=True, sharey=True,
                             constrained_layout=True)
    for col, (name, title) in enumerate(TERMS):
        absolute = [jet[name], ctrl[name]]
        delta = jet[name] - ctrl[name]
        lim_abs = robust_limit(absolute, r, z)
        lim_delta = robust_limit([delta], r, z)
        last_abs = None
        for row, (label, field) in enumerate((("JET", jet[name]), ("CTRL", ctrl[name]))):
            ax = axes[row, col]
            last_abs = ax.contourf(rr, zz, field / 1e3, levels=np.linspace(-lim_abs, lim_abs, 25) / 1e3,
                                   cmap="RdBu_r", extend="both")
            ax.set_title(title if row == 0 else "")
            if col == 0: ax.set_ylabel(f"{label}\nHeight (km)")
        ax = axes[2, col]
        last_delta = ax.contourf(rr, zz, delta / 1e3,
                                 levels=np.linspace(-lim_delta, lim_delta, 25) / 1e3,
                                 cmap="RdBu_r", extend="both")
        if col == 0: ax.set_ylabel("JET − CTRL\nHeight (km)")
        for row in range(3):
            a = axes[row, col]
            a.set_xlim(0, 900); a.set_ylim(0, 20); a.set_xlabel("Radius (km)")
            a.axhline(10, color="0.35", ls=":", lw=.7); a.axhline(17, color="0.35", ls=":", lw=.7)
            a.axvline(300, color="0.35", ls=":", lw=.7); a.axvline(600, color="0.35", ls=":", lw=.7)
        fig.colorbar(last_abs, ax=axes[:2, col], shrink=.72, pad=.01,
                     label=r"absolute ($10^3$ m$^2$ s$^{-2}$)")
        fig.colorbar(last_delta, ax=axes[2, col], shrink=.72, pad=.01,
                     label=r"difference ($10^3$ m$^2$ s$^{-2}$)")
    pair_text = ", ".join(f"J{float(p['jet_h']):g}/C{float(p['ctrl_h']):g}" for p in pairs)
    fig.suptitle(
        f"Matched absolute-angular-momentum transport: {phase.replace('_', '–')} h ({len(pairs)} pairs)\n"
        "Positive = local spin-up of $M_a$; resolved transport terms only, not a closed hourly budget\n"
        + pair_text, fontsize=13,
    )
    path = FIG / f"matched_{phase}_angular_momentum_transport.png"
    fig.savefig(path, dpi=210); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)
    return path


def volume_mean(field: np.ndarray, r: np.ndarray, z: np.ndarray,
                r_bounds: tuple[float, float], z_bounds: tuple[float, float]) -> float:
    rr, zz = np.meshgrid(r, z)
    mask = ((rr >= r_bounds[0]) & (rr <= r_bounds[1]) &
            (zz >= z_bounds[0]) & (zz <= z_bounds[1]) & np.isfinite(field))
    weight = np.maximum(rr, 0.5 * np.nanmedian(np.diff(r)))
    return float(np.sum(field[mask] * weight[mask]) / np.sum(weight[mask]))


def intensity_lookup() -> dict[tuple[str, float], dict[str, float]]:
    with (ROOT / "timeseries.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(r["case"], float(r["time_h"])): {k: float(r[k]) for k in ("vmax_ms", "rmw_km", "pmin_hpa")}
            for r in rows}


def domain_values(state: dict[str, np.ndarray], rmw: float, domain: str) -> dict[str, float]:
    if domain == "rmw_lowlevel":
        rb, zb = (max(5.0, 0.5 * rmw), max(15.0, 2.0 * rmw)), (0.5, 5.0)
    elif domain == "upper_outflow":
        rb, zb = (100.0, 600.0), (10.0, 17.0)
    else:
        raise ValueError(domain)
    return {name: volume_mean(state[name], state["r_km"], state["z_km"], rb, zb)
            for name, _ in TERMS}


def build_metrics() -> list[dict[str, object]]:
    lookup = intensity_lookup()
    rows: list[dict[str, object]] = []
    for comparison, phase in (("matched", "50_75"), ("matched", "80_150")):
        for ipair, p in enumerate(read_pairs(phase), start=1):
            for case in ("JET", "CTRL"):
                hour = float(p["jet_h"] if case == "JET" else p["ctrl_h"])
                state = load_state(case, hour); meta = lookup[(case, hour)]
                for domain in ("rmw_lowlevel", "upper_outflow"):
                    vals = domain_values(state, meta["rmw_km"], domain)
                    rows.append({"comparison": comparison, "phase": phase, "pair": ipair,
                                 "case": case, "hour": hour, "domain": domain,
                                 "vmax_ms": meta["vmax_ms"], "rmw_km": meta["rmw_km"], **vals})
    for hour in np.arange(50.0, 150.1, 5.0):
        for case in ("JET", "CTRL"):
            state = load_state(case, hour); meta = lookup[(case, hour)]
            for domain in ("rmw_lowlevel", "upper_outflow"):
                vals = domain_values(state, meta["rmw_km"], domain)
                rows.append({"comparison": "same_time", "phase": "50_150", "pair": "",
                             "case": case, "hour": hour, "domain": domain,
                             "vmax_ms": meta["vmax_ms"], "rmw_km": meta["rmw_km"], **vals})
    return rows


def plot_timeseries(rows: list[dict[str, object]]) -> Path:
    data = [r for r in rows if r["comparison"] == "same_time"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, constrained_layout=True)
    for row, domain in enumerate(("rmw_lowlevel", "upper_outflow")):
        d = [q for q in data if q["domain"] == domain]
        times = sorted({float(q["hour"]) for q in d})
        by = {(q["case"], float(q["hour"])): q for q in d}
        for case, color, ls in (("CTRL", "#303030", "--"), ("JET", "#b2182b", "-")):
            axes[row, 0].plot(times, [by[(case, t)]["resolved"] for t in times],
                              color=color, ls=ls, marker="o", ms=3, label=case)
        for name, _ in TERMS[:-1]:
            axes[row, 1].plot(times,
                              [(by[("JET", t)][name] - by[("CTRL", t)][name]) for t in times],
                              color=COLORS[name], marker="o", ms=3, label=name.replace("_", " "))
        axes[row, 0].set_title(("0.5–2 RMW, 0.5–5 km" if domain == "rmw_lowlevel" else
                                "100–600 km, 10–17 km") + ": resolved total")
        axes[row, 1].set_title(("Low-level RMW annulus" if domain == "rmw_lowlevel" else
                                "Upper outflow") + ": JET − CTRL components")
        for col in range(2):
            axes[row, col].axhline(0, color="0.4", lw=.8)
            axes[row, col].axvspan(50, 75, color="#fdae61", alpha=.08)
            axes[row, col].axvspan(80, 150, color="#2c7bb6", alpha=.06)
            axes[row, col].grid(alpha=.25)
            axes[row, col].set_ylabel(r"Domain mean (m$^2$ s$^{-2}$)")
    axes[0, 0].legend(frameon=False); axes[0, 1].legend(frameon=False, ncol=2, fontsize=9)
    axes[1, 0].set_xlabel("Same model time (h)"); axes[1, 1].set_xlabel("Same model time (h)")
    fig.suptitle("Same-time resolved absolute-angular-momentum transport\n"
                 "Positive = local spin-up; native hourly tendency closure is unavailable", fontsize=14)
    path = FIG / "same_time_angular_momentum_transport_timeseries.png"
    fig.savefig(path, dpi=210); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)
    return path


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    matched = [q for q in rows if q["comparison"] == "matched"]
    for phase in ("50_75", "80_150"):
        for domain in ("rmw_lowlevel", "upper_outflow"):
            d = [q for q in matched if q["phase"] == phase and q["domain"] == domain]
            pair_ids = sorted({int(q["pair"]) for q in d})
            by = {(q["case"], int(q["pair"])): q for q in d}
            for name, _ in TERMS:
                delta = np.array([float(by[("JET", p)][name]) - float(by[("CTRL", p)][name]) for p in pair_ids])
                out.append({"phase": phase, "domain": domain, "term": name, "n_pairs": len(delta),
                            "delta_mean_m2s2": float(np.mean(delta)),
                            "delta_median_m2s2": float(np.median(delta)),
                            "delta_q25_m2s2": float(np.percentile(delta, 25)),
                            "delta_q75_m2s2": float(np.percentile(delta, 75)),
                            "positive_fraction": float(np.mean(delta > 0))})
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(exist_ok=True)
    paths = [plot_matched("50_75"), plot_matched("80_150")]
    rows = build_metrics(); paths.append(plot_timeseries(rows))
    with (OUT / "angular_momentum_transport_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = summarize(rows)
    with (OUT / "matched_transport_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    report = {
        "definition": "M_a = r v_t + 0.5 f r^2; positive tendency spins up absolute angular momentum",
        "terms": {
            "mean_r": "-ubar_r dMbar_a/dr", "mean_z": "-wbar dMbar_a/dz",
            "eddy_r": "r times radial Reynolds-eddy tangential acceleration",
            "eddy_z": "r times vertical Reynolds-eddy tangential acceleration",
            "resolved": "sum of the four terms",
        },
        "averaging": "storm-centred azimuthal Reynolds mean; radial Gaussian sigma=1 bin",
        "matched_pairs": {p: [{"jet_h": float(q["jet_h"]), "ctrl_h": float(q["ctrl_h"])}
                              for q in read_pairs(p)] for p in ("50_75", "80_150")},
        "closure_status": "NOT CLOSED: CM1 ub/vb fields are hourly RK snapshots, not interval accumulations",
        "interpretation_boundary": "resolved instantaneous transport comparison; not full causal attribution",
        "figures": [str(p) for p in paths],
    }
    (OUT / "README.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"figures": [str(p) for p in paths], "summary_rows": len(summary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
