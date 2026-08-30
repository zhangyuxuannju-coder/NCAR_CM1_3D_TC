#!/usr/bin/env python3
"""Plot the time evolution of the regularized SE operator-only response.

The factorial definition follows run_se_forcing_operator_factorial_full.py:
operator effect = JC - CC, so forcing is held at CTRL while only the SE
operator changes from CTRL to JET.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm


DIV_CMAP = LinearSegmentedColormap.from_list(
    "operator_diverging", ["#3C5488", "#88A9CF", "#F7F7F4", "#E99B7B", "#B9363E"]
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jet15-dir", default="output/se_factorial_25N_jet15_eps1e_5")
    p.add_argument("--jet30-dir", default="output/se_factorial_25N_jet30_eps1e_5")
    p.add_argument(
        "--metrics-json",
        default="output/shear_vs_operator_intensity_25N/shear_operator_intensity_summary.json",
    )
    p.add_argument("--output-dir", default="output/operator_effect_evolution_25N")
    p.add_argument("--shared-percentile", type=float, default=98.5)
    p.add_argument("--local-percentile", type=float, default=98.0)
    return p.parse_args()


def robust_scale(a: np.ndarray, percentile: float, floor: float = 1.0e-12) -> float:
    finite = np.abs(np.asarray(a, dtype=float))
    finite = finite[np.isfinite(finite)]
    return max(float(np.percentile(finite, percentile)), floor) if finite.size else floor


def load_metrics(path: Path) -> dict[tuple[str, float], dict]:
    content = json.loads(path.read_text(encoding="utf-8"))
    return {(r["case"], float(r["hour"])): r for r in content["records"]}


def load_case(name: str, folder: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(folder.glob("full_domain_factorial_*h.npz")):
        with np.load(path) as d:
            record = {
                "case": name,
                "r_km": np.asarray(d["r_km"], dtype=float),
                "z_km": np.asarray(d["z_km"], dtype=float),
                "psi": np.asarray(d["psi_operator_effect"], dtype=float),
                "u": np.asarray(d["u_operator_effect"], dtype=float),
                "w": np.asarray(d["w_operator_effect"], dtype=float),
                "changed": np.asarray(d["changed_ctrl"], bool) | np.asarray(d["changed_jet"], bool),
            }
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        record["hour"] = float(meta["hour"])
        records.append(record)
    if not records:
        raise FileNotFoundError(f"No factorial NPZ files found in {folder}")
    return records


def style_axis(ax: plt.Axes, row: int, col: int, nrows: int) -> None:
    ax.set(xlim=(0, 300), ylim=(0, 20), xticks=[0, 100, 200, 300], yticks=[0, 5, 10, 15, 20])
    ax.axhline(10, color="0.35", lw=0.45, ls=":")
    ax.axhline(16, color="0.35", lw=0.45, ls=":")
    ax.grid(color="0.75", lw=0.35, alpha=0.28)
    if col == 0:
        ax.set_ylabel("Height (km)")
    if row == nrows - 1:
        ax.set_xlabel("Radius (km)")


def add_regularization_dots(ax: plt.Axes, rec: dict) -> None:
    zz, rr = np.where(rec["changed"])
    if zz.size:
        ax.scatter(
            rec["r_km"][rr], rec["z_km"][zz], s=2.2, c="0.15", marker=".",
            alpha=0.32, linewidths=0, rasterized=True,
        )


def add_case_label(ax: plt.Axes, name: str) -> None:
    ax.text(
        0.02, 0.97, name, transform=ax.transAxes, va="top", ha="left", fontsize=10,
        fontweight="bold", bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.2},
    )


def metric_title(rec: dict, metrics: dict[tuple[str, float], dict]) -> str:
    m = metrics[(rec["case"], rec["hour"])]
    return (
        f"{rec['hour']:.0f} h\nOUT {m['upper_outflow_u_mean_m_s']:+.2f}, "
        f"IN {m['low_inflow_strength_change_m_s']:+.2f} m s$^{{-1}}$"
    )


def plot_psi_shared(cases: dict[str, list[dict]], metrics: dict, out: Path, percentile: float) -> None:
    all_psi = np.concatenate([r["psi"].ravel() for rows in cases.values() for r in rows]) / 1.0e9
    vmax = robust_scale(all_psi, percentile)
    names = list(cases)
    hours = [r["hour"] for r in cases[names[0]]]
    fig, axes = plt.subplots(
        len(names), len(hours), figsize=(3.05 * len(hours), 6.45), sharex=True, sharey=True,
        constrained_layout=True, squeeze=False,
    )
    mappable = None
    for i, name in enumerate(names):
        for j, rec in enumerate(cases[name]):
            ax = axes[i, j]
            field = rec["psi"] / 1.0e9
            mappable = ax.pcolormesh(
                rec["r_km"], rec["z_km"], field, shading="auto", cmap=DIV_CMAP,
                norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax), rasterized=True,
            )
            if np.nanmin(field) < 0 < np.nanmax(field):
                ax.contour(rec["r_km"], rec["z_km"], field, levels=[0], colors="0.2", linewidths=0.65)
            add_regularization_dots(ax, rec)
            ax.set_title(metric_title(rec, metrics), fontsize=9.2)
            style_axis(ax, i, j, len(names))
            if j == 0:
                add_case_label(ax, name)
    cb = fig.colorbar(mappable, ax=axes, orientation="horizontal", shrink=0.72, pad=0.045)
    cb.set_label(
        f"Operator-only streamfunction effect, $\\psi_{{op}}$ ($10^9$ kg s$^{{-1}}$); "
        f"shared P{percentile:g}=±{vmax:.2f}"
    )
    fig.suptitle(
        "SE operator-only secondary-circulation evolution — shared amplitude scale\n"
        "positive ψ: direct cell (low-level inflow / inner ascent / upper outflow); dots: regularized coefficients",
        fontweight="bold",
    )
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_psi_normalized(cases: dict[str, list[dict]], metrics: dict, out: Path, percentile: float) -> None:
    names = list(cases)
    hours = [r["hour"] for r in cases[names[0]]]
    fig, axes = plt.subplots(
        len(names), len(hours), figsize=(3.05 * len(hours), 6.55), sharex=True, sharey=True,
        constrained_layout=True, squeeze=False,
    )
    mappable = None
    for i, name in enumerate(names):
        for j, rec in enumerate(cases[name]):
            ax = axes[i, j]
            local = robust_scale(rec["psi"], percentile)
            normalized = np.clip(rec["psi"] / local, -1.0, 1.0)
            mappable = ax.pcolormesh(
                rec["r_km"], rec["z_km"], normalized, shading="auto", cmap=DIV_CMAP,
                norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0), rasterized=True,
            )
            if np.nanmin(normalized) < 0 < np.nanmax(normalized):
                ax.contour(rec["r_km"], rec["z_km"], normalized, levels=[0], colors="0.2", linewidths=0.65)
            for lev, ls in ((-0.5, "--"), (0.5, "-")):
                if np.nanmin(normalized) <= lev <= np.nanmax(normalized):
                    ax.contour(
                        rec["r_km"], rec["z_km"], normalized, levels=[lev], colors="0.12",
                        linewidths=0.55, linestyles=ls,
                    )
            add_regularization_dots(ax, rec)
            ax.set_title(metric_title(rec, metrics), fontsize=9.2)
            ax.text(
                0.98, 0.03, f"P{percentile:g}={local / 1.0e9:.2f}×10⁹ kg s⁻¹",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.4,
                bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.0},
            )
            style_axis(ax, i, j, len(names))
            if j == 0:
                add_case_label(ax, name)
    cb = fig.colorbar(mappable, ax=axes, orientation="horizontal", shrink=0.72, pad=0.045)
    cb.set_label(f"$\\psi_{{op}}$/local P{percentile:g}(|$\\psi_{{op}}$|); solid/dashed contour = +0.5/−0.5")
    fig.suptitle(
        "SE operator-only cell structure — each panel normalized independently\n"
        "use the panel P-value for amplitude; positive ψ is a direct TC secondary-circulation cell",
        fontweight="bold",
    )
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_u_normalized(cases: dict[str, list[dict]], metrics: dict, out: Path, percentile: float) -> None:
    names = list(cases)
    hours = [r["hour"] for r in cases[names[0]]]
    fig, axes = plt.subplots(
        len(names), len(hours), figsize=(3.05 * len(hours), 6.65), sharex=True, sharey=True,
        constrained_layout=True, squeeze=False,
    )
    mappable = None
    for i, name in enumerate(names):
        for j, rec in enumerate(cases[name]):
            ax = axes[i, j]
            # Exclude the innermost 20 km when setting velocity scales.  The
            # u,w conversion contains 1/(rho*r), so axis-adjacent values are
            # not representative of the resolved secondary circulation.
            outer = rec["r_km"] >= 20.0
            uscale = robust_scale(rec["u"][:, outer], percentile)
            wscale = robust_scale(rec["w"][:, outer], percentile)
            un = np.clip(rec["u"] / uscale, -1.0, 1.0)
            wn = rec["w"] / wscale
            mappable = ax.pcolormesh(
                rec["r_km"], rec["z_km"], un, shading="auto", cmap=DIV_CMAP,
                norm=TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0), rasterized=True,
            )
            if np.nanmin(un) < 0 < np.nanmax(un):
                ax.contour(rec["r_km"], rec["z_km"], un, levels=[0], colors="0.2", linewidths=0.55)
            if np.nanmin(wn) <= -0.5:
                ax.contour(rec["r_km"], rec["z_km"], wn, levels=[-0.5], colors="#3C5488", linewidths=0.8, linestyles="--")
            if np.nanmax(wn) >= 0.5:
                ax.contour(rec["r_km"], rec["z_km"], wn, levels=[0.5], colors="#00876C", linewidths=0.8)
            add_regularization_dots(ax, rec)
            m = metrics[(name, rec["hour"])]
            ax.set_title(
                f"{rec['hour']:.0f} h\nOUT {m['upper_outflow_u_mean_m_s']:+.2f} | "
                f"IN {m['low_inflow_strength_change_m_s']:+.2f}",
                fontsize=8.8,
            )
            ax.text(
                0.98, 0.03,
                f"U P{percentile:g}=±{uscale:.2f} m s⁻¹\n"
                f"W P{percentile:g}=±{1000*wscale:.1f} mm s⁻¹\n"
                f"Wcore={1000*m['core_ascent_w_mean_m_s']:+.1f} mm s⁻¹",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.2,
                bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.0},
            )
            style_axis(ax, i, j, len(names))
            ax.set_xlim(20, 300)
            ax.set_xticks([20, 100, 200, 300])
            if j == 0:
                add_case_label(ax, name)
    cb = fig.colorbar(mappable, ax=axes, orientation="horizontal", shrink=0.72, pad=0.045)
    cb.set_label(
        f"Radial-wind operator effect / local P{percentile:g}; red=more outward, blue=more inward; "
        "green solid / blue dashed = upward / downward W at 0.5 local P"
    )
    fig.suptitle(
        "SE operator-only radial/vertical circulation structure — panel-normalized\n"
        "r≥20 km; OUT and IN are favourable-direction regional means; dots mark regularized coefficients",
        fontweight="bold",
    )
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_metrics(cases: dict[str, list[dict]], metrics: dict, out: Path, percentile: float) -> None:
    fields = [
        "case", "hour", "psi_p98_1e9_kg_s", "u_p98_m_s", "w_p98_mm_s",
        "upper_outflow_u_mean_m_s", "low_inflow_strength_change_m_s", "core_ascent_w_mean_mm_s",
        "regularized_union_fraction",
    ]
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name, rows in cases.items():
            for rec in rows:
                m = metrics[(name, rec["hour"])]
                outer = rec["r_km"] >= 20.0
                writer.writerow(
                    {
                        "case": name,
                        "hour": rec["hour"],
                        "psi_p98_1e9_kg_s": robust_scale(rec["psi"], percentile) / 1.0e9,
                        "u_p98_m_s": robust_scale(rec["u"][:, outer], percentile),
                        "w_p98_mm_s": 1000 * robust_scale(rec["w"][:, outer], percentile),
                        "upper_outflow_u_mean_m_s": m["upper_outflow_u_mean_m_s"],
                        "low_inflow_strength_change_m_s": m["low_inflow_strength_change_m_s"],
                        "core_ascent_w_mean_mm_s": 1000 * m["core_ascent_w_mean_m_s"],
                        "regularized_union_fraction": float(np.mean(rec["changed"])),
                    }
                )


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(Path(args.metrics_json))
    cases = {
        "JET15": load_case("JET15", Path(args.jet15_dir)),
        "JET30": load_case("JET30", Path(args.jet30_dir)),
    }
    expected = [r["hour"] for r in cases["JET15"]]
    if [r["hour"] for r in cases["JET30"]] != expected:
        raise ValueError("JET15 and JET30 factorial times differ")
    plot_psi_shared(cases, metrics, out / "operator_streamfunction_shared_scale.png", args.shared_percentile)
    plot_psi_normalized(cases, metrics, out / "operator_streamfunction_panel_normalized.png", args.local_percentile)
    plot_u_normalized(cases, metrics, out / "operator_radial_vertical_panel_normalized.png", args.local_percentile)
    write_metrics(cases, metrics, out / "operator_effect_structure_metrics.csv", args.local_percentile)
    print(json.dumps({
        "hours": expected,
        "outputs": [
            str(out / "operator_streamfunction_shared_scale.png"),
            str(out / "operator_streamfunction_panel_normalized.png"),
            str(out / "operator_radial_vertical_panel_normalized.png"),
            str(out / "operator_effect_structure_metrics.csv"),
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
