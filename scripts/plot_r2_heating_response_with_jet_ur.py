"""Plot matched-pair heating SE radial response overlaid with absolute JET ur."""
import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def inertial_file(heat_file: Path, inert: Path) -> Path:
    match = re.search(r"J(\d+)p0_C(\d+)p0", heat_file.name)
    if not match:
        raise ValueError(f"Cannot parse pair from {heat_file.name}")
    jet, ctrl = map(int, match.groups())
    path = inert / f"inertial_operator_J{jet:03d}_C{ctrl:03d}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heating-dir", type=Path, required=True)
    parser.add_argument("--inertial-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    heat, inert = args.heating_dir, args.inertial_dir
    out = heat / "figure_heating_SE_radial_response_with_absolute_jet_ur.png"
    files = sorted(heat.glob("J*p0_C*p0_diabatic_validation.npz"))
    if not files:
        raise FileNotFoundError(f"No heating results under {heat}")
    payload = [(p, np.load(p), np.load(inertial_file(p, inert))) for p in files]
    vmax = np.percentile(
        np.concatenate([np.abs(x[1]["Q_diabatic_u"]).ravel() for x in payload]), 99
    )
    vmax = max(float(vmax), 0.05)
    fig, axes = plt.subplots(1, len(payload), figsize=(4.5 * len(payload), 5.0),
                             sharex=True, sharey=True, constrained_layout=True)
    levels = np.array([-8, -4, -2, -1, -0.5, 0.5, 1, 2, 4, 8], dtype=float)
    for ax, (path, h, i) in zip(axes, payload):
        r, z = h["r_km"], h["z_km"]
        cf = ax.contourf(r, z, h["Q_diabatic_u"], levels=np.linspace(-vmax, vmax, 25),
                         cmap="RdBu_r", extend="both")
        # The heating and inertial diagnostics deliberately use different
        # grids.  Put the absolute JET wind onto the heating-response grid.
        interpolator = RegularGridInterpolator(
            (i["z_km"], i["r_km"]), i["ur_jet"], bounds_error=False, fill_value=np.nan
        )
        zz, rr = np.meshgrid(z, r, indexing="ij")
        ur = interpolator(np.column_stack([zz.ravel(), rr.ravel()])).reshape(zz.shape)
        # Absolute JET radial wind: solid inflow, dashed outflow.
        if np.nanmin(ur) < 0:
            ax.contour(r, z, ur, levels=levels[levels < 0], colors="k", linewidths=0.75)
        if np.nanmax(ur) > 0:
            ax.contour(r, z, ur, levels=levels[levels > 0], colors="k", linewidths=0.75,
                       linestyles="--")
        ax.axhspan(10, 17, facecolor="gold", alpha=0.12, zorder=0)
        pair = re.search(r"J(\d+)p0_C(\d+)p0", path.name).groups()
        ax.set_title(f"JET {pair[0]} h / CTRL {pair[1]} h")
        ax.set_xlim(0, 300)
        ax.set_ylim(0, 20)
        ax.set_xlabel("Radius (km)")
    axes[0].set_ylabel("Height (km)")
    cbar = fig.colorbar(cf, ax=axes, pad=0.02, shrink=0.92)
    cbar.set_label(r"$u_r$ from $\Delta Q_{diab}$ SE projection (m s$^{-1}$)")
    fig.suptitle(f"{args.label}: diabatic-heating SE radial response; contours: absolute JET radial wind\n"
                 "solid: inflow; dashed: outflow; shaded: 10–17 km", y=1.04)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
