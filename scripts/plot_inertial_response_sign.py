"""Postprocess the existing corrected SE response; no new SE solve."""
from pathlib import Path
import csv, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("output/se_complete_boundary_fixed_window3h_new01")
OUT = Path("output/inertial_response_sign_boundary_fixed_new02")
OUT.mkdir(exist_ok=False)

def mean(f, weight, mask):
    return float(np.sum(f[mask] * weight[mask]) / np.sum(weight[mask]))

rows = []
for hour in range(50, 151, 10):
    with np.load(ROOT / f"window_{hour:03d}.npz") as d:
        a = {key: d[key] for key in d.files}
    r, z, rho = a["r_km"], a["z_km"], a["rho"]
    rr, zz = np.meshgrid(r, z)
    weight = rho * rr * np.gradient(r)[None, :] * np.gradient(z)[:, None]
    masks = {
        "bl": (rr >= 20) & (rr <= 150) & (zz <= 2),
        "up": (rr >= 20) & (rr <= 150) & (zz >= 2) & (zz <= 12),
        "out": (rr >= 50) & (rr <= 400) & (zz >= 10) & (zz <= 17),
    }
    row = {
        "hour": hour,
        "bl_inertial": -mean(a["inertial_u"], weight, masks["bl"]),
        "up_inertial": mean(a["inertial_w"], weight, masks["up"]),
        "out_inertial": mean(a["inertial_u"], weight, masks["out"]),
        "bl_actual_delta": -mean(a["delta_ur"], weight, masks["bl"]),
        "out_actual_delta": mean(a["delta_ur"], weight, masks["out"]),
    }
    signs = np.sign([row["bl_inertial"], row["up_inertial"], row["out_inertial"]])
    row["cell_sign"] = "positive" if np.all(signs > 0) else "negative" if np.all(signs < 0) else "mixed"
    row["qa"] = "early_amplitude_anomaly" if hour <= 70 else "usable"
    rows.append(row)

with (OUT / "signed_branch_metrics.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader(); writer.writerows(rows)

hours = np.array([row["hour"] for row in rows])
fig, axes = plt.subplots(3, 1, figsize=(10.5, 9), sharex=True, constrained_layout=True)
spec = [
    ("bl_inertial", "bl_actual_delta", "Boundary-layer inflow contribution (+ = stronger inflow)"),
    ("up_inertial", None, "Inner-core vertical contribution (+ = stronger ascent)"),
    ("out_inertial", "out_actual_delta", "Upper-level outflow contribution (+ = stronger outflow)"),
]
for ax, (response, observed, title) in zip(axes, spec):
    ax.axhline(0, color="0.4", lw=0.8)
    ax.axvspan(49, 71, color="0.85", alpha=0.7, label="50-70 h amplitude-anomaly interval")
    ax.plot(hours, [row[response] for row in rows], "o-", color="#2166ac", lw=2,
            label="Inertial equivalent-forcing SE response")
    if observed:
        ax.plot(hours, [row[observed] for row in rows], "o--", color="black", lw=1.3,
                label="Actual JET-CTRL radial-wind difference")
    ax.set_title(title); ax.set_ylabel("m s$^{-1}$"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
axes[-1].set_xlabel("Window centre time (h); each point is a +/-3 h solve-then-average")
fig.suptitle("Sign of inertial-stability equivalent-forcing response\n"
             "Positive means strengthening of the canonical branch; native-grid metrics")
fig.savefig(OUT / "inertial_response_signed_branches.png", dpi=200)
plt.close(fig)

# A separate trustworthy-period view avoids letting the anomalous early amplitudes
# compress the 80-150 h signal toward zero.
late = [row for row in rows if row["hour"] >= 80]
late_hours = np.array([row["hour"] for row in late])
fig, axes = plt.subplots(3, 1, figsize=(10.5, 9), sharex=True, constrained_layout=True)
for ax, (response, observed, title) in zip(axes, spec):
    ax.axhline(0, color="0.4", lw=0.8)
    ax.plot(late_hours, [row[response] for row in late], "o-", color="#2166ac", lw=2,
            label="Inertial equivalent-forcing SE response")
    if observed:
        ax.plot(late_hours, [row[observed] for row in late], "o--", color="black", lw=1.3,
                label="Actual JET-CTRL radial-wind difference")
    ax.set_title(title); ax.set_ylabel("m s$^{-1}$"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
axes[-1].set_xlabel("Window centre time (h); each point is a +/-3 h solve-then-average")
fig.suptitle("Sign of inertial-stability equivalent-forcing response: 80-150 h zoom\n"
             "Positive means strengthening of the canonical branch; native-grid metrics")
fig.savefig(OUT / "inertial_response_signed_branches_80_150h.png", dpi=200)
plt.close(fig)

usable = [row for row in rows if row["qa"] == "usable"]
summary = {
    "source": str(ROOT), "new_SE_solves": 0,
    "definition": {
        "positive_BL": "more inward flow: -mass-weighted mean(inertial_u), 20-150 km, z<=2 km",
        "positive_up": "more ascent: mass-weighted mean(inertial_w), 20-150 km, 2-12 km",
        "positive_outflow": "more outward flow: mass-weighted mean(inertial_u), 50-400 km, 10-17 km",
    },
    "classification_rule": "positive iff all three canonical branches are positive; negative iff all are negative; otherwise mixed",
    "usable_80_150_counts": {name: sum(row["cell_sign"] == name for row in usable) for name in ["positive", "negative", "mixed"]},
    "qa": "50-70 h retained but excluded from physical sign count due to anomalous amplitudes in the prior SE-vs-actual audit",
    "interpretation_limit": "regularized balanced SE projection; sign consistency is not closure or unique causality",
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2)); print(OUT / "inertial_response_signed_branches.png")
