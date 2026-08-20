"""CLI adapter for raw CTRL/JET Bui SE applicability diagnostics."""

from __future__ import annotations

import json
from dataclasses import replace

from ._se_pipeline_single import azimuthal_average_from_3d
from .environmental_cli import config_from_args
from .se_applicability import compute_applicability_diagnostics, write_applicability_products


def run_from_args(args) -> None:
    if not args.jet_input_file:
        raise ValueError("--mode stability requires --jet-input-file")
    # I2/D and F_lambda_env need the 3-D state and direct eddy fluxes, not the
    # many optional CM1 budget arrays.  Skipping them greatly reduces server I/O.
    ctrl_cfg = replace(config_from_args(args), include_model_budget_terms=False)
    jet_cfg = replace(ctrl_cfg, input_file=args.jet_input_file)

    print("[INFO] Reading CTRL 3-D fields and diagnosing azimuthal eddy fluxes")
    ctrl = azimuthal_average_from_3d(ctrl_cfg)
    print("[INFO] Reading JET 3-D fields and diagnosing azimuthal eddy fluxes")
    jet = azimuthal_average_from_3d(jet_cfg)
    ctrl_time = float(ctrl["time_seconds_used"][0])
    jet_time = float(jet["time_seconds_used"][0])
    if abs(ctrl_time - jet_time) > 1.0:
        raise ValueError(
            "CTRL and JET selected times differ by more than one second: "
            f"CTRL={ctrl_time}, JET={jet_time}"
        )
    arrays, summary = compute_applicability_diagnostics(
        ctrl,
        jet,
        coriolis_f=ctrl_cfg.coriolis_f,
        theta_floor=ctrl_cfg.theta_floor,
        outer_smooth_window=ctrl_cfg.theta_outer_smooth_window,
        regularization_eps_ratio=ctrl_cfg.inertia_eps_ratio,
        regularization_margin=ctrl_cfg.elliptic_margin,
    )
    summary.update(
        {
            "ctrl_input": ctrl_cfg.input_file,
            "jet_input": args.jet_input_file,
            "ctrl_time_seconds": ctrl_time,
            "jet_time_seconds": jet_time,
            "eddy_average": args.eddy_average,
            "imposed_jet_axis_km": [
                args.stability_jet_axis_r_km,
                args.stability_jet_axis_z_km,
            ],
            "ctrl_center_km": [float(ctrl["center_x_km"][0]), float(ctrl["center_y_km"][0])],
            "jet_center_km": [float(jet["center_x_km"][0]), float(jet["center_y_km"][0])],
            "warning": (
                "Raw I2/D determine classical SE applicability. Regularized fields are "
                "only a nearby balanced-state comparison."
            ),
        }
    )
    outputs = write_applicability_products(
        arrays,
        summary,
        ctrl_cfg.output_dir,
        write_netcdf=ctrl_cfg.write_netcdf,
        make_plots=ctrl_cfg.plot_solution,
        outflow_threshold_ms=args.stability_outflow_threshold,
        jet_speed_threshold_ms=args.stability_jet_speed_threshold,
        forcing_contour_percentile=args.stability_forcing_percentile,
        jet_axis_r_km=args.stability_jet_axis_r_km,
        jet_axis_z_km=args.stability_jet_axis_z_km,
    )
    print("[INFO] Raw Bui SE applicability diagnostic complete")
    print(json.dumps({"outputs": outputs, "ctrl": summary["ctrl"], "jet": summary["jet"]}, indent=2))
