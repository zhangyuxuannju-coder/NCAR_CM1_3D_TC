"""CLI adapter for the environmental-eddy SE pipeline."""

from __future__ import annotations

from ._se_pipeline_environmental import run_environmental_pipeline
from ._se_pipeline_single import PipelineConfig, SourceMaskConfig, _parse_csv_names


def run_from_args(args) -> None:
    """Translate the unified CLI namespace into a single-case base config."""
    if not args.jet_input_file:
        raise ValueError("--mode env requires --jet-input-file")
    cfg = PipelineConfig(
        input_file=args.input_file,
        output_dir=args.output_dir,
        time_index=args.time_index,
        target_time_seconds=args.target_time_seconds,
        target_time_hours=args.target_time_hours,
        max_r_km=args.max_r_km,
        dr_km=args.dr_km,
        enforce_dr_not_finer_than_grid=not args.allow_fine_radial_bins,
        max_z_km=args.max_z_km,
        center_window=args.center_window,
        center_method=args.center_method,
        coriolis_f=args.f,
        theta_floor=args.theta_floor,
        theta_outer_smooth_window=args.theta_outer_smooth_window,
        elliptic_margin=args.elliptic_margin,
        inertia_eps_ratio=args.inertia_eps_ratio,
        regularization_max_iter=args.regularization_max_iter,
        sor_max_iter=args.sor_max_iter,
        sor_omega=args.sor_omega,
        sor_tol=args.sor_tol,
        sor_verbose_every=args.sor_verbose_every,
        write_netcdf=not args.no_write_netcdf,
        write_ieee=False,
        plot_solution=not args.no_plot_solution,
        u_name=args.u_name,
        v_name=args.v_name,
        w_name=args.w_name,
        prs_name=args.prs_name,
        rho_name=args.rho_name,
        theta_name=args.theta_name,
        psfc_name=args.psfc_name,
        q_name=args.q_name,
        fnu_name=args.fnu_name,
        u_candidates=_parse_csv_names(args.u_candidates),
        v_candidates=_parse_csv_names(args.v_candidates),
        w_candidates=_parse_csv_names(args.w_candidates),
        prs_candidates=_parse_csv_names(args.prs_candidates),
        rho_candidates=_parse_csv_names(args.rho_candidates),
        theta_candidates=_parse_csv_names(args.theta_candidates),
        psfc_candidates=_parse_csv_names(args.psfc_candidates),
        q_candidates=_parse_csv_names(args.q_candidates),
        fnu_candidates=_parse_csv_names(args.fnu_candidates),
        source_mask=SourceMaskConfig(),
        eddy_average=args.eddy_average,
    )
    run_environmental_pipeline(
        cfg,
        jet_input_file=args.jet_input_file,
        averaging=args.eddy_average,
        bui_baroclinic_scale=args.bui_baroclinic_scale,
    )

