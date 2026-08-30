"""JET-CTRL environmental-eddy Sawyer--Eliassen diagnostic pipeline."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from ._se_pipeline_single import (
    PipelineConfig,
    _rho_ext_from_rho_zr,
    _to_solver_layout_zr_to_rz,
    azimuthal_average_from_3d,
    psi_to_uw,
    solve_se_sor,
)
from .environmental_eddy import environmental_difference
from .se_bui import (
    assemble_operator,
    build_basic_state,
    build_forcing,
    invert_balanced_theta,
    regularize_ellipticity,
)


def _check_case_grids(ctrl: Dict[str, np.ndarray], jet: Dict[str, np.ndarray]) -> None:
    """Require identical SE grids before constructing a pointwise difference."""
    for name in ("r_km", "z_km"):
        a = np.asarray(ctrl[name], dtype=np.float64)
        b = np.asarray(jet[name], dtype=np.float64)
        if a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=1.0e-9):
            raise ValueError(f"JET and CTRL {name} grids differ; cannot form F_lambda_env")


def _solve_response(
    operator: Dict[str, np.ndarray],
    forcing_zr: np.ndarray,
    rho_zr: np.ndarray,
    r_m: np.ndarray,
    z_m: np.ndarray,
    cfg: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve one RHS with an already fixed CTRL operator."""
    arrays = {
        key: _to_solver_layout_zr_to_rz(operator[key])
        for key in ("A", "B", "C", "D", "E")
    }
    forcing_rz = _to_solver_layout_zr_to_rz(forcing_zr)
    dr = float(np.mean(np.diff(r_m)))
    dz = float(np.mean(np.diff(z_m)))
    psi = solve_se_sor(
        A=arrays["A"],
        B=arrays["B"],
        C=arrays["C"],
        D=arrays["D"],
        E=arrays["E"],
        F=forcing_rz,
        dr=dr,
        dz=dz,
        max_iter=cfg.sor_max_iter,
        omega=cfg.sor_omega,
        tol=cfg.sor_tol,
        verbose_every=cfg.sor_verbose_every,
    )
    rho_ext = _rho_ext_from_rho_zr(rho_zr)
    u_se, w_se = psi_to_uw(psi, rho_ext, r_m, dr, dz)
    return psi, u_se, w_se


def _plot_environmental_response(
    out_file: Path,
    r_km: np.ndarray,
    z_km: np.ndarray,
    f_env: np.ndarray,
    rhs_env: np.ndarray,
    u_env: np.ndarray,
    w_env: np.ndarray,
    u_ctrl: np.ndarray,
    u_ctrl_plus_env: np.ndarray,
) -> None:
    """Write a compact mechanism figure for the environmental response."""
    rr, zz = np.meshgrid(r_km, z_km)
    fields = [
        f_env,
        rhs_env,
        u_env[:, 1:-1].T,
        w_env[:, 1:-1].T,
        u_ctrl[:, 1:-1].T,
        u_ctrl_plus_env[:, 1:-1].T,
    ]
    titles = [
        r"$F_{\lambda,env}$ (m s$^{-2}$)",
        r"$-\partial_z(\chi\xi F_{\lambda,env})$",
        r"$U_{env}$ (m s$^{-1}$)",
        r"$W_{env}$ (m s$^{-1}$)",
        r"$U_{CTRL}$ (m s$^{-1}$)",
        r"$U_{CTRL+env}$ (m s$^{-1}$)",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, field, title in zip(axes.flat, fields, titles):
        finite = np.asarray(field)[np.isfinite(field)]
        vmax = float(np.nanpercentile(np.abs(finite), 99.0)) if finite.size else 1.0
        vmax = max(vmax, 1.0e-20)
        levels = np.linspace(-vmax, vmax, 25)
        im = ax.contourf(rr, zz, field, levels=levels, cmap="RdBu_r", extend="both")
        ax.set_title(title)
        ax.set_xlabel("Radius (km)")
        ax.set_ylabel("Height (km)")
        fig.colorbar(im, ax=ax, pad=0.02)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def _plot_environmental_forcing_components(
    out_file: Path,
    r_km: np.ndarray,
    z_km: np.ndarray,
    total: np.ndarray,
    radial: np.ndarray,
    vertical: np.ndarray,
) -> None:
    """Plot the requested JET-CTRL environmental eddy forcing by itself."""
    rr, zz = np.meshgrid(r_km, z_km)
    finite = np.concatenate(
        [np.asarray(field)[np.isfinite(field)] for field in (total, radial, vertical)]
    )
    vmax = float(np.nanpercentile(np.abs(finite), 99.0)) if finite.size else 1.0
    vmax = max(vmax, 1.0e-20)
    levels = np.linspace(-vmax, vmax, 25)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True, sharey=True)
    for ax, field, title in zip(
        axes,
        (total, radial, vertical),
        (r"$F_{\lambda,env}$", r"$F_{\lambda,env}^{(r)}$", r"$F_{\lambda,env}^{(z)}$"),
    ):
        im = ax.contourf(rr, zz, field, levels=levels, cmap="RdBu_r", extend="both")
        ax.set_title(title + r" (m s$^{-2}$)")
        ax.set_xlabel("Radius (km)")
        ax.set_ylabel("Height (km)")
        fig.colorbar(im, ax=ax, pad=0.02)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def _plot_rhs_decomposition(
    out_file: Path,
    r_km: np.ndarray,
    z_km: np.ndarray,
    thermal_ctrl: np.ndarray,
    momentum_ctrl: np.ndarray,
    environmental: np.ndarray,
    total_with_environment: np.ndarray,
) -> None:
    """Plot the traditional two RHS terms plus the isolated environment term."""
    rr, zz = np.meshgrid(r_km, z_km)
    fields = (thermal_ctrl, momentum_ctrl, environmental, total_with_environment)
    titles = (
        r"CTRL thermal $S_Q$",
        r"CTRL momentum $S_M$",
        r"extra environmental $S_{env}$",
        r"$S_Q+S_M+S_{env}$",
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True, sharex=True, sharey=True)
    for ax, field, title in zip(axes.flat, fields, titles):
        finite = np.asarray(field)[np.isfinite(field)]
        vmax = float(np.nanpercentile(np.abs(finite), 99.0)) if finite.size else 1.0
        vmax = max(vmax, 1.0e-30)
        levels = np.linspace(-vmax, vmax, 25)
        im = ax.contourf(rr, zz, field, levels=levels, cmap="RdBu_r", extend="both")
        ax.set_title(title)
        ax.set_xlabel("Radius (km)")
        ax.set_ylabel("Height (km)")
        fig.colorbar(im, ax=ax, pad=0.02)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def _plot_jet_activity_vs_torque(
    out_file: Path,
    r_km: np.ndarray,
    z_km: np.ndarray,
    eddy_speed_jet: np.ndarray,
    eddy_speed_env: np.ndarray,
    f_env: np.ndarray,
    rhs_env: np.ndarray,
    outer_zoom: bool = False,
) -> None:
    """Separate large non-axisymmetric wind amplitude from SE-eligible torque."""
    use_r = np.ones(r_km.size, dtype=bool)
    use_z = np.ones(z_km.size, dtype=bool)
    if outer_zoom:
        use_r = r_km >= 0.5 * float(np.nanmax(r_km))
        use_z = (z_km >= 8.0) & (z_km <= 18.0)
    r_show = r_km[use_r]
    z_show = z_km[use_z]
    rr, zz = np.meshgrid(r_show, z_show)
    fields = (
        eddy_speed_jet[np.ix_(use_z, use_r)],
        eddy_speed_env[np.ix_(use_z, use_r)],
        f_env[np.ix_(use_z, use_r)],
        rhs_env[np.ix_(use_z, use_r)],
    )
    titles = (
        r"JET non-axisymmetric speed $\sqrt{2EKE}$",
        r"JET$-$CTRL non-axisymmetric speed",
        r"SE-eligible $F_{\lambda,env}$",
        r"$S_{env}=-\partial_z(\chi\xi F_{\lambda,env})$",
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True, sharex=True, sharey=True)
    for index, (ax, field, title) in enumerate(zip(axes.flat, fields, titles)):
        finite = np.asarray(field)[np.isfinite(field)]
        if index == 0:
            vmax = float(np.nanpercentile(finite, 99.0)) if finite.size else 1.0
            levels = np.linspace(0.0, max(vmax, 1.0e-20), 25)
            cmap = "viridis"
        else:
            vmax = float(np.nanpercentile(np.abs(finite), 99.0)) if finite.size else 1.0
            vmax = max(vmax, 1.0e-30)
            levels = np.linspace(-vmax, vmax, 25)
            cmap = "RdBu_r"
        im = ax.contourf(rr, zz, field, levels=levels, cmap=cmap, extend="both")
        ax.set_title(title)
        ax.set_xlabel("Radius (km)")
        ax.set_ylabel("Height (km)")
        fig.colorbar(im, ax=ax, pad=0.02)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def run_environmental_pipeline(
    ctrl_cfg: PipelineConfig,
    jet_input_file: str,
    averaging: str = "reynolds",
    bui_baroclinic_scale: float = 1.0,
) -> None:
    """Solve the fixed-CTRL response to ``F_eddy(JET)-F_eddy(CTRL)``.

    The operator and the factors ``chi``/``xi`` on the momentum RHS are built
    only from CTRL.  Consequently ``U_env`` and ``W_env`` isolate the direct
    forcing pathway and do not include the JET-induced basic-state change.
    """
    out_dir = Path(ctrl_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctrl_cfg = replace(ctrl_cfg, eddy_average=averaging)
    jet_cfg = replace(ctrl_cfg, input_file=jet_input_file)

    print("[INFO] Diagnosing direct eddy forcing in CTRL")
    ctrl = azimuthal_average_from_3d(ctrl_cfg)
    print("[INFO] Diagnosing direct eddy forcing in JET")
    jet = azimuthal_average_from_3d(jet_cfg)
    _check_case_grids(ctrl, jet)

    r_km = np.asarray(ctrl["r_km"], dtype=np.float64)
    z_km = np.asarray(ctrl["z_km"], dtype=np.float64)
    r_m = r_km * 1000.0
    z_m = z_km * 1000.0
    f_env = environmental_difference(jet["F_lambda_eddy"], ctrl["F_lambda_eddy"])
    f_env_radial = environmental_difference(
        jet["F_lambda_eddy_radial"], ctrl["F_lambda_eddy_radial"]
    )
    f_env_vertical = environmental_difference(
        jet["F_lambda_eddy_vertical"], ctrl["F_lambda_eddy_vertical"]
    )
    eddy_speed_ctrl = np.sqrt(2.0 * np.maximum(ctrl["eddy_kinetic_energy"], 0.0))
    eddy_speed_jet = np.sqrt(2.0 * np.maximum(jet["eddy_kinetic_energy"], 0.0))
    eddy_speed_env = eddy_speed_jet - eddy_speed_ctrl

    theta_bal, tw_info = invert_balanced_theta(
        ctrl["ut"],
        ctrl["theta"],
        r_m,
        z_m,
        ctrl_cfg.coriolis_f,
        theta_floor=ctrl_cfg.theta_floor,
        outer_smooth_window=ctrl_cfg.theta_outer_smooth_window,
    )
    basic = build_basic_state(
        ctrl["ut"],
        theta_bal,
        ctrl["rho"],
        r_m,
        z_m,
        ctrl_cfg.coriolis_f,
        baroclinic_scale=bui_baroclinic_scale,
    )
    k1, k2, k3, reg_info = regularize_ellipticity(
        basic["K1_raw"],
        basic["K2_raw"],
        basic["K3_raw"],
        eps_ratio=ctrl_cfg.inertia_eps_ratio,
        margin=ctrl_cfg.elliptic_margin,
    )
    operator = assemble_operator(basic, k1, k2, k3, r_m, z_m)

    zero = np.zeros_like(f_env)
    env_rhs = build_forcing(basic, zero, f_env, r_m, z_m)
    ctrl_rhs = build_forcing(basic, ctrl["Q"], ctrl["Fnu"], r_m, z_m)
    forcing_ctrl_plus_env = ctrl_rhs["forcing_total"] + env_rhs["forcing_total"]

    print("[INFO] Solving fixed-CTRL operator for environmental forcing")
    psi_env, u_env, w_env = _solve_response(
        operator, env_rhs["forcing_total"], ctrl["rho"], r_m, z_m, ctrl_cfg
    )
    print("[INFO] Solving the same CTRL operator for baseline forcing")
    psi_ctrl, u_ctrl, w_ctrl = _solve_response(
        operator, ctrl_rhs["forcing_total"], ctrl["rho"], r_m, z_m, ctrl_cfg
    )
    psi_plus = psi_ctrl + psi_env
    u_plus = u_ctrl + u_env
    w_plus = w_ctrl + w_env

    np.savez_compressed(
        out_dir / "se_environmental_eddy_products.npz",
        r_km=r_km,
        z_km=z_km,
        theta_bal_ctrl=theta_bal,
        F_lambda_eddy_ctrl=ctrl["F_lambda_eddy"],
        F_lambda_eddy_jet=jet["F_lambda_eddy"],
        F_lambda_env=f_env,
        F_lambda_env_radial=f_env_radial,
        F_lambda_env_vertical=f_env_vertical,
        eddy_speed_ctrl=eddy_speed_ctrl,
        eddy_speed_jet=eddy_speed_jet,
        eddy_speed_env=eddy_speed_env,
        forcing_env=env_rhs["forcing_total"],
        forcing_ctrl=ctrl_rhs["forcing_total"],
        forcing_ctrl_thermal=ctrl_rhs["forcing_thermal"],
        forcing_ctrl_momentum=ctrl_rhs["forcing_momentum"],
        forcing_ctrl_plus_env=forcing_ctrl_plus_env,
        psi_env=psi_env,
        U_env=u_env,
        W_env=w_env,
        psi_ctrl=psi_ctrl,
        U_ctrl=u_ctrl,
        W_ctrl=w_ctrl,
        psi_ctrl_plus_env=psi_plus,
        U_ctrl_plus_env=u_plus,
        W_ctrl_plus_env=w_plus,
        **operator,
    )

    if ctrl_cfg.write_netcdf:
        ds = xr.Dataset(
            data_vars={
                "F_lambda_eddy_ctrl": (("z", "r"), ctrl["F_lambda_eddy"]),
                "F_lambda_eddy_jet": (("z", "r"), jet["F_lambda_eddy"]),
                "F_lambda_env": (("z", "r"), f_env),
                "F_lambda_env_radial": (("z", "r"), f_env_radial),
                "F_lambda_env_vertical": (("z", "r"), f_env_vertical),
                "eddy_speed_ctrl": (("z", "r"), eddy_speed_ctrl),
                "eddy_speed_jet": (("z", "r"), eddy_speed_jet),
                "eddy_speed_env": (("z", "r"), eddy_speed_env),
                "forcing_env": (("z", "r"), env_rhs["forcing_total"]),
                "forcing_ctrl_thermal": (("z", "r"), ctrl_rhs["forcing_thermal"]),
                "forcing_ctrl_momentum": (("z", "r"), ctrl_rhs["forcing_momentum"]),
                "forcing_ctrl_plus_env": (("z", "r"), forcing_ctrl_plus_env),
                "psi_env": (("r", "z"), psi_env[:, 1:-1]),
                "U_env": (("r", "z"), u_env[:, 1:-1]),
                "W_env": (("r", "z"), w_env[:, 1:-1]),
                "psi_ctrl": (("r", "z"), psi_ctrl[:, 1:-1]),
                "U_ctrl": (("r", "z"), u_ctrl[:, 1:-1]),
                "W_ctrl": (("r", "z"), w_ctrl[:, 1:-1]),
                "psi_ctrl_plus_env": (("r", "z"), psi_plus[:, 1:-1]),
                "U_ctrl_plus_env": (("r", "z"), u_plus[:, 1:-1]),
                "W_ctrl_plus_env": (("r", "z"), w_plus[:, 1:-1]),
            },
            coords={
                "r": r_km,
                "z": z_km,
            },
            attrs={
                "definition": "F_lambda_env = F_lambda_eddy(JET) - F_lambda_eddy(CTRL)",
                "operator": "fixed CTRL Bui et al. (2009) general SE operator",
                "eddy_average": averaging,
                "ctrl_input": ctrl_cfg.input_file,
                "jet_input": jet_input_file,
            },
        )
        ds.to_netcdf(out_dir / "se_environmental_eddy_products.nc")

    if ctrl_cfg.plot_solution:
        _plot_environmental_forcing_components(
            out_dir / "environmental_eddy_forcing.png",
            r_km,
            z_km,
            f_env,
            f_env_radial,
            f_env_vertical,
        )
        _plot_rhs_decomposition(
            out_dir / "se_rhs_three_pathway_decomposition.png",
            r_km,
            z_km,
            ctrl_rhs["forcing_thermal"],
            ctrl_rhs["forcing_momentum"],
            env_rhs["forcing_total"],
            forcing_ctrl_plus_env,
        )
        _plot_jet_activity_vs_torque(
            out_dir / "environmental_jet_activity_vs_torque.png",
            r_km,
            z_km,
            eddy_speed_jet,
            eddy_speed_env,
            f_env,
            env_rhs["forcing_total"],
        )
        if float(np.nanmax(r_km)) >= 600.0:
            _plot_jet_activity_vs_torque(
                out_dir / "environmental_jet_activity_vs_torque_outer_zoom.png",
                r_km,
                z_km,
                eddy_speed_jet,
                eddy_speed_env,
                f_env,
                env_rhs["forcing_total"],
                outer_zoom=True,
            )
        _plot_environmental_response(
            out_dir / "se_environmental_eddy_response.png",
            r_km,
            z_km,
            f_env,
            env_rhs["forcing_total"],
            u_env,
            w_env,
            u_ctrl,
            u_plus,
        )

    summary = {
        "ctrl_input": ctrl_cfg.input_file,
        "jet_input": jet_input_file,
        "environmental_forcing_png": str((out_dir / "environmental_eddy_forcing.png").as_posix()) if ctrl_cfg.plot_solution else "",
        "rhs_three_pathway_png": str((out_dir / "se_rhs_three_pathway_decomposition.png").as_posix()) if ctrl_cfg.plot_solution else "",
        "jet_activity_vs_torque_png": str((out_dir / "environmental_jet_activity_vs_torque.png").as_posix()) if ctrl_cfg.plot_solution else "",
        "eddy_average": averaging,
        "F_lambda_env_definition": "JET direct eddy flux convergence minus CTRL",
        "operator_definition": "CTRL fixed operator",
        "bui_baroclinic_scale": float(bui_baroclinic_scale),
        "thermal_wind": tw_info,
        "regularization": reg_info,
        "eddy_momentum_closure": {
            "ctrl_rms_m_s2": float(np.sqrt(np.nanmean(ctrl["F_lambda_eddy_closure_residual"] ** 2))),
            "jet_rms_m_s2": float(np.sqrt(np.nanmean(jet["F_lambda_eddy_closure_residual"] ** 2))),
            "definition": "budget reconstruction minus direct 3-D flux convergence",
        },
        "ctrl_center_km": [
            float(ctrl["center_x_km"][0]),
            float(ctrl["center_y_km"][0]),
        ],
        "jet_center_km": [
            float(jet["center_x_km"][0]),
            float(jet["center_y_km"][0]),
        ],
    }
    (out_dir / "environmental_eddy_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[INFO] Environmental-eddy products written to {out_dir}")

