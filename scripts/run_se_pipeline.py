#!/usr/bin/env python
"""
Sawyer-Eliassen (SE) 方程诊断流水线 — 统一入口。

完整的命令行接口，支持 README_SE_pipeline.md 中记录的所有参数。
三种模式: single（默认）、evap（蒸发冷却）、timeavg（时间段平均）。

用法:
  python scripts/run_se_pipeline.py --target-time-hours 72 --output-dir output/se_pipeline/72h

  python scripts/run_se_pipeline.py --mode evap --target-time-hours 72 \\
      --evap-q0 -2e-4 --output-dir output/se_pipeline/evap_72h

  python scripts/run_se_pipeline.py --mode timeavg \\
      --time-avg-start-hours 64 --time-avg-end-hours 72

  # 自定义源项 + 空间屏蔽
  python scripts/run_se_pipeline.py --target-time-hours 72 \\
      --q-override-file custom_Q.npy \\
      --source-mask-json '{"thermal_zero_boxes":[{"r_min_km":0,"r_max_km":50,"z_min_km":0,"z_max_km":3}]}'
"""

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SE 诊断全流程: 3D场 → 方位平均 → 热成风反算 → 六系数 → 正则化 → SE求解",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行示例:
  python scripts/run_se_pipeline.py --target-time-hours 48
  python scripts/run_se_pipeline.py --target-time-hours 72 --sor-omega 1.5 --sor-tol 1.5e-9
  python scripts/run_se_pipeline.py --mode evap --target-time-hours 72 --evap-q0 -2e-4
  python scripts/run_se_pipeline.py --mode timeavg --time-avg-start-hours 64 --time-avg-end-hours 72
""")

    # ===== 模式 =====
    p.add_argument("--mode", choices=["single","evap","timeavg"], default="single",
                   help="运行模式 (默认 single)")

    # ===== 路径 =====
    p.add_argument("--input-file", default="dataset/cm1out.nc", help="输入 NC 文件")
    p.add_argument("--output-dir", default="output/se_pipeline", help="输出目录")

    # ===== 时间 =====
    p.add_argument("--time-index", type=int, default=0)
    p.add_argument("--target-time-seconds", type=float, default=None)
    p.add_argument("--target-time-hours", type=float, default=None)
    p.add_argument("--time-avg-start-hours", type=float, default=None)
    p.add_argument("--time-avg-end-hours", type=float, default=None)

    # ===== 域 =====
    p.add_argument("--max-r-km", type=float, default=300.0)
    p.add_argument("--dr-km", type=float, default=2.0)
    p.add_argument("--allow-fine-radial-bins", action="store_true",
                   help="允许 dr 小于原始网格距（可能欠采样）")
    p.add_argument("--max-z-km", type=float, default=20.0)

    # ===== 中心定位 =====
    p.add_argument("--center-window", type=int, default=21)
    p.add_argument("--center-method", choices=["min","mean"], default="min")

    # ===== 物理参数 =====
    p.add_argument("--f", type=float, default=5.0e-5, help="科氏参数 (1/s)")
    p.add_argument("--theta-floor", type=float, default=150.0, help="位温下界 (K)")
    p.add_argument("--theta-outer-smooth-window", type=int, default=1)

    # ===== 正则化 =====
    p.add_argument("--elliptic-margin", type=float, default=0.0)
    p.add_argument("--inertia-eps-ratio", type=float, default=1.0e-3)
    p.add_argument("--regularization-max-iter", type=int, default=20)

    # ===== SOR 求解器 =====
    p.add_argument("--sor-max-iter", type=int, default=60000)
    p.add_argument("--sor-omega", type=float, default=1.8)
    p.add_argument("--sor-tol", type=float, default=1.0e-14)
    p.add_argument("--sor-verbose-every", type=int, default=500)
    p.add_argument("--baroclinic-scale", type=float, default=0.4,
                   help="NCL式斜压项缩放因子 (0.4 以增强椭圆性)")

    # ===== 源项 =====
    p.add_argument("--q-override-file", default="", help="外部热力源二维场 (.npy/.npz/.nc)")
    p.add_argument("--fnu-override-file", default="", help="外部动量源二维场")
    p.add_argument("--q-constant", type=float, default=0.0)
    p.add_argument("--fnu-constant", type=float, default=0.0)
    p.add_argument("--source-mask-json", default="",
                   help='源项屏蔽: \'{"thermal_zero_boxes":[{"r_min_km":0,...}]}\'')

    # ===== 变量名 =====
    p.add_argument("--u-name", default="u"); p.add_argument("--v-name", default="v")
    p.add_argument("--w-name", default="w"); p.add_argument("--prs-name", default="prs")
    p.add_argument("--rho-name", default="rho"); p.add_argument("--theta-name", default="th")
    p.add_argument("--psfc-name", default="psfc"); p.add_argument("--q-name", default="Q")
    p.add_argument("--fnu-name", default="Fnu")
    p.add_argument("--u-candidates", default="u,ua,uinterp")
    p.add_argument("--v-candidates", default="v,va,vinterp")
    p.add_argument("--w-candidates", default="w,wa,winterp")
    p.add_argument("--prs-candidates", default="prs,pres,p")
    p.add_argument("--rho-candidates", default="rho,rhoa,dens")
    p.add_argument("--theta-candidates", default="th,theta,thpert")
    p.add_argument("--psfc-candidates", default="psfc,sfcprs,ps")
    p.add_argument("--q-candidates", default="Q,q_diab,qheat,th_src")
    p.add_argument("--fnu-candidates", default="Fnu,fric_radial,mom_src,radial_drag")

    # ===== 输出 =====
    p.add_argument("--no-write-netcdf", action="store_true")
    p.add_argument("--no-write-ieee", action="store_true")
    p.add_argument("--ieee-prefix", default="SE")
    p.add_argument("--no-plot-solution", action="store_true")

    # ===== 蒸发冷却 (evap 模式) =====
    p.add_argument("--evap-q0", type=float, default=-2.0e-4, help="峰值冷却率 (K/s)")
    p.add_argument("--evap-r-center", type=float, default=145.0)
    p.add_argument("--evap-z-center", type=float, default=15.0)
    p.add_argument("--evap-r-half", type=float, default=105.0)
    p.add_argument("--evap-z-half", type=float, default=2.5)
    p.add_argument("--evap-dipole", action="store_true", help="加热-冷却偶极子")
    p.add_argument("--evap-dipole-q-factor", type=float, default=1.0)
    p.add_argument("--evap-dipole-sigma-z", type=float, default=1.5)

    return p


def _parse_csv_names(text: str):
    return tuple(x.strip() for x in text.split(",") if x.strip())


def _source_mask_from_json(text_or_path: str):
    import json
    if not text_or_path:
        return None
    p = Path(text_or_path)
    payload = json.loads(p.read_text(encoding="utf-8") if p.exists() else text_or_path)
    # Return dict, will be handled by the pipeline
    return payload


def _run_single(args):
    from src._se_pipeline_single import PipelineConfig, run_pipeline, SourceMaskConfig, _source_mask_from_json as _orig_mask, _parse_csv_names as _orig_csv
    mask_cfg = SourceMaskConfig()
    if args.source_mask_json:
        mask_cfg = _orig_mask(args.source_mask_json)
    cfg = PipelineConfig(
        input_file=args.input_file, output_dir=args.output_dir,
        time_index=args.time_index,
        target_time_seconds=args.target_time_seconds,
        target_time_hours=args.target_time_hours,
        max_r_km=args.max_r_km, dr_km=args.dr_km,
        enforce_dr_not_finer_than_grid=not args.allow_fine_radial_bins,
        max_z_km=args.max_z_km,
        center_window=args.center_window, center_method=args.center_method,
        coriolis_f=args.f, theta_floor=args.theta_floor,
        theta_outer_smooth_window=args.theta_outer_smooth_window,
        elliptic_margin=args.elliptic_margin,
        inertia_eps_ratio=args.inertia_eps_ratio,
        regularization_max_iter=args.regularization_max_iter,
        sor_max_iter=args.sor_max_iter, sor_omega=args.sor_omega,
        sor_tol=args.sor_tol, sor_verbose_every=args.sor_verbose_every,
        baroclinic_scale=args.baroclinic_scale,
        write_netcdf=not args.no_write_netcdf,
        write_ieee=not args.no_write_ieee, ieee_prefix=args.ieee_prefix,
        plot_solution=not args.no_plot_solution,
        u_name=args.u_name, v_name=args.v_name, w_name=args.w_name,
        prs_name=args.prs_name, rho_name=args.rho_name,
        theta_name=args.theta_name, psfc_name=args.psfc_name,
        q_name=args.q_name, fnu_name=args.fnu_name,
        u_candidates=_orig_csv(args.u_candidates),
        v_candidates=_orig_csv(args.v_candidates),
        w_candidates=_orig_csv(args.w_candidates),
        prs_candidates=_orig_csv(args.prs_candidates),
        rho_candidates=_orig_csv(args.rho_candidates),
        theta_candidates=_orig_csv(args.theta_candidates),
        psfc_candidates=_orig_csv(args.psfc_candidates),
        q_candidates=_orig_csv(args.q_candidates),
        fnu_candidates=_orig_csv(args.fnu_candidates),
        q_override_file=args.q_override_file,
        fnu_override_file=args.fnu_override_file,
        q_constant=args.q_constant, fnu_constant=args.fnu_constant,
        source_mask=mask_cfg,
    )
    run_pipeline(cfg)


def _run_evap(args):
    from src._se_pipeline_evap import PipelineConfig, run_pipeline, SourceMaskConfig, _source_mask_from_json as _orig_mask, _parse_csv_names as _orig_csv
    mask_cfg = SourceMaskConfig()
    if args.source_mask_json:
        mask_cfg = _orig_mask(args.source_mask_json)
    cfg = PipelineConfig(
        input_file=args.input_file, output_dir=args.output_dir,
        time_index=args.time_index,
        target_time_seconds=args.target_time_seconds,
        target_time_hours=args.target_time_hours,
        max_r_km=args.max_r_km, dr_km=args.dr_km,
        enforce_dr_not_finer_than_grid=not args.allow_fine_radial_bins,
        max_z_km=args.max_z_km,
        center_window=args.center_window, center_method=args.center_method,
        coriolis_f=args.f, theta_floor=args.theta_floor,
        theta_outer_smooth_window=args.theta_outer_smooth_window,
        elliptic_margin=args.elliptic_margin,
        inertia_eps_ratio=args.inertia_eps_ratio,
        regularization_max_iter=args.regularization_max_iter,
        sor_max_iter=args.sor_max_iter, sor_omega=args.sor_omega,
        sor_tol=args.sor_tol, sor_verbose_every=args.sor_verbose_every,
        baroclinic_scale=args.baroclinic_scale,
        write_netcdf=not args.no_write_netcdf,
        write_ieee=not args.no_write_ieee, ieee_prefix=args.ieee_prefix,
        plot_solution=not args.no_plot_solution,
        u_name=args.u_name, v_name=args.v_name, w_name=args.w_name,
        prs_name=args.prs_name, rho_name=args.rho_name,
        theta_name=args.theta_name, psfc_name=args.psfc_name,
        q_name=args.q_name, fnu_name=args.fnu_name,
        u_candidates=_orig_csv(args.u_candidates),
        v_candidates=_orig_csv(args.v_candidates),
        w_candidates=_orig_csv(args.w_candidates),
        prs_candidates=_orig_csv(args.prs_candidates),
        rho_candidates=_orig_csv(args.rho_candidates),
        theta_candidates=_orig_csv(args.theta_candidates),
        psfc_candidates=_orig_csv(args.psfc_candidates),
        q_candidates=_orig_csv(args.q_candidates),
        fnu_candidates=_orig_csv(args.fnu_candidates),
        q_override_file=args.q_override_file,
        fnu_override_file=args.fnu_override_file,
        q_constant=args.q_constant, fnu_constant=args.fnu_constant,
        source_mask=mask_cfg,
        evap_cooling_q0=args.evap_q0, evap_r_center=args.evap_r_center,
        evap_z_center=args.evap_z_center, evap_r_half=args.evap_r_half,
        evap_z_half=args.evap_z_half, evap_dipole=args.evap_dipole,
        evap_dipole_q_factor=args.evap_dipole_q_factor,
        evap_dipole_sigma_z=args.evap_dipole_sigma_z,
    )
    run_pipeline(cfg)


def _run_timeavg(args):
    from src._se_pipeline_timeavg import PipelineConfig, run_pipeline, SourceMaskConfig, _source_mask_from_json as _orig_mask, _parse_csv_names as _orig_csv
    mask_cfg = SourceMaskConfig()
    if args.source_mask_json:
        mask_cfg = _orig_mask(args.source_mask_json)
    cfg = PipelineConfig(
        input_file=args.input_file, output_dir=args.output_dir,
        time_index=args.time_index,
        target_time_seconds=args.target_time_seconds,
        target_time_hours=args.target_time_hours,
        time_avg_start_hours=args.time_avg_start_hours,
        time_avg_end_hours=args.time_avg_end_hours,
        max_r_km=args.max_r_km, dr_km=args.dr_km,
        enforce_dr_not_finer_than_grid=not args.allow_fine_radial_bins,
        max_z_km=args.max_z_km,
        center_window=args.center_window, center_method=args.center_method,
        coriolis_f=args.f, theta_floor=args.theta_floor,
        theta_outer_smooth_window=args.theta_outer_smooth_window,
        elliptic_margin=args.elliptic_margin,
        inertia_eps_ratio=args.inertia_eps_ratio,
        regularization_max_iter=args.regularization_max_iter,
        sor_max_iter=args.sor_max_iter, sor_omega=args.sor_omega,
        sor_tol=args.sor_tol, sor_verbose_every=args.sor_verbose_every,
        baroclinic_scale=args.baroclinic_scale,
        write_netcdf=not args.no_write_netcdf,
        write_ieee=not args.no_write_ieee, ieee_prefix=args.ieee_prefix,
        plot_solution=not args.no_plot_solution,
        u_name=args.u_name, v_name=args.v_name, w_name=args.w_name,
        prs_name=args.prs_name, rho_name=args.rho_name,
        theta_name=args.theta_name, psfc_name=args.psfc_name,
        q_name=args.q_name, fnu_name=args.fnu_name,
        u_candidates=_orig_csv(args.u_candidates),
        v_candidates=_orig_csv(args.v_candidates),
        w_candidates=_orig_csv(args.w_candidates),
        prs_candidates=_orig_csv(args.prs_candidates),
        rho_candidates=_orig_csv(args.rho_candidates),
        theta_candidates=_orig_csv(args.theta_candidates),
        psfc_candidates=_orig_csv(args.psfc_candidates),
        q_candidates=_orig_csv(args.q_candidates),
        fnu_candidates=_orig_csv(args.fnu_candidates),
        q_override_file=args.q_override_file,
        fnu_override_file=args.fnu_override_file,
        q_constant=args.q_constant, fnu_constant=args.fnu_constant,
        source_mask=mask_cfg,
    )
    run_pipeline(cfg)


def main() -> None:
    args = build_parser().parse_args()
    print(f"[INFO] SE Pipeline mode={args.mode}, input={args.input_file}, output={args.output_dir}")

    if args.mode == "single":
        _run_single(args)
    elif args.mode == "evap":
        _run_evap(args)
    elif args.mode == "timeavg":
        _run_timeavg(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print("Pipeline completed.")


if __name__ == "__main__":
    main()
