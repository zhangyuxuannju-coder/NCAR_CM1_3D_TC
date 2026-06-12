#!/usr/bin/env python
"""
动量收支诊断绘图工具 — 统一入口。

从 u_budget_diagnostic_groupplot.ipynb 和 radial_diagnostic_singlepage.ipynb
中提取的绘图逻辑，支持:

  1. 单个诊断项 R-Z 填色图
  2. 分组面板图（径向/切向 budget 各项分组对比）
  3. 径向剖面图（指定高度层）
  4. 收支闭合残差图

用法:
  # 绘制单个诊断项
  python scripts/plot_budget_terms.py --input dataset/budget.nc \\
      --var U_mr --time 48 --output figure/U_mr.png

  # 绘制分组面板图
  python scripts/plot_budget_terms.py --input dataset/budget.nc \\
      --mode grouped --time 48 --output figure/budget_panel.png
"""

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from netCDF4 import Dataset
from src.plotting import (
    plot_single_diagnostic_rz,
    plot_budget_grouped_panels,
    find_index,
)
from src.coordinates import nearest_index_1d


# 默认分组定义（与 notebook 一致）
DEFAULT_RADIAL_GROUPS = {
    "Mean Advection": ["U_mr", "U_mv"],
    "Eddy Advection": ["U_eh", "U_ev"],
    "Mean AGF": ["U_magf"],
    "Eddy AGF": ["U_eagf"],
    "Diffusion": ["U_dh", "U_dv"],
    "Damping": ["ramp"],
}

DEFAULT_TANGENTIAL_GROUPS = {
    "Mean Advection": ["V_mr", "V_mv"],
    "Eddy Advection": ["V_eh", "V_ev"],
    "Mean AGF": ["V_magf"],
    "Eddy AGF": ["V_eagf"],
    "Diffusion": ["V_dh", "V_dv"],
    "Damping": ["tramp"],
}


def load_nc_budget(nc_file: str) -> dict:
    """加载 budget NC 文件，返回坐标和所有变量。"""
    data = {}
    with Dataset(nc_file, "r") as nc:
        data["r_km"] = np.asarray(nc.variables["r"][:], dtype=float)
        data["z_km"] = np.asarray(nc.variables["z"][:], dtype=float)
        if "time" in nc.variables:
            data["time_vals"] = np.asarray(nc.variables["time"][:], dtype=float)

        for var_name in nc.variables:
            if var_name in ("r", "z", "time"):
                continue
            data[var_name] = np.asarray(nc.variables[var_name][:], dtype=float)
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="动量收支诊断绘图工具"
    )
    parser.add_argument("--input", default="dataset/typhoon_azimuthal_avg_budget.nc",
                        help="budget NC 文件")
    parser.add_argument("--var", default=None, help="单个变量名（single 模式）")
    parser.add_argument("--time", type=float, default=None,
                        help="时间索引或时间值")
    parser.add_argument("--zh", type=float, default=None,
                        help="目标高度 (km)，用于径向剖面模式")
    parser.add_argument("--mode", choices=["single", "grouped", "radial_profile"],
                        default="single", help="绘图模式")
    parser.add_argument("--budget-type", choices=["radial", "tangential"],
                        default="radial", help="budget 类型（grouped 模式）")
    parser.add_argument("--output", default="figure/budget_diag.png",
                        help="输出图片路径")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data = load_nc_budget(args.input)

    r_km = data["r_km"]
    z_km = data["z_km"]

    # 解析时间索引
    if args.time is not None:
        if "time_vals" in data:
            t_idx = nearest_index_1d(data["time_vals"], args.time)
        else:
            t_idx = int(args.time)
    else:
        t_idx = 0

    out_path = Path(args.output)

    if args.mode == "single":
        if args.var is None:
            print("[ERROR] single 模式需要指定 --var")
            sys.exit(1)
        field = data[args.var][t_idx] if data[args.var].ndim == 3 else data[args.var]
        plot_single_diagnostic_rz(
            r_km, z_km, field, out_path,
            var_name=args.var, dpi=args.dpi,
        )

    elif args.mode == "grouped":
        groups = (DEFAULT_RADIAL_GROUPS if args.budget_type == "radial"
                  else DEFAULT_TANGENTIAL_GROUPS)
        terms = {}
        for group_terms in groups.values():
            for tname in group_terms:
                if tname in data:
                    terms[tname] = data[tname][t_idx]

        plot_budget_grouped_panels(
            r_km, z_km, terms, out_path,
            z_target=args.zh,
            term_groups=groups,
            title=f"{args.budget_type.capitalize()} Momentum Budget",
            dpi=args.dpi,
        )

    elif args.mode == "radial_profile":
        if args.zh is None or args.var is None:
            print("[ERROR] radial_profile 模式需要指定 --zh 和 --var")
            sys.exit(1)
        iz = find_index(z_km, args.zh)
        profile = data[args.var][t_idx, iz, :]

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5), dpi=args.dpi)
        ax.plot(r_km, profile, 'b-', linewidth=2)
        ax.axhline(y=0, color='grey', linestyle=':', alpha=0.5)
        ax.set_xlabel("Radius (km)")
        ax.set_ylabel(args.var)
        ax.set_title(f"{args.var} at z≈{z_km[iz]:.1f} km, t_idx={t_idx}")
        ax.grid(True, alpha=0.3)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"[INFO] 图片已保存: {out_path}")


if __name__ == "__main__":
    main()
