#!/usr/bin/env python
"""
沿 X 轴提取径向剖面数据。

从 extract_profile_along_x.py 提取。

用法:
  python scripts/extract_profile.py --input dataset/cm1out.nc --time 400 --zh 2.0 \\
      --stop-x-km 1000 --output profile.txt
"""

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
from netCDF4 import Dataset
from src.center_finder import find_smoothed_min_point
from src.coordinates import ensure_2d_xy, nearest_index_1d


def profile_along_x(
    nc_file: str,
    time_idx: int = 400,
    prs_var: str = "prs",
    prs_zh_value: float = 2.0,
    psfc_var: str = "psfc",
    psfc_time_key=400,
    x_dim: str = "xh",
    y_dim: str = "yh",
    stop_x_km: float = 1000.0,
    verbose: bool = False,
) -> np.ndarray:
    """沿 X 轴提取径向剖面，返回 (N, 2) 数组 [r_km, prs_value]。"""
    res = find_smoothed_min_point(
        nc_file, psfc_time_key, var_name=psfc_var,
        x_dim=x_dim, y_dim=y_dim, window=21, verbose=verbose,
    )
    ix_min = int(res["ix"])
    iy_min = int(res["iy"])

    with Dataset(nc_file, "r") as nc:
        xh = np.asarray(nc.variables[x_dim][:], dtype=float)
        yh = np.asarray(nc.variables[y_dim][:], dtype=float)
        nx = len(xh)
        ny = len(yh)

        zh_arr = np.asarray(nc.variables["zh"][:], dtype=float)
        zh_idx = nearest_index_1d(zh_arr, prs_zh_value)

        prs_slice_raw = np.asarray(
            nc.variables[prs_var][time_idx, zh_idx, ...], dtype=float
        )
        prs_slice = ensure_2d_xy(prs_slice_raw, nx, ny)

    out_list = []
    ix = ix_min
    while ix < nx:
        x_val = float(xh[ix])
        prs_val = float(prs_slice[iy_min, ix])
        r = abs(x_val - float(xh[ix_min]))
        out_list.append((r, prs_val))
        if verbose:
            print(f"  r={r:.1f} km, prs={prs_val:.2f}")
        if x_val > stop_x_km:
            break
        ix += 1

    return np.array(out_list)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="沿 X 轴提取径向剖面")
    parser.add_argument("--input", default="dataset/cm1out.nc")
    parser.add_argument("--time", type=int, default=400, help="时间索引")
    parser.add_argument("--zh", type=float, default=2.0, help="目标高度 (km)")
    parser.add_argument("--prs-var", default="prs")
    parser.add_argument("--psfc-var", default="psfc")
    parser.add_argument("--stop-x-km", type=float, default=1000.0)
    parser.add_argument("--output", default="output_profile.txt")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    arr = profile_along_x(
        nc_file=args.input, time_idx=args.time,
        prs_var=args.prs_var, prs_zh_value=args.zh,
        psfc_var=args.psfc_var, psfc_time_key=args.time,
        stop_x_km=args.stop_x_km, verbose=args.verbose,
    )
    np.savetxt(args.output, arr, header="r_km prs_value", fmt="%.6f")
    print(f"[INFO] 剖面已保存至: {args.output} (共 {len(arr)} 点)")


if __name__ == "__main__":
    main()
