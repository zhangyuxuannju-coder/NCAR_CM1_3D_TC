"""
配置数据类 (Configuration Dataclasses)

集中管理所有流水线的配置参数。支持从 YAML 文件加载配置。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


# ==============================================================================
# 方位角平均与收支诊断配置
# ==============================================================================

@dataclass
class TransformConfig:
    """柱坐标转换与方位角平均配置。"""
    input_file: str = "dataset/cm1out.nc"
    output_file: str = "dataset/typhoon_azimuthal_avg_budget.nc"
    max_r_km: float = 300.0
    dr_km: float = 2.0
    max_z_km: float = 20.0
    center_window: int = 21
    center_method: str = "min"

    # 内核稳定（进阶功能）
    enable_core_stabilization: bool = False
    core_radius_km: float = 6.0
    center_time_smooth_window: int = 11
    subtract_translation_speed: bool = False

    # 分组残差分配（进阶功能）
    grouped_residual: bool = True
    residual_weight_alpha: float = 1.0
    residual_weight_eps: float = 1.0e-8

    max_times: Optional[int] = None
    verbose: bool = True


@dataclass(frozen=True)
class BudgetPair:
    """U/V 预算项配对。"""
    suffix: str
    u_name: str
    v_name: str


# ==============================================================================
# SE 方程求解配置
# ==============================================================================

@dataclass
class SourceMaskConfig:
    """源项遮罩配置。"""
    thermal_scale: float = 1.0
    momentum_scale: float = 1.0
    thermal_zero_boxes: List[Dict[str, float]] = field(default_factory=list)
    momentum_zero_boxes: List[Dict[str, float]] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """
    SE 方程诊断流水线统一配置。

    支持三种模式：
      - "single"  : 单时刻诊断（基础版）
      - "evap"    : 含蒸发冷却强迫项
      - "timeavg" : 时间段平均后诊断
    """
    # --- 模式 ---
    mode: str = "single"  # "single", "evap", "timeavg"

    # --- 路径 ---
    input_file: str = "dataset/cm1out.nc"
    output_dir: str = "se_pipeline_output"

    # --- 时间选择 ---
    time_index: int = 0
    target_time_seconds: Optional[float] = None
    target_time_hours: Optional[float] = None

    # --- 时间平均（timeavg 模式） ---
    time_avg_start_hours: Optional[float] = None
    time_avg_end_hours: Optional[float] = None

    # --- 变量名 ---
    u_name: str = "u"
    v_name: str = "v"
    w_name: str = "w"
    prs_name: str = "prs"
    rho_name: str = "rho"
    theta_name: str = "th"
    psfc_name: str = "psfc"
    q_name: str = "Q"
    fnu_name: str = "Fnu"

    u_candidates: Tuple[str, ...] = ("u", "ua", "uinterp")
    v_candidates: Tuple[str, ...] = ("v", "va", "vinterp")
    w_candidates: Tuple[str, ...] = ("w", "wa", "winterp")
    prs_candidates: Tuple[str, ...] = ("prs", "pres", "p")
    rho_candidates: Tuple[str, ...] = ("rho", "rhoa", "dens")
    theta_candidates: Tuple[str, ...] = ("th", "theta", "thpert")
    psfc_candidates: Tuple[str, ...] = ("psfc", "sfcprs", "ps")
    q_candidates: Tuple[str, ...] = ("Q", "q_diab", "qheat", "th_src")
    fnu_candidates: Tuple[str, ...] = (
        "Fnu", "fric_radial", "mom_src", "radial_drag"
    )

    q_override_file: str = ""
    fnu_override_file: str = ""
    q_constant: float = 0.0
    fnu_constant: float = 0.0

    # --- 域 ---
    max_r_km: float = 300.0
    dr_km: float = 2.0
    enforce_dr_not_finer_than_grid: bool = True
    max_z_km: float = 20.0

    # --- 中心定位 ---
    center_window: int = 21
    center_method: str = "min"

    # --- 物理常数 ---
    coriolis_f: float = 5.0e-5
    theta_floor: float = 150.0
    theta_outer_smooth_window: int = 1

    # --- 正则化 ---
    elliptic_margin: float = 0.0
    inertia_eps_ratio: float = 1.0e-3
    regularization_max_iter: int = 20

    # --- SOR 求解器 ---
    sor_max_iter: int = 60000
    sor_omega: float = 1.8
    sor_tol: float = 1.0e-14
    sor_verbose_every: int = 500

    # --- 输出 ---
    write_netcdf: bool = True
    write_ieee: bool = True
    ieee_prefix: str = "SE"
    plot_solution: bool = True

    # --- 斜压项 ---
    baroclinic_scale: float = 0.4

    # --- 蒸发冷却（evap 模式） ---
    evap_cooling_enabled: bool = True
    evap_cooling_q0: float = -2.0e-4     # 峰值冷却率 (K/s)
    evap_r_center: float = 145.0          # 椭圆中心半径 (km)
    evap_z_center: float = 15.0           # 椭圆中心高度 (km)
    evap_r_half: float = 105.0            # 径向半轴 (km)
    evap_z_half: float = 2.5              # 垂直半轴 (km)
    evap_dipole: bool = False             # 偶极子模式
    evap_dipole_q_factor: float = 1.0
    evap_dipole_sigma_z: float = 1.5

    # --- 源项遮罩 ---
    source_mask: SourceMaskConfig = field(default_factory=SourceMaskConfig)


# ==============================================================================
# 绘图配置
# ==============================================================================

@dataclass
class PlotParams:
    """水平场绘图参数。"""
    nc_file: str = "dataset/cm1out.nc"
    var_name: str = "prs"
    target_zh: float = 1000.0
    target_time: int = 0

    start_time: Optional[object] = None
    end_time: Optional[object] = None

    zh_dim: str = "zh"
    time_dim: str = "time"
    x_dim: str = "xh"
    y_dim: str = "yh"

    cmap = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    fps: int = 5
    video_dir: str = "video"
    out_name: Optional[str] = None
    save_video: bool = False
    use_memory_read: bool = False
    verbose: bool = True

    figure_dir: str = "figure"
    figure_name: Optional[str] = None
    xy_limit: Optional[float] = None


# ==============================================================================
# YAML 配置加载
# ==============================================================================

def load_pipeline_config_from_yaml(yaml_path: str | Path) -> PipelineConfig:
    """
    从 YAML 文件加载 PipelineConfig。

    YAML 结构应与 config/default.yaml 一致，未指定的字段使用默认值。
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    cfg = PipelineConfig()

    # 顶层字段
    if "paths" in data:
        p = data["paths"]
        if "input_file" in p:
            cfg.input_file = p["input_file"]
        if "se_output_dir" in p:
            cfg.output_dir = p["se_output_dir"]

    if "domain" in data:
        d = data["domain"]
        if "max_r_km" in d:
            cfg.max_r_km = d["max_r_km"]
        if "dr_km" in d:
            cfg.dr_km = d["dr_km"]
        if "max_z_km" in d:
            cfg.max_z_km = d["max_z_km"]

    if "center" in data:
        c = data["center"]
        if "method" in c:
            cfg.center_method = c["method"]
        if "window" in c:
            cfg.center_window = c["window"]

    if "physics" in data:
        ph = data["physics"]
        if "coriolis_f" in ph:
            cfg.coriolis_f = ph["coriolis_f"]

    if "se_solver" in data:
        s = data["se_solver"]
        if "mode" in s:
            cfg.mode = s["mode"]
        if "sor_max_iter" in s:
            cfg.sor_max_iter = s["sor_max_iter"]
        if "sor_omega" in s:
            cfg.sor_omega = s["sor_omega"]
        if "sor_tol" in s:
            cfg.sor_tol = s["sor_tol"]
        if "baroclinic_scale" in s:
            cfg.baroclinic_scale = s["baroclinic_scale"]
        if "plot_solution" in s:
            cfg.plot_solution = s["plot_solution"]

        if "evap_cooling" in s:
            ec = s["evap_cooling"]
            cfg.evap_cooling_enabled = ec.get("enabled", True)
            cfg.evap_cooling_q0 = ec.get("q0", -2.0e-4)
            cfg.evap_r_center = ec.get("r_center", 145.0)
            cfg.evap_z_center = ec.get("z_center", 15.0)
            cfg.evap_r_half = ec.get("r_half", 105.0)
            cfg.evap_z_half = ec.get("z_half", 2.5)
            cfg.evap_dipole = ec.get("dipole", False)
            cfg.evap_dipole_q_factor = ec.get("dipole_q_factor", 1.0)
            cfg.evap_dipole_sigma_z = ec.get("dipole_sigma_z", 1.5)

        if "time_avg" in s:
            ta = s["time_avg"]
            cfg.time_avg_start_hours = ta.get("start_hours")
            cfg.time_avg_end_hours = ta.get("end_hours")

    return cfg


def load_transform_config_from_yaml(yaml_path: str | Path) -> TransformConfig:
    """从 YAML 文件加载 TransformConfig。"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    cfg = TransformConfig()

    if "paths" in data:
        p = data["paths"]
        if "input_file" in p:
            cfg.input_file = p["input_file"]
        if "budget_output_file" in p:
            cfg.output_file = p["budget_output_file"]

    if "domain" in data:
        d = data["domain"]
        if "max_r_km" in d:
            cfg.max_r_km = d["max_r_km"]
        if "dr_km" in d:
            cfg.dr_km = d["dr_km"]
        if "max_z_km" in d:
            cfg.max_z_km = d["max_z_km"]

    if "center" in data:
        c = data["center"]
        if "method" in c:
            cfg.center_method = c["method"]
        if "window" in c:
            cfg.center_window = c["window"]

    if "budget" in data:
        b = data["budget"]
        if "grouped_residual" in b:
            cfg.grouped_residual = b["grouped_residual"]
        if "enable_core_stabilization" in b:
            cfg.enable_core_stabilization = b["enable_core_stabilization"]
        if "subtract_translation_speed" in b:
            cfg.subtract_translation_speed = b["subtract_translation_speed"]
        if "max_times" in b:
            cfg.max_times = b["max_times"]

    return cfg
