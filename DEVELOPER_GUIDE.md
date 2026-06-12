# CM1 Typhoon Diagnostic Toolkit — 开发者指南 (Developer Guide)

> **目标读者**：未来的 AI 编程助手。读完本文档后，你应该能理解项目的全部科学背景、代码架构、以及如何添加新功能。

---

## 1. 项目是什么？（科学背景）

这是一个**台风数值模拟诊断分析工具包**，用于分析 CM1（Cloud Model 1）数值模式的输出。

### 1.1 科学流水线

```
CM1 原始输出 (cm1out.nc)
  └─ 笛卡尔网格 (xh, yh, zh) 上的 u, v, w, prs, rho, th, psfc, ub_*, vb_*
       │
       ├── [流水线 A] 柱坐标转换 + 方位角平均 + 动量收支诊断
       │      └─ 输出: budget.nc — 径向(u)和切向(v)动量方程各项的 Mean/Eddy 分解
       │
       ├── [流水线 B] Sawyer-Eliassen (SE) 方程求解
       │      └─ 输出: se_pipeline_products.npz — 次级环流流函数 ψ、径向环流 U_se、垂直环流 W_se
       │
       └── [绘图] 单页诊断图、水平风场、敏感性实验对比
```

### 1.2 关键物理概念

| 概念 | 含义 | 对应变量 |
|:---|:---|:---|
| **方位角平均** (azimuthal average) | 以台风中心为原点，对等半径环做平均 | `ur`, `ut`, `w` |
| **Mean/Eddy 分解** | 将任意量分解为方位角平均 + 扰动 | `U_mr`(mean) vs `U_eh`(eddy) |
| **动量收支** (momentum budget) | 径向/切向动量方程各项的闭合诊断 | `U_mr, U_eh, U_mv, U_ev, U_magf, U_eagf, U_dh, U_dv, ramp` |
| **SE 方程** (Sawyer-Eliassen) | 从平衡涡旋求解次级环流（横向-垂直环流）的椭圆型 PDE | ψ, U_se, W_se |
| **椭圆性正则化** | 确保 SE 系数矩阵判别式 Δ = 4AC - B² > 0，否则 SOR 发散 | |
| **SOR 求解器** | Successive Over-Relaxation，迭代求解大型稀疏椭圆方程 | |
| **热成风平衡** | 利用梯度风关系从切向风场反算平衡位温场 θ_bal | `invert_theta_from_thermal_wind` |
| **内核轴对称约束** | 在 r ≈ 0 处强制 ur~O(r), ut~O(r)，消除柱坐标奇点噪声 | `apply_core_axisymmetric_constraints` |
| **分组残差分配** | 将动量方程闭合残差按比例分配至各组内的 mean/eddy 项 | `grouped_residual_allocation` |

### 1.3 CM1 网格约定

- CM1 使用 **Arakawa-C 交错网格**：u 在 xf 面、v 在 yf 面、w 在 zf 面，标量在 xh/yh/zh
- `destagger` = 交错网格 → 标量网格的算术平均
- 预算项 `ub_*`/`vb_*` 是 CM1 内建诊断输出，在 `(time, zh, yh, xh)` 网格上

---

## 2. 代码架构 (`refactor/`)

```
refactor/
├── src/                              ← 核心库（所有可复用代码）
│   ├── center_finder.py              ← 台风中心定位（3 种方法）
│   ├── coordinates.py                ← 坐标变换、去交错、梯度
│   ├── config.py                     ← 配置数据类 + YAML 加载
│   ├── io.py                         ← netCDF/IEEE 读写 + Windows 路径
│   ├── azimuthal_avg.py              ← 流水线 A：方位角平均 + 收支诊断
│   ├── se_equation.py                ← 流水线 B：SE 方程核心算法
│   ├── _se_pipeline_single.py        ← SE 流水线完整版（单时刻）
│   ├── _se_pipeline_evap.py          ← SE 流水线完整版（蒸发冷却）
│   ├── _se_pipeline_timeavg.py       ← SE 流水线完整版（时均）
│   └── plotting.py                   ← 统一绘图函数
│
├── scripts/                          ← 可执行入口（每个脚本完成一个完整步骤）
│   ├── run_budget_diagnostic.py      ← 运行流水线 A
│   ├── run_se_pipeline.py            ← 运行流水线 B（三合一入口）
│   ├── plot_horizontal_field.py      ← 水平场绘图/视频
│   ├── plot_singlepage_diagnostics.py← 单页诊断图（径向/切向/组合）
│   ├── plot_budget_terms.py          ← 收支项快速绘图
│   ├── track_centers.py              ← 中心追踪
│   ├── extract_profile.py            ← 径向剖面提取
│   ├── profile_centers.py            ← 3D 中心轨迹
│   └── make_video.py                 ← PNG→MP4
│
├── experiments/                      ← 敏感性实验分析（后处理，不产生新数据）
│   ├── compare_evap.py               ← CTRL vs NOEVAP vs EVAP_ONLY
│   ├── sensitivity_evap.py           ← 蒸发冷却强度敏感性
│   └── dipole_analysis.py            ← 偶极子加热/冷却敏感性
│
├── config/default.yaml               ← 全局默认参数
├── notebooks/                        ← Jupyter Notebook（探索性分析）
├── code/                             ← Fortran 源码（CM1 初始场、namelist）
└── README.md                         ← 用户操作手册
```

### 2.1 模块依赖图

```
                    center_finder.py
                    coordinates.py
                    config.py
                    io.py
                    plotting.py
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
  azimuthal_avg.py  se_equation.py  (experiments/)
         │               │
         ▼               ▼
  run_budget_       run_se_pipeline.py
  diagnostic.py     (导入 _se_pipeline_*.py)
```

### 2.2 命名约定

| 前缀 | 含义 | 示例 |
|:---|:---|:---|
| `U_*` | 径向动量收支项 | `U_mr`, `U_eh`, `U_magf` |
| `V_*` | 切向动量收支项 | `V_mr`, `V_eh`, `V_magf` |
| `_raw` 后缀 | 残差分配前的原始项 | `U_mr_raw` |
| `_avg` 后缀 | 时间段平均后的项 | `U_mr_avg` |
| `br_*` | 径向预算投影项 | `br_hadv` |
| `bt_*` | 切向预算投影项 | `bt_hadv` |
| `ub_*` / `vb_*` | CM1 内建诊断项（笛卡尔网格） | `ub_hadv`, `vb_pgrad` |

---

## 3. 如何添加新功能

### 3.1 新增一个诊断项

场景：你想在动量收支中添加一个新的诊断项（如 `U_newterm`）。

**修改位置**：
1. `src/azimuthal_avg.py` — 在 `DIAG_META` 字典中添加新项的描述
2. `src/azimuthal_avg.py` — 在 `run_budget_diagnostic()` 函数的计算逻辑中添加新项的计算代码
3. `src/azimuthal_avg.py` — 在输出变量创建循环中添加新项的 NC 变量定义
4. `scripts/plot_singlepage_diagnostics.py` — 在 `preferred_diag_all` 列表中添加新变量名
5. `scripts/plot_budget_terms.py` — 在 `DEFAULT_RADIAL_GROUPS` 或 `DEFAULT_TANGENTIAL_GROUPS` 中添加新项

### 3.2 新增一个绘图面板类型

场景：你想添加一个新的单页诊断图（如"涡度收支单页图"）。

**修改位置**：
1. `scripts/plot_singlepage_diagnostics.py` — 添加一个新的 `plot_xxx_diagnostics()` 函数，复用已有的辅助函数（`get_var_2d`, `symmetric_levels_from_data`, `make_symlog_levels` 等）
2. `scripts/plot_singlepage_diagnostics.py` — 在 `build_parser()` 的 `--panel` choices 中添加新选项
3. `scripts/plot_singlepage_diagnostics.py` — 在 `main()` 中添加对应的调用分支

### 3.3 新增一个 SE 方程变体

场景：你想添加一个新的 SE 求解模式（如"仅热力强迫"模式）。

**方法 A（推荐）**：在现有 `_se_pipeline_single.py` 的基础上修改，通过 `PipelineConfig` 增加新参数控制行为。例如：
1. `src/config.py` — 在 `PipelineConfig` 中添加新字段
2. `src/_se_pipeline_single.py` — 在 `run_pipeline()` 中根据新字段分支
3. `scripts/run_se_pipeline.py` — 添加对应的 CLI 参数

**方法 B**：复制 `_se_pipeline_single.py` 为新文件 `_se_pipeline_xxx.py`，修改内部逻辑，然后在 `scripts/run_se_pipeline.py` 中添加新的 mode 选项。

### 3.4 新增一个敏感性实验

场景：你想添加新的实验对比分析。

**修改位置**：
1. `experiments/` — 添加新的 `.py` 文件，参照 `compare_evap.py` 的结构
2. 实验脚本通常读取 `se_pipeline_products.npz` 或 budget NC 文件，做对比绘图
3. 无需修改 `src/` 核心库（除非需要新的数据处理函数）

### 3.5 新增一个数据读取格式

场景：你需要读取 GRIB、CSV 等非 netCDF 格式的数据。

**修改位置**：
1. `src/io.py` — 添加新的读取函数（如 `read_grib()`）
2. 确保函数返回的数组格式与现有代码兼容（`(nz, nr)` 或 `(time, z, y, x)`）

### 3.6 新增一个物理常数或全局参数

**修改位置**：
1. `config/default.yaml` — 在对应段落添加新参数
2. `src/config.py` — 如果是 PipelineConfig 相关，在对应的 dataclass 中添加字段
3. 使用该参数的脚本中引用配置值

---

## 4. 关键技术实现细节

### 4.1 方位角平均（流水线 A 的核心）

```
输入: CM1 笛卡尔网格 3D 场 (zh, yh, xh)
步骤:
  1. 对每个时次，用 psfc 场的平滑最小值定位台风中心
  2. 计算每个网格点到中心的距离 r2d = √((x-xc)²+(y-yc)²) 和方位角 θ
  3. 将 u,v 投影到径向/切向: ur = u·cosθ + v·sinθ, ut = -u·sinθ + v·cosθ
  4. 按径向 bin 做方位角平均: np.bincount
  5. 计算 Mean/Eddy 分解: 对每个诊断项分别做平均和方差
  6. 分组残差分配: 将闭合残差均分到各组内的 mean/eddy 项
输出: budget.nc — (time, z, r) 二维诊断场
```

### 4.2 SE 方程求解（流水线 B 的核心）

```
输入: 方位角平均场 (ut, theta, rho, Q, Fnu)
步骤:
  1. azimuthal_average_from_3d() — 从 CM1 输出计算方位角平均场 + Q/Fnu
  2. invert_theta_from_thermal_wind() — 热成风平衡位温反演
  3. build_se_diagnostic_fields() — 构建 chi, C, ct, xi, zeta, inertial_stability
  4. regularize_inertial_stability_for_ellipticity() — 海绵层正则化确保椭圆性
  5. build_se_coefficients() — 构建 A/B/C/D/E/F 六系数矩阵
     (方程: A·∂²ψ/∂r² + B·∂²ψ/∂r∂z + C·∂²ψ/∂z² + D·∂ψ/∂r + E·∂ψ/∂z = F)
  6. solve_se_sor() — SOR 迭代求解 ψ
  7. psi_to_uw() — 从 ψ 计算 U_se, W_se
输出: se_pipeline_products.npz — (r, z) 二维解场
```

### 4.3 SOR 求解器细节

- 边界条件: r=0 处 dψ/dr=0（轴对称），r=Rmax 处 dψ/dr=0（远场），z=0 处 ψ=0（地面），z=Zmax 处 dψ/dz=0（自由滑移顶）
- 超松弛因子 ω=1.8 是默认值，发散时自动降到 1.0（Gauss-Seidel）
- 最多重试 4 次，每次 ω 减半
- 收敛判据: max_res < tol（默认 1e-14）

### 4.4 涡动分解（Mean/Eddy Separation）

对于径向动量方程:
```
∂u/∂t = -[u·∂u/∂r] - [u'·∂u'/∂r]  ← 水平平流 (mean + eddy)
        -[w·∂u/∂z] - [w'·∂u'/∂z]  ← 垂直平流 (mean + eddy)
        +[v²/r] + [v'²/r]          ← 曲率项 (mean + eddy)
        +f·v - (1/ρ)·∂p/∂r          ← 科氏力 + 气压梯度力
        +D_u                         ← 扩散/湍流/阻尼
```

其中 `U_mr = u·∂u/∂r`（mean），`U_eh = u'·∂u'/∂r`（eddy），`U_magf = v²/r + f·v - (1/ρ)·∂p/∂r`（mean 加速度梯度力组）。

### 4.5 诊断项命名对应关系

| NC 文件变量 | 含义 | 符号 |
|:---|:---|:---|
| `U_mr` | mean 径向平流 | ū·∂ū/∂r |
| `U_eh` | eddy 水平平流 | u'·∂u'/∂r（从 hadv 反算） |
| `U_mv` | mean 垂直平流 | w̄·∂ū/∂z |
| `U_ev` | eddy 垂直平流 | w'·∂u'/∂z（从 vadv 反算） |
| `U_magf` | mean 加速度梯度力组 | v̄²/r + f·v̄ - (1/ρ̄)·∂p̄/∂r |
| `U_eagf` | eddy 加速度梯度力组 | v'²/r - (1/ρ̄)·∂p'/∂r |
| `U_dh` | 水平扩散+湍流 | hidiff + hturb |
| `U_dv` | 垂直扩散+湍流 | vidiff + vturb |
| `ramp` | 径向阻尼 | rdamp |
| `V_mr/V_eh/V_mv/V_ev` | 切向对应项 | 同上，切向版本 |
| `V_magf/V_eagf` | 切向加速度梯度力组 | -ūv̄/r + f·ū（mean）/ -u'v'/r + pgrad_t（eddy） |

---

## 5. 配置文件说明 (`config/default.yaml`)

所有可调参数都在这个文件中。新增参数时遵循以下约定：

```yaml
# 参数按功能分组: paths, physics, domain, center, budget, se_solver, variables, plotting
# 每个参数都有中文注释说明含义和单位

paths:          # 文件路径（相对路径基于 refactor/）
physics:        # 物理常数（重力加速度、科氏参数）
domain:         # 计算域（max_r_km, dr_km, max_z_km）
center:         # 台风中心定位参数（方法、窗口大小）
budget:         # 收支诊断参数（残差分配、内核约束开关）
se_solver:      # SE 求解器参数（SOR 迭代、正则化、蒸发冷却）
variables:      # 变量名候选列表（自动匹配不同 CM1 输出的变量命名）
plotting:       # 绘图默认值（fps, dpi, xy_limit）
```

---

## 6. 常见调试场景

### 6.1 SOR 不收敛
- 症状：`max_res` 不下降，或在 1e-8 附近震荡
- 诊断：检查判别式 D = 4AC - B² 是否全正（`discriminant` 变量）
- 修复：降低 `--sor-omega` 至 1.5 或 1.0；增大 `--baroclinic-scale`；降低 `--dr-km`

### 6.2 收支不闭合（残差过大）
- 症状：`residual_after_allocation` 量级 > 1e-5
- 原因：内核区 (r < 6 km) 柱坐标奇点噪声
- 修复：启用 `--enable-core-stabilization`

### 6.3 中心定位异常
- 症状：台风中心在时间序列中跳动
- 修复：增大 `--center-window`（如 31）；使用 `--center-method centroid`；启用 `--center-time-smooth-window`

### 6.4 径向条纹/欠采样
- 症状：诊断图中出现同心圆条纹
- 原因：`dr_km` 小于原始网格距，某些环内样本不足
- 修复：增大 `--dr-km`（如 3-5 km），代码会打印 `native_grid_min` 供参考

---

## 7. 添加新功能的标准流程

1. **确定改动范围**：是需要新数据处理、新诊断项、新绘图还是新实验
2. **找参照**：在现有代码中找最相似的功能，复制其结构
3. **修改核心库**：在 `src/` 中添加/修改函数
4. **添加 CLI 参数**：在对应 `scripts/` 文件中添加 argparse 参数
5. **更新配置**：如有新参数，在 `config/default.yaml` 中添加
6. **测试**：用 `--max-times 2` 做快速冒烟测试
7. **更新文档**：在 `README.md`（用户手册）和本文档（开发者指南）中添加说明

---

## 8. pip 依赖

```
numpy, scipy, xarray, netCDF4, matplotlib, pyyaml, opencv-python
```

Fortran 源码 (`code/`) 中的 `SE-solver.f90` 和 `init3d.F` 不需要编译——所有求解都在 Python 中完成。

---

## 9. 快速参考：常用命令

```bash
# 方位角平均 + 收支诊断（快速测试，仅 2 时次）
python scripts/run_budget_diagnostic.py --max-times 2

# SE 方程诊断（单时刻）
python scripts/run_se_pipeline.py --mode single --target-time-hours 72

# SE 方程诊断（蒸发冷却模式）
python scripts/run_se_pipeline.py --mode evap --target-time-hours 72

# SE 方程诊断（时间段平均模式）
python scripts/run_se_pipeline.py --mode timeavg --time-avg-start-hours 64 --time-avg-end-hours 72

# 单页诊断图
python scripts/plot_singlepage_diagnostics.py --panel radial --input output/budget/budget.nc --mode time_range --start-hour 42 --end-hour 74

# 自定义组合图
python scripts/plot_singlepage_diagnostics.py --panel combo --combo-terms "1.0,U_magf -1.0,U_mr"

# 水平风场图
python scripts/plot_horizontal_field.py --var prs --zh 0 --time 48 --xy-limit 200
```
