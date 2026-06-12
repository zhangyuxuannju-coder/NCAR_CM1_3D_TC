# CM1 Typhoon Diagnostic Toolkit — 操作手册

台风数值模拟诊断分析工具包，覆盖从 CM1 原始输出到论文图表的完整科研流水线。

---

## 目录

1. [环境准备](#1-环境准备)
2. [项目结构](#2-项目结构)
3. [功能一：台风中心定位与追踪](#3-功能一台风中心定位与追踪)
4. [功能二：方位角平均与动量收支诊断](#4-功能二方位角平均与动量收支诊断)
5. [功能三：SE 方程次级环流诊断](#5-功能三se-方程次级环流诊断)
4. [功能四：动量收支诊断单页图 ⭐](#6-功能四动量收支诊断单页图-核心绘图)
7. [功能五：水平风场绘图与视频](#7-功能五水平风场绘图与视频)
8. [功能六：径向剖面提取](#8-功能六径向剖面提取)
9. [功能七：敏感性实验](#9-功能七敏感性实验)
10. [功能八：图片合成视频](#10-功能八图片合成视频)
11. [常见问题](#11-常见问题)
12. [从原始代码迁移对照](#12-从原始代码迁移对照)

---

## 1. 环境准备

### 1.1 安装依赖

```bash
cd refactor
conda env create -f environment.yml
conda activate cm1-typhoon-diagnostics
```

或手动安装：

```bash
pip install numpy scipy xarray netCDF4 matplotlib pyyaml opencv-python
```

### 1.2 连接数据

```bash
cd refactor
ln -s ../dataset dataset
```

项目已配置 `.gitignore`，`dataset/` 和 `output/` 不会被提交到 Git。

### 1.3 输出文件结构

所有输出统一放在 `output/` 目录下，脚本会自动创建所需子目录：

```
output/
├── budget/        # 方位角平均诊断 NC 文件
├── se_pipeline/   # SE 方程诊断结果（NC + PNG + IEEE）
├── experiments/   # 敏感性实验对比图
├── figures/       # 通用图片
│   └── az_avg/    # 方位角平均时序图
├── videos/        # 视频
├── tracks/        # 台风中心轨迹 CSV + 图
└── profiles/      # 径向剖面文本
```

配置集中在 `config/default.yaml`，可复制后自定义：

```bash
cp config/default.yaml config/my_exp.yaml
# 编辑 my_exp.yaml 修改路径和参数
```

---

## 2. 项目结构

```
refactor/
├── src/                          # 核心库（7 个模块）
│   ├── center_finder.py          # 台风中心定位（3 种方法）
│   ├── coordinates.py            # 坐标变换、去交错
│   ├── config.py                 # 配置数据类
│   ├── azimuthal_avg.py          # 方位角平均 + 动量收支诊断
│   ├── se_equation.py            # SE 方程系数构建 + SOR 求解
│   ├── io.py                     # netCDF/IEEE 读写
│   └── plotting.py               # 统一绘图（6 类图）
│
├── scripts/                      # 可执行脚本（9 个）
│   ├── run_budget_diagnostic.py  # 方位角平均 + 收支诊断
│   ├── run_se_pipeline.py        # SE 方程诊断（三合一入口）
│   ├── plot_singlepage_diagnostics.py  # 单页诊断图（径向/切向/组合）
│   ├── plot_horizontal_field.py  # 水平场填色图/视频
│   ├── plot_budget_terms.py      # 收支项快速绘图
│   ├── track_centers.py          # 中心追踪
│   ├── extract_profile.py        # 径向剖面提取
│   ├── profile_centers.py        # 3D 中心轨迹
│   └── make_video.py             # PNG→MP4 视频合成
│
├── experiments/                  # 敏感性实验分析
│   ├── compare_evap.py           # CTRL vs NOEVAP vs EVAP_ONLY
│   ├── sensitivity_evap.py       # 蒸发冷却强度敏感性
│   └── dipole_analysis.py        # 偶极子加热/冷却敏感性
│
├── config/default.yaml           # 全局默认配置
├── notebooks/                    # Jupyter Notebook（10 个）
├── code/                         # Fortran 源码
└── environment.yml               # Conda 环境
```

---

## 3. 功能一：台风中心定位与追踪

### 3.1 批量追踪所有时次

```bash
python scripts/track_centers.py \
    --input dataset/cm1out.nc \
    --output output/tracks/center_track.csv \
    --plot --plot-output output/tracks/center_tracks.png
```

输出：`output/tracks/center_track.csv`（每时次的 t, x, y, psfc_min）和轨迹图。

### 3.2 3D 中心轨迹（中心随高度变化）

```bash
python scripts/profile_centers.py \
    --input dataset/cm1out.nc \
    --time 400 --z-min 0.5 --z-max 20 --z-step 0.5 \
    --output output/figures/centers_3d.png
```

---

## 4. 功能二：方位角平均与动量收支诊断

将 CM1 笛卡尔网格输出转换到柱坐标，做方位角平均，计算径向 (u) 和切向 (v) 动量收支的 mean/eddy 分解。

### 4.1 快速测试（仅处理 5 个时次）

```bash
python scripts/run_budget_diagnostic.py \
    --input dataset/cm1out.nc \
    --output output/budget/budget_test.nc \
    --max-times 5
```

### 4.2 完整运行

```bash
python scripts/run_budget_diagnostic.py \
    --input dataset/cm1out.nc \
    --output output/budget/typhoon_azimuthal_avg_budget.nc \
    --max-r-km 300 --dr-km 2 --max-z-km 20
```

### 4.3 进阶运行（分组残差 + 内核约束 + 移速消减）

```bash
python scripts/run_budget_diagnostic.py \
    --input dataset/cm1out.nc \
    --output output/budget/budget_advanced.nc \
    --grouped-residual \
    --enable-core-stabilization --core-radius-km 6.0 \
    --subtract-translation-speed
```

### 4.4 关键参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `--input` | `dataset/cm1out.nc` | 输入 CM1 输出 NC 文件 |
| `--output` | `output/budget/...` | 输出诊断 NC 文件 |
| `--max-r-km` | 300 | 柱坐标最大半径 (km) |
| `--dr-km` | 2 | 径向分箱间隔 (km) |
| `--max-z-km` | 20 | 最大分析高度 (km) |
| `--max-times` | null | 仅处理前 N 个时次（调试用） |
| `--center-method` | min | min / centroid / streamfunction |
| `--grouped-residual` | false | 分组残差分配（提高闭合精度） |
| `--enable-core-stabilization` | false | 抑制 r=0 奇点噪声 |
| `--subtract-translation-speed` | false | 消减台风平移速度 |

### 4.5 输出变量

输出 NC 文件包含 **基本场**（ur, ut, w, prs, rho）和完整的**径向+切向动量收支项**：

| 前缀 | 含义 |
|:---|:---|
| `U_mr`, `U_eh` | 水平平流（mean + eddy） |
| `U_mv`, `U_ev` | 垂直平流（mean + eddy） |
| `U_magf`, `U_eagf` | 加速度梯度力组 |
| `U_dh`, `U_dv` | 扩散+湍流 |
| `ramp` / `tramp` | 径向/切向阻尼 |
| `curv_mean`, `curv_eddy` | 曲率项 |
| `pgrad_mean`, `pgrad_eddy` | 气压梯度力分解 |
| `br_total_raw` / `bt_total_raw` | 所有预算项总和 |
| `residual_after_allocation` | 残差分配后闭合残差 |

切向对应 `V_*` 和 `bt_*` 系列。原始项（调整前）以 `_raw` 后缀保存。

---

## 5. 功能三：SE 方程次级环流诊断

### 5.1 单时刻诊断（标准模式）

```bash
python scripts/run_se_pipeline.py \
    --mode single \
    --input dataset/cm1out.nc \
    --target-time-hours 48 \
    --output-dir output/se_pipeline/single_48h \
    --sor-omega 1.8
```

输出：`se_pipeline_products.npz`、`se_solution_fields.png`、`se_forcing_terms.png`。

### 5.2 蒸发冷却模式

```bash
python scripts/run_se_pipeline.py \
    --mode evap \
    --input dataset/cm1out.nc \
    --target-time-hours 72 \
    --output-dir output/se_pipeline/evap_72h \
    --evap-q0 -2e-4 \
    --evap-r-center 145 --evap-z-center 15 \
    --evap-r-half 105 --evap-z-half 2.5 \
    --sor-omega 1.8
```

偶极子模式：追加 `--evap-dipole --evap-q0 -5e-4`。

### 5.3 时间段平均模式

```bash
python scripts/run_se_pipeline.py \
    --mode timeavg \
    --input dataset/cm1out.nc \
    --time-avg-start-hours 64 --time-avg-end-hours 72 \
    --output-dir output/se_pipeline/avg_64_72h \
    --sor-omega 1.5
```

### 5.4 求解器调优

| 问题 | 操作 |
|:---|:---|
| SOR 不收敛 | 降低 `--sor-omega` 至 1.5 或 1.0 |
| 解场太粗糙 | 降低 `--dr-km` 至 1.5 或 1.0 |
| 出流层异常 | 检查 `--baroclinic-scale`（默认 0.4） |
| 蒸发冷却无响应 | 确认 `--evap-q0` 设置正确（负值=冷却） |

---

## 6. 功能四：动量收支诊断单页图（⭐ 核心绘图）

**前置条件**：需先运行功能二生成 budget NC 文件（或已有处理好的 NC）。

本功能忠实复现 `radial_diagnostic_singlepage.ipynb` 的全部绘图逻辑，
包括三组 colorbar、symlog 色阶、径向高斯平滑、诊断项加权求和等。

### 6.1 径向动量收支单页图（时间段平均）

```bash
python scripts/plot_singlepage_diagnostics.py \
    --panel radial \
    --input output/budget/typhoon_azimuthal_avg_budget.nc \
    --mode time_range --start-hour 42 --end-hour 74 \
    --output output/figures/radial_diag_42_74h.png
```

**输出**：单页多面板图，包含：

| 面板 | 内容 |
|:---|:---|
| 第 1–3 列 | 径向风 ur 起止 + 趋势 ∂u/∂t |
| 第 4 列 | diag_sum = 全部独立收支项加权和 |
| 第 5+ 列 | 各项诊断（U_mr, U_eh, U_mv, U_ev, U_magf, U_eagf, U_dh, U_dv, ramp, coriolis, pgrad_mean, pgrad_eddy, curv_mean, curv_eddy, br_total_raw, tendency, residual...） |

**绘图特性**：
- 🔴🔵 三组独立 colorbar：风场 / 普通诊断项 / 大项（sum + pgrad + curv + tendency）
- 📐 `SymLogNorm` 色阶（兼顾眼墙极值和流出层弱信号）或线性 `BoundaryNorm`
- 🧹 径向高斯平滑（沿 R 方向 `gaussian_filter1d`，sigma=2.0），针对 pgrad 等锯齿项
- 🏷️ 等高线标签 + 底部信息栏

### 6.2 切向动量收支单页图

```bash
python scripts/plot_singlepage_diagnostics.py \
    --panel tangential \
    --input output/budget/typhoon_azimuthal_avg_budget.nc \
    --mode time_range --start-hour 42 --end-hour 74 \
    --output output/figures/tangential_diag_42_74h.png
```

结构与径向图对称，使用切向诊断项（V_mr, V_eh, V_magf, coriolis_t, pgrad_t, vcurv_mean 等）。

### 6.3 单时刻模式

```bash
python scripts/plot_singlepage_diagnostics.py \
    --panel radial \
    --input output/budget/typhoon_azimuthal_avg_budget.nc \
    --mode time_point --target-hour 60 \
    --output output/figures/radial_diag_60h.png
```

### 6.4 自定义线性组合单窗图

绘制任意诊断项的加权和，例如 `U_magf - U_mr`：

```bash
python scripts/plot_singlepage_diagnostics.py \
    --panel combo \
    --input output/budget/typhoon_azimuthal_avg_budget.nc \
    --mode time_range --start-hour 42 --end-hour 74 \
    --combo-terms "1.0,U_magf -1.0,U_mr" \
    --output output/figures/combo_magf_minus_mr.png
```

`--combo-terms` 格式：`"系数1,变量1 系数2,变量2 ..."`（空格分隔，符号需显式写负号）。

例如绘制 `U_magf - U_mv - U_eh + U_dh`：
```bash
--combo-terms "1.0,U_magf -1.0,U_mv -1.0,U_eh 1.0,U_dh"
```

### 6.5 高级参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `--max-r-km` | 300 | 绘图域最大半径 (km) |
| `--max-z-km` | 20 | 绘图域最大高度 (km) |
| `--no-smoothing` | false | 关闭径向平滑 |
| `--no-symlog` | false | 使用线性色阶替代 symlog |

---

## 7. 功能五：水平风场绘图与视频

### 7.1 单帧填色图

```bash
# 海平面气压
python scripts/plot_horizontal_field.py \
    --input dataset/cm1out.nc --var prs \
    --zh 0 --time 48 --xy-limit 200 \
    --output output/figures/psfc_48h.png

# 5km 高度 u 风场
python scripts/plot_horizontal_field.py \
    --input dataset/cm1out.nc --var u \
    --zh 5000 --time 48 --xy-limit 200
```

### 7.2 时间序列视频

```bash
python scripts/plot_horizontal_field.py \
    --input dataset/cm1out.nc --var prs \
    --zh 1000 --save-video --fps 5 \
    --start-time 0 --end-time 144
```

| 参数 | 说明 |
|:---|:---|
| `--save-video` | 切换为视频模式 |
| `--fps` | 帧率 |
| `--xy-limit` | 裁剪域半径 (km) |
| `--cmap` | colormap 名称 |
| `--vmin/--vmax` | 手动色阶 |

---

## 8. 功能六：径向剖面提取

沿 X 轴从台风中心出发提取变量一维剖面：

```bash
python scripts/extract_profile.py \
    --input dataset/cm1out.nc \
    --time 400 --zh 2.0 \
    --var prs --stop-x-km 1000 \
    --output output/profiles/prs_profile_2km.txt
```

---

## 9. 功能七：敏感性实验

### 9.1 不同数据集的风场对比（标准流程）

假设有两个 CM1 输出：`cm1out_Morrison.nc` 和 `cm1out_Thompson.nc`。

**步骤 1：运行收支诊断**

```bash
python scripts/run_budget_diagnostic.py \
    --input dataset/cm1out_Morrison.nc \
    --output output/budget/budget_Morrison.nc

python scripts/run_budget_diagnostic.py \
    --input dataset/cm1out_Thompson.nc \
    --output output/budget/budget_Thompson.nc
```

**步骤 2：分别绘图对比**

```bash
python scripts/plot_budget_terms.py \
    --input output/budget/budget_Morrison.nc \
    --mode grouped --time 48 \
    --output output/figures/budget_Morrison_48h.png

python scripts/plot_budget_terms.py \
    --input output/budget/budget_Thompson.nc \
    --mode grouped --time 48 \
    --output output/figures/budget_Thompson_48h.png
```

**步骤 3：强度时间序列对比**

使用 `notebooks/typhoon_intensity_comparison.ipynb`，在 datasets 字典中修改文件路径，运行即可得到多曲线对比图。

**步骤 4：SE 响应对比**

```bash
python scripts/run_se_pipeline.py --mode single \
    --input dataset/cm1out_Morrison.nc --target-time-hours 48 \
    --output-dir output/se_pipeline/se_Morrison

python scripts/run_se_pipeline.py --mode single \
    --input dataset/cm1out_Thompson.nc --target-time-hours 48 \
    --output-dir output/se_pipeline/se_Thompson
```

### 9.2 蒸发冷却敏感性（CTRL vs NOEVAP vs EVAP_ONLY）

**步骤 1：运行各实验组**

```bash
# CTRL
python scripts/run_se_pipeline.py --mode single \
    --input dataset/cm1out.nc --target-time-hours 72 \
    --output-dir output/experiments/exp_ctrl

# NOEVAP
python scripts/run_se_pipeline.py --mode single \
    --input dataset/cm1out_noevap.nc --target-time-hours 72 \
    --output-dir output/experiments/exp_noevap

# EVAP_ONLY
python scripts/run_se_pipeline.py --mode evap \
    --input dataset/cm1out.nc --target-time-hours 72 \
    --output-dir output/experiments/exp_evap_only
```

**步骤 2：对比分析**

修改 `experiments/compare_evap.py` 中的 `BASE` 为 `Path("output/experiments")`，然后：

```bash
python experiments/compare_evap.py
```

### 9.3 蒸发冷却强度敏感性

```bash
python experiments/sensitivity_evap.py
```

### 9.4 偶极子敏感性

```bash
python experiments/dipole_analysis.py
```

---

## 10. 功能八：图片合成视频

```bash
python scripts/make_video.py \
    --input-dir output/figures/az_avg \
    --pattern "az_avg_*.png" \
    --output output/videos/az_avg_animation.mp4 \
    --fps 15
```

---

## 11. 常见问题

**Q: 运行时找不到模块 `src.xxx`？**
确保在 `refactor/` 目录下运行：`cd refactor && python scripts/xxx.py`

**Q: SOR 不收敛？**
依次尝试：降低 `--sor-omega` 至 1.5→1.0；增大 `--dr-km` 至 3；减小 `--max-r-km` 至 200

**Q: 输出文件太大？**
用 `--max-times 5` 仅处理少量时次测试

**Q: 变量名不同？**
系统自动从候选列表匹配，无需修改。候选列表在 `config/default.yaml` 的 `variables` 段

**Q: 如何在服务器运行？**
复制 `refactor/` 到服务器 → `ln -s /path/to/data dataset` → 安装环境 → 运行

---

## 12. 从原始代码迁移对照

| 原始文件 | 新位置 | 说明 |
|:---|:---|:---|
| `se_diagnostic_pipeline.py` | `scripts/run_se_pipeline.py --mode single` | 三合一 |
| `se_diagnostic_pipeline_evap.py` | `scripts/run_se_pipeline.py --mode evap` | |
| `se_diagnostic_pipeline_timeavg.py` | `scripts/run_se_pipeline.py --mode timeavg` | |
| `cm1_azimuthal_avg_budget_full*.py` | `scripts/run_budget_diagnostic.py` | 二合一 |
| `cm1_out_nc_plot.py` | `scripts/plot_horizontal_field.py` | |
| `plot_single_frame.py` | `scripts/plot_horizontal_field.py`（不加 `--save-video`） | |
| `radial_diagnostic_singlepage.ipynb` | `scripts/plot_singlepage_diagnostics.py` | 完整复现，功能更强 |
| `u_budget_diagnostic_groupplot.ipynb` | `scripts/plot_budget_terms.py --mode grouped` | |
| `convert_to_pptx.py` | 保留在原位 | 独立文档工具 |

所有原始文件保持不动，可继续独立使用。
