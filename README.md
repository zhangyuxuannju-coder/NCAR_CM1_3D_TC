# CM1 Typhoon Diagnostic Toolkit — 操作手册

台风数值模拟诊断分析工具包，覆盖从 CM1 原始输出到论文图表的完整科研流水线。

---

## 目录

1. [环境准备](#1-环境准备)
2. [项目结构](#2-项目结构)
3. [功能一：台风中心定位与追踪](#3-功能一台风中心定位与追踪)
4. [功能二：方位角平均与动量收支诊断](#4-功能二方位角平均与动量收支诊断)
5. [功能三：SE 方程次级环流诊断](#5-功能三se-方程次级环流诊断)
6. [功能四：动量收支诊断单页图 ⭐](#6-功能四动量收支诊断单页图-核心绘图)
7. [功能五：水平风场绘图与视频](#7-功能五水平风场绘图与视频)
8. [功能六：径向剖面提取](#8-功能六径向剖面提取)
9. [功能七：敏感性实验](#9-功能七敏感性实验)
10. [功能八：图片合成视频](#10-功能八图片合成视频)
11. [常见问题](#11-常见问题)
12. [从原始代码迁移对照](#12-从原始代码迁移对照)
13. [JET–CTRL 环境涡动 SE 与适用性诊断](#13-jetctrl-环境涡动-se-诊断)

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

### 1.4 JupyterLab 远程开发环境

服务器上已配置 JupyterLab + GPU conda 环境，支持远程浏览器访问。

#### Mac 浏览器访问

> ⚠️ 校园网防火墙封锁服务器 8888 端口，必须通过 SSH 隧道访问，不能直连公网 IP。

Mac 端 `~/.ssh/config` 配置：

```
Host nju-server
    HostName 114.212.48.225
    User zhangyx
    LocalForward 8888 localhost:8888
```

在 Mac 终端执行 `ssh nju-server` 登录后，端口转发自动生效，浏览器访问 `http://localhost:8888` 即可。

#### Windows 浏览器访问

Windows 10/11 自带 OpenSSH，PowerShell 中执行：

```powershell
ssh -L 8888:localhost:8888 zhangyx@114.212.48.225
```

保持窗口不关，浏览器访问 `http://localhost:8888`。

#### 服务端管理

JupyterLab 通过 systemd 用户服务实现开机自启，内核为 `Python 3 (cm1_tc + PyTorch CUDA)`。

```bash
# 服务管理
systemctl --user status jupyterlab     # 查看状态
systemctl --user restart jupyterlab    # 重启
systemctl --user stop jupyterlab       # 停止

# 日志
tail -f /data1/home/zhangyx/.jupyter/jupyterlab.log

# 修改密码
conda activate cm1_tc
jupyter server password
systemctl --user restart jupyterlab
```

#### 关键配置文件

| 文件 | 用途 |
|:---|:---|
| `~/.config/systemd/user/jupyterlab.service` | systemd 服务定义 |
| `~/.jupyter/jupyter_lab_config.py` | JupyterLab 配置（端口 8888、root_dir、密码） |
| `~/.local/share/jupyter/kernels/cm1_tc/kernel.json` | 内核定义（含 LD_LIBRARY_PATH） |
| `~/miniconda3/envs/cm1_tc/etc/conda/activate.d/env_vars.sh` | conda 激活钩子 |

#### GLIBCXX 版本问题

系统 libstdc++ 仅到 `GLIBCXX_3.4.25`（Rocky 8 的 GCC 8），而 numpy 2.4 需 `GLIBCXX_3.4.29`。修复方案：

- conda 环境的 `libstdc++.so.6.0.34` 已包含所需符号
- 通过 conda 激活钩子和 kernel.json 的 `env.LD_LIBRARY_PATH` 优先加载

#### pip/conda 镜像

```bash
# NJU pip 镜像
pip install <package> -i https://mirror.nju.edu.cn/pypi/web/simple

# conda 镜像已在 ~/.condarc 配置 NJU 源
conda install <package>
```

#### 激活环境

```bash
conda activate cm1_tc   # Python 3.12 + PyTorch 2.12 + CUDA 13.2 + xarray/netCDF4
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
    --input-file dataset/cm1out.nc \
    --target-time-hours 48 \
    --output-dir output/se_pipeline/single_48h \
    --sor-omega 1.8
```

输出：`se_pipeline_products.npz`、`se_solution_fields.png`、`se_forcing_terms.png`。

### 5.2 蒸发冷却模式

```bash
python scripts/run_se_pipeline.py \
    --mode evap \
    --input-file dataset/cm1out.nc \
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
    --input-file dataset/cm1out.nc \
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

---

## 13. JET–CTRL 环境涡动 SE 诊断

新增 `--mode env`，直接从 CTRL/JET 三维风场计算 Reynolds 涡动角动量通量
辐合，构造 `F_lambda_env = F_eddy(JET) - F_eddy(CTRL)`，并使用固定 CTRL
的 Bui general SE operator 求解直接环境强迫响应。

所有 SE 模式现在都从原始三维 CM1 场先做 TC 中心柱坐标分解，再形成二维
方位平均基本态和强迫。总 `Q`/`Fnu` 包含 eddy、扩散/PBL、非绝热及其他
CM1 显式源项；`hadv`/`vadv` 仅用于闭合检查，不与直接 eddy flux convergence
重复相加。各分量保存在 `se_pipeline_products.npz/.nc` 中。

完整公式、命令、输出变量和解释见
[`ENVIRONMENTAL_EDDY_SE.md`](ENVIRONMENTAL_EDDY_SE.md)。

在正式求解前，建议先运行 `--mode stability`，绘制 CTRL/JET 未正则化的
`I2_raw`、`D_raw`，并检查它们与 `F_lambda_env`、上层出流和急流位置的重合。
命令、分类含义和输出说明见 [`SE_APPLICABILITY.md`](SE_APPLICABILITY.md)。

本次代码修改包括：

- 新增 `src/se_applicability.py`：计算实际基本态、平衡投影和正则化比较三套稳定性场；
- 新增 `src/stability_cli.py`：把稳定性诊断接入统一命令行；
- `scripts/run_se_pipeline.py` 新增 `--mode stability` 和全部绘图参数；
- `src/environmental_eddy.py` 新增方位 eddy kinetic energy，用作急流/非对称风速位置代理；
- stability 模式跳过无关 CM1 budget 读取，并检查 CTRL/JET 时次一致性；
- 新增稳定性分类、强迫重合统计、NetCDF/NPZ/JSON 输出和回归测试。

### 13.1 推荐工作流程

建议不要直接从 regularization 后的 SE 解开始解释，而按以下顺序运行：

1. 使用 `--mode stability` 检查 CTRL 和 JET 的原始 (I^2,D)；
2. 检查 `F_lambda_env` 最大区是否与 `D_raw<=0` 重合；
3. 如果固定 CTRL 基本态在目标区满足 `K1>0, I2>0, D>0`，再运行 `--mode env`；
4. 对 `D_raw<=0` 区，只把正则化 SE 解解释为 balanced projection，不解释为原始模拟的真实次级环流。

主诊断使用 CM1 实际方位平均的 `theta/v`：

\[
K_1=-g\frac{\partial\chi}{\partial z},
\qquad
I^2=\chi\xi(\zeta+f)+C_g\frac{\partial\chi}{\partial r},
\]

\[
D=K_1I^2-
\left[\frac{\partial(\chi C_g)}{\partial z}\right]^2.
\]

程序严格区分：

- `*_raw`：CM1 实际方位平均基本态，没有热成风投影或正则化；
- `*_balanced_projection`：使用热成风反演位温，但没有正则化；
- `*_regularized`：为得到椭圆算子而调整后的比较场，不代表原始稳定性。

### 13.2 在服务器运行稳定性诊断

先更新代码并进入仓库：

```bash
git pull origin main
cd NCAR_CM1_3D_TC
```

然后运行：

```bash
python scripts/run_se_pipeline.py \
  --mode stability \
  --input-file /path/to/CTRL/cm1out.nc \
  --jet-input-file /path/to/JET/cm1out.nc \
  --target-time-hours 72 \
  --eddy-average reynolds \
  --f 5.0e-5 \
  --dr-km 12 \
  --max-r-km 1200 \
  --max-z-km 20 \
  --center-window 21 \
  --center-method min \
  --stability-outflow-threshold 2 \
  --stability-jet-speed-threshold 20 \
  --stability-forcing-percentile 90 \
  --stability-jet-axis-r-km 888 \
  --stability-jet-axis-z-km 12 \
  --output-dir output/se_applicability/72h
```

上例中的 `888 km` 和 `12 km` 分别对应当前 CM1 namelist 的
`jet_y_dist=888000 m` 和 `jet_z_ctr=12000 m`。如果改变了急流配置，应同步
修改这两个绘图参数。为了在图中显示急流轴，`--max-r-km` 必须大于急流轴
距 TC 中心的距离；如果只研究内核区，可以缩小半径并省略急流轴参数。

该模式不运行 SOR/SE 求解器，并主动跳过不需要的 CM1 `ptb_*`、`ub_*`、
`vb_*` budget 读取；主要计算成本来自两组原始三维风、密度和位温的读取及
TC 中心柱坐标方位分解。

### 13.3 stability 模式参数说明

#### 必需路径和模式参数

| 参数 | 默认值 | 含义 |
|:---|:---|:---|
| `--mode stability` | `single` | 启用 CTRL/JET 原始 SE 适用性诊断；该模式只诊断，不求解 SE 方程。 |
| `--input-file PATH` | `dataset/cm1out.nc` | CTRL 试验的 CM1 原始三维 NetCDF 文件。 |
| `--jet-input-file PATH` | 空 | JET 试验的 CM1 原始三维 NetCDF 文件；stability 模式必须提供。 |
| `--output-dir DIR` | `output/se_pipeline` | PNG、NetCDF、NPZ 和 JSON 的输出目录。建议每个时次使用独立目录。 |

#### 时间选择参数

| 参数 | 默认值 | 含义 |
|:---|:---|:---|
| `--time-index N` | `0` | 按 NetCDF 时间维下标选择时次。未指定目标时间时使用。 |
| `--target-time-seconds SEC` | 空 | 按秒选择最近的输出时次。不能与 `--target-time-hours` 同时使用。 |
| `--target-time-hours HOUR` | 空 | 按小时选择最近的输出时次，例如 `72`。优先于 `--time-index`。 |

CTRL 和 JET 最终选中的时间必须相差不超过 1 秒，否则程序会停止，避免把
不同演变阶段误作 JET−CTRL 差值。

#### 方位平均网格和台风中心参数

| 参数 | 默认值 | 含义 |
|:---|:---|:---|
| `--max-r-km R` | `300` | 诊断最大半径。若要显示当前 888 km 的急流轴，建议至少设为 `1000–1200`。 |
| `--dr-km DR` | `2` | 方位平均径向分箱宽度。若小于原始水平网格距，程序默认自动提高到原始分辨率。 |
| `--allow-fine-radial-bins` | 关闭 | 允许 `dr` 小于原始网格距；可能产生空环带和条纹，一般不建议。 |
| `--max-z-km Z` | `20` | 保留的最大高度。应覆盖急流、TC outflow 和其上方稳定层。 |
| `--center-window N` | `21` | 最低平滑海平面/地面气压定位前使用的水平平滑窗口，单位为格点。 |
| `--center-method {min,mean}` | `min` | 台风中心定位方法；`min` 使用平滑气压最低点，`mean` 使用中心定位模块的平均方法。 |

#### 物理和涡动分解参数

| 参数 | 默认值 | 含义 |
|:---|:---|:---|
| `--f VALUE` | `5.0e-5 s^-1` | 科氏参数。应与 CM1 试验设置或研究纬度一致。直接影响 `xi`、绝对涡度和 `I2`。 |
| `--eddy-average {reynolds,favre}` | `reynolds` | 涡动方位分解。`reynolds` 与 Bui 的 prime 定义一致；`favre` 是质量加权敏感性方案。 |
| `--theta-floor K` | `150` | 热成风平衡投影中的最低允许位温，只影响 `*_balanced_projection` 和后续比较，不覆盖 `*_raw`。 |
| `--theta-outer-smooth-window N` | `1` | 热成风反演时外边界位温廓线的垂直平滑窗口；`1` 表示不平滑。 |
| `--inertia-eps-ratio X` | `1.0e-3` | 正则化比较中 `K1/I2` 正下限相对于场幅值的比例；不影响 `*_raw`。 |
| `--elliptic-margin X` | `0` | 正则化比较要求的判别式绝对下限；不影响 `*_raw`。 |

`stability` 模式计算原始 (I^2,D) 时始终使用物理的
`baroclinic_scale=1.0`。全局参数 `--baroclinic-scale` 和
`--bui-baroclinic-scale` 不会改变本模式的 `I2_raw/D_raw`。

#### 图形叠加参数

| 参数 | 默认值 | 含义 |
|:---|:---|:---|
| `--stability-outflow-threshold MS` | `2 m s^-1` | 绿色等值线：JET 方位平均径向风达到该正值的位置，用于标记上层 outflow。 |
| `--stability-jet-speed-threshold MS` | `20 m s^-1` | 橙色等值线：`sqrt(2*eddy_kinetic_energy)` 达到该值的位置，是急流/非对称风速代理，不等同于 imposed jet 本身。 |
| `--stability-forcing-percentile P` | `90` | 在 `I2/D` 图上叠加的正、负 `F_lambda_env` 强值等值线百分位，范围 `0–100`。 |
| `--stability-jet-axis-r-km R` | 空 | 可选 imposed jet 轴到 TC 中心的半径；与高度同时给出时绘制金色星号。 |
| `--stability-jet-axis-z-km Z` | 空 | 可选 imposed jet 轴高度；与半径同时给出时绘制金色星号。 |

#### 输出控制参数

| 参数 | 默认值 | 含义 |
|:---|:---|:---|
| `--no-write-netcdf` | 关闭 | 不写 `se_applicability_products.nc`，但仍写压缩 NPZ 和 JSON。 |
| `--no-plot-solution` | 关闭 | 在 stability 模式中表示不生成两张 PNG，只写数值产品。 |

以下全局参数在 `stability` 模式中不参与计算：`--sor-*`、
`--regularization-max-iter`、`--q-override-file`、`--fnu-override-file`、
`--q-constant`、`--fnu-constant`、`--q-name`、`--fnu-name`、
`--q-candidates`、`--fnu-candidates`、`--source-mask-json`、
`--no-write-ieee`、`--ieee-prefix`、`--evap-*`、
`--time-avg-start-hours` 和 `--time-avg-end-hours`。

#### CM1 变量名映射参数

通常无需设置，程序会依次搜索候选变量。只有您的 CM1 输出变量名不同才需要覆盖。

| 参数 | 默认首选变量 | 含义 |
|:---|:---|:---|
| `--u-name` / `--u-candidates` | `u` / `u,ua,uinterp` | x 方向风及其候选列表。 |
| `--v-name` / `--v-candidates` | `v` / `v,va,vinterp` | y 方向风及其候选列表。 |
| `--w-name` / `--w-candidates` | `w` / `w,wa,winterp` | 垂直速度及其候选列表。 |
| `--prs-name` / `--prs-candidates` | `prs` / `prs,pres,p` | 三维气压；用于统一核心输入检查。 |
| `--rho-name` / `--rho-candidates` | `rho` / `rho,rhoa,dens` | 三维密度。 |
| `--theta-name` / `--theta-candidates` | `th` / `th,theta,thpert` | 三维位温。必须确认选中的变量是程序所需的总位温，而不是未经恢复的扰动量。 |
| `--psfc-name` / `--psfc-candidates` | `psfc` / `psfc,sfcprs,ps` | 地面气压，用于逐试验定位 TC 中心；缺失时回退到计算域中心。 |

候选列表使用英文逗号分隔，例如：

```bash
--theta-name theta --theta-candidates theta,th
```

### 13.4 稳定性分类和图中线条

`stability_class_ctrl/jet` 使用四种互斥分类：

| 编码 | 条件 | 解释 |
|:---:|:---|:---|
| `0` | `K1>0, I2>0, D>0` | 经典 SE 椭圆区。 |
| `1` | `K1>0, I2<=0` | 惯性不稳定区。 |
| `2` | `K1>0, I2>0, D<=0` | 惯性上稳定但对称/强切变非椭圆区。 |
| `3` | `K1<=0` | 静力不稳定区；分类时具有最高优先级。 |

图中叠加含义：

- 黑色实线：正的强 `F_lambda_env`；
- 黑色虚线：负的强 `F_lambda_env`；
- 绿色线：JET 方位平均径向 outflow；
- 橙色线：急流/非对称风速代理；
- 金色星号：用户输入的 imposed jet 轴；
- 蓝色实线：CTRL 的 `D_raw=0`；
- 红色虚线：JET 的 `D_raw=0`。

### 13.5 输出文件和关键变量

| 输出文件 | 内容 |
|:---|:---|
| `se_applicability_I2_D.png` | CTRL/JET 的原始 `I2`、`D` 和 JET−CTRL 差值，并叠加强迫、outflow 和急流位置。 |
| `se_applicability_classes.png` | CTRL/JET 四类稳定性区，以及 `F_lambda_env` 与两组 `D=0` 边界的重合。 |
| `se_applicability_products.nc` | 便于 xarray/NCL 读取的全部二维诊断场。 |
| `se_applicability_products.npz` | 与 NetCDF 对应的压缩 NumPy 产品；始终写出。 |
| `se_applicability_summary.json` | 输入文件、中心、时次、面积比例、强迫重合比例、最大强迫位置和输出路径。 |

关键变量：

| 变量 | 含义 |
|:---|:---|
| `I2_raw_ctrl`, `I2_raw_jet` | CTRL/JET 实际 CM1 方位平均基本态的广义惯性稳定度。 |
| `D_raw_ctrl`, `D_raw_jet` | CTRL/JET 未正则化 Bui 判别式。 |
| `I2_balanced_projection_*`, `D_balanced_projection_*` | 热成风平衡投影后的未正则化场。 |
| `I2_regularized_*`, `D_regularized_*` | 正则化比较场，不得当作原始模拟稳定性。 |
| `F_lambda_env` | `F_lambda_eddy(JET)-F_lambda_eddy(CTRL)`。 |
| `F_lambda_env_radial`, `F_lambda_env_vertical` | 环境 eddy 动量强迫的径向、垂直通量散度贡献。 |
| `F_lambda_env_ctrl_raw_elliptic` | 只保留实际 CTRL 原始椭圆区内的环境强迫。 |
| `F_lambda_env_ctrl_balanced_projection_elliptic` | 只保留 CTRL 平衡投影椭圆区内的环境强迫。 |
| `I2_vorticity_component_raw_*` | (I^2) 中 `chi*xi*(zeta+f)` 的贡献。 |
| `I2_baroclinic_component_raw_*` | (I^2) 中 `Cg*dchi/dr` 的贡献。 |
| `D_static_inertial_product_raw_*` | 判别式的 `K1*I2` 正/负贡献。 |
| `D_shear_penalty_raw_*` | 判别式中始终被减去的切变平方项。 |
| `stability_class_*` | 上述四类稳定性编码。 |
| `regularization_changed_mask_*` | 哪些格点被正则化修改。 |
| `outflow_jet` | JET 方位平均径向风。 |
| `eddy_speed_jet` | `sqrt(2*EKE)` 急流/非对称风速代理。 |

JSON 中最关键的量是：

- `nonelliptic_abs_forcing_fraction`：按柱坐标体积权重，
  `abs(F_lambda_env)` 位于非椭圆区的比例；
- `strong_forcing_nonelliptic_area_fraction`：最强 10% 非零环境强迫区中，
  非椭圆区域的比例；
- `maximum_abs_environmental_forcing`：最大绝对强迫点的位置，以及该点 CTRL/JET
  的原始和平衡投影 `I2,D`。

### 13.6 稳定性检查后运行环境 SE 响应

若 CTRL 目标区的原始/平衡投影满足椭圆条件，可以继续运行：

```bash
python scripts/run_se_pipeline.py \
  --mode env \
  --input-file /path/to/CTRL/cm1out.nc \
  --jet-input-file /path/to/JET/cm1out.nc \
  --target-time-hours 72 \
  --eddy-average reynolds \
  --bui-baroclinic-scale 1.0 \
  --dr-km 12 \
  --max-r-km 1200 \
  --max-z-km 20 \
  --output-dir output/se_environmental/72h
```

`env` 模式使用固定 CTRL operator 求解 `F_lambda_env` 的 balanced response。
即使正则化后求解成功，位于原始 `D_raw<=0` 区的结果仍只能解释为修改后
邻近平衡态的 balanced component，不能替代 CM1 中真实的惯性/对称不稳定动力过程。

