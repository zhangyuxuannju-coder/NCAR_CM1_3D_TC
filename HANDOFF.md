# TC_dynamic 研究交接文档

最后更新：2026-09-03
项目主目录：`/data1/home/zhangyx/project/TC_dynamic`
Windows SSHFS 映射：`Z:\`
GitHub：<https://github.com/zhangyuxuannju-coder/NCAR_CM1_3D_TC>

> 本文档写给完全没有上下文的新会话。接手后先阅读本文件、`AGENTS.md`、`CLAUDE.md`、`README.md` 和 `TODO.md`，不要直接覆盖或清理当前工作树。

## 0. 一分钟状态摘要

我们正在研究：**位于热带气旋高层出流附近的环境西风急流，如何通过非对称角动量输送、基本态惯性稳定度变化以及 Sawyer–Eliassen（SE）平衡响应，改变台风次级环流和强度演变。**

目前最可靠的证据链是：

```text
环境 jet
  -> 早期非对称/eddy 角动量输送
  -> 后续平均环流与出流结构调整
  -> 径向绝对角动量梯度 dM/dr 改变
  -> 惯性稳定度 I² 与 SE 左侧算子改变
  -> 强迫在不同基本态上激发不同的平衡次级环流
  -> 最终影响 TC 增强过程
```

已经确认：

- JET 初始场中的急流位置和强度与 namelist 一致；
- 72 h 的惯性稳定度差异几乎完全由径向绝对角动量梯度/绝对涡度差异实现，而不是简单的风速差异；
- 强度差只能解释部分稳定度差异，强度匹配后仍有明显结构残差；
- 角动量预算支持“早期 eddy 输送建立差异、随后平均环流接管”的阶段性路径；
- SE 全域交换试验显示，高层实际出流区在 55、80、110 h 的直接强迫主效应大于直接算子主效应，但强迫—算子交互项很大；
- 25N强度匹配诊断中，惯性稳定度算子扰动可产生有组织的高层径向环流投影；这是支持当前机制假设的证据，但不是独立因果证明；
- 25N CTRL在72–80 h的减弱不符合完整眼墙置换：没有持续第二切向风峰，仅82 h左右有短暂外侧反射率次峰；
- 真正未完成的核心任务，是建立惯性稳定度/角动量结构与快速增强（RI）和强度变化之间的时间因果关系。

## 1. 任务目标与总体思路

### 1.1 科学问题

两组 CM1 试验只有一个关键外部差异：一组添加位于 TC 北侧高层出流附近的环境西风急流（JET），另一组不添加（noJET）。目标是回答：

1. jet 是否通过环境 eddy 动量通量改变 TC 的次级环流？
2. jet 是否首先塑造非对称出流，再通过角动量重分布改变轴对称基本态？
3. JET/noJET 的惯性稳定度差异，是 jet 的直接结构作用，还是 TC 强度和出流强弱差异的间接结果？
4. SE 方程右端强迫变化与左侧基本态算子变化，哪一个对径向次级环流更重要？
5. 惯性不稳定/低惯性稳定度区域与 TC 快速增强、强度倾向有什么关系？

### 1.2 关键物理量

绝对角动量：

$$
M=rv+\frac12fr^2.
$$

经典惯性稳定度：

$$
I_{\rm classic}=\xi\eta,
\qquad
\xi=f+\frac{2v}{r},
\qquad
\eta=f+\zeta=\frac1r\frac{\partial M}{\partial r}.
$$

Bui 广义惯性稳定度：

$$
I^2_{\rm Bui}=\chi\xi(f+\zeta)+C\frac{\partial\chi}{\partial r},
\qquad \chi=\frac1\theta.
$$

SE 椭圆判别式：

$$
D=K_1 I^2_{\rm Bui}-K_2^2.
$$

只有原始基本态满足 `D>0` 时，经典诊断型 SE 方程才是严格椭圆边值问题。`D<=0` 区域的正则化解只能称为 **regularized balanced projection**，不能称为原始不稳定区域的真实次级环流。

SE 方程概念上写为：

$$
\mathcal L(\psi)=\mathcal F(Q,F_\lambda),
$$

其中左侧算子由静力稳定度、惯性稳定度和斜压/垂直切变控制；右侧包括加热和切向动量强迫。环境 eddy 强迫的目标形式是：

$$
\mathcal L_{\rm CTRL}(\psi_{\lambda,\rm env})
=-\frac{\partial}{\partial z}
\left(\chi_{\rm CTRL}\xi_{\rm CTRL}F_{\lambda,\rm env}\right).
$$

但当前 `F_lambda_env = eddy(JET)-eddy(noJET)` 包含直接环境急流、jet–TC 相互作用以及 jet 诱发的 TC 内部非对称响应，不能称为“纯外部急流通量”。

### 1.3 总体分析框架

研究被拆为四条相互补充的路径：

1. **状态结构**：比较 JET/noJET 的三维风场、轴对称风场、PV、绝对角动量、`dM/dr`、经典/Bui 惯性稳定度和 SE 判别式。
2. **预算机制**：用 CM1 原始预算项和直接三维 eddy 通量建立绝对角动量预算，判断谁改变了 `M` 和 `dM/dr`。
3. **SE 平衡投影**：分别诊断环境 eddy 强迫以及“强迫—算子交换试验”，区分右端强迫主效应、左侧算子主效应和交互项。
4. **强度/RI 联系**：将稳定度与角动量结构的时间序列同最低气压、最大风和增强率配准，做领先—滞后和阶段分析。这一部分仍待完成。

## 2. 数据、代码和已完成内容

### 2.1 服务器和数据文件

当前主要数据目录：

```text
/data/zhangyx/DATA
```

本研究核心文件：

```text
/data/zhangyx/DATA/cm1out_22N_nojet.nc
/data/zhangyx/DATA/cm1out_22N_8o_jet.nc
```

其他已复制到新目录的数据：

```text
cm1out_25N_5o_jet.nc
cm1out_25N_nojet.nc
cm1out_27N_8o_jet.nc
cm1out_27N_nojet.nc
cm1out_Morrison.nc
cm1out_thompson.nc
cm1out_ptype=1.nc
cm1out_km.nc
cm1outQzyx.nc
```

原 CM1 修改版源码在用户 Windows 电脑：

```text
E:\projects\cm1r20.3_jet
```

jet 配置为：

```text
jet_umax   = 45 m s-1
jet_z_ctr  = 12 km
jet_z_rad  = 2 km
jet_y_dist = 888 km（TC 北侧约 8°）
jet_y_rad  = 444 km
jet_z_bot  = 3 km
jet_z_top  = 16 km
```

### 2.2 数据路径修改

代码默认数据目录已经由旧路径改为 `/data/zhangyx/DATA`。已确认这些位置使用新路径：

- `README.md`
- `CLAUDE.md`
- `config/default.yaml`
- `scripts/run_se_pipeline.py`
- `src/_se_pipeline_single.py`
- `src/_se_pipeline_timeavg.py`
- `src/_se_pipeline_evap.py`
- 常用绘图和 profile/budget 脚本

`scripts/migrate_data1_to_data.sh` 中保留旧路径是有意的，因为它就是迁移脚本。

### 2.3 急流结构确认

脚本：

```text
scripts/plot_jet_yz_cross_section.py
scripts/plot_jet_horizontal_wind.py
```

最新 y-z 剖面：

```text
output/jet_structure/jet_yz_cross_section_t000h.png
output/jet_structure/jet_yz_cross_section_t000h.json
```

0 h 的 `JET-noJET` 最大西风为 `44.30 m s-1`，位于相对 TC 中心 `y=883.97 km`、`z=12.25 km`，与设置的 `45 m s-1 / 888 km / 12 km` 一致。总风场中心附近的低层红蓝偶极是 TC 环流，相减后消失。

### 2.4 强度与风场演变

主要脚本：

```text
scripts/plot_tc_intensity.py
scripts/plot_axisymmetric_wind_evolution.py
scripts/plot_horizontal_wind_comparison.py
scripts/plot_pv_streamlines_comparison.py
```

主要输出目录：

```text
output/figures
output/axisymmetric_wind_22N_8deg
output/horizontal_wind_22N_8deg
output/pv_streamlines_22N_8deg
```

已经绘制：

- JET/noJET 最低气压随时间变化；
- 25、55、80、110 h 的轴对称垂直风场；
- 300 km 内高层水平风和流线；
- 13 km 附近 PV 与流线；
- 500 km 内水平惯性稳定度与三维径向风叠加图。

### 2.5 惯性稳定度归因

脚本：

```text
scripts/analyze_inertial_stability_attribution.py
scripts/analyze_mgradient_i2_overlap.py
scripts/plot_horizontal_3d_inertial_stability.py
```

权威输出：

```text
output/inertial_attribution_22N_8deg
output/mgradient_i2_overlap
output/horizontal_3d_inertial_stability_72h
```

72 h、内出流层 `r=50–350 km, z=10–16 km` 的最新结果：

- noJET 原始 `I2<0` 面积比例：`35.43%`；
- JET 原始 `I2<0` 面积比例：`0.85%`；
- `I2<0` 与 `dM/dr<0` 的逐格 Jaccard：两组均为 `1.000`；
- `corr(delta I2, delta vorticity/M-gradient term)=0.999885`；
- Bui 斜压项差异只占总差异 RMS 的 `1.52%`；
- 经典分解中，直接风速因子贡献为负，而绝对涡度/角动量梯度贡献为正且更大。

因此目前可以非常有力地说：

> JET/noJET 的惯性稳定度差异在状态变量层面几乎完全由径向绝对角动量梯度变化实现，而不是“JET 台风较弱，所以风速较小，稳定度自然更高”。

但 `I2` 与 `dM/dr` 是诊断恒等关系，这不能单独证明谁在时间上造成了 `dM/dr` 变化。上游因果必须依赖角动量预算。

强度匹配结果位于：

```text
output/inertial_attribution_22N_8deg/analysis_report.md
```

JET 72 h 最接近 noJET 60 h。强度匹配能解释部分稳定度差异，但仍保留明显结构残差；所以当前结论是“强度是重要中介，但不是全部”。

### 2.6 角动量预算及闭合

脚本：

```text
scripts/analyze_angular_momentum_budget.py
scripts/summarize_angular_momentum_budget.py
scripts/plot_explicit_budget_closure.py
scripts/diagnose_cartesian_budget_closure.py
scripts/audit_cm1_budget_variables.py
```

主要输出：

```text
output/angular_momentum_budget_dense_22N_8deg
output/angular_momentum_budget_dense_22N_8deg/validation
```

预算形式：

$$
\frac{\partial\bar M}{\partial t}
=T_{\rm mean,r}+T_{\rm mean,z}
+T_{\rm eddy,r}+T_{\rm eddy,z}
+T_{\rm pgrad}+T_{\rm diff}+T_{\rm rdamp}+R_M.
$$

当前阶段性结论：

- 45–55 h：JET−noJET 的 eddy 角动量输送首先建立正差异；
- 55–65 h：平均环流/强度—出流结构路径接管；
- 65–75 h：平均流和 eddy 均为正，但受到非轴对称压力梯度力抵消；
- 在闭合较可靠的 45–70 h，区域积分中平均平流约 48%、eddy 约 43%、压力梯度约 2%、平均残差约 7%，只能作为该阶段归因，不能当作普适百分比。

72 h 显式预算闭合：

- noJET 区域平均 observation `7.478`，全部项之和 `7.076`；
- JET 区域平均 observation `5.713`，全部项之和 `5.563`；
- 逐格归一化 RMS 残差仍约 45–50%，所以不能只看区域平均闭合。

质量状态：25–60 h 多数通过；65、75 h 谨慎；70、72 h 可用；80 h noJET 和 110 h 两组失败，不能用于主要预算归因。

注意：预算中出现压力梯度项并非程序凭空添加。理论上的严格轴对称切向平均可消掉某些压力梯度贡献，但 CM1 的三维非轴对称 Cartesian 预算转换到 storm-centered angular momentum 后，方位相关压力梯度及坐标转换残差不会自动为零。

### 2.7 SE 适用性与环境 eddy 强迫

核心入口：

```text
scripts/run_se_pipeline.py
src/se_applicability.py
src/se_equation.py
```

主要输出：

```text
output/se_applicability
output/se_environmental
```

已完成：

- 计算经典和 Bui 惯性稳定度、`D`、原始非椭圆区与正则化区；
- 诊断 JET−noJET 的环境 eddy 切向动量强迫；
- 分离其径向和垂直 eddy flux contribution；
- 在稳定区域求环境 eddy 强迫对应的 SE 平衡响应；
- 绘制 50–300 km 内以及包含 jet 区域的结果。

重要解释边界：

- `D<=0` 处不能把原始 SE 当作经典椭圆诊断问题；
- 正则化只是求邻近平衡态的 balanced response；
- SE 只能解释平衡分量，CM1−SE 的差还包含惯性/对称不稳定、波动、瞬变调整和非线性过程；
- 当前 forcing difference 不是纯环境外部通量。

### 2.8 SE 强迫—算子交换试验

脚本：

```text
scripts/run_se_forcing_operator_factorial.py
scripts/run_se_forcing_operator_factorial_full.py
scripts/summarize_se_factorial_regularization_sensitivity.py
```

定义：

$$
\psi_{CC}=\mathcal L_C^{-1}\mathcal F_C,
\quad
\psi_{CJ}=\mathcal L_C^{-1}\mathcal F_J,
\quad
\psi_{JC}=\mathcal L_J^{-1}\mathcal F_C,
\quad
\psi_{JJ}=\mathcal L_J^{-1}\mathcal F_J.
$$

$$
\Delta u_F=u_{CJ}-u_{CC},
\quad
\Delta u_L=u_{JC}-u_{CC},
$$

$$
\Delta u_{\rm int}=u_{JJ}-u_{JC}-u_{CJ}+u_{CC}.
$$

最新主结果目录：

```text
output/se_factorial_full_regularized_eps_1e_5
output/se_factorial_full_regularization_sensitivity
```

全域为 `r=6–294 km, z=0.025–19.75 km`，时次为 20、55、80、110 h。径向风分解闭合误差约 `1e-14 m s-1`。

最弱正则化 `eps=1e-5` 的强迫/算子 RMS 比：

| 时次 | 全域 | 高层实际出流区 |
|---|---:|---:|
| 20 h | 2.16 | 1.00 |
| 55 h | 1.14 | 4.20 |
| 80 h | 3.44 | 5.93 |
| 110 h | 0.69 | 1.34 |

三组正则化 `1e-3, 1e-4, 1e-5` 下，高层出流区结论稳健：20 h 两者相当，55/80/110 h 直接强迫主效应更大。但 55–110 h 的强迫—算子交互很大，尤其 80 h，所以不能说“左侧算子不重要”；更准确的说法是：

> jet 既改变强迫，也改变基本态；强迫直接幅度在高层出流区更大，而算子决定该强迫如何投影成平衡径向环流。

`eps=1e-3` 会大面积修改后期系数，应只作敏感性。当前主图使用 `eps=1e-5`。


### 2.9 惯性稳定度算子扰动：等效强迫与单项 SE 响应

本阶段把 JET–CTRL 基本态差异造成的 SE 左侧算子变化一阶线性化，并等效移到
右端。将静力稳定度、斜压/垂直切变和惯性稳定度系数分别记为 A、B 和 I²，
以强度匹配 CTRL 的次级环流 (u_C,w_C) 为参考，代码计算：

$$
S_A=-\partial_r(\Delta A\,w_C),
\qquad
S_I=\partial_z(\Delta I^2\,u_C),
$$

$$
S_B=-\partial_r(\Delta B\,u_C)+\partial_z(\Delta B\,w_C).
$$

惯性稳定度单项试验只保留 S_I，并求解：

$$
\mathcal L_{C,\mathrm{reg}}(\psi_I)=S_I.
$$

实现文件：

- `scripts/solve_matched_operator_forcing_outer100.py`：读取场、构造/正则化匹配
  CTRL算子、计算三个等效强迫并做稀疏SE反演；
- `scripts/plot_inertial_operator_evolution_matched.py`：只施加 S_I，绘制多时次
  强迫、径向风和垂直速度响应；
- `scripts/plot_matched_operator_regularization_sensitivity.py`：比较不同椭圆性
  下限下的响应稳健性。

最新输入与设置：

~~~text
CTRL: /data/zhangyx/DATA/cm1out_25N_nojet.nc
JET : /data/zhangyx/DATA/cm1out_25N_9o_jet_30_15km.nc
f=6.1636e-5 s-1; dr=12 km; r=0–1200 km; z=0–20 km
regularization eps_ratio=1e-5
~~~

匹配限制为JET 70–90 h、CTRL 80–140 h。当前配对：

| JET/CTRL | ΔP | ΔV | ΔRMW | 使用边界 |
|---|---:|---:|---:|---|
| 70/85 h | -1.03 hPa | +0.63 m s-1 | -12 km | 仅结构参考 |
| 75/95 h | +0.93 hPa | +0.40 m s-1 | 0 km | 最佳配对 |
| 80/110 h | -1.24 hPa | -0.57 m s-1 | 0 km | 最佳配对 |
| 85/135 h | -3.99 hPa | +1.04 m s-1 | 0 km | 弱匹配 |
| 90/140 h | -5.01 hPa | +2.11 m s-1 | 0 km | 弱匹配 |

权威输出：

~~~text
output/inertial_operator_evolution_25N_9deg_30_15km_J70_90_C80_140
output/inertial_operator_evolution_25N_9deg_30_15km_J70_90_C80_140_allradius
output/static_operator_only_25N_9deg_30_15km_J70_90_C80_140
~~~

当只保留100 km外 S_I 时，内出流区SE径向风RMS约0.16–0.50 m s-1；不去除
100 km内强迫时可达0.98–5.54 m s-1。因此内核强度/眼墙差异会显著污染算子
归因，主结果应使用 `r>=100 km` 强迫，全半径结果只作敏感性。

彩色SE响应是非局地反演结果，不等同于急流轴附近局地强迫。急流高度上移后，
强迫可上移，而响应仍优先投影到TC自身的出流层。强迫与响应必须分图解释。

当前可表述为：强度近似匹配后，惯性稳定度系数差异仍能产生结构清晰、量级
不可忽略的平衡次级环流投影，支持“jet通过惯性稳定度算子改变次级环流”作为
优先机制假设。不能表述为已经证明它主导了JET–CTRL强度差。

### 2.10 25N CTRL 72–80 h 强度突降

脚本与输出：

~~~text
scripts/diagnose_ctrl_eyewall_cycle.py
scripts/diagnose_ctrl_eyewall_cycle.sbatch
output/ctrl_eyewall_cycle_25N_66_88h/
~~~

诊断显示：72–80 h最低气压约由974.1升至980.1 hPa；最大风从约50–54降至
43.4 m s-1；RMW仅由22.5扩到28.5 km后重新收缩。66–88 h始终只有一个显著
低层切向风峰；82 h附近虽短暂出现约64.5 km反射率次峰，但没有持续外侧风峰、
外眼墙收缩接管和内眼墙消亡。因此目前不支持完整眼墙置换，更像内部非对称
眼墙调整、雨带/涡旋罗斯贝波扰动或失败的次眼墙形成。

为得到不被单成员突降主导的可信CTRL基线，优先顺序是：

1. 找回生成数据的最终namelist；`code/namelist.input` 的fcor与NetCDF不一致；
2. CTRL/JET各做3–5个微扰集合成员；
3. 做 `dx_inner=1.5 km`，条件允许增加1 km的分辨率敏感性；
4. 做当前微物理与 `ptype=5` Morrison的成对敏感性；
5. 若启用一维海洋，比较固定SST、原混合层和更深混合层；
6. 60–90 h把三维输出加密到15–30 min；
7. 最后才测试 `kdiff6`，不能靠增强扩散直接“调平”强度曲线。

### 2.11 当前两条最高优先级工作线

#### A. 建立可信 CTRL 基线

目标不是挑一条平滑曲线，而是判断突降属于可重复物理响应、内部随机变率还是
参数/分辨率伪影。任何新CTRL设置都必须与JET成对保持一致。

#### B. 建立 JET–CTRL 强度差的机制闭环

当前优先假设：

~~~text
jet -> eddy角动量输送 -> dM/dr与I²改变
    -> 惯性稳定度SE算子改变 -> 平衡次级环流改变 -> 强度倾向差异
~~~

下一步要验证 S_I/SE响应是否领先强度差、与CM1实际径向风和上升运动是否同号，
并与传统热力强迫、动量强迫、静力和斜压算子项统一量级比较。还必须检验强度/
RMW匹配、时间平均、正则化与集合稳健性，并通过jet高度、距离、宽度和强度的
CM1因子试验建立因果证据。

### 2.12 本会话本地预览目录

服务器输出是权威、持久版本。本会话为了在 Codex 中显示图片，还复制了一份到：

```text
C:\Users\14351\.codex\visualizations\2026\08\19\01a019ea-4c33-7c32-ba97-54036ac1ab4d
```

其中包含全域 SE 图、正则化敏感性、急流 y-z 剖面以及早期诊断图。新会话优先从服务器 `output/` 读取，必要时再复制到新的可视化目录。

## 3. 当前问题和下一步任务

### 3.1 最核心的科学缺口：惯性稳定度与 RI/强度变化的关系

目前已经说明“jet 改变了 `dM/dr` 和 `I2`”，但还没有说明这种稳定度变化到底促进还是抑制 TC 快速增强，也没有建立时间先后。

下一步建议：

1. 构建逐小时或 3 h 时间序列：
   - 最低气压、最大切向风、RMW；
   - 6/12/24 h 强度倾向；
   - 内出流层 `I2<0` 面积比例、`I2` 中位数/分位数；
   - `dM/dr<0` 面积、出流强度和出流半径；
   - eddy/mean angular-momentum tendency。
2. 做领先—滞后分析，不只做同位相相关：稳定度改变是否领先强度倾向 3–24 h？
3. 按 pre-RI、RI、post-RI 或 45–55、55–65、65–75 h 阶段积分，而不是全生命周期平均。
4. 同时做相同时次和强度匹配比较。强度是 jet 的中介变量，匹配结果只能解释为“强度条件化结构残差”，不能代替总效应。
5. 将 `I2`、`M_r` 和预算项按 `r/RMW` 以及 outflow radius 配准，减少结构位置错位。
6. 最终区分两种可能路径：
   - 低稳定度允许径向流更容易穿越 `M` 面，可能促进响应；
   - 真正的 `I2<0/D<0` 表示平衡理论失效，可能触发非平衡混合并重排角动量，不能简单等同于增强有利。

### 3.2 3D 非对称结构仍需理论上谨慎

用户提出的工作假设是：jet 首先使高层出流非对称，额外 eddy 项随后改变轴对称 `M_r/I2`。这个方向合理，但“经典惯性稳定度”本质上来自轴对称基本态。水平 3D 图可以画局地 storm-relative 径向梯度代理，但不能未经说明就把它当作严格的三维惯性稳定判据。

建议下一步把问题拆成：

- 三维非对称输送：直接画 `u'M'`、`w'M'`、eddy torque 和出流方位结构；
- 轴对称基本态响应：画方位平均 `M_r`、`I2` 和 SE operator；
- 用预算连接前者到后者，而不是强行定义一个唯一的“3D SE 惯性稳定度”。

### 3.3 “纯环境 eddy 强迫”尚未完全隔离

目前 `eddy(JET)-eddy(noJET)` 是 jet-induced total eddy response。若要进一步隔离外部环境，建议：

- 在远离 TC 的环境区定义 background jet，并做背景/涡旋分解；
- 追踪通量从 750–1100 km jet 区向 50–350 km 内出流区传播和累积；
- 做 source–receptor 诊断：固定 CTRL operator，只施加按空间来源分区的 eddy forcing；
- 将径向和垂直 eddy flux 同时保留；
- 用更多 jet 距离/强度试验或 ensemble 检验系统响应。

### 3.4 SE 全域求解的数学限制

不能真正“不管惯性不稳定直接求原始方程”。原始 `D<=0` 时 PDE 是混合型或双曲型，经典椭圆边值求解没有数学保证。当前全域结果经过正则化，只能解释为 balanced projection。

后续论文建议同时报告：

- 原始 `D<=0` 区域；
- 实际被正则化的系数区域；
- 不同正则化下的敏感性；
- 稳定区严格 SE 结果与全域 regularized 结果分别讨论。

### 3.5 角动量预算后期闭合不足

80 h noJET 和 110 h 两组逐格闭合失败。不要用这些时次做主要预算归因。需要检查：

- 移动中心和固定中心的坐标倾向；
- 时间积分窗口和输出频率；
- 垂直/径向边界通量；
- 未输出的微物理、湍流或数值过程；
- 将 `M` 预算再求径向导数时的噪声放大。

### 3.6 数据迁移尚未完成，`/data1` 仍然满

当前磁盘：

```text
/data1: 11 T，总体显示 100%，仅约 85 G 可用
/data : 84 T，约 51 T 可用
```

虽然所有主要大文件已经复制到 `/data/zhangyx/DATA`，但旧目录 `/data1/home/zhangyx/data` 中仍保留约 11 个大文件，尚未删除或替换成软链接，因此当前是“双份占用”。

不要直接 `rm`。先处理 `scripts/migrate_data1_to_data.sh` 的一个问题：该脚本要求目标目录文件总数与源目录完全一致，但目标目录额外包含 `cm1out_25N_5o_jet.nc`，所以当前直接重跑可能因 file-count mismatch 停止。安全方案有两种：

1. 修改脚本，只对源目录清单中的相对路径做大小和 SHA-256 比较，允许目标目录存在额外文件；或
2. 临时把目标目录的额外文件移到 `/data/zhangyx/` 其他安全目录，完成迁移，再移回。

只有出现以下标志后才能确认迁移完成：

```text
/data/zhangyx/.tc_dynamic_data_migration/MIGRATION_COMPLETE
```

完成后旧路径应成为软链接：

```text
/data1/home/zhangyx/data -> /data/zhangyx/DATA
```

### 3.7 Git 工作树很脏，尚未整理提交

当前存在大量 modified/untracked 文件，还有两个旧 figure 被删除。不要运行 `git reset --hard`、`git clean` 或覆盖式 checkout。当前服务器 Git remote 只有：

```text
mac = ssh://zhangyuxuan@localhost:2222/Users/zhangyuxuan/git_mirror/NCAR_CM1_3D_TC.git
```

它依赖未开机的 Mac 中转站，当前不可用。服务器本身不能正常访问外网，因此 GitHub 同步应在 Windows 端通过 `Z:` 工作树完成，或先复制到 Windows 本地再提交/推送。GitHub 地址见本文开头。

## 4. 踩过的坑和经验教训

### 4.1 不要把 Bui 简化方程和 general SE 混用

研究目标是 Bui general SE Eq. (14)，不是 Boussinesq 简化 Eq. (8)。Challa 最终式是 Eq. (4)。涉及公式、边界条件和符号时必须回到原论文核对。

### 4.2 惯性稳定度的两种定义用途不同

- `I_classic = (f+2v/r)(f+zeta)`：方便直接解释风速因子和 `M_r`；
- `I2_Bui = chi*xi*(f+zeta)+C*chi_r`：与 general SE operator 一致。

项目中旧路径曾使用过不一致定义，已增加 legacy test。画图时要明确使用哪个量，不能都叫 `I2` 而不注明。

### 4.3 `I2>0` 不等于 SE 一定可解

最终必须检查 `D>0`。即使 `I2>0`，强斜压/切变也可能使 `D<=0`。

### 4.4 正则化不是物理证明

`eps=1e-3` 在后期会修改大量网格，可能人工压低算子效应。当前使用 `1e-5` 作为最弱主方案，同时保留 `1e-3/1e-4` 敏感性。正则化结果必须叫“修改后基本态的平衡投影”。

### 4.5 流函数正负不能直接解释成入流/出流

径向风由流函数的垂直梯度决定，不由 `psi` 本身正负决定。交换试验必须把 `psi` 转成 `u_SE` 和 `w_SE` 后解释。

### 4.6 forcing effect 不是纯环境 eddy effect

当前交换的是总 `Q` 和总 `F_lambda`，所以包含加热、总动量强迫、jet 诱发对流和内部 eddy 响应。命名时不能直接写成 `F_lambda_env` 独立贡献。

### 4.7 区域平均闭合会掩盖逐格残差

预算正负残差会抵消。必须同时报告二维图、空间相关、RMS 残差和区域平均，不能只画 observation 与 sum 的两条区域平均曲线。

### 4.8 压力梯度项不能随意删除

三维非轴对称 CM1 Cartesian budget 转换到 storm-centered angular momentum 后，压力梯度项可能非零。删除它会破坏闭合，也会漏掉 jet–TC 非对称结构的一部分。

### 4.9 `I2` 与 `M_r` 的高相关是恒等关系，不是上游因果

逐格 Jaccard=1 和相关接近 1 证明惯性稳定度差异通过哪个状态变量实现，但不能证明 eddy torque 是唯一原因。谁改变 `M_r` 必须由预算和时间先后回答。

### 4.10 3D 局地惯性稳定度需要谨慎命名

经典公式假设轴对称基本态。三维水平截面上的 storm-relative 局地代理适合展示非对称结构，但不能不加说明地称为严格的三维 SE 惯性稳定度。

### 4.11 服务器 Python 环境容易用错

- `python` 命令可能不存在；
- `/usr/bin/python3` 是 Python 3.6.8，无法运行使用 `from __future__ import annotations` 的新脚本；
- 推荐解释器：`/data1/home/zhangyx/miniconda3/envs/cm1_tc/bin/python`（Python 3.12.13）；
- 也可用 `/data1/home/zhangyx/miniconda3/bin/python`（Python 3.13.12）。

### 4.12 SSHFS 路径和权限

- 正确 UNC 形式是 `\\sshfs.k\user@host\remote-path`，不是 `\\sshfs\...`；
- Z: 断开时先删除旧映射再重连；
- Codex sandbox 有时能读 Z: 但写入提示 Access denied，需要用户批准提升权限，或用 `scp` 上传；
- 不要把私钥复制到聊天、仓库或服务器。

### 4.13 图片在 Codex 中不可见时

服务器路径不能直接渲染。把 PNG 复制到当前会话允许写入的本地 visualization 目录，然后在回复中使用绝对 Windows 路径。

## 5. 服务器连接、文件传输、编辑和运行

### 5.1 连接信息

```text
主机：114.212.48.225
用户：zhangyx
SSH alias：zhangyx-server（如果本机 ~/.ssh/config 保留）
私钥：C:\Users\14351\.ssh\id_ed25519_codex
服务器项目：/data1/home/zhangyx/project/TC_dynamic
服务器数据：/data/zhangyx/DATA
```

Windows PowerShell 直接连接：

```powershell
ssh -i "$env:USERPROFILE\.ssh\id_ed25519_codex" zhangyx@114.212.48.225
```

测试连接：

```powershell
ssh -i "$env:USERPROFILE\.ssh\id_ed25519_codex" `
  -o BatchMode=yes -o ConnectTimeout=15 `
  zhangyx@114.212.48.225 "hostname; whoami; pwd"
```

### 5.2 挂载服务器项目为 Z:

先清理失效映射：

```powershell
net use Z: /delete /yes
```

重新挂载：

```powershell
net use Z: \\sshfs.k\zhangyx@114.212.48.225\project\TC_dynamic /persistent:no
```

验证：

```powershell
Get-ChildItem Z:\
Get-Content Z:\HANDOFF.md -TotalCount 20
```

Z: 是服务器项目的实时映射，不是本地副本。直接编辑 `Z:\scripts\...` 就是在修改服务器文件。

### 5.3 从服务器拉取文件到 Windows

单文件：

```powershell
scp -i "$env:USERPROFILE\.ssh\id_ed25519_codex" `
  zhangyx@114.212.48.225:/data1/home/zhangyx/project/TC_dynamic/output/jet_structure/jet_yz_cross_section_t000h.png `
  C:\Users\14351\Downloads\
```

整个目录：

```powershell
scp -i "$env:USERPROFILE\.ssh\id_ed25519_codex" -r `
  zhangyx@114.212.48.225:/data1/home/zhangyx/project/TC_dynamic/output/se_factorial_full_regularized_eps_1e_5 `
  C:\Users\14351\Downloads\
```

大数据不要通过本机中转；计算和大文件复制应在服务器 `/data1` 与 `/data` 之间完成。

### 5.4 上传脚本或文件

```powershell
scp -i "$env:USERPROFILE\.ssh\id_ed25519_codex" `
  C:\path\to\script.py `
  zhangyx@114.212.48.225:/data1/home/zhangyx/project/TC_dynamic/scripts/
```

在 Codex 中应优先用 `apply_patch` 修改文件。若 SSHFS 写权限在 sandbox 内被拒绝，可以先在获准的本地工作目录用 `apply_patch` 生成文件，再经过用户批准复制到 Z:，或者用 `scp` 上传。

### 5.5 运行环境

进入项目：

```bash
cd /data1/home/zhangyx/project/TC_dynamic
```

推荐解释器：

```bash
/data1/home/zhangyx/miniconda3/envs/cm1_tc/bin/python
```

不要依赖 `python` 或 `/usr/bin/python3`。

### 5.6 关键可复现命令

急流 y-z 初始剖面：

```bash
/data1/home/zhangyx/miniconda3/envs/cm1_tc/bin/python \
  scripts/plot_jet_yz_cross_section.py \
  --jet /data/zhangyx/DATA/cm1out_22N_8o_jet.nc \
  --nojet /data/zhangyx/DATA/cm1out_22N_nojet.nc \
  --time-hours 0 \
  --max-z-km 20 \
  --y-half-width-km 2500 \
  --jet-offset-km 888 \
  --jet-z-km 12 \
  --output output/jet_structure/jet_yz_cross_section_t000h.png
```

全域 SE 强迫—算子交换：

```bash
/data1/home/zhangyx/miniconda3/envs/cm1_tc/bin/python \
  scripts/run_se_forcing_operator_factorial_full.py \
  --nojet /data/zhangyx/DATA/cm1out_22N_nojet.nc \
  --jet /data/zhangyx/DATA/cm1out_22N_8o_jet.nc \
  --hours 20 55 80 110 \
  --max-r-km 300 \
  --max-z-km 20 \
  --dr-km 12 \
  --eps-ratio 1e-5 \
  --output-dir output/se_factorial_full_regularized_eps_1e_5
```

角动量预算：

```bash
/data1/home/zhangyx/miniconda3/envs/cm1_tc/bin/python \
  scripts/analyze_angular_momentum_budget.py \
  --nojet /data/zhangyx/DATA/cm1out_22N_nojet.nc \
  --jet /data/zhangyx/DATA/cm1out_22N_8o_jet.nc \
  --hours 25,45,50,55,60,65,70,72,75,80,110 \
  --difference-hours 2 \
  --output-dir output/angular_momentum_budget_dense_22N_8deg

/data1/home/zhangyx/miniconda3/envs/cm1_tc/bin/python \
  scripts/summarize_angular_momentum_budget.py \
  --input-dir output/angular_momentum_budget_dense_22N_8deg \
  --output-dir output/angular_momentum_budget_dense_22N_8deg/validation
```

长任务后台运行模板：

```bash
mkdir -p output/logs
nohup /data1/home/zhangyx/miniconda3/envs/cm1_tc/bin/python \
  scripts/example.py [arguments] \
  > output/logs/example.log 2>&1 &
```

查看进度：

```bash
tail -f output/logs/example.log
```

### 5.7 GitHub 同步方案

服务器当前 remote `mac` 不可用，服务器也不能直接访问外网。推荐：

1. 在 Windows 上挂载 Z:；
2. 在普通用户 PowerShell 中通过 Z: 检查和提交；
3. 使用 Windows 的 GitHub 凭据推送到 GitHub。

SSHFS 仓库若出现 `dubious ownership`，在用户自己的 PowerShell 中仅对该明确路径添加 safe.directory，不要设置通配全局信任：

```powershell
git config --global --add safe.directory "//sshfs.k/zhangyx@114.212.48.225/project/TC_dynamic"
```

随后：

```powershell
git -C Z:\ status
git -C Z:\ diff --check
```

在提交前必须人工审查当前大量修改和删除；不要把 500 GB NetCDF、`output/` 大产物、私钥或临时缓存加入 Git。

## 6. 新会话建议的接手顺序

1. 连接服务器并读取 `HANDOFF.md`、`AGENTS.md`、`README.md`、`TODO.md`。
2. 用 `git status --short` 记录当前脏工作树；不要清理。
3. 检查 `/data1` 与 `/data`，先修复并完成安全数据迁移，释放 `/data1` 空间。
4. 打开以下三份权威报告，理解当前科学结论：
   - `output/inertial_attribution_22N_8deg/analysis_report.md`
   - `output/mgradient_i2_overlap/mgradient_i2_overlap_report.md`
   - `output/angular_momentum_budget_dense_22N_8deg/validation/angular_momentum_budget_validation_report.md`
5. 从“稳定度—RI 时间联系”开始下一阶段，不要继续只堆单时次空间图。
6. 第一批新图建议包括：
   - JET/noJET 强度和 6/12/24 h 强度倾向；
   - 内出流 `I2<0`/`M_r<0` 比例与强度倾向的联合时间序列；
   - 领先—滞后相关或事件复合；
   - eddy、mean、pressure-gradient 累计角动量贡献与 `M_r/I2` 的阶段对照；
   - 标出预算 PASS/CAUTION/FAIL，避免对失败时次做机制解释。
7. 最后再决定论文主线是：
   - “jet 通过非对称角动量输送重塑基本态并影响增强”；还是
   - “jet 诱发平衡和非平衡两条响应路径”。

## 7. 结论边界

目前可以说：

- jet 明确改变 TC 上层出流、角动量梯度、惯性稳定度和 SE 平衡响应；
- 稳定度差异主要通过绝对涡度/`M_r` 实现；
- 强度差是重要中介但不是全部；
- 早期 eddy 输送和后期平均环流共同作用；
- 高层出流区的 SE 直接强迫效应通常大于直接算子效应，但交互不可忽略；
- 25N强度匹配样本中，惯性稳定度算子单项产生有组织的平衡环流投影；
- CTRL 72–80 h减弱目前不支持完整眼墙置换。

目前还不能说：

- 局地环境 eddy torque 是唯一原因；
- 正则化 SE 解就是真实不稳定区次级环流；
- 低惯性稳定度必然促进或抑制 RI；
- 单组 JET/noJET 足以排除内部变率；
- 3D 水平代理就是严格的三维惯性稳定判据；
- operator-only SE响应已证明惯性稳定度算子主导JET–CTRL强度差；
- 通过挑选单个平滑CTRL或增加扩散，可以无代价地消除72–80 h突降。

后续所有论文表述应保留这些边界。
