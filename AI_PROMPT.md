# SE 诊断管线完整操作手册 — 给 AI 的教学文档

## 0. 先搞清楚这段代码在干什么

这段代码实现了台风 **Sawyer-Eliassen (SE) 方程** 的诊断求解。简单来说：

```
cm1out.nc (3D 模式输出)
    → 找到台风中心
    → 把风场投影到柱坐标 (径向风 ur, 切向风 vt)
    → 按半径做方位角平均 → 得到 (z, r) 二维平均场
    → 从 CM1 的 ptb_mp 变量提取非绝热加热率 Q
    → 从 CM1 的 ub_*/vb_* 预算项诊断动量源 Fnu
    → 用热成风关系反算"平衡位温" θ_bal
    → 构建 SE 方程 6 个系数 (A, B, C, D, E, F)
    → 用 SOR 迭代求解流函数 ψ
    → ψ → (U_se, W_se) 次级环流速度
```

**核心理念**：SE 方程描述的是轴对称涡旋对热力/动量强迫的平衡响应。求解出的 U_se 和 W_se 就是台风次级环流（径向入流/出流 + 垂直上升/下沉）。

## 1. 代码结构

```
refactor/
├── scripts/
│   └── run_se_pipeline.py        ← 统一入口，你只运行这个
├── src/
│   ├── config.py                  ← PipelineConfig 参数配置
│   ├── se_equation.py             ← 核心算法：SE系数、SOR求解、加热场构造
│   ├── _se_pipeline_single.py     ← single 模式的完整管线
│   ├── _se_pipeline_evap.py       ← evap 模式（支持自定义加热场）
│   ├── _se_pipeline_timeavg.py    ← timeavg 模式（多时次平均）
│   ├── azimuthal_avg.py           ← 方位角平均工具
│   ├── coordinates.py             ← 坐标变换、C网格去交错
│   ├── io.py                      ← NetCDF 读写
│   └── plotting.py                ← 绘图
├── config/
│   └── default.yaml               ← 默认参数
└── output/                        ← 所有输出放这里
```

**你只需要运行 `scripts/run_se_pipeline.py`**，所有复杂逻辑都在 `src/` 下自动调用。

## 2. 运行模式

### mode=single — 单时次诊断（最常用）

用于诊断某个特定时刻的次级环流。

```bash
python scripts/run_se_pipeline.py \
    --mode single \
    --target-time-hours 72 \       # 选第72小时
    --output-dir output/my_run \
    --dr-km 12 \                   # 径向分辨率 12km (必须 >= 原始 dx)
    --sor-omega 1.5 \              # SOR 松弛因子
    --sor-tol 1.5e-9               # 收敛容差
```

### mode=timeavg — 多时次平均

对一段时间内多个时次分别诊断后求平均。

```bash
python scripts/run_se_pipeline.py \
    --mode timeavg \
    --time-avg-start-hours 64 \
    --time-avg-end-hours 72 \
    --output-dir output/avg_64_72h \
    --dr-km 12 --sor-omega 1.5 --sor-tol 1.5e-9
```

## 3. 关键参数解释

| 参数 | 含义 | 如何选择 |
|------|------|----------|
| `--target-time-hours` | 选哪个模拟时刻 | CM1 输出间隔是什么就用什么 |
| `--dr-km` | 径向分箱宽度 (km) | **必须 >= 原始网格距**，原始 dx~12km 就用 12 |
| `--sor-omega` | SOR 松弛因子 | 1.5 对 12km 稳定；如果 NaN，降到 1.0 |
| `--sor-tol` | SOR 收敛容差 | 1.5e-9 严格；如果收敛太慢可放宽到 1e-8 |
| `--baroclinic-scale` | 斜压项缩放 (0~1) | 默认 0.4，越小=越椭圆=越稳定 |
| `--max-r-km` | 最大诊断半径 | 默认 300 |
| `--max-z-km` | 最大诊断高度 | 默认 20 |
| `--f` | 科氏参数 (1/s) | 默认 5e-5，如果你的台风在不同纬度需要改 |

## 4. 如何验证结果正确

### 4.1 终端输出基准值

CTRL (无屏蔽, 72h, dr=12km) 必须是：

```
SOR converged at iter=2266, max_res=1.500e-09
center_x_km: -10.500000953674316
center_y_km: -7.500000476837158
regularization: bad_D_points_before=148.0, bad_D_points_after=33.0
momentum_budget_pairs: 9 pairs
```

**如果 SOR 迭代数不是 2266，参数或数据有问题。**

### 4.2 数值验证

```python
import numpy as np
d = np.load('output/verify_ctrl/se_pipeline_products.npz')
r, z = d['r_km'], d['z_km']
U = d['U_se'][:, 1:-1].T    # 去掉 ghost cells

iz = np.argmin(np.abs(z - 11))
ir = np.argmin(np.abs(r - 150))
print(f"U_se@z~11km r=150km: {U[iz, ir]:.4f}")  # 应为 -1.5390

iz = np.argmin(np.abs(z - 15))
print(f"U_se@z~15km r=150km: {U[iz, ir]:.4f}")  # 应为 +2.2376
```

## 5. 如何对比多组实验

每组实验输出一个 `se_pipeline_products.npz`：

```python
import numpy as np
import matplotlib.pyplot as plt

ctrl = np.load('output/exp_ctrl/se_pipeline_products.npz')
expt = np.load('output/exp_treatment/se_pipeline_products.npz')
r, z = ctrl['r_km'], ctrl['z_km']

# 差值图
diff = expt['U_se'] - ctrl['U_se']

# 入流+出流层廓线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for ax, zt in [(ax1, 11), (ax2, 15)]:
    iz = np.argmin(np.abs(z - zt))
    ax.plot(r, ctrl['U_se'][:,1:-1].T[iz,:], 'k-', label='CTRL')
    ax.plot(r, expt['U_se'][:,1:-1].T[iz,:], 'r--', label='EXP')
    ax.set_title(f'z~{z[iz]:.1f} km'); ax.legend()
plt.show()
```

## 6. 输出文件

```
output/my_run/
├── se_solution_fields.png      # ψ, U_se, W_se, Vt 四图
├── se_forcing_terms.png        # 强迫项图
├── se_pipeline_products.npz    # 所有诊断场 (numpy)
├── se_pipeline_products.nc     # NetCDF (可选)
└── summary.json                # 运行参数
```

## 7. 关键物理限制

在高空 (z>10 km)，χ² = (1/θ)² ≈ 8×10⁻⁶ 将热力强迫压缩约 10⁵ 倍。**这意味着**：高空的 Q 变化对 U_se 的影响被极大衰减。如果要做高空热力敏感性实验，需要考虑这个衰减效应。

## 9. 实操 checklist

- [ ] `conda env create -f environment.yml && conda activate cm1-typhoon-diagnostics`
- [ ] `ln -s ../dataset dataset`
- [ ] CTRL 基准：`--mode single --target-time-hours 72 --dr-km 12 --sor-omega 1.5 --sor-tol 1.5e-9`
- [ ] 验证：SOR iter=2266, max_res=1.500e-09
- [ ] 修改参数跑你的实验
- [ ] 用第 6 节方法对比 CTRL vs EXP
