# Bui general SE 适用性诊断

该程序在求解 SE 方程之前，检查 CTRL/JET 的 CM1 实际方位平均基本态是否满足经典椭圆型
边值问题的条件。主诊断场始终是未正则化的

\[
K_1=-g\chi_z,
\qquad I^2=\chi\xi(\zeta+f)+C_g\chi_r,
\]

\[
D=K_1I^2-\left[(\chi C_g)_z\right]^2.
\]

程序不会先用热成风反演结果替代主诊断场，也不会用 regularization 后的量
覆盖 `I2_raw` 或 `D_raw`。三套场严格分开：

- `*_raw`：CM1 实际方位平均 `theta/v`，没有平衡投影或正则化；
- `*_balanced_projection`：使用热成风反演位温，但尚未正则化；
- `*_regularized`：为了求解椭圆问题而调整后的比较场。

## 服务器运行

```bash
python scripts/run_se_pipeline.py \
  --mode stability \
  --input-file /path/to/CTRL/cm1out.nc \
  --jet-input-file /path/to/JET/cm1out.nc \
  --target-time-hours 72 \
  --eddy-average reynolds \
  --dr-km 12 \
  --max-r-km 1200 \
  --max-z-km 20 \
  --stability-outflow-threshold 2 \
  --stability-jet-speed-threshold 20 \
  --stability-jet-axis-r-km 888 \
  --stability-jet-axis-z-km 12 \
  --output-dir output/se_applicability/72h
```

其中绿色等值线表示 JET 方位平均径向出流；橙色等值线表示
`sqrt(2*eddy_kinetic_energy)`，它是相对于 TC 中心的急流/非对称风速位置代理。
黑色实线和虚线是环境 eddy 强迫的正、负强值等值线。可选的金色星号是
namelist 中给定的 imposed jet 轴；上例对应当前 `jet_y_dist=888 km`、
`jet_z_ctr=12 km`。如果只关注 TC 附近的相互作用区，可以减小 `max-r-km`，
但急流轴可能落在图外。

## 稳定性分类

- `0`：`K1>0, I2>0, D>0`，经典 SE 椭圆区；
- `1`：`K1>0, I2<=0`，惯性不稳定区；
- `2`：`K1>0, I2>0, D<=0`，对称/强切变非椭圆区；
- `3`：`K1<=0`，静力不稳定区。

## 输出

- `se_applicability_I2_D.png`：CTRL/JET 的原始 `I2`、`D` 及其差值；
- `se_applicability_classes.png`：物理稳定性分类以及 `F_lambda_env` 与 `D=0` 的重合；
- `se_applicability_products.nc/.npz`：所有原始场、正则化比较场和掩膜；
- `se_applicability_summary.json`：面积比例、强迫重合比例和最大强迫位置。

重点统计量 `nonelliptic_abs_forcing_fraction` 表示按柱坐标体积权重计算时，
`abs(F_lambda_env)` 有多少比例位于非椭圆区。实际模式态和热成风平衡投影
分别输出 `F_lambda_env_ctrl_raw_elliptic` 与
`F_lambda_env_ctrl_balanced_projection_elliptic`；JET 的 `D_raw<=0` 区则应解释为
真实模拟可能包含非平衡调整，而不是 regularization 后 SE 已重新适用。

NetCDF/NPZ 还保存 `I2_vorticity_component_raw`、
`I2_baroclinic_component_raw`、`D_static_inertial_product_raw` 和
`D_shear_penalty_raw`，可继续判断负值主要来自惯性项还是强垂直切变项。
