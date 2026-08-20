# JET–CTRL 环境涡动 Sawyer–Eliassen 诊断

该模式用于回答：在相同的 CTRL 平衡台风基本态上，JET 试验相对于 CTRL
新增的环境涡动角动量输送会驱动怎样的平衡次级环流？

## 1. 环境涡动动量强迫

程序首先以各自的台风中心进行柱坐标转换。默认使用与 Bui et al. (2009)
prime 定义一致的 Reynolds 方位平均。直接从三维扰动通量诊断

\[
F_{\lambda,\mathrm{eddy}}
=-\frac{1}{\bar\rho r^2}\frac{\partial}{\partial r}
 \left(r^2\overline{\rho u_r''v_t''}\right)
-\frac{1}{\bar\rho}\frac{\partial}{\partial z}
 \left(\overline{\rho w''v_t''}\right).
\]

环境强迫定义为

\[
F_{\lambda,\mathrm{env}}
=F_{\lambda,\mathrm{eddy}}^{\mathrm{JET}}
-F_{\lambda,\mathrm{eddy}}^{\mathrm{CTRL}}.
\]

这一定义包括 imposed jet、jet–TC 相互作用以及 jet 引起的非轴对称 TC
响应，因此应称为 **jet-induced environmental eddy forcing**，而不是纯粹的
背景急流通量。

## 2. 固定 CTRL 算子

使用 CTRL 的切向风、平衡位温和密度构建 Bui et al. (2009) general SE
operator，并只在右端加入环境动量强迫：

\[
\mathcal L_{\mathrm{CTRL}}(\psi_{\mathrm{env}})
=-\frac{\partial}{\partial z}
 \left(\chi_{\mathrm{CTRL}}\xi_{\mathrm{CTRL}}
 F_{\lambda,\mathrm{env}}\right).
\]

因此求得的 \(U_{\mathrm{env}}\) 和 \(W_{\mathrm{env}}\) 不混入 JET 对静力稳定度、
惯性稳定度和基本涡旋结构的改变。

该模式使用完整梯度风项

\[
C_g=\frac{v_t^2}{r}+fv_t
\]

以及热力强迫

\[
g\frac{\partial(\chi^2Q)}{\partial r}
+\frac{\partial(C_g\chi^2Q)}{\partial z}.
\]

默认不使用旧 NCL 路径中的 15 km 指数海绵，斜压系数缩放默认为物理原式
的 1.0。`single`、`evap` 和 `timeavg` 现在共用同一套修正后的 Bui 强迫组装；
时间平均模式先逐时次计算非线性涡动通量，再对诊断结果做时间平均。

完整基线强迫按下式组织：

\[
Q=Q_{\rm eddy}+Q_{\rm diffusion}+Q_{\rm diabatic}+Q_{\rm other},
\qquad
F_\lambda=F_{\lambda,\rm eddy}+F_{\lambda,\rm diffusion}+F_{\lambda,\rm other}.
\]

CM1 的 `hadv`/`vadv` 只用于独立闭合检验，不再加入总强迫，因而不会与
直接从三维扰动通量求得的 eddy convergence 重复计数。

## 3. 运行命令

```bash
python scripts/run_se_pipeline.py \
  --mode env \
  --input-file dataset/cm1out_CTRL.nc \
  --jet-input-file dataset/cm1out_JET.nc \
  --target-time-hours 72 \
  --eddy-average reynolds \
  --bui-baroclinic-scale 1.0 \
  --dr-km 12 \
  --output-dir output/se_pipeline/env_72h
```

`--input-file` 始终表示 CTRL，`--jet-input-file` 表示 JET。两个文件必须具有
相同的水平/垂直网格和可比较的输出时间。

可使用 `--eddy-average favre` 做质量加权敏感性试验，但只有在基本态也采用
一致的 Favre 平均时才能视为严格自洽的可压缩扩展；主结果建议保留默认的
`reynolds`。

## 4. 输出

- `se_environmental_eddy_products.npz`：完整算子、强迫和解场；
- `se_environmental_eddy_products.nc`：便于 NCL/xarray 读取的主要结果；
- `se_environmental_eddy_response.png`：强迫与次级环流机制图；
- `environmental_eddy_forcing.png`：单独绘制总、径向和垂直环境 eddy 强迫；
- `environmental_eddy_summary.json`：中心位置、平衡残差和正则化统计。

核心变量包括：

- `F_lambda_env`、`F_lambda_env_radial`、`F_lambda_env_vertical`；
- `forcing_env`；
- `psi_env`、`U_env`、`W_env`；
- `psi_ctrl`、`U_ctrl`、`W_ctrl`；
- `psi_ctrl_plus_env`、`U_ctrl_plus_env`、`W_ctrl_plus_env`。

由于 SE 方程对给定算子是线性的，理论上

\[
\psi_{\mathrm{CTRL+env}}-
\psi_{\mathrm{CTRL}}=\psi_{\mathrm{env}}.
\]
