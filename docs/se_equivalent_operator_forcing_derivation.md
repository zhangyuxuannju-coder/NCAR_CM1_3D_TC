# JET–CTRL Sawyer–Eliassen 等效算子强迫推导

本文档记录当前 CM1 JET–CTRL 诊断中三个等效算子强迫的定义、推导、数值实现和解释边界，供后续论文方法与附录写作使用。

## 1. General SE 方程与符号

令

$$
\chi=\theta^{-1},\qquad C_g=\frac{\bar v^2}{r}+f\bar v,
$$

$$
\xi=f+\frac{2\bar v}{r},\qquad
\zeta_a=f+\frac{1}{r}\frac{\partial(r\bar v)}{\partial r}.
$$

Bui general Sawyer–Eliassen（SE）方程的左端写为

$$
\begin{aligned}
\mathcal L(\psi)=&\frac{\partial}{\partial r}\left[-g\chi_z\frac{1}{\rho r}\psi_r-\frac{\partial(\chi C_g)}{\partial z}\frac{1}{\rho r}\psi_z\right]\\
&+\frac{\partial}{\partial z}\left[I^2\frac{1}{\rho r}\psi_z-\frac{\partial(\chi C_g)}{\partial z}\frac{1}{\rho r}\psi_r\right],
\end{aligned}
$$

其中

$$
I^2=\chi\xi\zeta_a+C_g\chi_r.
$$

流函数约定为

$$
u=-\frac{1}{\rho r}\psi_z,\qquad w=\frac{1}{\rho r}\psi_r.
$$

定义三个主系数

$$
K_1=-g\chi_z,\qquad
K_2=-\frac{\partial(\chi C_g)}{\partial z},\qquad
K_3=I^2.
$$

于是左端算子可写成便于扰动分析的通量形式：

$$
\boxed{\mathcal L(\psi)=\partial_r(K_1w-K_2u)+\partial_z(-K_3u+K_2w)}.
$$

## 2. JET–CTRL 算子的一阶线性化

将 JET 基本态和响应写成

$$
\mathcal L_J=\mathcal L_C+\delta\mathcal L,\qquad
\psi_J=\psi_C+\delta\psi,
$$

并定义

$$
\Delta K_i=K_{i,J}-K_{i,C}.
$$

JET 与 CTRL 分别满足

$$
\mathcal L_J(\psi_J)=F_J,\qquad \mathcal L_C(\psi_C)=F_C,
$$

其中 $F$ 表示传统 SE 右端的热力和切向动量强迫。展开 JET 方程：

$$
\mathcal L_C\psi_C+\mathcal L_C\delta\psi+\delta\mathcal L\psi_C+\delta\mathcal L\delta\psi=F_J.
$$

忽略二阶项 $\delta\mathcal L\delta\psi$，并利用 CTRL 方程，得到

$$
\boxed{\mathcal L_C\delta\psi=\Delta F-\delta\mathcal L\psi_C}.
$$

因此，将左端算子改变搬到右端后定义的等效算子强迫为

$$
\boxed{S_{\rm op}=-\delta\mathcal L\psi_C}.
$$

负号来自移项，而不是人为选择。

## 3. 三个等效算子强迫

以 CTRL 次级环流 $u_C,w_C$ 为参考，算子扰动为

$$
\begin{aligned}
\delta\mathcal L\psi_C
=&\partial_r(\Delta K_1w_C-\Delta K_2u_C)\\
&+\partial_z(-\Delta K_3u_C+\Delta K_2w_C).
\end{aligned}
$$

故

$$
\begin{aligned}
S_{\rm op}
=&-\partial_r(\Delta K_1w_C)+\partial_r(\Delta K_2u_C)\\
&+\partial_z(\Delta K_3u_C)-\partial_z(\Delta K_2w_C).
\end{aligned}
$$

### 3.1 静力稳定度算子强迫

令

$$
\Delta A\equiv\Delta K_1=\Delta(-g\chi_z),
$$

则

$$
\boxed{S_A=-\partial_r(\Delta A\,w_C)}.
$$

展开为

$$
S_A=-w_C\partial_r\Delta A-\Delta A\,\partial_rw_C.
$$

该项要求静力稳定度差异与 CTRL 垂直运动共同存在。若 $w_C=0$，即使 $\Delta A$ 很大，这个一阶等效强迫仍为零。

### 3.2 惯性稳定度算子强迫

由于 $K_3=I^2$，定义

$$
\Delta I^2=I_J^2-I_C^2.
$$

则

$$
\boxed{S_I=\partial_z(\Delta I^2\,u_C)}.
$$

展开为

$$
\boxed{S_I=u_C\partial_z\Delta I^2+\Delta I^2\partial_zu_C}.
$$

因此，$S_I$ 的最大值不必与急流轴或 $\Delta I^2$ 最大值重合；其高度和符号还受到 CTRL 径向风及其垂直切变控制。这也是急流上移后，强迫或 SE 响应不一定刚性上移相同距离的原因。

惯性稳定度单项试验求解

$$
\boxed{\mathcal L_{C,\mathrm{reg}}(\psi_I)=S_I},
$$

并由

$$
u_I^{SE}=-\frac{1}{\rho_Cr}\partial_z\psi_I,\qquad
w_I^{SE}=\frac{1}{\rho_Cr}\partial_r\psi_I
$$

得到对应的正则化平衡响应。

### 3.3 斜压交叉算子强迫

若直接使用 $\Delta K_2=K_{2,J}-K_{2,C}$，则

$$
\boxed{S_{K_2}=\partial_r(\Delta K_2u_C)-\partial_z(\Delta K_2w_C)}.
$$

当前代码为保持既有绘图符号，定义

$$
\Delta B\equiv-\Delta K_2=\Delta\left[\partial_z(\chi C_g)\right].
$$

因此代码中的等价表达是

$$
\boxed{S_B=-\partial_r(\Delta B\,u_C)+\partial_z(\Delta B\,w_C)}.
$$

展开为

$$
S_B=-u_C\partial_r\Delta B-\Delta B\partial_ru_C+w_C\partial_z\Delta B+\Delta B\partial_zw_C.
$$

## 4. 总的一阶差分方程

三个算子贡献之和为

$$
\boxed{S_{\rm op}=S_A+S_I+S_B}.
$$

若同时考虑传统热力和动量强迫的 JET–CTRL 差异，则完整的一阶诊断写为

$$
\boxed{\mathcal L_{C,\mathrm{reg}}\delta\psi=\Delta F_{\rm thermal}+\Delta F_{\rm momentum}+S_A+S_I+S_B}.
$$

该表达把两类效应明确分开：

1. **右端强迫效应**：JET 改变加热和动量输送；
2. **左端算子效应**：JET 改变基本态稳定度，使同一强迫对应不同的平衡响应。

## 5. 与当前代码的对应关系

主要实现位于：

- `src/se_bui.py`：构造 $K_1,K_2,K_3$，执行椭圆性正则化并组装 CTRL 算子；
- `scripts/solve_matched_operator_forcing_outer100.py`：计算三个等效强迫并分别反演；
- `scripts/plot_inertial_operator_evolution_matched.py`：多时次计算和绘制 $S_I$ 及其 SE 响应；
- `config/inertial_matches_J70_90_C80_140.json`：当前 JET–CTRL 强度匹配时次。

代码中的核心表达为

```python
static = -grad(d_a * w0, r_m, axis=1)
inertial = grad(d_i2 * u0, z_m, axis=0)
baroclinic = (
    -grad(d_b * u0, r_m, axis=1)
    + grad(d_b * w0, z_m, axis=0)
)
```

对应

$$
S_A=-\partial_r(\Delta A w_C),\qquad
S_I=\partial_z(\Delta I^2u_C),
$$

$$
S_B=-\partial_r(\Delta B u_C)+\partial_z(\Delta B w_C).
$$

注意：代码变量 `d_a` 对应的是 $\Delta K_1$，并非将通量形式展开后矩阵中的 $A=K_1/(\rho r)$。

## 6. 近似条件与论文表述边界

这套诊断是一阶算子扰动方法，需要明确以下限制：

1. 忽略二阶项 $\delta\mathcal L\delta\psi$；当算子变化或环流响应很大时，应检查线性近似误差。
2. 使用 CTRL 的 $u_C,w_C$ 作为参考响应，回答“JET 改变的算子如何作用于 CTRL 环流”。
3. 当前三个分量只考虑 $K_1,K_2,K_3$ 的改变，没有把 $1/(\rho r)$ 的 JET–CTRL 差异单独定义为第四个算子项。
4. JET 与 CTRL 应尽量按强度、RMW 和结构匹配；否则内核区 $\Delta I^2$ 可能主要反映气旋强度差，而非环境急流作用。
5. 对 $r<100\ \mathrm{km}$ 的屏蔽是 source-region attribution，不是 SE 求解域裁剪；求解仍在完整半径域进行。
6. 椭圆性正则化后的解只能解释为 **regularized balanced projection**，不能代表惯性或对称不稳定区中的完整非平衡动力过程。
7. 算子单项响应能够提供动力一致性证据，但不能单独证明它造成了 JET–CTRL 强度差。因果归因还需要固定强迫/固定算子试验、时间领先关系和专门的 CM1 敏感性试验。

建议论文中使用“equivalent operator forcing”或“first-order operator-perturbation forcing”，避免将其直接称为独立的外部动量或热力强迫。
