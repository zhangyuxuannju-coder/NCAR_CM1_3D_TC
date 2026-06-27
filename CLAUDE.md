# CLAUDE.md — TC_dynamic 项目规范

## 用户身份

- 大气科学科研从业者，研究方向为热带气旋（TC）动力学
- 使用 NCAR CM1 (Cloud Model 1) 进行三维理想化 TC 模拟
- 编程语言：Fortran（模式代码）、Python（数据处理与可视化）、NCL（部分诊断）
- 熟悉气象学概念：SE (surface entropy)、角动量、涡旋动力学、边界层过程

## 项目环境

- **服务器**：Rocky Linux 8.10，`/data1/home/zhangyx/project/TC_dynamic`
- **硬件**：AMD EPYC 9554 (128核256线程)，566 GB 内存，11TB NVMe，**2× RTX 5090 (32GB 显存/卡，共 64GB)**
- **Python**：Python 3.12.13，**conda 环境 `cm1_tc`**（激活：`conda activate cm1_tc`）
- **GPU 库**：PyTorch 2.12.1+cu130（通过 NJU 镜像安装），CUDA 驱动 13.2
- **CM1 编译**：gfortran + OpenMP
- **模型配置**：`code/namelist.input`，探空文件 `code/input_sounding`
- **数据目录**：`dataset/`（不进入 Git）；输入数据：`/data1/home/zhangyx/data/cm1out_Thompson.nc`
- **输出目录**：`output/`（不进入 Git）

### JupyterLab 远程开发环境

**Mac 浏览器访问**：`http://localhost:8888`，密码在服务器上配置（可通过 `jupyter server password` 修改）。

> ⚠️ 校园网防火墙会封锁服务器 8888 端口，必须通过 SSH 隧道访问，不能直连公网 IP。

**Mac 端 SSH 配置**（`~/.ssh/config`）：

```
Host nju-server
    HostName 114.212.48.225
    User zhangyx
    LocalForward 8888 localhost:8888
```

在 Mac 终端执行 `ssh nju-server` 登录后，端口转发自动生效，浏览器访问 `http://localhost:8888` 即可。SSH 连接断开后 JupyterLab 仍可在下次 SSH 登录时继续使用（服务端持续运行）。

JupyterLab 通过 systemd 用户服务实现开机自启，内核为 `Python 3 (cm1_tc + PyTorch CUDA)`，内置 xarray/netCDF4/PyTorch 全套科研工具链。

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

**关键配置文件**：
- 服务：`~/.config/systemd/user/jupyterlab.service`
- Jupyter 配置：`~/.jupyter/jupyter_lab_config.py`
- 内核定义：`~/.local/share/jupyter/kernels/cm1_tc/kernel.json`
- conda 激活钩子（修复 GLIBCXX 版本冲突）：`~/miniconda3/envs/cm1_tc/etc/conda/activate.d/env_vars.sh`

**GLIBCXX 问题说明**：系统 libstdc++ 仅到 `GLIBCXX_3.4.25`（Rocky 8 的 GCC 8），而 numpy 2.4 需 `GLIBCXX_3.4.29`。修复方案：conda 环境的 `libstdc++.so.6.0.34` 已包含所需符号，通过 conda 激活钩子和 kernel.json 的 `env.LD_LIBRARY_PATH` 优先加载。

### pip/conda 镜像

```bash
# NJU pip 镜像（速度快，避免超时）
pip install <package> -i https://mirror.nju.edu.cn/pypi/web/simple

# conda 镜像已在 ~/.condarc 配置 NJU 源
conda install <package>
```

## 代码规范

### Fortran (CM1 模式代码)
- 自由格式 Fortran 90+，缩进 2 空格
- 主初始化文件：`code/init3d.F`
- 求解器：`code/SE-solver.f90`
- 修改物理参数后需重新编译 CM1

### Python (数据处理)
- 遵循 PEP 8，使用 4 空格缩进
- 核心模块在 `src/` 目录：
  - `config.py` — 配置文件
  - `io.py` — 数据读写
  - `azimuthal_avg.py` — 方位角平均
  - `center_finder.py` — 台风中心定位
  - `coordinates.py` — 坐标转换
  - `plotting.py` — 绘图工具
  - `se_equation.py` — SE 方程诊断
- 分析脚本在 `experiments/`，流水线脚本在 `scripts/`
- Jupyter Notebook 在 `notebooks/`

### 命名约定
- 变量使用 snake_case
- 类使用 CamelCase
- 函数名应描述其功能：`calc_azimuthal_mean()`, `find_tc_center()`

## 每次任务完成后必须执行（重要！）

完成代码修改或数据处理后，**必须**执行以下 Git 工作流：

```bash
# 1. 查看修改内容
git status
git diff

# 2. 提交（用中文描述修改内容）
git add .
git commit -m "描述具体改了什么"

# 3. 推送到 Mac 中转 → GitHub
git push mac main
```

> Mac 的 post-receive 钩子会自动将代码同步到 https://github.com/zhangyuxuannju-coder/NCAR_CM1_3D_TC

## 文档同步更新

如果修改影响了以下内容，需同步更新对应文档：

- 新增/修改函数 → 更新函数 docstring
- 新增配置参数 → 更新 `config/default.yaml`
- 修改工作流 → 更新 `DEVELOPER_GUIDE.md`
- 修改 namelist 参数 → 在提交信息中说明物理意义

## CM1 模式注意事项

- 数据文件（.nc, .grib, .dat）**绝不**进入 Git，已在 `.gitignore` 排除
- 修改 `namelist.input` 参数时注意：
  - `nx`, `ny`, `nz` — 网格分辨率
  - `dtl` — 大时间步长（秒）
  - `soundtep` — 声波步数
  - `cfl` — CFL 数上限
- 输出文件命名格式：`cm1out_<实验名>_<时间步>.nc`
- 编译命令：进入 `code/` 目录，参考 CM1 官方文档编译

## 常用命令

```bash
# CM1 编译
cd code && ./configure && make

# 运行模拟
./cm1 > run.log 2>&1 &

# Python 分析
python3 experiments/sensitivity_evap.py
python3 scripts/run_se_pipeline.py

# 查看输出
ls output/
ncdump -h output/cm1out_*.nc
```

## 交互方式

- 我会用中文或英文描述任务需求
- 你在和我交流时尽量使用中文，给我提交运行审核的时候用中文告诉我你在干什么
- 你直接修改代码文件，不需要先展示代码再等我确认
- 修改完成后给出简洁总结，列出改了什么
- 如有文件需要创建/删除，直接操作
