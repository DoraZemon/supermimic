# 实验室主机验证记录

日期：2026-11-16。此文件记录实验室主机实际结果，区别于准备电脑的历史验证。

## 本机基线

- 工作目录：`/home/user/zhengshaozheng/supermimic_ws/supermimic`。
- 分支：`experiment/single-motion`，初始提交 `7f4f688fed20cc976458c2869801e2ddae4fc5bb`。
- mjlab_research：`dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd`，同级目录，工作区干净。
- GPU：RTX 5080，16303 MiB；驱动 `580.178.04`。
- Python：3.13.15；uv：0.12.14；项目环境：`.venv`。
- 动作文件：`../data/motions/run1_subject2.npz`，直接下载 Release 附件，未重新转换。
- 动作 SHA256：`3583c360c38ecedbee2487e3a45e51346c6de0639dc40c722aca400bb8945930`。
- 动作检查通过：11891 帧、29 关节、50 Hz，数值字段全部有限。

## 首次干净安装发现的依赖冲突

原锁文件 SHA256 为 `aa0bc916642b70ed85ddf550efe896a25488ac1ae34ab8422767baf773914404`。
`uv sync --locked` 成功，但 MuJoCo 导入和训练入口失败：

```text
OSError: libmujoco.so.3.16.0: cannot open shared object file: No such file or directory
```

依赖树及 wheel 文件记录确认：`mjviser 0.0.17` 要求 `mujoco-core>=4.0.1`，
后者与 `mujoco 3.16.0` 写入同一 `mujoco/` 包目录。
`readelf` 证实本机扩展链接 4.0.1，而遗留 elasticity 插件链接 3.16.0。
这是包文件冲突，不能仅靠补充动态库搜索路径解决。

核查 PyPI 版本元数据后，在主项目 `pyproject.toml` 的 `[tool.uv]` 中加入
`constraint-dependencies = ["mjviser==0.0.16"]`。
该版本满足 mjlab 的 `mjviser>=0.0.16`，依赖 `mujoco>=3.6.0`，不会引入 mujoco-core。
执行：

```bash
cd /home/user/zhengshaozheng/supermimic_ws/supermimic
uv lock
uv sync --locked --reinstall-package mujoco
```

重装同版本 MuJoCo 是为恢复被重叠安装/卸载影响的文件。
依赖版本变化仅为 mjviser 0.0.17 → 0.0.16、移除 mujoco-core 4.0.1；
锁文件还新增约束，当前 uv 将格式 revision 从 5 写为 3。
PyTorch、MuJoCo、Warp、RSL-RL 版本未变，mjlab 源码未改。
修复后锁文件 SHA256：`440b41dc796c086f504c7519b745e275b773ad9008b8eb5e07a159e4ea4ba981`。
后续仍使用 `uv sync --locked`，不需要重复 `uv lock`。

## 已验证及待办

已通过：

- PyTorch 2.15.0+cu132 在 RTX 5080 上的 CUDA 矩阵运算与有限值检查。
- MuJoCo 实际运行版本 3.16.0，mjlab 和 MotionLoader 导入。
- Warp 1.19.0 CUDA 设备初始化。
- 官方训练入口 `SuperMimic-GenCoord-G1 --help`。
- GPU 物理仿真最小检查：同一任务、本地动作、uniform 采样、8 个环境，
  零动作执行 10 个环境步，奖励全部有限，进程正常结束。
  这只验证仿真执行，不包含世界模型或策略学习。

仿真检查出现 `linesearch iterations limit reached`，当前 `ls_iterations=20`。
检查仍正常结束；短训练中需观察告警频率及物理状态，暂保留官方求解器配置。
另有 TorchScript 弃用和 Triton FP fusion 默认值提示。

W&B 登录与在线日志上传已在本机通过，账号为 `zemon`。
连接检查 run：`zemon-xi-an-jiaotong-university-/connection-check/90ywyjiu`，
正常结束并同步 5 个文件，尚不代表 model artifact 已验证。

1001 步短训练已完成：1024 环境、WM batch 512、policy batch 1024、
save interval 500；日志为 `logs/manual/smoke-20261116-1607.log`。
run：`zemon-xi-an-jiaotong-university-/supermimic/re66rgm3`，云端状态 `finished`。
日志包含 WM 和 Policy 更新，未检出 traceback、OOM、CUDA error 或 NaN/Inf 字样。
日志 Mean reward 首末为 0.58 / 15.97，Mean episode length 为 20.53 / 453.15；
Loss/state 首末为 2.4326 / 0.0440。这些不是严格评测分数。
日志训练计时约 6 分 6 秒，不包含所有初始化开销。

本地模型目录：`logs/SuperMimic-GenCoord-G1/2026-11-16_16-08-18/`。
云端 model artifact 有 v0/v1/v2，latest 为 v2；已通过官方下载辅助函数下载，
actor、world_model、critic 三个文件 SHA256 均与本地一致，权重张量均有限。
v2 对应 step=1000 保存点。

短训练共记录 8542 条线搜索上限、599 条求解器上限告警，不能换算为失败环境比例。
保留官方配置，先回放检查实际行为，再评估长训练。
短训练回放已启动，日志 `logs/manual/play-smoke-20261116.log`。
官方 Viser 已监听 8080，本机 `http://127.0.0.1:8080/` 返回 HTTP 200。
模型加载和 viewer 服务启动通过，浏览器场景显示和机器人动作仍待用户观察。

待验证：20001 步体验训练及本机浏览器回放。

用户已通过浏览器截图确认短训练回放场景显示。随后按用户要求通过 SIGINT
关闭回放，进程退出且 8080 端口释放。回放日志出现 `broadphase overflow`，
提示 nconmax/naconmax 超过 35，下一次回放前需要处理；短训练日志未发现该告警。

用户随后明确要求启动 20001 步体验训练，已启动新 run `run1_subject2_trial`：
1024 环境、WM batch 512、policy batch 1024、save interval 1000，从头训练。
启动前 GPU 空闲显存 14266 MiB，无其他 SuperMimic 训练或回放进程。
日志路径记录在 `logs/manual/trial-current-log.txt`，启动时间记录在
`logs/manual/trial-start-time.txt`；结束后退出码写入 `logs/manual/trial-exit-code.txt`。
体验训练现已完成，退出码 0，日志训练计时 44 分 51 秒。
run：`zemon-xi-an-jiaotong-university-/supermimic/0am3tdg0`，云端状态 finished。
本地模型：`logs/SuperMimic-GenCoord-G1/2026-11-16_17-18-37/`。
latest artifact 为 `model-0am3tdg0:v20`，对应 step=20000；三个权重下载后
与本地 SHA256 一致，张量有限值检查通过。

末次日志：Mean reward 42.27、Mean episode length 490.48、Loss/state 0.0084、
error_anchor_pos 0.1542、error_body_pos 0.0342；不是独立评测或论文分数。
完整日志 53095140 bytes，未检出 traceback、OOM、CUDA error、NaN/Inf 字样、
模型上传失败或 broadphase overflow；有 261865 条线搜索上限和 6293 条求解器上限告警。

## 体验模型回放

为解决单环境回放的接触容量不足，在 `src/supermimic/scripts/play.py` 增加可选
`--nconmax` 参数，默认 None 保持原配置；本次回放传入 256（原配置 35）。
该设置只改变回放缓冲区容量，不更改已训练权重或训练设置。
参数检查拒绝非正数。已通过该文件 ruff lint/format 检查、入口 --help 和 git diff --check。

已启动以下回放，日志 `logs/manual/play-trial-20261116.log`；需继续观察是否有容量告警：

```bash
cd /home/user/zhengshaozheng/supermimic_ws/supermimic
CUDA_VISIBLE_DEVICES=0 bash ../lab-run.sh /home/user/.local/bin/uv run --locked --no-sync \
  src/supermimic/scripts/play.py SuperMimic-GenCoord-G1 \
  --wandb-path zemon-xi-an-jiaotong-university-/supermimic/0am3tdg0 \
  --motion-file /home/user/zhengshaozheng/supermimic_ws/data/motions/run1_subject2.npz \
  --nconmax 256 --viewer viser
```

原始检查输出保存在 `logs/manual/`；该目录已被 Git 忽略。
所有后续命令使用上述真实工作目录与 `uv run --locked`，不依赖 Conda。

## 用户限定的写入范围

用户要求后续工作仅在 `/home/user/zhengshaozheng/` 内写入。
工作区根目录创建了 `../lab-run.sh`，为子进程设置项目内 HOME、XDG、uv、W&B、
Warp、Torch、Triton、CUDA、Python 字节码缓存和临时目录。
运行时数据位于 `../.runtime/`，凭据位于 `../.runtime/home/.netrc`，不提交凭据。
命令使用既有环境 `uv run --locked --no-sync`，避免重复同步和下载。
示例：

```bash
cd /home/user/zhengshaozheng/supermimic_ws/supermimic
bash ../lab-run.sh /home/user/.local/bin/uv run --locked --no-sync <入口与参数>
```

该脚本是进程环境配置，不是系统级文件系统沙箱。
约束提出之前已存在用户级 uv Python/包缓存、Warp 缓存等外部写入，不自行清理。
OpenCode 宿主自身日志位置由宿主管理，已向用户说明。
