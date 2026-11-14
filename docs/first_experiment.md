# 单动作训练：版本基线

更新日期：2026-11-15。详细操作见仓库根目录 [复刻指南.md](../复刻指南.md)。

## 仓库与版本

| 项目 | 仓库 | 固定版本 |
|---|---|---|
| SuperMimic 官方源码基线 | https://github.com/HybridRobotics/supermimic | `abc124c1bd5976341034668af3d12c087f2f1f55` |
| mjlab_research 依赖 | https://github.com/HybridRobotics/mjlab_research | `dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd` |

个人 fork：https://github.com/DoraZemon/supermimic

实验分支：`experiment/single-motion`。

SuperMimic 的版本号是修改前的官方基线。个人实验改动提交在上述分支；每次实际训练还应记录该分支当时的 `git rev-parse HEAD`，便于精确复现。

## 实验室主机准备

两个仓库必须处于同级目录，因为 `pyproject.toml` 使用本地依赖路径 `../mjlab_research`。

```bash
mkdir -p ~/supermimic_ws
cd ~/supermimic_ws

git clone --branch experiment/single-motion https://github.com/DoraZemon/supermimic.git
git clone https://github.com/HybridRobotics/mjlab_research.git

git -C mjlab_research checkout --detach dd51a9fcc98791a2f0b018b3c7f71a3eefd5ecdd

cd supermimic
git remote add upstream https://github.com/HybridRobotics/supermimic.git
uv sync --locked
```

使用本实验分支更新后的 `uv.lock`，首次实验不主动升级依赖。官方基线中的旧锁文件对应 mjlab 1.5.0，与上述依赖提交不匹配；本机已执行 `uv lock`，解析到 mjlab 1.6.0、PyTorch 2.15.0、MuJoCo / MuJoCo Warp 3.16.0、Warp 1.19.0、RSL-RL 5.5.2。锁文件中的 mjlab 是本地 editable 路径，不会自动固定其 Git 提交，因此上面的依赖检出步骤不可省略。

训练前记录实际代码版本及工作区状态：

```bash
git rev-parse HEAD
git status --short
git -C ../mjlab_research rev-parse HEAD
git -C ../mjlab_research status --short
```

本机已完成依赖安装、基础模块与动作转换模块导入、训练入口 `--help` 检查，以及 run1_subject2.csv 的 CPU 转换与输出结构检查。尚未验证 GPU 仿真、训练、模型 artifact 上传及回放，不应将此环境标为完整训练验证通过。
