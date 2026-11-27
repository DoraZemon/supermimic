# SuperMimic

[Project page](https://supermimic.github.io) | [Paper](https://arxiv.org/abs/2610.28480)

SuperMimic trains a humanoid whole-body motion tracker with one differentiable world model, built on [mjlab](https://github.com/mujocolab/mjlab).

本实验分支的中文说明：[使用指南](使用指南.md)（工程与算法、主机运行、训练回放、评估和自定义任务迁移）。

## Setup

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone both repos as siblings:

```bash
mkdir supermimic_ws && cd supermimic_ws
git clone git@github.com:HybridRobotics/supermimic.git
git clone git@github.com:HybridRobotics/mjlab_research.git
cd supermimic
```

## Usage

Available task IDs:

- `SuperMimic-GenCoord-G1` — generalized coordinates representation
- `SuperMimic-MaxCoord-G1` — maximal coordinates representation
- `SuperMimic-Basket-V1` — Unitree G1 + Dex3-1 stationary single-hand `1.10 kg` fruit basket grasp-and-lift (closed-loop RSL-RL PPO, single & multi-GPU)

### Motion Tracking (World Model)

Train a world model and policy:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> uv run src/supermimic/scripts/train.py \
  <TASK_ID> --runner.run_name <description>
```

Play back a trained model:

```bash
uv run src/supermimic/scripts/play.py \
  <TASK_ID> --wandb-path <RUN_PATH>
```

### Fruit Basket V1 (`SuperMimic-Basket-V1`)

Single-GPU smoke test:

```bash
uv run python -m supermimic.scripts.train_basket \
  --gpu-ids 0 --env.scene.num-envs 256 --agent.max-iterations 20 --agent.logger tensorboard
```

Multi-GPU DDP training (e.g., 4 or 8× RTX 3090):

```bash
uv run python -m supermimic.scripts.train_basket \
  --gpu-ids all --env.scene.num-envs 2048 --agent.max-iterations 10000 --agent.logger tensorboard
```

Play back a local checkpoint:

```bash
uv run python -m supermimic.scripts.play_basket \
  --checkpoint-file logs/rsl_rl/g1_basket_v1/<timestamp>/model_xxxx.pt
```

### Remote GPU Server Deployment (Quick Checklist)

Both `supermimic/` and `mjlab_research/` must be placed under the same parent directory on the server (because `pyproject.toml` references `../mjlab_research`). Detailed beginner-friendly step-by-step instructions (SSH, `rsync`, `uv sync`, `tmux`, TensorBoard port forwarding, and downloading checkpoints) are in [使用指南.md — 第 9 节](使用指南.md#9-g1--dex3-1-果篮抓取任务supermimic-basket-v1与多卡-gpu-服务器完整部署指南):

1. **Check GPU & driver on server**: `ssh -p <PORT> <USER>@<HOST>` then `nvidia-smi` (recommended NVIDIA Driver `>= 535`, CUDA `>= 12.1`).
2. **Upload both repos from local machine** (excluding `.venv`/`logs`):
   ```bash
   ssh -p <PORT> <USER>@<HOST> "mkdir -p ~/project/supermimic"
   rsync -avz --progress -e "ssh -p <PORT>" --exclude '.venv' --exclude '__pycache__' --exclude 'logs' \
     ./supermimic/ <USER>@<HOST>:~/project/supermimic/supermimic/
   rsync -avz --progress -e "ssh -p <PORT>" --exclude '.venv' --exclude '__pycache__' \
     ./mjlab_research/ <USER>@<HOST>:~/project/supermimic/mjlab_research/
   ```
3. **Install `uv` & dependencies on server**:
   ```bash
   which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
   source "$HOME/.local/bin/env"
   cd ~/project/supermimic/supermimic && uv sync
   ```
4. **Run 1-minute single-GPU smoke test** before launching multi-GPU training inside `tmux`.
5. **Download checkpoints back to local machine** via `rsync` and visualize with `play_basket`.

## Acknowledgements

SuperMimic contains code inspired by these projects:

- [ControlVAE](https://github.com/heyuanYao-pku/Control-VAE) (BSD-3-Clause)
- [robotic_world_model](https://github.com/leggedrobotics/robotic_world_model) (Apache-2.0)
- [DreamerV3](https://github.com/danijar/dreamerv3) (MIT)

## License

Copyright 2026 The SuperMimic Authors.

Released under the [Apache License 2.0](LICENSE).
