"""Train G1 + Dex3-1 fruit basket V1 policy with RSL-RL PPO (single or multi-GPU)."""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import mjlab
import mjlab.tasks  # noqa: F401
import tyro
from mjlab.scripts.train import TrainConfig, run_train
from mjlab.utils.gpu import select_gpus

import supermimic.tasks  # noqa: F401

TASK_ID = "SuperMimic-Basket-V1"


def _run_worker(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  """Worker entrypoint ensuring SuperMimic tasks are registered in every DDP rank."""
  import importlib

  importlib.import_module("mjlab.tasks")
  importlib.import_module("supermimic.tasks")
  run_train(task_id, cfg, log_dir)


def launch_basket_training(task_id: str, args: TrainConfig) -> None:
  """Launch single-GPU or multi-GPU (torchrunx DDP) training for Basket V1."""
  log_root_path = (Path(args.log_root) / args.agent.experiment_name).resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  selected_gpus, num_gpus = select_gpus(args.gpu_ids)
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))

  if num_gpus <= 1:
    _run_worker(task_id, args, log_dir)
    return

  import torchrunx

  logging.basicConfig(level=logging.INFO)
  if "TORCHRUNX_LOG_DIR" not in os.environ:
    if args.torchrunx_log_dir is not None:
      os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
    else:
      os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

  print(f"[INFO] Launching {task_id} training with {num_gpus} GPUs", flush=True)
  torchrunx.Launcher(
    hostnames=["localhost"],
    workers_per_host=num_gpus,
    backend=None,
    copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
  ).run(_run_worker, task_id, args, log_dir)


def main() -> None:
  args = tyro.cli(
    TrainConfig,
    default=TrainConfig.from_task(TASK_ID),
    prog=f"{sys.argv[0]} ({TASK_ID})",
    config=mjlab.TYRO_FLAGS,
  )
  launch_basket_training(task_id=TASK_ID, args=args)


if __name__ == "__main__":
  main()
