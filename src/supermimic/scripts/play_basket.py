"""Evaluate / visualize a trained G1 + Dex3-1 fruit basket V1 policy."""

from __future__ import annotations

import sys

import mjlab
import mjlab.tasks  # noqa: F401
import tyro
from mjlab.scripts.play import PlayConfig, run_play

import supermimic.tasks  # noqa: F401

TASK_ID = "SuperMimic-Basket-V1"


def main() -> None:
  args = tyro.cli(
    PlayConfig,
    default=PlayConfig(),
    prog=f"{sys.argv[0]} ({TASK_ID})",
    config=mjlab.TYRO_FLAGS,
  )
  run_play(task_id=TASK_ID, cfg=args)


if __name__ == "__main__":
  main()
