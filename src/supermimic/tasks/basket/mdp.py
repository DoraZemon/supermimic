"""MDP terms (actions, observations, rewards, terminations) for G1 Basket V1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from .assets import RIGHT_HAND_CLOSED_TARGETS

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


TABLE_A_TOP_Z: float = 0.74
_DEFAULT_ROBOT_CFG = SceneEntityCfg("robot")
_DEFAULT_BASKET_CFG = SceneEntityCfg("basket")


# ---------------------------------------------------------------------------
# 1D Right-Hand Dex3-1 Open/Close Switch Action (+ fixed open left hand)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class RightHandSwitchActionCfg(ActionTermCfg):
  """Map a 1D scalar action to the 7 right-hand Dex3-1 joints while keeping
  the 7 left-hand joints naturally open (target = 0.0)."""

  entity_name: str = "robot"

  def build(self, env: ManagerBasedRlEnv) -> RightHandSwitchAction:
    return RightHandSwitchAction(self, env)


class RightHandSwitchAction(ActionTerm):
  """1-DoF continuous gripper command ``g in [0, 1]`` for Dex3-1 right hand."""

  cfg: RightHandSwitchActionCfg
  _entity: Entity

  def __init__(self, cfg: RightHandSwitchActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    self._action_dim = 1
    self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
    self._gripper_cmd = torch.zeros(self.num_envs, 1, device=self.device)

    rh_names = list(RIGHT_HAND_CLOSED_TARGETS.keys())
    rh_ids, resolved_rh = self._entity.find_joints_by_actuator_names(rh_names)
    self._rh_joint_ids = torch.tensor(rh_ids, device=self.device, dtype=torch.long)
    self._rh_closed_targets = torch.tensor(
      [RIGHT_HAND_CLOSED_TARGETS[n] for n in resolved_rh],
      device=self.device,
      dtype=torch.float32,
    )

    lh_ids, _ = self._entity.find_joints_by_actuator_names((r"left_hand_.*_joint",))
    self._lh_joint_ids = torch.tensor(lh_ids, device=self.device, dtype=torch.long)
    self._lh_open_targets = torch.zeros(
      self.num_envs, len(lh_ids), device=self.device, dtype=torch.float32
    )

  @property
  def action_dim(self) -> int:
    return self._action_dim

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def gripper_cmd(self) -> torch.Tensor:
    """Normalized gripper closure in ``[0, 1]``, shape ``(num_envs, 1)``."""
    return self._gripper_cmd

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    # Map raw policy output [-1, 1] -> smooth closure g in [0, 1].
    self._gripper_cmd[:] = torch.clamp(0.5 * (actions + 1.0), 0.0, 1.0)

  def apply_actions(self) -> None:
    rh_targets = self._gripper_cmd * self._rh_closed_targets.unsqueeze(0)
    self._entity.set_joint_position_target(rh_targets, joint_ids=self._rh_joint_ids)
    if len(self._lh_joint_ids) > 0:
      self._entity.set_joint_position_target(
        self._lh_open_targets, joint_ids=self._lh_joint_ids
      )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_actions[env_ids] = -1.0
    self._gripper_cmd[env_ids] = 0.0


# ---------------------------------------------------------------------------
# Helpers for palm & basket handle geometry
# ---------------------------------------------------------------------------


def _palm_and_handle_pose(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg,
  basket_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return ``(palm_pos_w, palm_quat_w, handle_pos_w, basket_quat_w)``."""
  robot: Entity = env.scene[robot_cfg.name]
  basket: Entity = env.scene[basket_cfg.name]

  palm_pos_w = robot.data.site_pos_w[:, robot_cfg.site_ids][:, 0]
  palm_quat_w = robot.data.site_quat_w[:, robot_cfg.site_ids][:, 0]
  handle_pos_w = basket.data.site_pos_w[:, basket_cfg.site_ids][:, 0]
  basket_quat_w = basket.data.root_link_quat_w
  return palm_pos_w, palm_quat_w, handle_pos_w, basket_quat_w


# ---------------------------------------------------------------------------
# Observations (closed-loop relative handle pose + basket state)
# ---------------------------------------------------------------------------


def handle_pos_in_palm(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg,
  basket_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Vector from right_palm_grasp to basket_handle_center in palm local frame (3D)."""
  palm_pos_w, palm_quat_w, handle_pos_w, _ = _palm_and_handle_pose(
    env, robot_cfg, basket_cfg
  )
  return quat_apply_inverse(palm_quat_w, handle_pos_w - palm_pos_w)


def handle_pos_in_root(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg,
  basket_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Vector from robot root to basket_handle_center in robot root frame (3D)."""
  robot: Entity = env.scene[robot_cfg.name]
  _, _, handle_pos_w, _ = _palm_and_handle_pose(env, robot_cfg, basket_cfg)
  return quat_apply_inverse(
    robot.data.root_link_quat_w, handle_pos_w - robot.data.root_link_pos_w
  )


def handle_axis_in_palm(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg,
  basket_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Basket handle bar axis (local +Y of basket) expressed in palm local frame (3D)."""
  _, palm_quat_w, _, basket_quat_w = _palm_and_handle_pose(env, robot_cfg, basket_cfg)
  y_local = torch.zeros_like(basket_quat_w[:, :3])
  y_local[:, 1] = 1.0
  handle_axis_w = quat_apply(basket_quat_w, y_local)
  return quat_apply_inverse(palm_quat_w, handle_axis_w)


def basket_height_above_table(
  env: ManagerBasedRlEnv,
  basket_cfg: SceneEntityCfg = _DEFAULT_BASKET_CFG,
) -> torch.Tensor:
  """Height of basket bottom above Table A surface (1D)."""
  basket: Entity = env.scene[basket_cfg.name]
  return (basket.data.root_link_pos_w[:, 2] - TABLE_A_TOP_Z).unsqueeze(-1)


def basket_lin_vel_in_root(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
  basket_cfg: SceneEntityCfg = _DEFAULT_BASKET_CFG,
) -> torch.Tensor:
  """Basket linear velocity expressed in robot root frame (3D)."""
  robot: Entity = env.scene[robot_cfg.name]
  basket: Entity = env.scene[basket_cfg.name]
  return quat_apply_inverse(robot.data.root_link_quat_w, basket.data.root_link_lin_vel_w)


# ---------------------------------------------------------------------------
# Rewards (V1: reach -> align 3-finger pocket -> close -> lift & hold + balance)
# ---------------------------------------------------------------------------


def reach_handle_reward(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg,
  basket_cfg: SceneEntityCfg,
  std: float = 0.20,
) -> torch.Tensor:
  """Smooth distance reward pulling right_palm_grasp onto basket_handle_center."""
  palm_pos_w, _, handle_pos_w, _ = _palm_and_handle_pose(env, robot_cfg, basket_cfg)
  dist = torch.norm(handle_pos_w - palm_pos_w, dim=-1)
  return torch.exp(-dist / std)


def align_grip_axis_reward(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg,
  basket_cfg: SceneEntityCfg,
  near_dist: float = 0.14,
) -> torch.Tensor:
  """Reward aligning Dex3-1 grip-hole axis (palm +Z) with handle bar (basket ±Y)
  when close to the handle, encouraging genuine 2-finger vs thumb opposition."""
  palm_pos_w, palm_quat_w, handle_pos_w, basket_quat_w = _palm_and_handle_pose(
    env, robot_cfg, basket_cfg
  )
  dist = torch.norm(handle_pos_w - palm_pos_w, dim=-1)
  near_gate = torch.clamp(1.0 - dist / near_dist, 0.0, 1.0)

  z_local = torch.zeros_like(palm_pos_w)
  z_local[:, 2] = 1.0
  y_local = torch.zeros_like(palm_pos_w)
  y_local[:, 1] = 1.0

  palm_z_w = quat_apply(palm_quat_w, z_local)
  handle_y_w = quat_apply(basket_quat_w, y_local)
  alignment = torch.abs(torch.sum(palm_z_w * handle_y_w, dim=-1))
  return near_gate * alignment


def grasp_closure_reward(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg,
  basket_cfg: SceneEntityCfg,
  close_dist: float = 0.05,
  open_dist: float = 0.12,
) -> torch.Tensor:
  """Reward closing the 1D right-hand switch when palm pocket is around the handle,
  and keeping the hand open while still far away."""
  palm_pos_w, _, handle_pos_w, _ = _palm_and_handle_pose(env, robot_cfg, basket_cfg)
  dist = torch.norm(handle_pos_w - palm_pos_w, dim=-1)

  rh_action = env.action_manager.get_term("right_hand")
  assert isinstance(rh_action, RightHandSwitchAction)
  g = rh_action.gripper_cmd.squeeze(-1)

  want_close = (dist < close_dist).float()
  want_open = (dist > open_dist).float()
  return want_close * g + 0.25 * want_open * (1.0 - g)


def lift_basket_reward(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg,
  basket_cfg: SceneEntityCfg,
  target_lift: float = 0.12,
) -> torch.Tensor:
  """Dense reward for lifting the basket cleanly above Table A (gated by proximity
  to the handle so knocking the basket does not earn lift reward)."""
  palm_pos_w, _, handle_pos_w, _ = _palm_and_handle_pose(env, robot_cfg, basket_cfg)
  basket: Entity = env.scene[basket_cfg.name]

  dist = torch.norm(handle_pos_w - palm_pos_w, dim=-1)
  grasp_gate = torch.exp(-torch.clamp(dist - 0.04, min=0.0) / 0.08)
  dz = torch.clamp(basket.data.root_link_pos_w[:, 2] - TABLE_A_TOP_Z, 0.0, target_lift)
  return grasp_gate * (dz / target_lift)


def lift_hold_bonus(
  env: ManagerBasedRlEnv,
  robot_cfg: SceneEntityCfg,
  basket_cfg: SceneEntityCfg,
  min_lift: float = 0.08,
) -> torch.Tensor:
  """Steady-hold bonus once the basket clears the 4 cm rail by >= 8 cm."""
  palm_pos_w, _, handle_pos_w, _ = _palm_and_handle_pose(env, robot_cfg, basket_cfg)
  basket: Entity = env.scene[basket_cfg.name]

  dist = torch.norm(handle_pos_w - palm_pos_w, dim=-1)
  dz = basket.data.root_link_pos_w[:, 2] - TABLE_A_TOP_Z
  lifted = (dz >= min_lift) & (dist < 0.08)
  basket_speed = torch.norm(basket.data.root_link_lin_vel_w, dim=-1)
  return lifted.float() * torch.exp(-basket_speed / 0.3)


def upright_torso_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Keep G1 torso upright (projected gravity z close to -1)."""
  robot: Entity = env.scene[asset_cfg.name]
  proj_g = robot.data.projected_gravity_b
  tilt_err = torch.sum(proj_g[:, :2] ** 2, dim=-1)
  return torch.exp(-tilt_err / 0.15)


def root_height_reward(
  env: ManagerBasedRlEnv,
  target_height: float = 0.78,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Keep G1 pelvis near nominal standing height."""
  robot: Entity = env.scene[asset_cfg.name]
  err = robot.data.root_link_pos_w[:, 2] - target_height
  return torch.exp(-(err**2) / 0.02)


def root_vel_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Penalize drifting/stepping away during V1 stationary grasp-and-lift."""
  robot: Entity = env.scene[asset_cfg.name]
  lin_sq = torch.sum(robot.data.root_link_lin_vel_w[:, :2] ** 2, dim=-1)
  ang_sq = robot.data.root_link_ang_vel_w[:, 2] ** 2
  return lin_sq + 0.5 * ang_sq


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


def robot_fell_over(
  env: ManagerBasedRlEnv,
  min_height: float = 0.48,
  max_tilt_cos: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ROBOT_CFG,
) -> torch.Tensor:
  """Terminate if G1 falls or collapses."""
  robot: Entity = env.scene[asset_cfg.name]
  low = robot.data.root_link_pos_w[:, 2] < min_height
  tilted = -robot.data.projected_gravity_b[:, 2] < max_tilt_cos
  return low | tilted


def basket_dropped(
  env: ManagerBasedRlEnv,
  min_z: float = 0.55,
  basket_cfg: SceneEntityCfg = _DEFAULT_BASKET_CFG,
) -> torch.Tensor:
  """Terminate if the basket is knocked off Table A."""
  basket: Entity = env.scene[basket_cfg.name]
  return basket.data.root_link_pos_w[:, 2] < min_z
