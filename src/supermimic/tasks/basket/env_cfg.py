"""Environment configuration for G1 + Dex3-1 V1 fruit basket grasp-and-lift."""

from __future__ import annotations

import mjlab.envs.mdp as mdp
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from . import mdp as basket_mdp
from .assets import (
  BODY_29_ACTUATOR_EXPRS,
  G1_BODY_ACTION_SCALE,
  get_basket_cfg,
  get_g1_dex3_robot_cfg,
  get_table_a_cfg,
  get_table_b_cfg,
)


def g1_basket_v1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create V1 closed-loop grasp-and-lift environment for G1 + Dex3-1.

  - Action space: 30 DoF (29 body joint positions + 1D right-hand open/close switch;
    left hand stays naturally open).
  - Observations: proprioception + closed-loop relative handle pose/axis in palm & root.
  - Resets: collision-free stance outside Table A (straight to diagonal yaw), small
    randomization of basket position/yaw on Table A.
  """
  palm_cfg = SceneEntityCfg("robot", site_names=("right_palm_grasp",))
  handle_cfg = SceneEntityCfg("basket", site_names=("basket_handle_center",))

  actor_terms = {
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.base_ang_vel,
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "handle_pos_in_palm": ObservationTermCfg(
      func=basket_mdp.handle_pos_in_palm,
      params={"robot_cfg": palm_cfg, "basket_cfg": handle_cfg},
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    "handle_pos_in_root": ObservationTermCfg(
      func=basket_mdp.handle_pos_in_root,
      params={"robot_cfg": palm_cfg, "basket_cfg": handle_cfg},
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    "handle_axis_in_palm": ObservationTermCfg(
      func=basket_mdp.handle_axis_in_palm,
      params={"robot_cfg": palm_cfg, "basket_cfg": handle_cfg},
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
    "basket_height": ObservationTermCfg(
      func=basket_mdp.basket_height_above_table,
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    **actor_terms,
    "basket_lin_vel": ObservationTermCfg(
      func=basket_mdp.basket_lin_vel_in_root,
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=not play
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms, concatenate_terms=True, enable_corruption=False
    ),
  }

  actions: dict[str, ActionTermCfg] = {
    "body_joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=BODY_29_ACTUATOR_EXPRS,
      scale=G1_BODY_ACTION_SCALE,
      use_default_offset=True,
    ),
    "right_hand": basket_mdp.RightHandSwitchActionCfg(
      entity_name="robot",
    ),
  }

  # Safe collision-free resets:
  # - Robot default pos=(0.35, 0.28, 0.785), yaw=-pi/2 -> delta keeps y >= 0.22 m
  #   (strictly outside Table A's y <= 0.0 boundary) and yaw in [-90 deg, -30 deg].
  # - Basket default pos=(0.325, -0.18, 0.742) -> stays inside Table A rails.
  events = {
    "reset_robot_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "pose_range": {
          "x": (-0.12, 0.12),
          "y": (-0.04, 0.08),
          "yaw": (0.0, 1.05),  # -90 deg (straight) to -30 deg (diagonal toward B)
        },
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "position_range": (-0.03, 0.03),
        "velocity_range": (0.0, 0.0),
      },
    ),
    "reset_basket": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("basket"),
        "pose_range": {
          "x": (-0.06, 0.06),
          "y": (-0.04, 0.04),
          "yaw": (-0.20, 0.20),
        },
        "velocity_range": {},
      },
    ),
  }

  rewards = {
    "reach_coarse": RewardTermCfg(
      func=basket_mdp.reach_handle_reward,
      weight=1.5,
      params={"robot_cfg": palm_cfg, "basket_cfg": handle_cfg, "std": 0.25},
    ),
    "reach_fine": RewardTermCfg(
      func=basket_mdp.reach_handle_reward,
      weight=2.0,
      params={"robot_cfg": palm_cfg, "basket_cfg": handle_cfg, "std": 0.06},
    ),
    "align_grip_axis": RewardTermCfg(
      func=basket_mdp.align_grip_axis_reward,
      weight=1.5,
      params={"robot_cfg": palm_cfg, "basket_cfg": handle_cfg, "near_dist": 0.14},
    ),
    "grasp_closure": RewardTermCfg(
      func=basket_mdp.grasp_closure_reward,
      weight=1.0,
      params={
        "robot_cfg": palm_cfg,
        "basket_cfg": handle_cfg,
        "close_dist": 0.05,
        "open_dist": 0.12,
      },
    ),
    "lift_basket": RewardTermCfg(
      func=basket_mdp.lift_basket_reward,
      weight=6.0,
      params={"robot_cfg": palm_cfg, "basket_cfg": handle_cfg, "target_lift": 0.12},
    ),
    "lift_hold_bonus": RewardTermCfg(
      func=basket_mdp.lift_hold_bonus,
      weight=4.0,
      params={"robot_cfg": palm_cfg, "basket_cfg": handle_cfg, "min_lift": 0.08},
    ),
    "upright_torso": RewardTermCfg(
      func=basket_mdp.upright_torso_reward,
      weight=1.0,
    ),
    "root_height": RewardTermCfg(
      func=basket_mdp.root_height_reward,
      weight=1.0,
      params={"target_height": 0.78},
    ),
    "root_vel_penalty": RewardTermCfg(
      func=basket_mdp.root_vel_penalty,
      weight=-0.5,
    ),
    "action_rate_l2": RewardTermCfg(
      func=mdp.action_rate_l2,
      weight=-0.01,
    ),
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-2.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(func=basket_mdp.robot_fell_over),
    "basket_dropped": TerminationTermCfg(func=basket_mdp.basket_dropped),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={
        "robot": get_g1_dex3_robot_cfg(),
        "table_a": get_table_a_cfg(),
        "table_b": get_table_b_cfg(),
        "basket": get_basket_cfg(),
      },
      num_envs=2048,
      env_spacing=0.0,
    ),
    observations=observations,
    actions=actions,
    commands={},
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=2.0,
      elevation=-18.0,
      azimuth=135.0,
    ),
    sim=SimulationCfg(
      nconmax=120,
      njmax=800,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=20,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=4,  # 50 Hz control (matches 50 Hz teleop dataset)
    episode_length_s=12.0,
  )

  if play:
    cfg.scene.num_envs = 1
    cfg.episode_length_s = 30.0
    cfg.observations["actor"].enable_corruption = False

  return cfg
