"""Robot and scene assets for the G1 + Dex3-1 fruit basket V1 task."""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ACTUATOR_4010,
  G1_ACTUATOR_5020,
  G1_ACTUATOR_7520_14,
  G1_ACTUATOR_7520_22,
  G1_ACTUATOR_ANKLE,
  G1_ACTUATOR_WAIST,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

# Package-local copy of G1 29-DoF + Dex3-1 14-DoF model so training is portable
# across local workstations and remote GPU servers without hardcoded paths.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_LOCAL_G1_DEX3_XML = _REPO_ROOT / "assets" / "g1_dex3" / "g1_29dof_with_hand.xml"

# Verified right-hand closed targets from 261020.zip teleop dataset (action.wbc).
RIGHT_HAND_CLOSED_TARGETS: dict[str, float] = {
  "right_hand_thumb_0_joint": 0.0,
  "right_hand_thumb_1_joint": -0.7,
  "right_hand_thumb_2_joint": -0.7,
  "right_hand_index_0_joint": 1.0,
  "right_hand_index_1_joint": 1.5,
  "right_hand_middle_0_joint": 1.0,
  "right_hand_middle_1_joint": 1.5,
}

# Body-only joint patterns (29 DoF) controlled directly by the policy.
BODY_29_ACTUATOR_EXPRS: tuple[str, ...] = (
  r".*_hip_.*_joint",
  r".*_knee_joint",
  r".*_ankle_.*_joint",
  r"waist_.*_joint",
  r".*_shoulder_.*_joint",
  r".*_elbow_joint",
  r".*_wrist_.*_joint",
)

# Dex3-1 finger position actuators matching g1_29dof_with_hand.xml torque limits.
DEX3_THUMB0_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(r".*_hand_thumb_0_joint",),
  stiffness=10.0,
  damping=0.3,
  effort_limit=2.45,
  armature=0.001,
)

DEX3_FINGER_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    r".*_hand_thumb_1_joint",
    r".*_hand_thumb_2_joint",
    r".*_hand_index_0_joint",
    r".*_hand_index_1_joint",
    r".*_hand_middle_0_joint",
    r".*_hand_middle_1_joint",
  ),
  stiffness=6.0,
  damping=0.15,
  effort_limit=0.7,
  armature=0.001,
)

G1_DEX3_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
    DEX3_THUMB0_ACTUATOR,
    DEX3_FINGER_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.9,
)

# Body action scales (identical to mjlab G1_ACTION_SCALE for the 29 body joints).
G1_BODY_ACTION_SCALE: dict[str, float] = {}
for _act in (
  G1_ACTUATOR_5020,
  G1_ACTUATOR_7520_14,
  G1_ACTUATOR_7520_22,
  G1_ACTUATOR_4010,
  G1_ACTUATOR_WAIST,
  G1_ACTUATOR_ANKLE,
):
  assert _act.effort_limit is not None
  for _expr in _act.target_names_expr:
    G1_BODY_ACTION_SCALE[_expr] = 0.25 * _act.effort_limit / _act.stiffness


def resolve_g1_dex3_xml() -> Path:
  """Locate g1_29dof_with_hand.xml (env override -> repo-local assets)."""
  env_override = os.environ.get("G1_DEX3_XML")
  if env_override:
    p = Path(env_override).expanduser().resolve()
    if p.exists():
      return p
  if _LOCAL_G1_DEX3_XML.exists():
    return _LOCAL_G1_DEX3_XML
  raise FileNotFoundError(
    f"Could not find g1_29dof_with_hand.xml at {_LOCAL_G1_DEX3_XML}. "
    "Set G1_DEX3_XML or copy assets/g1_dex3/ into the repository."
  )


def get_g1_dex3_spec() -> mujoco.MjSpec:
  """Load G1 29-DoF + Dex3-1 MjSpec, strip raw motor actuators, and add grasp site."""
  xml_path = resolve_g1_dex3_xml()
  spec = mujoco.MjSpec.from_file(str(xml_path))
  spec.meshdir = str(xml_path.parent / "meshes")

  # Remove pre-existing torque motors so mjlab attaches BuiltinPositionActuatorCfg.
  for act in list(spec.actuators):
    spec.delete(act)

  # Add right_palm_grasp site inside the open Dex3-1 C-ring pocket (between
  # index+middle and thumb) in right_wrist_yaw_link local coordinates.
  right_wrist = spec.body("right_wrist_yaw_link")
  if right_wrist is not None:
    right_wrist.add_site(
      name="right_palm_grasp",
      pos=(0.125, 0.035, 0.0),
      size=(0.01, 0.01, 0.01),
      rgba=(0.1, 0.9, 0.2, 0.8),
    )

  return spec


# Safe collision-free initial stance outside Table A (Table A occupies y in [-0.44, 0]).
# Facing Table A (-Y, yaw = -pi/2 -> quat w=cos(-pi/4)=0.7071, z=sin(-pi/4)=-0.7071).
G1_BASKET_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.35, 0.28, 0.785),
  rot=(0.7071068, 0.0, 0.0, -0.7071068),
  joint_pos={
    ".*_hip_pitch_joint": -0.20,
    ".*_knee_joint": 0.42,
    ".*_ankle_pitch_joint": -0.22,
    ".*_elbow_joint": 0.60,
    "left_shoulder_roll_joint": 0.20,
    "left_shoulder_pitch_joint": 0.20,
    "right_shoulder_roll_joint": -0.15,
    "right_shoulder_pitch_joint": 0.15,
    ".*_hand_.*_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)


def get_g1_dex3_robot_cfg() -> EntityCfg:
  """Create a fresh G1 29-DoF + Dex3-1 EntityCfg."""
  return EntityCfg(
    init_state=G1_BASKET_INIT_STATE,
    spec_fn=get_g1_dex3_spec,
    articulation=G1_DEX3_ARTICULATION,
  )


def get_table_a_spec() -> mujoco.MjSpec:
  """Table A (cart with 4 cm perimeter rail): 0.65 x 0.44 x 0.74 m."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table_a")
  rgba = (0.86, 0.85, 0.83, 1.0)
  body.add_geom(
    name="table_a_top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.0, 0.0, 0.73),
    size=(0.325, 0.22, 0.01),
    rgba=rgba,
    friction=(0.8, 0.005, 0.0001),
  )
  # 4 cm rails (z: 0.74 ~ 0.78 m)
  body.add_geom(
    name="table_a_rail_front",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.0, 0.2125, 0.76),
    size=(0.325, 0.0075, 0.02),
    rgba=rgba,
  )
  body.add_geom(
    name="table_a_rail_back",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.0, -0.2125, 0.76),
    size=(0.325, 0.0075, 0.02),
    rgba=rgba,
  )
  body.add_geom(
    name="table_a_rail_left",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(-0.3175, 0.0, 0.76),
    size=(0.0075, 0.22, 0.02),
    rgba=rgba,
  )
  body.add_geom(
    name="table_a_rail_right",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.3175, 0.0, 0.76),
    size=(0.0075, 0.22, 0.02),
    rgba=rgba,
  )
  body.add_geom(
    name="table_a_shelf",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.0, 0.0, 0.14),
    size=(0.30, 0.20, 0.01),
    rgba=rgba,
  )
  for idx, (lx, ly) in enumerate(
    [(-0.30, -0.20), (0.30, -0.20), (-0.30, 0.20), (0.30, 0.20)], start=1
  ):
    body.add_geom(
      name=f"table_a_leg_{idx}",
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      pos=(lx, ly, 0.36),
      size=(0.015, 0.36, 0.0),
      rgba=rgba,
    )
  return spec


def get_table_a_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=(0.325, -0.22, 0.0)),
    spec_fn=get_table_a_spec,
  )


def get_table_b_spec() -> mujoco.MjSpec:
  """Table B (folding table): 0.60 x 0.40 x 0.61 m."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table_b")
  rgba = (0.93, 0.94, 0.95, 1.0)
  body.add_geom(
    name="table_b_top",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.0, 0.0, 0.60),
    size=(0.30, 0.20, 0.01),
    rgba=rgba,
    friction=(0.8, 0.005, 0.0001),
  )
  body.add_geom(
    name="table_b_shelf",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.0, 0.0, 0.20),
    size=(0.28, 0.18, 0.01),
    rgba=rgba,
  )
  for idx, (lx, ly) in enumerate(
    [(-0.26, -0.16), (0.26, -0.16), (-0.26, 0.16), (0.26, 0.16)], start=1
  ):
    body.add_geom(
      name=f"table_b_leg_{idx}",
      type=mujoco.mjtGeom.mjGEOM_CYLINDER,
      pos=(lx, ly, 0.295),
      size=(0.014, 0.295, 0.0),
      rgba=rgba,
    )
  return spec


def get_table_b_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=(1.50, -0.24, 0.0)),
    spec_fn=get_table_b_spec,
  )


def get_basket_spec() -> mujoco.MjSpec:
  """1.10 kg fruit basket with hollow sideways-arch handle (bar along Y)."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="basket")
  body.add_freejoint(name="basket_freejoint")
  body.mass = 1.10
  body.ipos = (0.0, 0.0, 0.06)
  body.inertia = (0.012, 0.012, 0.016)
  body.explicitinertial = True

  basket_rgba = (0.82, 0.46, 0.18, 1.0)
  handle_rgba = (0.76, 0.40, 0.14, 1.0)

  # Bottom + 4 sloped walls (mass=0 since explicit inertial defines total 1.10 kg).
  body.add_geom(
    name="basket_bottom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=(0.0, 0.0, 0.006),
    size=(0.10, 0.075, 0.006),
    mass=0.0,
    rgba=basket_rgba,
    friction=(0.9, 0.01, 0.001),
  )
  walls = [
    ("basket_wall_px", (0.125, 0.0, 0.064), (0.0, 0.38, 0.0), (0.006, 0.10, 0.062)),
    ("basket_wall_nx", (-0.125, 0.0, 0.064), (0.0, -0.38, 0.0), (0.006, 0.10, 0.062)),
    ("basket_wall_py", (0.0, 0.10, 0.064), (-0.38, 0.0, 0.0), (0.125, 0.006, 0.062)),
    ("basket_wall_ny", (0.0, -0.10, 0.064), (0.38, 0.0, 0.0), (0.125, 0.006, 0.062)),
  ]
  for name, pos, euler, size in walls:
    g = body.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=pos,
      size=size,
      mass=0.0,
      rgba=basket_rgba,
    )
    g.alt.euler = euler

  # Visual-only fruits inside basket.
  fruits = [
    ("fruit_1", (-0.04, 0.02, 0.055), 0.038, (0.95, 0.48, 0.08, 1.0)),
    ("fruit_2", (0.04, 0.03, 0.055), 0.036, (0.82, 0.16, 0.14, 1.0)),
    ("fruit_3", (0.02, -0.03, 0.052), 0.035, (0.90, 0.76, 0.16, 1.0)),
    ("fruit_4", (-0.03, -0.03, 0.050), 0.034, (0.95, 0.48, 0.08, 1.0)),
  ]
  for name, pos, radius, rgba in fruits:
    body.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      pos=pos,
      size=(radius, 0.0, 0.0),
      mass=0.0,
      contype=0,
      conaffinity=0,
      rgba=rgba,
    )

  # 5-segment hollow arch handle: roots on ±Y long walls, top bar along Y, opening ±X.
  handle_segs = [
    (
      "handle_front_post",
      (0.0, 0.114, 0.170),
      (0.14, 0.0, 0.0),
      (0.0125, 0.006, 0.062),
    ),
    (
      "handle_back_post",
      (0.0, -0.114, 0.170),
      (-0.14, 0.0, 0.0),
      (0.0125, 0.006, 0.062),
    ),
    (
      "handle_front_shoulder",
      (0.0, 0.0775, 0.2625),
      (0.72, 0.0, 0.0),
      (0.0125, 0.006, 0.045),
    ),
    (
      "handle_back_shoulder",
      (0.0, -0.0775, 0.2625),
      (-0.72, 0.0, 0.0),
      (0.0125, 0.006, 0.045),
    ),
    (
      "handle_top_bar",
      (0.0, 0.0, 0.294),
      (0.0, 0.0, 0.0),
      (0.0125, 0.055, 0.006),
    ),
  ]
  for name, pos, euler, size in handle_segs:
    g = body.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      pos=pos,
      size=size,
      mass=0.0,
      rgba=handle_rgba,
      friction=(1.2, 0.01, 0.001),
    )
    if euler != (0.0, 0.0, 0.0):
      g.alt.euler = euler

  body.add_site(
    name="basket_handle_center",
    pos=(0.0, 0.0, 0.294),
    size=(0.01, 0.01, 0.01),
    rgba=(0.0, 1.0, 0.0, 0.8),
  )
  return spec


def get_basket_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=(0.325, -0.18, 0.742)),
    spec_fn=get_basket_spec,
  )
