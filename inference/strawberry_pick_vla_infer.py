"""
Strawberry Picking — VLA Multi-Scene Evaluation (3-Camera)
============================================================
Runs the fine-tuned pi0.5 policy across multiple randomized scenes
with ripe and unripe strawberry distractors.

Prerequisites:
  1. Policy server (in openpi/ directory):
     XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/serve_policy.py \
         policy:checkpoint --policy.config=pi05_strawberry_3c \
         --policy.dir=<checkpoint_path>
  2. Run (in a separate terminal):
     $ISAAC_SIM_DIR/python.sh inference/strawberry_pick_vla_infer.py
"""

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import omni
import omni.kit.app
from PIL import Image
import os
import json
import time
from datetime import datetime

from pxr import UsdPhysics, PhysxSchema, Sdf, Gf, UsdGeom, UsdShade
from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import open_stage, get_current_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.sensor import Camera

import sys
import csv


# =============================================================================
# TUNABLE PARAMETERS
# =============================================================================

# -- Paths (auto-resolved from project structure) --
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_THIS_DIR)  # strawberry_vla_pro/
ISAAC_SIM_DIR = os.environ.get("ISAAC_SIM_DIR", "/path/to/isaac-sim")
OPENPI_DIR = os.path.join(BASE_DIR, "openpi")
SCENE_PATH_TEMPLATE = os.path.join(BASE_DIR, "scene", "scene_assembled_manual.usd")
FIXED_SCENE_PATH_TEMPLATE = os.path.join(_THIS_DIR, "scene_assembled_fixed_infer.usd")
STRAWBERRY_MODEL_PATH = os.path.join(BASE_DIR, "strawberry", "Strawberry_gltf.gltf")
PLANT_USD_PATH = os.path.join(BASE_DIR, "plant", "ficus_obj.usd")

# -- Evaluation --
N_SCENES = 50
MAX_STEPS_PER_ATTEMPT = 600
NO_ATTACH_TIMEOUT = 400
MAX_ATTEMPTS_PER_BERRY = 2
SETTLE_STEPS = 120
ACTION_EXEC_STEPS = 8

# -- Pseudo-grasp --
ATTACH_X_OFFSET = 0.18       # EE x-offset for attach check point (added to ee_x)
ATTACH_Z_OFFSET = 0.005       # EE z-offset for attach check point (added to ee_z)
ATTACH_RADIUS = 0.02         # distance threshold from offset point to berry mesh
GRIPPER_CLOSE_THRESHOLD = 38.0
GRIPPER_OPEN_THRESHOLD = 32.0

# -- VLA / Policy --
POLICY_HOST = "localhost"
POLICY_PORT = 8000
VLA_IMAGE_SIZE = (224, 224)
PROMPT = "pick the topmost rightmost strawberry and place it in the box"

# -- Scene generation --
PLANT_TRANSLATE = Gf.Vec3d(0.65, -0.05, 1.38)
PLANT_ROTATE_X = 90.0
PLANT_TARGET_HEIGHT = 0.5
STRAWBERRY_TARGET_DIAMETER = 0.03
MAX_BERRIES_PER_SCENE = 15
BERRY_MIN_DIST = 0.075
BERRY_X_RANGE = (0.76, 0.81)
BERRY_Y_RANGE = (-0.20, 0.10)
BERRY_Z_RANGE = (1.03, 1.33)
BERRY_SCALE_RANGE = (1.4, 1.8)       # ripe berry scale
BERRY_SCALE_RANGE_UNRIPE = (0.9, 1.3)  # unripe berry scale (smaller)

# Unripe strawberry colors (visual distractors - not picked)
UNRIPE_YELLOW_COLOR = Gf.Vec3f(0.92, 0.78, 0.20)  # half-ripe yellow-orange
UNRIPE_GREEN_COLOR  = Gf.Vec3f(0.30, 0.65, 0.15)  # fully unripe green

# -- Robot --
GRIPPER_OPEN_DEG = 30.0
GRIPPER_CLOSE_DEG = 45.0
HOME_JOINT_ANGLES = np.array([
    -0.4777, -1.4451, 2.2133, 2.3354, -1.0323, 0.0014,
     0.5236, 0.5236,
])

# -- Motion control --
POS_TOLERANCE = 0.008
STALL_TOLERANCE = 0.0002
MAX_MOVE_STEPS = 400
MIN_MOVE_STEPS = 5
WAYPOINT_SPACING = 0.15
RENDER_EVERY = 4
IK_MAX_JOINT_DELTA = 0.5

# -- Place box --
BOX_X, BOX_Y, BOX_Z_BASE = 0.95, 0.25, 0.80
BOX_W, BOX_D, BOX_H = 0.17, 0.17, 0.10
BOX_WALL_T = 0.005

# -- Output --
RESULTS_DIR = os.path.join(_THIS_DIR, "results")
SAVE_DIR = os.path.join(_THIS_DIR, "captures")


# =============================================================================
# DERIVED / FIXED CONSTANTS
# =============================================================================

SCENE_PATH = SCENE_PATH_TEMPLATE
FIXED_SCENE_PATH = FIXED_SCENE_PATH_TEMPLATE

ROBOT_PRIM_PATH = "/World/UR5e"
GRIPPER_ROOT = "/World/Robotiq_2F_140_physics_edit"
GRIPPER_BASE_LINK = f"{GRIPPER_ROOT}/robotiq_base_link"
WRIST_3_LINK = "/World/UR5e/wrist_3_link"
FINGER_JOINT = f"{GRIPPER_ROOT}/finger_joint"
RIGHT_OUTER_KNUCKLE_JOINT = f"{GRIPPER_ROOT}/right_outer_knuckle_joint"
CAM1_PATH = "/World/Cam1"
CAM2_PATH = "/World/UR5e/wrist_3_link/Cam2"
CAM3_PATH = "/World/Cam3"
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480

PRE_GRASP_OFFSET = 0.25
GRASP_OFFSET = 0.20
UR5E_REACH = 0.85
SAFE_RATIO = 0.82

q_ry = np.array([0.7071, 0.0, -0.7071, 0.0])
q_rz = np.array([0.7071, 0.0, 0.0, 0.7071])
def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2
    ])
HOME_QUAT = quat_multiply(q_ry, q_rz)
HOME_QUAT = HOME_QUAT / np.linalg.norm(HOME_QUAT)
HOME_POS = np.array([1.0, 0.0, 1.1])

_PLANT_CENTER = Gf.Vec3d(311.5, 2252.9, 2504.4)
_PLANT_MAX_DIM = 854.2
_PLANT_SCALE = PLANT_TARGET_HEIGHT / _PLANT_MAX_DIM

_KEEP_PLANT_MESHES = {
    "ficus_lyrata_054", "ficus_lyrata_053", "ficus_lyrata_052",
    "ficus_lyrata_051", "ficus_lyrata_05", "ficus_lyrata_061",
    "ficus_lyrata_06", "ficus_lyrata_07",
}


# =============================================================================
# VLA CLIENT
# =============================================================================

sys.path.insert(0, f"{OPENPI_DIR}/packages/openpi-client/src")
from openpi_client.websocket_client_policy import WebsocketClientPolicy


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_prim(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim: {path}")
    return prim

def ensure_drive(jp):
    d = UsdPhysics.DriveAPI.Get(jp, "angular")
    if not d: d = UsdPhysics.DriveAPI.Apply(jp, "angular")
    return d

def configure_active_drive(jp, stiff=50.0, damp=1.0, mf=60.0):
    d = ensure_drive(jp)
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(stiff)
    d.CreateDampingAttr(damp)
    d.CreateMaxForceAttr(mf)
    PhysxSchema.PhysxJointAPI.Apply(jp).CreateMaxJointVelocityAttr(130.0)

def configure_arm_drive(jp, stiff=1e4, damp=1e2, mf=200.0):
    d = ensure_drive(jp)
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(stiff)
    d.CreateDampingAttr(damp)
    d.CreateMaxForceAttr(mf)
    PhysxSchema.PhysxJointAPI.Apply(jp).CreateMaxJointVelocityAttr(3.14)

def set_target_deg(jp, deg):
    ensure_drive(jp).GetTargetPositionAttr().Set(float(deg))

def quat_conjugate(q): return np.array([q[0],-q[1],-q[2],-q[3]])
def quat_inverse(q): return quat_conjugate(q)/np.dot(q,q)
def rotate_vec_by_quat(v, q):
    vq = np.array([0,v[0],v[1],v[2]])
    return quat_multiply(quat_multiply(q,vq),quat_inverse(q))[1:4]

def _get_all_descendants(prim):
    result = []
    for child in prim.GetAllChildren():
        result.append(child)
        result.extend(_get_all_descendants(child))
    return result

def _find_meshes_recursive(prim):
    meshes = []
    if prim.IsA(UsdGeom.Mesh):
        meshes.append(prim)
    for child in prim.GetAllChildren():
        meshes.extend(_find_meshes_recursive(child))
    return meshes

def _has_any_material(prim):
    api = UsdShade.MaterialBindingAPI(prim)
    mat, _ = api.ComputeBoundMaterial()
    if mat.GetPrim().IsValid():
        return True
    for desc in _get_all_descendants(prim):
        api2 = UsdShade.MaterialBindingAPI(desc)
        mat2, _ = api2.ComputeBoundMaterial()
        if mat2.GetPrim().IsValid():
            return True
    return False

def _apply_strawberry_material(prim, color):
    """Apply a solid color material to strawberry meshes (for unripe berries)."""
    mat_path = prim.GetPath().AppendChild("_ColorMat")
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path.AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    for desc in _get_all_descendants(prim):
        if desc.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(desc).Bind(mat)
    if prim.IsA(UsdGeom.Mesh):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

def _apply_fallback_strawberry_material(prim):
    mat_path = prim.GetPath().AppendChild("_FallbackMat")
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path.AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.85, 0.10, 0.07))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    bound_count = 0
    for desc in _get_all_descendants(prim):
        if desc.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(desc).Bind(mat)
            bound_count += 1
    if bound_count == 0 and prim.IsA(UsdGeom.Mesh):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
        bound_count = 1
    return bound_count

def min_dist_to_berry_mesh(berry_prim, point):
    """Compute min distance from a point to any vertex on the berry meshes."""
    xf_cache = UsdGeom.XformCache(0)
    min_d = float('inf')
    for mesh_prim in _find_meshes_recursive(berry_prim):
        mesh = UsdGeom.Mesh(mesh_prim)
        pts_attr = mesh.GetPointsAttr()
        if not pts_attr: continue
        local_pts = np.array(pts_attr.Get(), dtype=np.float64)
        if len(local_pts) == 0: continue
        world_xf = xf_cache.GetLocalToWorldTransform(mesh_prim)
        mat = np.array(world_xf, dtype=np.float64)
        ones = np.ones((local_pts.shape[0], 1), dtype=np.float64)
        pts_h = np.hstack([local_pts, ones])
        pts_world = (pts_h @ mat)[:, :3]
        dists = np.linalg.norm(pts_world - point, axis=1)
        d = np.min(dists)
        if d < min_d: min_d = d
    return min_d


# =============================================================================
# PASS 1: Fix scene
# =============================================================================

print("=" * 60)
print("PASS 1: Fix scene")
print("=" * 60)

open_stage(SCENE_PATH)
for _ in range(60): omni.kit.app.get_app().update()
stage = get_current_stage()

old_joint_prim = stage.GetPrimAtPath("/World/UR5e/joints/robot_gripper_joint")
w3_xf = XFormPrim(WRIST_3_LINK)
bl_xf = XFormPrim(GRIPPER_BASE_LINK)
w3_pos, w3_quat = [np.array(x, np.float64) for x in w3_xf.get_world_pose()]
bl_pos, bl_quat = [np.array(x, np.float64) for x in bl_xf.get_world_pose()]
lp0_gf = old_joint_prim.GetAttribute("physics:localPos0").Get()
lr0_gf = old_joint_prim.GetAttribute("physics:localRot0").Get()
lp0 = np.array([lp0_gf[0], lp0_gf[1], lp0_gf[2]], np.float64)
lr0 = np.array([lr0_gf.GetReal(), *lr0_gf.GetImaginary()], np.float64)
jwp = w3_pos + rotate_vec_by_quat(lp0, w3_quat)
jwr = quat_multiply(w3_quat, lr0)
bqi = quat_inverse(bl_quat)
lr1 = quat_multiply(bqi, jwr)
lp1 = rotate_vec_by_quat(jwp - bl_pos, bqi)

br = old_joint_prim.GetRelationship("physics:body1")
if not br: br = old_joint_prim.CreateRelationship("physics:body1")
br.SetTargets([Sdf.Path(GRIPPER_BASE_LINK)])
old_joint_prim.GetAttribute("physics:jointEnabled").Set(True)
old_joint_prim.GetAttribute("physics:localPos1").Set(Gf.Vec3f(*[float(x) for x in lp1]))
old_joint_prim.GetAttribute("physics:localRot1").Set(Gf.Quatf(*[float(x) for x in lr1]))
old_joint_prim.GetAttribute("physics:excludeFromArticulation").Set(False)

grp = stage.GetPrimAtPath(GRIPPER_ROOT)
if grp.HasAPI(UsdPhysics.ArticulationRootAPI): grp.RemoveAPI(UsdPhysics.ArticulationRootAPI)
for jp in [FINGER_JOINT, RIGHT_OUTER_KNUCKLE_JOINT]:
    p = stage.GetPrimAtPath(jp)
    if p.IsValid():
        lo = p.GetAttribute("physics:lowerLimit"); hi = p.GetAttribute("physics:upperLimit")
        if lo.IsValid(): lo.Set(-75.0)
        if hi.IsValid(): hi.Set(75.0)

print("\n--- Removing pre-existing strawberries ---")
_n_removed = 0
for _i in range(30):
    for _pat in [f"/World/Strawberry_{_i:02d}", f"/World/Strawberry_{_i}",
                 f"/World/strawberry_{_i:02d}", f"/World/strawberry_{_i}"]:
        _p = stage.GetPrimAtPath(_pat)
        if _p.IsValid():
            stage.RemovePrim(_pat)
            _n_removed += 1
print(f"  Removed {_n_removed} strawberry prims")

stage.GetRootLayer().Export(FIXED_SCENE_PATH)
print("Fixed scene saved")


# =============================================================================
# PASS 2: Initialize simulation
# =============================================================================

print("\n" + "=" * 60)
print("PASS 2: Initialize simulation")
print("=" * 60)

open_stage(FIXED_SCENE_PATH)
for _ in range(60): omni.kit.app.get_app().update()
stage = get_current_stage()

world = World(stage_units_in_meters=1.0)
world.reset()

finger_joint_prim = get_prim(stage, FINGER_JOINT)
right_outer_knuckle_prim = get_prim(stage, RIGHT_OUTER_KNUCKLE_JOINT)
configure_active_drive(finger_joint_prim)
configure_active_drive(right_outer_knuckle_prim)

robot = Articulation(ROBOT_PRIM_PATH)
try: world.scene.add(robot)
except: pass

world.reset()
for _ in range(100): world.step(render=True)
robot.initialize()
for _ in range(40): world.step(render=True)

dof_names = list(robot.dof_names)
n_dofs = robot.num_dof
print(f"DOFs ({n_dofs}): {dof_names}")

UR5E_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
ur5e_idx = np.array([dof_names.index(j) for j in UR5E_JOINTS], dtype=int)
finger_idx = 6
rok_idx = dof_names.index("right_outer_knuckle_joint")

print("--- Configuring UR5e arm drives ---")
for jname in UR5E_JOINTS:
    jp_path = f"{ROBOT_PRIM_PATH}/{jname}"
    jp = stage.GetPrimAtPath(jp_path)
    if jp.IsValid():
        configure_arm_drive(jp)
    else:
        for link_name in ["base_link", "shoulder_link", "upper_arm_link",
                          "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]:
            alt_path = f"{ROBOT_PRIM_PATH}/{link_name}/{jname}"
            jp2 = stage.GetPrimAtPath(alt_path)
            if jp2.IsValid():
                configure_arm_drive(jp2)
                break

from omni.isaac.motion_generation import (
    RmpFlow, ArticulationMotionPolicy,
    LulaKinematicsSolver, ArticulationKinematicsSolver,
)

RMP_BASE = f"{ISAAC_SIM_DIR}/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots/ur5e"
rmpflow = RmpFlow(
    robot_description_path=RMP_BASE + "/rmpflow/ur5e_robot_description.yaml",
    urdf_path=RMP_BASE + "/ur5e.urdf",
    rmpflow_config_path=RMP_BASE + "/rmpflow/ur5e_rmpflow_config.yaml",
    end_effector_frame_name="tool0", maximum_substep_size=0.00334,
)
art_rmpflow = ArticulationMotionPolicy(robot, rmpflow, 1.0 / 60.0)
kin_solver = LulaKinematicsSolver(
    robot_description_path=RMP_BASE + "/rmpflow/ur5e_robot_description.yaml",
    urdf_path=RMP_BASE + "/ur5e.urdf",
)
art_kin = ArticulationKinematicsSolver(robot, kin_solver, "tool0")

base_xf = XFormPrim("/World/UR5e/base_link")
base_pos, base_quat = [np.array(x, np.float64) for x in base_xf.get_world_pose()]
rmpflow.set_robot_base_pose(base_pos, base_quat)
kin_solver.set_robot_base_pose(base_pos, base_quat)
print(f"Base: {np.round(base_pos, 4)}")

def setup_cameras():
    c1 = Camera(prim_path=CAM1_PATH, resolution=(IMAGE_WIDTH, IMAGE_HEIGHT))
    c2 = Camera(prim_path=CAM2_PATH, resolution=(IMAGE_WIDTH, IMAGE_HEIGHT))
    c3 = Camera(prim_path=CAM3_PATH, resolution=(IMAGE_WIDTH, IMAGE_HEIGHT))
    c1.initialize(); c2.initialize(); c3.initialize()
    return c1, c2, c3

cam1, cam2, cam3 = setup_cameras()
for _ in range(20): world.step(render=True)
print("Cameras ready")


# =============================================================================
# MOTION HELPERS
# =============================================================================

def get_ee_pos():
    pos, _ = art_kin.compute_end_effector_pose()
    return np.array(pos, np.float64)

def generate_waypoints(start, end, spacing=WAYPOINT_SPACING):
    dist = np.linalg.norm(end - start)
    if dist <= spacing: return [end]
    n_segs = int(np.ceil(dist / spacing))
    return [start * (1 - i / n_segs) + end * (i / n_segs) for i in range(1, n_segs + 1)]

def rmpflow_step_arm(render=True):
    actions = art_rmpflow.get_next_articulation_action()
    if actions.joint_positions is not None:
        raw = np.array(actions.joint_positions, np.float64)
        arm = raw[:6].copy()
        if not np.any(np.isnan(arm)):
            robot.apply_action(ArticulationAction(joint_positions=arm, joint_indices=ur5e_idx))
    world.step(render=render)

def move_rmpflow(target_pos, target_quat, label="",
                 tol=POS_TOLERANCE, max_steps=MAX_MOVE_STEPS):
    cur_pos = get_ee_pos()
    waypoints = generate_waypoints(cur_pos, target_pos)
    total_steps = 0
    for wp in waypoints:
        rmpflow.set_end_effector_target(wp, target_quat)
        prev_pos = get_ee_pos()
        stall_count = 0
        steps_wp = 0
        for step in range(max_steps):
            rmpflow_step_arm(render=(step % RENDER_EVERY == 0))
            steps_wp += 1; total_steps += 1
            if steps_wp >= MIN_MOVE_STEPS and step % 5 == 0:
                now_pos = get_ee_pos()
                if np.linalg.norm(wp - now_pos) < tol: break
                if np.linalg.norm(now_pos - prev_pos) < STALL_TOLERANCE:
                    stall_count += 1
                else:
                    stall_count = 0
                prev_pos = now_pos
                if stall_count >= 6:
                    action, success = art_kin.compute_inverse_kinematics(wp, target_quat)
                    if success and action.joint_positions is not None:
                        arm_pos = np.array(action.joint_positions[:6], np.float64)
                        cur_joints = robot.get_joint_positions()[:6]
                        if np.max(np.abs(arm_pos - cur_joints)) <= IK_MAX_JOINT_DELTA:
                            for bi in range(20):
                                alpha = (bi + 1) / 20.0
                                blended = cur_joints * (1 - alpha) + arm_pos * alpha
                                robot.apply_action(ArticulationAction(
                                    joint_positions=blended, joint_indices=ur5e_idx))
                                world.step(render=(bi % 4 == 0))
                                total_steps += 1
                    stall_count = 0
                    if np.linalg.norm(wp - get_ee_pos()) < tol: break
    world.step(render=True)
    final_pos = get_ee_pos()
    err = np.linalg.norm(target_pos - final_pos)
    if label:
        tag = "OK" if err < tol * 2 else "WARN"
        print(f"      [{tag}] {label}: err={err:.4f}m ({total_steps} steps)")
    return final_pos

def reset_to_home_joints():
    """Smoothly blend robot to exact HOME_JOINT_ANGLES, then enforce gripper drive."""
    _cur = robot.get_joint_positions().copy()
    _tgt = _cur.copy()
    _tgt[ur5e_idx] = HOME_JOINT_ANGLES[:6]
    _tgt[finger_idx] = HOME_JOINT_ANGLES[6]
    _tgt[rok_idx] = HOME_JOINT_ANGLES[7]
    for gi in range(6, n_dofs):
        if gi != finger_idx and gi != rok_idx: _tgt[gi] = 0.0
    for bi in range(40):
        alpha = (bi + 1) / 40.0
        robot.set_joint_positions(_cur * (1 - alpha) + _tgt * alpha)
        set_target_deg(finger_joint_prim, GRIPPER_OPEN_DEG)
        set_target_deg(right_outer_knuckle_prim, GRIPPER_OPEN_DEG)
        world.step(render=(bi % 4 == 0))
    open_rad = np.radians(GRIPPER_OPEN_DEG)
    gi_idx = np.array([finger_idx, rok_idx], dtype=int)
    gi_tgt = np.array([open_rad, open_rad], dtype=np.float64)
    for _ in range(30):
        robot.apply_action(ArticulationAction(joint_positions=gi_tgt, joint_indices=gi_idx))
        world.step(render=True)

def return_to_home():
    """Move arm back to HOME via RMPFlow then exact joint reset."""
    move_rmpflow(HOME_POS, HOME_QUAT, label="return-HOME", max_steps=MAX_MOVE_STEPS)
    reset_to_home_joints()


# =============================================================================
# INITIAL HOME
# =============================================================================

print("\n--- Initial HOME ---")
move_rmpflow(HOME_POS, HOME_QUAT, label="HOME-init", max_steps=MAX_MOVE_STEPS)
reset_to_home_joints()
for _ in range(30): world.step(render=True)


# =============================================================================
# STRAWBERRY TEMPLATE MEASUREMENT
# =============================================================================

print("\n--- Measuring strawberry template ---")
_template_path = "/World/_StrawberryTemplate"
_tp = stage.GetPrimAtPath(_template_path)
if _tp.IsValid(): stage.RemovePrim(_template_path)
_template_prim = stage.DefinePrim(_template_path, "Xform")
_template_prim.GetReferences().AddReference(STRAWBERRY_MODEL_PATH)
for _ in range(20): world.step(render=True)
_bbox_cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
_bbox = _bbox_cache.ComputeWorldBound(_template_prim)
_range = _bbox.ComputeAlignedRange()
_bb_min = np.array([_range.GetMin()[i] for i in range(3)])
_bb_max = np.array([_range.GetMax()[i] for i in range(3)])
_bb_size = _bb_max - _bb_min
_max_extent = max(_bb_size)
_scale_factor = STRAWBERRY_TARGET_DIAMETER / _max_extent if _max_extent > 0 else 1.0
_template_has_mat = _has_any_material(_template_prim)
print(f"  Scale={_scale_factor:.6f}, Material={'YES' if _template_has_mat else 'NO'}")
stage.RemovePrim(_template_path)


# =============================================================================
# PLACE BOX
# =============================================================================

print("\n--- Creating place box ---")
for p in ["/World/PlaceBox_floor", "/World/PlaceBox_wall_L", "/World/PlaceBox_wall_R",
          "/World/PlaceBox_wall_F", "/World/PlaceBox_wall_B"]:
    pp = stage.GetPrimAtPath(p)
    if pp.IsValid(): stage.RemovePrim(p)

def make_box_part(name, pos, scale, color=(0.3, 0.8, 0.3), opacity=0.25):
    path = f"/World/{name}"
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xf.AddScaleOp().Set(Gf.Vec3d(*scale))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    mp = path + "/_Mat"
    mat = UsdShade.Material.Define(stage, mp)
    shader = UsdShade.Shader.Define(stage, mp + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)

bx, by, bz = BOX_X, BOX_Y, BOX_Z_BASE
bw, bd, bh, wt = BOX_W, BOX_D, BOX_H, BOX_WALL_T
make_box_part("PlaceBox_floor",  (bx, by, bz + wt/2), (bw, bd, wt))
make_box_part("PlaceBox_wall_L", (bx, by - bd/2 + wt/2, bz + bh/2), (bw, wt, bh))
make_box_part("PlaceBox_wall_R", (bx, by + bd/2 - wt/2, bz + bh/2), (bw, wt, bh))
make_box_part("PlaceBox_wall_F", (bx - bw/2 + wt/2, by, bz + bh/2), (wt, bd, bh))
make_box_part("PlaceBox_wall_B", (bx + bw/2 - wt/2, by, bz + bh/2), (wt, bd, bh))

ped = UsdGeom.Cube.Define(stage, "/World/PlaceBox_pedestal")
ped.GetSizeAttr().Set(1.0)
pxf = UsdGeom.Xformable(ped.GetPrim())
pxf.ClearXformOpOrder()
pxf.AddTranslateOp().Set(Gf.Vec3d(bx, by, bz - 0.03))
pxf.AddScaleOp().Set(Gf.Vec3d(bw * 0.9, bd * 0.9, 0.06))
ped.CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.35, 0.25)])
for _ in range(30): world.step(render=True)
print("  Box ready")


# =============================================================================
# STEM HELPERS
# =============================================================================

def _get_branch_world_points(stage):
    for prim in stage.Traverse():
        if prim.GetName() == "ficus_lyrata_05" and prim.GetTypeName() == "Mesh":
            mesh = UsdGeom.Mesh(prim)
            pts_attr = mesh.GetPointsAttr()
            if not pts_attr: return None
            local_pts = np.array(pts_attr.Get(), dtype=np.float64)
            xf_cache = UsdGeom.XformCache(0)
            world_xf = xf_cache.GetLocalToWorldTransform(prim)
            mat = np.array(world_xf, dtype=np.float64)
            ones = np.ones((local_pts.shape[0], 1), dtype=np.float64)
            return (np.hstack([local_pts, ones]) @ mat)[:, :3]
    return None

def _find_nearest_branch_point(branch_pts, target, min_z_above=0.20):
    mask = branch_pts[:, 2] >= target[2] + min_z_above
    candidates = branch_pts[mask] if np.any(mask) else branch_pts[np.argsort(branch_pts[:, 2])[::-1][:max(1, len(branch_pts)//10)]]
    dists = np.linalg.norm(candidates - target, axis=1)
    return candidates[np.argmin(dists)], np.min(dists)

def _create_stem_curve(stage, stem_path, start_pos, end_pos, num_segments=12, radius=0.0015):
    horiz_dist = np.linalg.norm(end_pos[:2] - start_pos[:2])
    sag = horiz_dist * 0.35 + 0.02
    ctrl1 = np.array([end_pos[0]+(start_pos[0]-end_pos[0])*0.3, end_pos[1]+(start_pos[1]-end_pos[1])*0.3, end_pos[2]-sag*0.5])
    ctrl2 = np.array([start_pos[0]+(end_pos[0]-start_pos[0])*0.3, start_pos[1]+(end_pos[1]-start_pos[1])*0.3, start_pos[2]+0.01])
    pts, ws = [], []
    for i in range(num_segments + 1):
        t = i / num_segments
        pt = (1-t)**3*start_pos + 3*(1-t)**2*t*ctrl2 + 3*(1-t)*t**2*ctrl1 + t**3*end_pos
        pts.append(Gf.Vec3f(*pt.astype(float)))
        ws.append(radius * (1.0 + 0.6 * t))
    c = UsdGeom.BasisCurves.Define(stage, stem_path)
    c.CreateTypeAttr("cubic"); c.CreateBasisAttr("catmullRom"); c.CreateWrapAttr("nonperiodic")
    c.CreatePointsAttr(pts); c.CreateCurveVertexCountsAttr([len(pts)])
    c.CreateWidthsAttr(ws); c.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    mp = stem_path + "/_StemMat"
    mat = UsdShade.Material.Define(stage, mp)
    sh = UsdShade.Shader.Define(stage, mp + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.18, 0.55, 0.12))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mat)


# =============================================================================
# SCENE GENERATION / CLEANUP
# =============================================================================

def cleanup_scene(stage, n_berries):
    for i in range(max(n_berries, 20)):
        for path in [f"/World/Strawberry_{i:02d}", f"/World/Stem_{i:02d}"]:
            p = stage.GetPrimAtPath(path)
            if p.IsValid(): stage.RemovePrim(path)
    for path in ["/World/PlantAssembly", "/World/PlantPedestal"]:
        p = stage.GetPrimAtPath(path)
        if p.IsValid(): stage.RemovePrim(path)
    for _ in range(20): world.step(render=True)

def generate_scene(stage):
    _seed = np.random.randint(0, 100000)
    np.random.seed(_seed)

    pa = stage.DefinePrim("/World/PlantAssembly", "Xform")
    pa_xf = UsdGeom.Xformable(pa); pa_xf.ClearXformOpOrder()
    plant_rz = float(np.random.uniform(0, 360))
    pa_xf.AddTranslateOp().Set(PLANT_TRANSLATE)
    pa_xf.AddRotateZOp().Set(plant_rz)
    pa_xf.AddRotateXOp().Set(PLANT_ROTATE_X)
    pa_xf.AddScaleOp().Set(Gf.Vec3f(_PLANT_SCALE, _PLANT_SCALE, _PLANT_SCALE))

    ficus = stage.DefinePrim("/World/PlantAssembly/ficus", "Xform")
    ficus.GetReferences().AddReference(PLANT_USD_PATH)
    UsdGeom.Xformable(ficus).ClearXformOpOrder()
    UsdGeom.Xformable(ficus).AddTranslateOp().Set(_PLANT_CENTER)
    for _ in range(60): world.step(render=True)

    for p in stage.Traverse():
        if p.GetTypeName() == "Mesh" and "ficus_lyrata" in p.GetName():
            if p.GetName() not in _KEEP_PLANT_MESHES:
                UsdGeom.Imageable(p).MakeInvisible()

    for p in stage.Traverse():
        if not str(p.GetPath()).startswith("/World/PlantAssembly"): continue
        for api in [UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI,
                    UsdPhysics.RigidBodyAPI, PhysxSchema.PhysxCollisionAPI, PhysxSchema.PhysxRigidBodyAPI]:
            if p.HasAPI(api): p.RemoveAPI(api)
        for an in ["physics:collisionEnabled", "physics:rigidBodyEnabled"]:
            a = p.GetAttribute(an)
            if a.IsValid(): a.Clear()
    for _ in range(10): world.step(render=True)

    pc = UsdGeom.Cylinder.Define(stage, "/World/PlantPedestal")
    pc.CreateRadiusAttr(0.08); pc.CreateHeightAttr(0.13); pc.CreateAxisAttr("Z")
    pcx = UsdGeom.Xformable(pc.GetPrim()); pcx.ClearXformOpOrder()
    pcx.AddTranslateOp().Set(Gf.Vec3d(float(PLANT_TRANSLATE[0]), float(PLANT_TRANSLATE[1]), 0.83 + 0.065))
    pc.CreateDisplayColorAttr([Gf.Vec3f(0.30, 0.18, 0.08)])

    configs = []
    attempts = 0
    while len(configs) < MAX_BERRIES_PER_SCENE and attempts < 2000:
        attempts += 1
        sx = np.random.uniform(*BERRY_X_RANGE)
        sy = np.random.uniform(*BERRY_Y_RANGE)
        sz = np.random.uniform(*BERRY_Z_RANGE)
        ok = True
        for offset in [0.05, GRASP_OFFSET, PRE_GRASP_OFFSET]:
            if np.linalg.norm(np.array([sx + offset, sy, sz]) - base_pos) > 0.92 * UR5E_REACH:
                ok = False; break
        if ok:
            if not any(np.linalg.norm(np.array([sx,sy,sz]) - np.array(e)) < BERRY_MIN_DIST for e in configs):
                configs.append((sx, sy, sz))

    # ── Assign maturity (ripe/yellow/green)  ──
    n_total = len(configs)
    maturity_map = {i: "ripe" for i in range(n_total)}
    if n_total >= 5:
        all_indices = list(range(n_total))
        np.random.shuffle(all_indices)
        n_yellow = np.random.randint(1, 3)  # 1-2 yellow
        n_green  = np.random.randint(1, 3)  # 1-2 green
        for idx in all_indices[:n_yellow]:
            maturity_map[idx] = "yellow"
        for idx in all_indices[n_yellow:n_yellow + n_green]:
            maturity_map[idx] = "green"
    elif n_total >= 3:
        all_indices = list(range(n_total))
        np.random.shuffle(all_indices)
        maturity_map[all_indices[0]] = "yellow"
        maturity_map[all_indices[1]] = "green"
    n_ripe = sum(1 for v in maturity_map.values() if v == "ripe")
    n_yellow_ct = sum(1 for v in maturity_map.values() if v == "yellow")
    n_green_ct = sum(1 for v in maturity_map.values() if v == "green")
    print(f"  Maturity: {n_ripe} ripe, {n_yellow_ct} yellow, {n_green_ct} green")

    positions = []
    for i, (sx, sy, sz) in enumerate(configs):
        path = f"/World/Strawberry_{i:02d}"
        prim = stage.DefinePrim(path, "Xform")
        prim.GetReferences().AddReference(STRAWBERRY_MODEL_PATH)
        xf = UsdGeom.Xformable(prim); xf.ClearXformOpOrder()
        for ch in _get_all_descendants(prim):
            try: UsdGeom.Xformable(ch).ClearXformOpOrder()
            except: pass
        xf.AddTranslateOp().Set(Gf.Vec3d(sx, sy, sz))
        xf.AddOrientOp().Set(Gf.Quatf(0.7071068, Gf.Vec3f(0.0, -0.7071068, 0.0)))
        # Scale: unripe berries are smaller, ripe are normal
        if maturity_map.get(i, "ripe") in ("yellow", "green"):
            s = _scale_factor * np.random.uniform(*BERRY_SCALE_RANGE_UNRIPE)
        else:
            s = _scale_factor * np.random.uniform(*BERRY_SCALE_RANGE)
        xf.AddScaleOp().Set(Gf.Vec3d(s, s, s))
        UsdPhysics.RigidBodyAPI.Apply(prim)
        prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        for m in _find_meshes_recursive(prim):
            UsdPhysics.CollisionAPI.Apply(m)
            UsdPhysics.MeshCollisionAPI.Apply(m).CreateApproximationAttr("convexDecomposition")
        if not _find_meshes_recursive(prim):
            UsdPhysics.CollisionAPI.Apply(prim)
        positions.append(np.array([sx, sy, sz]))
    for _ in range(30): world.step(render=True)

    # Materials: apply color based on maturity
    for i in range(len(configs)):
        prim = stage.GetPrimAtPath(f"/World/Strawberry_{i:02d}")
        if not prim.IsValid():
            continue
        maturity = maturity_map.get(i, "ripe")
        if maturity == "yellow":
            _apply_strawberry_material(prim, UNRIPE_YELLOW_COLOR)
        elif maturity == "green":
            _apply_strawberry_material(prim, UNRIPE_GREEN_COLOR)
        elif not _template_has_mat or not _has_any_material(prim):
            _apply_fallback_strawberry_material(prim)
    for _ in range(10): world.step(render=True)

    bpts = _get_branch_world_points(stage)
    if bpts is not None and len(bpts) > 0:
        for i, (sx, sy, sz) in enumerate(configs):
            bp, _ = _find_nearest_branch_point(bpts, np.array([sx, sy, sz + STRAWBERRY_TARGET_DIAMETER * 0.5]))
            _create_stem_curve(stage, f"/World/Stem_{i:02d}", np.array([sx, sy, sz + STRAWBERRY_TARGET_DIAMETER*0.5]), bp)
        for i in range(len(configs)):
            sp = stage.GetPrimAtPath(f"/World/Stem_{i:02d}")
            if sp.IsValid():
                if sp.HasAPI(UsdPhysics.CollisionAPI): sp.RemoveAPI(UsdPhysics.CollisionAPI)
                if sp.HasAPI(UsdPhysics.RigidBodyAPI): sp.RemoveAPI(UsdPhysics.RigidBodyAPI)
    for _ in range(10): world.step(render=True)
    return positions, configs, _seed, plant_rz, maturity_map


# =============================================================================
# VLA OBSERVATION / ACTION
# =============================================================================

print(f"\n--- Connecting to policy server at {POLICY_HOST}:{POLICY_PORT} ---")
policy_client = WebsocketClientPolicy(host=POLICY_HOST, port=POLICY_PORT)
print("Connected!")

def get_vla_observation():
    for _ in range(3): world.step(render=True)
    r1, r2, r3 = cam1.get_rgba(), cam2.get_rgba(), cam3.get_rgba()
    img1 = r1[:,:,:3] if r1 is not None else np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    img2 = r2[:,:,:3] if r2 is not None else np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    img3 = r3[:,:,:3] if r3 is not None else np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    img1 = np.array(Image.fromarray(img1).resize(VLA_IMAGE_SIZE, Image.LANCZOS))
    img2 = np.array(Image.fromarray(img2).resize(VLA_IMAGE_SIZE, Image.LANCZOS))
    img3 = np.array(Image.fromarray(img3).resize(VLA_IMAGE_SIZE, Image.LANCZOS))
    js = robot.get_joint_positions()
    state = np.zeros(8, dtype=np.float32)
    for i, idx in enumerate(ur5e_idx): state[i] = float(js[idx])
    state[6] = float(js[finger_idx]); state[7] = float(js[rok_idx])
    return {"observation/cam1": img1, "observation/cam2": img2, "observation/cam3": img3,
            "observation/state": state, "prompt": PROMPT}


# =============================================================================
# PSEUDO-GRASP MECHANISM
# =============================================================================

attached_berry_idx = None
attached_berry_xf = None
attached_berry_offset = None
gripper_was_closed = False
berry_states = []
n_berries = 0

def reset_grasp_state():
    global attached_berry_idx, attached_berry_xf, attached_berry_offset, gripper_was_closed
    attached_berry_idx = None; attached_berry_xf = None
    attached_berry_offset = None; gripper_was_closed = False

def check_attach():
    global attached_berry_idx, attached_berry_xf, attached_berry_offset, gripper_was_closed
    if attached_berry_idx is not None: return
    ee_pos = get_ee_pos()
    # Attach check point: offset EE by -0.16 in x, +0.02 in z
    check_point = np.array([ee_pos[0] - ATTACH_X_OFFSET, ee_pos[1], ee_pos[2] + ATTACH_Z_OFFSET])
    for i in range(n_berries):
        if berry_states[i] != "FREE": continue
        bp = f"/World/Strawberry_{i:02d}"
        bprim = stage.GetPrimAtPath(bp)
        if not bprim.IsValid(): continue
        dist = min_dist_to_berry_mesh(bprim, check_point)
        if dist < ATTACH_RADIUS:
            attached_berry_idx = i
            attached_berry_xf = XFormPrim(bp)
            berry_states[i] = "ATTACHED"
            gripper_was_closed = False
            bpos, _ = attached_berry_xf.get_world_pose()
            w3p, w3q = [np.array(x, np.float64) for x in XFormPrim(WRIST_3_LINK).get_world_pose()]
            attached_berry_offset = rotate_vec_by_quat(np.array(bpos, np.float64) - w3p, quat_inverse(w3q))
            for m in _find_meshes_recursive(bprim):
                if m.HasAPI(UsdPhysics.CollisionAPI): m.GetAttribute("physics:collisionEnabled").Set(False)
            if bprim.HasAPI(UsdPhysics.CollisionAPI): bprim.GetAttribute("physics:collisionEnabled").Set(False)
            print(f"\n  ** ATTACHED S{i:02d} (offset_dist={dist:.4f}m)")
            return

def update_attached_berry():
    if attached_berry_idx is None: return
    w3p, w3q = [np.array(x, np.float64) for x in XFormPrim(WRIST_3_LINK).get_world_pose()]
    attached_berry_xf.set_world_pose(
        position=w3p + rotate_vec_by_quat(attached_berry_offset, w3q),
        orientation=np.array([1, 0, 0, 0], dtype=np.float64))

def check_release():
    global attached_berry_idx, attached_berry_xf, attached_berry_offset, gripper_was_closed
    if attached_berry_idx is None: return False
    js = robot.get_joint_positions()
    fdeg = float(np.degrees(js[finger_idx]))
    if fdeg >= GRIPPER_CLOSE_THRESHOLD: gripper_was_closed = True
    if not gripper_was_closed: return False
    if fdeg > GRIPPER_OPEN_THRESHOLD: return False
    i = attached_berry_idx
    bp = f"/World/Strawberry_{i:02d}"
    bprim = stage.GetPrimAtPath(bp)
    for m in _find_meshes_recursive(bprim):
        if m.HasAPI(UsdPhysics.CollisionAPI): m.GetAttribute("physics:collisionEnabled").Set(True)
    if bprim.HasAPI(UsdPhysics.CollisionAPI): bprim.GetAttribute("physics:collisionEnabled").Set(True)
    ka = bprim.GetAttribute("physics:kinematicEnabled")
    if ka.IsValid(): ka.Set(False)
    print(f"  ** RELEASED S{i:02d} (finger={fdeg:.1f} deg)")
    berry_states[i] = "RELEASED"
    attached_berry_idx = None; attached_berry_xf = None
    attached_berry_offset = None; gripper_was_closed = False
    return True

def check_berry_in_box(berry_idx):
    bxf = XFormPrim(f"/World/Strawberry_{berry_idx:02d}")
    bpos, _ = bxf.get_world_pose()
    bpos = np.array(bpos, np.float64)
    in_x = (BOX_X - BOX_W/2 - 0.03) <= bpos[0] <= (BOX_X + BOX_W/2 + 0.03)
    in_y = (BOX_Y - BOX_D/2 - 0.03) <= bpos[1] <= (BOX_Y + BOX_D/2 + 0.03)
    in_z = BOX_Z_BASE - 0.02 <= bpos[2] <= BOX_Z_BASE + BOX_H + 0.10
    return (in_x and in_y and in_z), bpos

def apply_vla_action_with_grasp(action_8dim):
    robot.apply_action(ArticulationAction(
        joint_positions=np.array(action_8dim[:6], dtype=np.float64), joint_indices=ur5e_idx))
    gi = np.array([finger_idx, rok_idx], dtype=int)
    robot.apply_action(ArticulationAction(
        joint_positions=np.array(action_8dim[6:8], dtype=np.float64), joint_indices=gi))
    set_target_deg(finger_joint_prim, float(np.degrees(action_8dim[6])))
    set_target_deg(right_outer_knuckle_prim, float(np.degrees(action_8dim[7])))
    world.step(render=True)
    check_attach(); update_attached_berry()
    return check_release()


# =============================================================================
# MAIN EVALUATION LOOP
# =============================================================================

os.makedirs(RESULTS_DIR, exist_ok=True)
all_scene_results = []
global_total_queries = 0

# ── Open CSV file for real-time per-berry logging ──
CSV_COLUMNS = [
    "scene_idx", "seed", "berry_idx", "spawn_x", "spawn_y", "spawn_z",
    "maturity", "attached", "in_box", "note"
]
csv_path = os.path.join(RESULTS_DIR, f"berry_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(CSV_COLUMNS)
csv_file.flush()
print(f"  CSV log: {csv_path}")

def _csv_update_berry(scene_idx, seed, berry_idx, pos, maturity, attached, in_box, note=""):
    """Append an updated row for a berry event to the CSV (real-time)."""
    csv_writer.writerow([
        scene_idx, seed, berry_idx,
        f"{pos[0]:.4f}", f"{pos[1]:.4f}", f"{pos[2]:.4f}",
        maturity,
        "TRUE" if attached else "FALSE",
        "TRUE" if in_box else "FALSE",
        note
    ])
    csv_file.flush()

print("\n" + "=" * 60)
print(f"MULTI-SCENE EVALUATION: {N_SCENES} scenes")
print("=" * 60)
print(f"  Attach: mesh dist to (ee_x-{ATTACH_X_OFFSET}, ee_y, ee_z+{ATTACH_Z_OFFSET}) < {ATTACH_RADIUS}m")
print(f"  Release: close >= {GRIPPER_CLOSE_THRESHOLD} deg, open < {GRIPPER_OPEN_THRESHOLD} deg")
print(f"  No-attach timeout: {NO_ATTACH_TIMEOUT} steps")
print(f"  Max steps/attempt: {MAX_STEPS_PER_ATTEMPT}")
print(f"  Max attempts/scene: n_berries * {MAX_ATTEMPTS_PER_BERRY}")
print(f"  Berry X range: {BERRY_X_RANGE}")
print(f"  Basket size: {BOX_W}x{BOX_D}m")

try:
    for scene_idx in range(N_SCENES):
        print(f"\n{'#' * 60}")
        print(f"SCENE {scene_idx + 1}/{N_SCENES}")
        print(f"{'#' * 60}")

        if scene_idx > 0:
            cleanup_scene(stage, n_berries)

        strawberry_positions, strawberry_configs, scene_seed, plant_rz, maturity_map = generate_scene(stage)
        n_berries = len(strawberry_positions)
        # All berries (ripe + unripe) are FREE targets — all can be picked
        berry_states = ["FREE"] * n_berries
        n_ripe = sum(1 for v in maturity_map.values() if v == "ripe")
        n_unripe_total = n_berries - n_ripe
        print(f"  {n_berries} berries ({n_ripe} ripe, {n_unripe_total} unripe, ALL pickable), seed={scene_seed}, plant_rz={plant_rz:.1f}")

        # ── Write spawn info to CSV immediately ──
        for i in range(n_berries):
            pos = strawberry_positions[i]
            maturity = maturity_map.get(i, "ripe")
            csv_writer.writerow([
                scene_idx, scene_seed, i, f"{pos[0]:.4f}", f"{pos[1]:.4f}", f"{pos[2]:.4f}",
                maturity, "FALSE", "FALSE", ""
            ])
            csv_file.flush()

        for _ in range(60): world.step(render=True)

        return_to_home()
        for _ in range(30): world.step(render=True)

        max_attempts = n_ripe * MAX_ATTEMPTS_PER_BERRY
        scene_attempt_results = []

        # Per-berry CSV tracking: attached / in_box status
        berry_csv_attached = [False] * n_berries
        berry_csv_in_box = [False] * n_berries

        for attempt_idx in range(max_attempts):
            # Stop when all ripe berries are no longer FREE (even if unripe remain)
            n_ripe_free = sum(1 for i in range(n_berries)
                              if berry_states[i] == "FREE" and maturity_map.get(i, "ripe") == "ripe")
            if n_ripe_free == 0:
                print(f"\n  All ripe berries done, stopping.")
                break

            n_free = sum(1 for s in berry_states if s == "FREE")
            print(f"\n  --- Attempt {attempt_idx + 1}/{max_attempts} ({n_ripe_free} ripe FREE, {n_free} total FREE) ---")
            reset_grasp_state()
            attempt_steps = 0
            did_attach = False
            did_release = False
            released_berry_idx = None

            while attempt_steps < MAX_STEPS_PER_ATTEMPT:
                obs = get_vla_observation()
                result = policy_client.infer(obs)
                action_chunk = result["actions"]
                global_total_queries += 1

                if global_total_queries % 20 == 1:
                    ee = get_ee_pos()
                    js = robot.get_joint_positions()
                    grip = [float(np.degrees(js[finger_idx])), float(np.degrees(js[rok_idx]))]
                    att = f"S{attached_berry_idx:02d}" if attached_berry_idx is not None else "none"
                    cl = "Y" if gripper_was_closed else "N"
                    print(f"    [Q{global_total_queries}] step={attempt_steps}, "
                          f"EE=({ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f}), "
                          f"grip=[{grip[0]:.1f},{grip[1]:.1f}], att={att}, cl={cl}")

                for i in range(min(ACTION_EXEC_STEPS, len(action_chunk))):
                    released = apply_vla_action_with_grasp(action_chunk[i])
                    attempt_steps += 1
                    # Detect new attach event and log to CSV immediately
                    if attached_berry_idx is not None and not did_attach:
                        did_attach = True
                        berry_csv_attached[attached_berry_idx] = True
                        _csv_update_berry(scene_idx, scene_seed, attached_berry_idx,
                                          strawberry_positions[attached_berry_idx],
                                          maturity_map.get(attached_berry_idx, "ripe"),
                                          attached=True, in_box=False, note="ATTACHED")
                    if released:
                        did_release = True
                        for bi in range(n_berries):
                            if berry_states[bi] == "RELEASED":
                                released_berry_idx = bi; break
                        break
                if did_release: break

                # Early exit: no attach within timeout
                if not did_attach and attempt_steps >= NO_ATTACH_TIMEOUT:
                    print(f"    No attach after {attempt_steps} steps, ending attempt.")
                    break

            # Evaluate attempt
            if did_release and released_berry_idx is not None:
                for _ in range(SETTLE_STEPS): world.step(render=True)
                in_box, fpos = check_berry_in_box(released_berry_idx)
                res = "IN_BOX" if in_box else "PLACE_FAIL"
                print(f"  >> S{released_berry_idx:02d}: {res} at ({fpos[0]:.3f},{fpos[1]:.3f},{fpos[2]:.3f})")
                scene_attempt_results.append({
                    "attempt": attempt_idx, "berry": released_berry_idx,
                    "result": res, "steps": attempt_steps, "final_pos": fpos.tolist()})
                # CSV update: mark in_box result
                if in_box:
                    berry_csv_in_box[released_berry_idx] = True
                _csv_update_berry(scene_idx, scene_seed, released_berry_idx,
                                  strawberry_positions[released_berry_idx],
                                  maturity_map.get(released_berry_idx, "ripe"),
                                  attached=True, in_box=in_box, note=res)
            elif did_attach:
                if attached_berry_idx is not None:
                    released_berry_idx = attached_berry_idx
                    bp = f"/World/Strawberry_{attached_berry_idx:02d}"
                    bprim = stage.GetPrimAtPath(bp)
                    for m in _find_meshes_recursive(bprim):
                        if m.HasAPI(UsdPhysics.CollisionAPI): m.GetAttribute("physics:collisionEnabled").Set(True)
                    if bprim.HasAPI(UsdPhysics.CollisionAPI): bprim.GetAttribute("physics:collisionEnabled").Set(True)
                    ka = bprim.GetAttribute("physics:kinematicEnabled")
                    if ka.IsValid(): ka.Set(False)
                    berry_states[attached_berry_idx] = "RELEASED"
                print(f"  >> S{released_berry_idx:02d}: PLACE_FAIL (timeout holding)")
                scene_attempt_results.append({
                    "attempt": attempt_idx, "berry": released_berry_idx,
                    "result": "PLACE_FAIL", "steps": attempt_steps, "final_pos": [0,0,0]})
                # CSV update: attached but place fail
                if released_berry_idx is not None:
                    _csv_update_berry(scene_idx, scene_seed, released_berry_idx,
                                      strawberry_positions[released_berry_idx],
                                      maturity_map.get(released_berry_idx, "ripe"),
                                      attached=True, in_box=False, note="PLACE_FAIL_TIMEOUT")
            else:
                print(f"  >> NO_ATTACH ({attempt_steps} steps)")
                scene_attempt_results.append({
                    "attempt": attempt_idx, "berry": None,
                    "result": "NO_ATTACH", "steps": attempt_steps, "final_pos": [0,0,0]})

            return_to_home()
            for _ in range(30): world.step(render=True)

        # Per-berry results
        berry_results = []
        for i in range(n_berries):
            bp = f"/World/Strawberry_{i:02d}"
            bprim = stage.GetPrimAtPath(bp)
            maturity = maturity_map.get(i, "ripe")
            if not bprim.IsValid():
                berry_results.append({"berry_idx": i, "spawn_pos": strawberry_positions[i].tolist(),
                                      "final_pos": [0,0,0], "status": "INVALID", "maturity": maturity})
                continue
            bxf = XFormPrim(bp)
            bpos, _ = bxf.get_world_pose()
            bpos = np.array(bpos, np.float64)
            in_box, _ = check_berry_in_box(i)
            if in_box:
                status = "IN_BOX"
            elif berry_states[i] == "RELEASED":
                status = "PLACE_FAIL"
            elif berry_states[i] == "ATTACHED":
                status = "ATTACHED_STUCK"
            else:
                status = "FREE"
            berry_results.append({"berry_idx": i, "spawn_pos": strawberry_positions[i].tolist(),
                                  "final_pos": bpos.tolist(), "status": status, "maturity": maturity})

        n_inbox = sum(1 for b in berry_results if b["status"] == "IN_BOX")
        n_pfail = sum(1 for b in berry_results if b["status"] == "PLACE_FAIL")
        n_free_final = sum(1 for b in berry_results if b["status"] == "FREE")
        n_ripe_inbox = sum(1 for b in berry_results if b["status"] == "IN_BOX" and b["maturity"] == "ripe")
        n_unripe_inbox = sum(1 for b in berry_results if b["status"] == "IN_BOX" and b["maturity"] != "ripe")

        scene_result = {
            "scene_idx": scene_idx, "seed": int(scene_seed), "plant_rz": float(plant_rz),
            "n_berries": n_berries, "n_ripe": n_ripe, "n_unripe": n_unripe_total,
            "n_in_box": n_inbox, "n_ripe_in_box": n_ripe_inbox, "n_unripe_in_box": n_unripe_inbox,
            "n_place_fail": n_pfail,
            "n_free": n_free_final, "berries": berry_results, "attempts": scene_attempt_results,
        }
        all_scene_results.append(scene_result)

        print(f"\n  Scene {scene_idx+1} summary: {n_inbox}/{n_berries} IN_BOX "
              f"(ripe:{n_ripe_inbox}, unripe:{n_unripe_inbox}), "
              f"{n_pfail} PLACE_FAIL, {n_free_final} FREE")

except KeyboardInterrupt:
    print("\n[Interrupted by user]")


# =============================================================================
# FINAL SUMMARY + JSON OUTPUT
# =============================================================================

print(f"\n{'=' * 60}")
print("FINAL EVALUATION SUMMARY")
print(f"{'=' * 60}")

total_berries = sum(s["n_berries"] for s in all_scene_results)
total_ripe = sum(s.get("n_ripe", s["n_berries"]) for s in all_scene_results)
total_inbox = sum(s["n_in_box"] for s in all_scene_results)
total_ripe_inbox = sum(s.get("n_ripe_in_box", 0) for s in all_scene_results)
total_unripe_inbox = sum(s.get("n_unripe_in_box", 0) for s in all_scene_results)
total_pfail = sum(s["n_place_fail"] for s in all_scene_results)
total_free = sum(s["n_free"] for s in all_scene_results)
total_unripe = sum(s.get("n_unripe", 0) for s in all_scene_results)
n_scenes_done = len(all_scene_results)

print(f"  Scenes completed:  {n_scenes_done}/{N_SCENES}")
print(f"  Total berries:     {total_berries} ({total_ripe} ripe, {total_unripe} unripe)")
print(f"  IN_BOX:            {total_inbox}  ({total_inbox/max(total_berries,1)*100:.1f}% of all)")
print(f"    Ripe IN_BOX:     {total_ripe_inbox}  ({total_ripe_inbox/max(total_ripe,1)*100:.1f}% of ripe)")
print(f"    Unripe IN_BOX:   {total_unripe_inbox}  ({total_unripe_inbox/max(total_unripe,1)*100:.1f}% of unripe)")
print(f"  PLACE_FAIL:        {total_pfail}  ({total_pfail/max(total_berries,1)*100:.1f}% of all)")
print(f"  FREE (never att):  {total_free}  ({total_free/max(total_berries,1)*100:.1f}% of all)")
print(f"  Total VLA queries: {global_total_queries}")

print(f"\n  Per-scene breakdown:")
print(f"  {'Scene':>5s}  {'Seed':>6s}  {'Ripe':>5s}  {'Total':>5s}  {'InBox':>5s}  {'RpBox':>5s}  {'UrBox':>5s}  {'PFail':>5s}  {'Free':>5s}")
for s in all_scene_results:
    print(f"  {s['scene_idx']+1:5d}  {s['seed']:6d}  {s.get('n_ripe', s['n_berries']):5d}  "
          f"{s['n_berries']:5d}  "
          f"{s['n_in_box']:5d}  {s.get('n_ripe_in_box',0):5d}  {s.get('n_unripe_in_box',0):5d}  "
          f"{s['n_place_fail']:5d}  {s['n_free']:5d}")

summary = {
    "timestamp": datetime.now().isoformat(),
    "n_scenes": n_scenes_done,
    "total_berries": total_berries,
    "total_ripe": total_ripe,
    "total_unripe": total_unripe,
    "total_in_box": total_inbox,
    "total_ripe_in_box": total_ripe_inbox,
    "total_unripe_in_box": total_unripe_inbox,
    "total_place_fail": total_pfail,
    "total_free": total_free,
    "total_vla_queries": global_total_queries,
    "params": {
        "N_SCENES": N_SCENES,
        "MAX_STEPS_PER_ATTEMPT": MAX_STEPS_PER_ATTEMPT,
        "NO_ATTACH_TIMEOUT": NO_ATTACH_TIMEOUT,
        "MAX_ATTEMPTS_PER_BERRY": MAX_ATTEMPTS_PER_BERRY,
        "ATTACH_X_OFFSET": ATTACH_X_OFFSET,
        "ATTACH_Z_OFFSET": ATTACH_Z_OFFSET,
        "ATTACH_RADIUS": ATTACH_RADIUS,
        "GRIPPER_CLOSE_THRESHOLD": GRIPPER_CLOSE_THRESHOLD,
        "GRIPPER_OPEN_THRESHOLD": GRIPPER_OPEN_THRESHOLD,
        "ACTION_EXEC_STEPS": ACTION_EXEC_STEPS,
        "BERRY_X_RANGE": list(BERRY_X_RANGE),
        "BERRY_Y_RANGE": list(BERRY_Y_RANGE),
        "BERRY_Z_RANGE": list(BERRY_Z_RANGE),
        "PLANT_TRANSLATE": [float(PLANT_TRANSLATE[0]), float(PLANT_TRANSLATE[1]), float(PLANT_TRANSLATE[2])],
        "BOX_W": BOX_W, "BOX_D": BOX_D,
        "PROMPT": PROMPT,
    },
    "scenes": all_scene_results,
}

json_path = os.path.join(RESULTS_DIR, f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n  Results saved to: {json_path}")
print(f"  CSV log saved to: {csv_path}")
print(f"{'=' * 60}")

# Close CSV file
csv_file.close()

print("\nSimulation running. Press Ctrl+C to exit.")
while simulation_app.is_running():
    world.step(render=True)
simulation_app.close()
