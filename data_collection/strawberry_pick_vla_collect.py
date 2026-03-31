"""
Strawberry Picking — VLA Data Collection (3-Camera, 15Hz)
==========================================================
Expert demonstration collection for VLA training.
Uses RMPFlow planner with 3 RGB cameras (2 static + 1 wrist).
Records at 15Hz (RENDER_EVERY=4 from 60Hz physics).

Usage:
  $ISAAC_SIM_DIR/python.sh data_collection/strawberry_pick_vla_collect.py
"""

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})  # set True for headless collection

import numpy as np
import omni
import omni.kit.app
from PIL import Image
import os
import csv
import time
from datetime import datetime

from pxr import UsdPhysics, PhysxSchema, Sdf, Gf, UsdGeom, UsdShade
from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import open_stage, get_current_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.sensor import Camera

# VLA data recording
from vla_data_collector import EpisodeRecorder

# ===== PATHS (auto-resolved from project structure) =====
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)  # strawberry_vla_pro/
ISAAC_SIM_DIR = os.environ.get("ISAAC_SIM_DIR", "/path/to/isaac-sim")

SCENE_PATH = os.path.join(_PROJECT_DIR, "scene", "scene_assembled_manual.usd")
FIXED_SCENE_PATH = os.path.join(_THIS_DIR, "scene_assembled_fixed_vla.usd")
SAVE_DIR = os.path.join(_THIS_DIR, "captures")
STRAWBERRY_MODEL_PATH = os.path.join(_PROJECT_DIR, "strawberry", "Strawberry_gltf.gltf")
STRAWBERRY_TARGET_DIAMETER = 0.03

# Unripe strawberry colors (visual distractors - not picked)
UNRIPE_YELLOW_COLOR = Gf.Vec3f(0.92, 0.78, 0.20)  # half-ripe yellow-orange
UNRIPE_GREEN_COLOR  = Gf.Vec3f(0.30, 0.65, 0.15)  # fully unripe green

# Plant model
PLANT_USD_PATH = os.path.join(_PROJECT_DIR, "plant", "ficus_obj.usd")
PLANT_TRANSLATE = Gf.Vec3d(0.65, -0.05, 1.38)
PLANT_ROTATE_X = 90.0
PLANT_TARGET_HEIGHT = 0.5  # meters

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

GRIPPER_OPEN_DEG = 30.0
GRIPPER_CLOSE_DEG = 45.0

# HOME pose
# Ry(-90) * Rz(90) -> gripper pointing -X
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

# Fixed HOME joint angles — reset to these before every episode
# to prevent wrist joint drift across episodes
HOME_JOINT_ANGLES = np.array([
    -0.4777,   # shoulder_pan
    -1.4451,   # shoulder_lift
     2.2133,   # elbow
     2.3354,   # wrist_1
    -1.0323,   # wrist_2
     0.0014,   # wrist_3
     0.5236,   # finger (30 deg = open)
     0.5236,   # right_outer_knuckle (30 deg = open)
])

# Grasp parameters
PRE_GRASP_OFFSET = 0.25   # v9.5: x+0.20
GRASP_OFFSET = 0.20       # v9.84: x+0.20

# Collection box position
BOX_CENTER = np.array([1.1, 0.25, 0.85])

# ── Motion control parameters ──
POS_TOLERANCE    = 0.008   # position convergence threshold (m)
STALL_TOLERANCE  = 0.0002  # per-step displacement below this = stalled
MAX_MOVE_STEPS   = 400     # max steps per motion segment
MIN_MOVE_STEPS   = 5       # min steps before checking convergence
WAYPOINT_SPACING = 0.15    # waypoint spacing (m)
RENDER_EVERY     = 4       # render every N steps (physics runs every step)
IK_MAX_JOINT_DELTA = 0.5   # max joint delta for IK fallback (rad)
ADVANCE_ERR_THRESHOLD = 0.01  # advance error above this = grasp failure

# ===== Collection config =====
NUM_ROUNDS = 1
LOG_FILE = os.path.join(_THIS_DIR, "picking_log.csv")

# ===== VLA recording config =====
VLA_EPISODE_DIR = os.path.join(_THIS_DIR, "episodes")
VLA_RECORD_EVERY = 1   # v4: no second-layer filtering; RENDER_EVERY controls rate → 15Hz


# ===== Utility functions =====

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
    """High-stiffness drive for UR5e joints — reduces oscillation, faster response"""
    d = ensure_drive(jp)
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(stiff)
    d.CreateDampingAttr(damp)
    d.CreateMaxForceAttr(mf)
    PhysxSchema.PhysxJointAPI.Apply(jp).CreateMaxJointVelocityAttr(3.14)  # ~180 deg/s

def set_target_deg(jp, deg):
    ensure_drive(jp).GetTargetPositionAttr().Set(float(deg))

def quat_conjugate(q): return np.array([q[0],-q[1],-q[2],-q[3]])
def quat_inverse(q): return quat_conjugate(q)/np.dot(q,q)
def rotate_vec_by_quat(v, q):
    vq = np.array([0,v[0],v[1],v[2]])
    return quat_multiply(quat_multiply(q,vq),quat_inverse(q))[1:4]
def rot_matrix_to_quat(R):
    R = np.array(R, np.float64)
    tr = R[0,0]+R[1,1]+R[2,2]
    if tr>0:
        s=0.5/np.sqrt(tr+1.0); w=0.25/s; x=(R[2,1]-R[1,2])*s; y=(R[0,2]-R[2,0])*s; z=(R[1,0]-R[0,1])*s
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]:
        s=2*np.sqrt(1+R[0,0]-R[1,1]-R[2,2]); w=(R[2,1]-R[1,2])/s; x=0.25*s; y=(R[0,1]+R[1,0])/s; z=(R[0,2]+R[2,0])/s
    elif R[1,1]>R[2,2]:
        s=2*np.sqrt(1+R[1,1]-R[0,0]-R[2,2]); w=(R[0,2]-R[2,0])/s; x=(R[0,1]+R[1,0])/s; y=0.25*s; z=(R[1,2]+R[2,1])/s
    else:
        s=2*np.sqrt(1+R[2,2]-R[0,0]-R[1,1]); w=(R[1,0]-R[0,1])/s; x=(R[0,2]+R[2,0])/s; y=(R[1,2]+R[2,1])/s; z=0.25*s
    q=np.array([w,x,y,z]); return q/np.linalg.norm(q)


# ===== Cameras =====
def setup_cameras():
    c1=Camera(prim_path=CAM1_PATH, resolution=(IMAGE_WIDTH,IMAGE_HEIGHT))
    c2=Camera(prim_path=CAM2_PATH, resolution=(IMAGE_WIDTH,IMAGE_HEIGHT))
    c3=Camera(prim_path=CAM3_PATH, resolution=(IMAGE_WIDTH,IMAGE_HEIGHT))
    c1.initialize(); c2.initialize(); c3.initialize()
    return c1, c2, c3

def capture_images(c1, c2, c3, world, step_id=0):
    for _ in range(3): world.step(render=True)
    r1,r2,r3 = c1.get_rgba(), c2.get_rgba(), c3.get_rgba()
    i1 = r1[:,:,:3] if r1 is not None else None
    i2 = r2[:,:,:3] if r2 is not None else None
    i3 = r3[:,:,:3] if r3 is not None else None
    if i1 is not None and i2 is not None and i3 is not None:
        os.makedirs(SAVE_DIR, exist_ok=True)
        Image.fromarray(i1).save(f"{SAVE_DIR}/step{step_id:04d}_cam1.png")
        Image.fromarray(i2).save(f"{SAVE_DIR}/step{step_id:04d}_cam2.png")
        Image.fromarray(i3).save(f"{SAVE_DIR}/step{step_id:04d}_cam3.png")
    return i1, i2, i3


# ============================================================
# PASS 1: Fix scene joints
# ============================================================
print("="*60); print("PASS 1: Fix scene"); print("="*60)

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
lp0 = np.array([lp0_gf[0],lp0_gf[1],lp0_gf[2]], np.float64)
lr0 = np.array([lr0_gf.GetReal(),*lr0_gf.GetImaginary()], np.float64)
jwp = w3_pos + rotate_vec_by_quat(lp0, w3_quat)
jwr = quat_multiply(w3_quat, lr0)
bqi = quat_inverse(bl_quat)
lr1 = quat_multiply(bqi, jwr)
lp1 = rotate_vec_by_quat(jwp-bl_pos, bqi)

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
        lo=p.GetAttribute("physics:lowerLimit"); hi=p.GetAttribute("physics:upperLimit")
        if lo.IsValid(): lo.Set(-75.0)
        if hi.IsValid(): hi.Set(75.0)

# ── Remove pre-existing strawberries to start clean ──
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
print(f"✅ Fixed scene saved")


# ============================================================
# PASS 2: Run
# ============================================================
print("\n"+"="*60); print("PASS 2: Run"); print("="*60)

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
print(f"✅ DOFs ({n_dofs}): {dof_names}")

UR5E_JOINTS = ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint",
               "wrist_1_joint","wrist_2_joint","wrist_3_joint"]
ur5e_idx = np.array([dof_names.index(j) for j in UR5E_JOINTS], dtype=int)
finger_idx = 6
rok_idx = dof_names.index("right_outer_knuckle_joint")
print(f"✅ UR5e indices: {ur5e_idx}, finger={finger_idx}, rok={rok_idx}")

# Configure high-stiffness drives for UR5e arm joints
print("--- Configuring UR5e arm drives (high stiffness) ---")
for jname in UR5E_JOINTS:
    jp_path = f"{ROBOT_PRIM_PATH}/{jname}"
    jp = stage.GetPrimAtPath(jp_path)
    if jp.IsValid():
        configure_arm_drive(jp)
        print(f"  ✅ {jname}: stiff=1e4, damp=1e2")
    else:
        # Joint may be nested under link prims, try alternate paths
        found = False
        for link_name in ["base_link", "shoulder_link", "upper_arm_link",
                          "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]:
            alt_path = f"{ROBOT_PRIM_PATH}/{link_name}/{jname}"
            jp2 = stage.GetPrimAtPath(alt_path)
            if jp2.IsValid():
                configure_arm_drive(jp2)
                print(f"  ✅ {jname} (at {alt_path}): stiff=1e4, damp=1e2")
                found = True
                break
        if not found:
            print(f"  ⚠️ {jname}: prim not found, skipping drive config")

# RMPFlow + Kinematics
from omni.isaac.motion_generation import (
    RmpFlow, ArticulationMotionPolicy,
    LulaKinematicsSolver, ArticulationKinematicsSolver,
)

RMP_BASE = f"{ISAAC_SIM_DIR}/exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots/ur5e"
rmpflow = RmpFlow(
    robot_description_path=RMP_BASE+"/rmpflow/ur5e_robot_description.yaml",
    urdf_path=RMP_BASE+"/ur5e.urdf",
    rmpflow_config_path=RMP_BASE+"/rmpflow/ur5e_rmpflow_config.yaml",
    end_effector_frame_name="tool0", maximum_substep_size=0.00334,
)
art_rmpflow = ArticulationMotionPolicy(robot, rmpflow, 1.0/60.0)
kin_solver = LulaKinematicsSolver(
    robot_description_path=RMP_BASE+"/rmpflow/ur5e_robot_description.yaml",
    urdf_path=RMP_BASE+"/ur5e.urdf",
)
art_kin = ArticulationKinematicsSolver(robot, kin_solver, "tool0")

base_xf = XFormPrim("/World/UR5e/base_link")
base_pos, base_quat = [np.array(x, np.float64) for x in base_xf.get_world_pose()]
rmpflow.set_robot_base_pose(base_pos, base_quat)
kin_solver.set_robot_base_pose(base_pos, base_quat)
print(f"✅ Base: {np.round(base_pos,4)}")

# ── Workspace reachability check ──
UR5E_REACH = 0.85
SAFE_RATIO = 0.82  # > 82% reach = warning
def check_reach(name, pos):
    d = np.linalg.norm(np.array(pos) - base_pos)
    ratio = d / UR5E_REACH
    tag = "OK" if ratio < SAFE_RATIO else ("MARGINAL" if ratio < 0.92 else "OUT OF REACH")
    print(f"  {tag} {name}: dist={d:.3f}m ({ratio*100:.1f}% reach)")
    return ratio < 0.92

print("--- Workspace safety check ---")
# Check key target positions
check_reach("HOME", HOME_POS)
check_reach("ABOVE_BOX", [1.1, 0.25, 0.95])
check_reach("LIFT_OFF(1.1,0.25,1.1)", [1.1, 0.25, 1.1])
check_reach("RETRACT(1.2,0.25,0.95)", [1.2, 0.25, 0.95])

cam1, cam2, cam3 = setup_cameras()
for _ in range(20): world.step(render=True)
print("✅ Cameras ready")

# ===== [VLA] Initialize episode recorder =====
recorder = EpisodeRecorder(
    save_dir=VLA_EPISODE_DIR,
    cam1=cam1,
    cam2=cam2,
    cam3=cam3,
    robot=robot,
    world=world,
    ur5e_idx=ur5e_idx,
    finger_idx=finger_idx,
    rok_idx=rok_idx,
    art_kin=art_kin,
    record_every=VLA_RECORD_EVERY,
)
print(f"✅ VLA Recorder ready (3-cam): {VLA_EPISODE_DIR}")


# ============================================================
# ============================================================
# Core control functions
# ============================================================

def get_ee_pos():
    """Get current end-effector position."""
    pos, _ = art_kin.compute_end_effector_pose()
    return np.array(pos, np.float64)


def generate_waypoints(start, end, spacing=WAYPOINT_SPACING):
    """
    Generate intermediate waypoints between start and end.
    If distance <= spacing, returns [end] directly.
    """
    dist = np.linalg.norm(end - start)
    if dist <= spacing:
        return [end]
    n_segs = int(np.ceil(dist / spacing))
    waypoints = []
    for i in range(1, n_segs + 1):
        alpha = i / n_segs
        waypoints.append(start * (1 - alpha) + end * alpha)
    return waypoints


def rmpflow_step_arm(render=True):
    """
    Execute one RMPFlow step, applying only to UR5e's 6 arm joints.
    [VLA] Records state + action + image on render steps.
    """
    actions = art_rmpflow.get_next_articulation_action()
    valid = False
    action_8dim = None
    if actions.joint_positions is not None:
        raw = np.array(actions.joint_positions, np.float64)
        arm = raw[:6].copy()
        if not np.any(np.isnan(arm)):
            robot.apply_action(ArticulationAction(
                joint_positions=arm, joint_indices=ur5e_idx))
            valid = True
            # [VLA] Build 8-dim action vector
            js = robot.get_joint_positions()
            action_8dim = np.zeros(8, dtype=np.float32)
            action_8dim[:6] = arm.astype(np.float32)
            action_8dim[6] = float(js[finger_idx])
            action_8dim[7] = float(js[rok_idx])
    world.step(render=render)
    # [VLA] Only record on render frames (no images on non-render frames)
    if valid and render and action_8dim is not None:
        recorder.record_step(action_8dim)
    return valid


def move_rmpflow(target_pos, target_quat, label="",
                 tol=POS_TOLERANCE, max_steps=MAX_MOVE_STEPS,
                 min_steps=MIN_MOVE_STEPS):
    """
    Smooth motion to target pose using RMPFlow.
      1. Large waypoint spacing (0.15m)
      2. Render every N steps (physics runs every step)
      3. min_steps=5, stop as soon as converged
      4. IK fallback with safety check (reject if joint delta > 0.5rad)
    Returns: final EE position
    """
    cur_pos = get_ee_pos()
    waypoints = generate_waypoints(cur_pos, target_pos)
    total_steps = 0

    for wi, wp in enumerate(waypoints):
        rmpflow.set_end_effector_target(wp, target_quat)

        prev_pos = get_ee_pos()
        stall_count = 0
        steps_this_wp = 0

        for step in range(max_steps):
            # Render every N steps, physics runs every step
            do_render = (step % RENDER_EVERY == 0)
            rmpflow_step_arm(render=do_render)
            steps_this_wp += 1
            total_steps += 1

            # Check convergence every 5 steps (after min_steps)
            if steps_this_wp >= min_steps and step % 5 == 0:
                now_pos = get_ee_pos()
                err = np.linalg.norm(wp - now_pos)

                # Within tolerance -> next waypoint
                if err < tol:
                    break

                # Check for stall
                movement = np.linalg.norm(now_pos - prev_pos)
                if movement < STALL_TOLERANCE:
                    stall_count += 1
                else:
                    stall_count = 0
                prev_pos = now_pos

                # Consecutive stalls -> IK fallback (with safety check)
                if stall_count >= 6:
                    action, success = art_kin.compute_inverse_kinematics(wp, target_quat)
                    if success and action.joint_positions is not None:
                        arm_pos = np.array(action.joint_positions[:6], np.float64)
                        cur_joints = robot.get_joint_positions()[:6]
                        max_delta = np.max(np.abs(arm_pos - cur_joints))

                        # Safety check: large joint delta means different IK config
                        if max_delta <= IK_MAX_JOINT_DELTA:
                            # Safe IK solution, blend to it
                            for blend_i in range(20):
                                alpha = (blend_i + 1) / 20.0
                                blended = cur_joints * (1 - alpha) + arm_pos * alpha
                                robot.apply_action(ArticulationAction(
                                    joint_positions=blended, joint_indices=ur5e_idx))
                                world.step(render=(blend_i % 4 == 0))
                                total_steps += 1
                        else:
                            pass  # Reject this IK solution, let RMPFlow continue

                    stall_count = 0
                    now_pos = get_ee_pos()
                    if np.linalg.norm(wp - now_pos) < tol:
                        break

    # Final render to sync visuals
    world.step(render=True)

    final_pos = get_ee_pos()
    err = np.linalg.norm(target_pos - final_pos)
    if label:
        status = "✅" if err < tol * 2 else "⚠️"
        print(f"      {status} {label}: err={err:.4f}m ({total_steps} steps)")
    # [VLA] Mark anomaly if motion error > 5cm
    if err > 0.05:
        recorder.mark_anomaly(f"{label} err={err:.4f}m")
    return final_pos


def move_rmpflow_with_berry(target_pos, target_quat, berry_xf, berry_offset,
                            label="", tol=POS_TOLERANCE, max_steps=MAX_MOVE_STEPS):
    """
    Smooth motion with RMPFlow + berry following.
    Berry follows wrist_3_link every step (including non-render steps).
    """
    cur_pos = get_ee_pos()
    waypoints = generate_waypoints(cur_pos, target_pos)
    total_steps = 0

    for wi, wp in enumerate(waypoints):
        rmpflow.set_end_effector_target(wp, target_quat)

        prev_pos = get_ee_pos()
        stall_count = 0
        steps_this_wp = 0

        for step in range(max_steps):
            do_render = (step % RENDER_EVERY == 0)
            rmpflow_step_arm(render=do_render)
            steps_this_wp += 1
            total_steps += 1

            # Update berry position every step (render or not)
            w3p, w3q = [np.array(x, np.float64) for x in XFormPrim(WRIST_3_LINK).get_world_pose()]
            berry_xf.set_world_pose(
                position=w3p + rotate_vec_by_quat(berry_offset, w3q),
                orientation=np.array([1,0,0,0], dtype=np.float64))

            # Check convergence every 5 steps
            if steps_this_wp >= MIN_MOVE_STEPS and step % 5 == 0:
                now_pos = get_ee_pos()
                err = np.linalg.norm(wp - now_pos)
                if err < tol:
                    break

                movement = np.linalg.norm(now_pos - prev_pos)
                if movement < STALL_TOLERANCE:
                    stall_count += 1
                else:
                    stall_count = 0
                prev_pos = now_pos

                if stall_count >= 6:
                    action, success = art_kin.compute_inverse_kinematics(wp, target_quat)
                    if success and action.joint_positions is not None:
                        arm_pos = np.array(action.joint_positions[:6], np.float64)
                        cur_joints = robot.get_joint_positions()[:6]
                        max_delta = np.max(np.abs(arm_pos - cur_joints))

                        if max_delta <= IK_MAX_JOINT_DELTA:
                            for blend_i in range(20):
                                alpha = (blend_i + 1) / 20.0
                                blended = cur_joints * (1 - alpha) + arm_pos * alpha
                                robot.apply_action(ArticulationAction(
                                    joint_positions=blended, joint_indices=ur5e_idx))
                                world.step(render=(blend_i % 4 == 0))
                                total_steps += 1
                                w3p, w3q = [np.array(x, np.float64) for x in XFormPrim(WRIST_3_LINK).get_world_pose()]
                                berry_xf.set_world_pose(
                                    position=w3p + rotate_vec_by_quat(berry_offset, w3q),
                                    orientation=np.array([1,0,0,0], dtype=np.float64))

                    stall_count = 0
                    now_pos = get_ee_pos()
                    if np.linalg.norm(wp - now_pos) < tol:
                        break

    # Final render frame
    world.step(render=True)

    final_pos = get_ee_pos()
    err = np.linalg.norm(target_pos - final_pos)
    if label:
        status = "✅" if err < tol * 2 else "⚠️"
        print(f"      {status} {label}: err={err:.4f}m ({total_steps} steps)")
    # [VLA] Mark anomaly if motion error > 5cm
    if err > 0.05:
        recorder.mark_anomaly(f"{label} err={err:.4f}m")
    return final_pos


def set_gripper_for_steps(deg, n_steps):
    """
    Time-based gripper control with dual drive targets.
    Uses RENDER_EVERY to control render/record rate at 15Hz.
    """
    target_rad = np.radians(float(deg))
    gripper_indices = np.array([finger_idx, rok_idx], dtype=int)
    gripper_targets = np.array([target_rad, target_rad], dtype=np.float64)

    for step in range(n_steps):
        set_target_deg(finger_joint_prim, deg)
        set_target_deg(right_outer_knuckle_prim, deg)
        robot.apply_action(ArticulationAction(
            joint_positions=gripper_targets,
            joint_indices=gripper_indices,
        ))
        do_render = (step % RENDER_EVERY == 0)
        world.step(render=do_render)
        # [VLA] Only record on render frames (consistent with arm motion)
        if do_render:
            js = robot.get_joint_positions()
            action_8dim = np.zeros(8, dtype=np.float32)
            for i, idx in enumerate(ur5e_idx):
                action_8dim[i] = float(js[idx])
            action_8dim[6] = target_rad
            action_8dim[7] = target_rad
            recorder.record_step(action_8dim)

    js = robot.get_joint_positions()
    f = np.degrees(js[finger_idx])
    r = np.degrees(js[rok_idx])
    delta = abs(f - r)
    if delta > 3.0:
        print(f"    ⚠️ Gripper asymmetry Δ={delta:.1f}°, re-enforcing...")
        # [VLA] Mark anomaly if asymmetry > 10 deg
        if delta > 10.0:
            recorder.mark_anomaly(f"gripper asymmetry {delta:.1f}°")
        for _ in range(60):
            set_target_deg(finger_joint_prim, deg)
            set_target_deg(right_outer_knuckle_prim, deg)
            robot.apply_action(ArticulationAction(
                joint_positions=gripper_targets, joint_indices=gripper_indices))
            world.step(render=True)
        js = robot.get_joint_positions()
        f = np.degrees(js[finger_idx])
        r = np.degrees(js[rok_idx])
    return f, r


def hard_reset_gripper():
    """
    Smooth gripper reset.
    Uses drive targets + blending instead of instant teleport
    to eliminate jumps during HOME return.
    """
    open_rad = np.radians(GRIPPER_OPEN_DEG)
    gripper_indices = np.array([finger_idx, rok_idx], dtype=int)
    gripper_targets = np.array([open_rad, open_rad], dtype=np.float64)

    # ── Phase 1: Drive finger + rok to open position smoothly ──
    # Set drive targets and let physics engine converge naturally
    set_target_deg(finger_joint_prim, GRIPPER_OPEN_DEG)
    set_target_deg(right_outer_knuckle_prim, GRIPPER_OPEN_DEG)
    robot.apply_action(ArticulationAction(
        joint_positions=gripper_targets, joint_indices=gripper_indices))

    # Allow time for drive convergence (~60 steps = 1s at 60Hz)
    for _ in range(60):
        robot.apply_action(ArticulationAction(
            joint_positions=gripper_targets, joint_indices=gripper_indices))
        world.step(render=True)

    # ── Phase 2: Smooth blend internal gripper joints to 0 ──
    # These are passive joints (inner_knuckle, inner_finger, etc.)
    # Direct teleport causes jumps, so we blend instead
    BLEND_STEPS = 30
    cur_joints = robot.get_joint_positions().copy()
    target_joints = cur_joints.copy()
    # Blend internal joints to 0, keep finger/rok at open_rad
    for gi in range(6, n_dofs):
        if gi == finger_idx or gi == rok_idx:
            target_joints[gi] = open_rad
        else:
            target_joints[gi] = 0.0
    # Keep arm joints unchanged
    target_joints[:6] = cur_joints[:6]

    for blend_i in range(BLEND_STEPS):
        alpha = (blend_i + 1) / BLEND_STEPS
        blended = cur_joints * (1 - alpha) + target_joints * alpha
        robot.set_joint_positions(blended)
        # Keep drive targets active to hold finger/rok stable
        set_target_deg(finger_joint_prim, GRIPPER_OPEN_DEG)
        set_target_deg(right_outer_knuckle_prim, GRIPPER_OPEN_DEG)
        robot.apply_action(ArticulationAction(
            joint_positions=gripper_targets, joint_indices=gripper_indices))
        world.step(render=(blend_i % 3 == 0))

    # ── Phase 3: Settle ──
    for _ in range(20):
        robot.apply_action(ArticulationAction(
            joint_positions=gripper_targets, joint_indices=gripper_indices))
        world.step(render=True)

    js = robot.get_joint_positions()
    f = np.degrees(js[finger_idx])
    r = np.degrees(js[rok_idx])
    delta = abs(f - r)
    if delta < 5.0 and abs(f - GRIPPER_OPEN_DEG) < 10.0:
        print(f"      gripper reset OK: f={f:.1f}°, r={r:.1f}°")
    else:
        print(f"      ⚠️ gripper reset incomplete: f={f:.1f}°, r={r:.1f}° (Δ={delta:.1f}°)")


# ===== Move to HOME =====
print("\n--- MOVE TO HOME ---")
# Use RMPFlow to move to HOME
final = move_rmpflow(HOME_POS, HOME_QUAT, label="HOME", max_steps=400)
print(f"  HOME: {np.round(final,4)}, err={np.linalg.norm(HOME_POS-final):.4f}m")

ee_rot = art_kin.compute_end_effector_pose()[1]
if hasattr(ee_rot,'shape') and ee_rot.shape==(3,3):
    tz = ee_rot[:,2]
else:
    tz = np.array([0,0,1])
print(f"  tool0 Z: {np.round(tz,3)} ({'✅ -X' if tz[0]<-0.5 else '⚠️'})")

for _ in range(60): world.step(render=True)


# ============================================================
# Plant model constants (computed once)
# ============================================================
_PLANT_CENTER = Gf.Vec3d(311.5, 2252.9, 2504.4)
_PLANT_MAX_DIM = 854.2
_PLANT_SCALE = PLANT_TARGET_HEIGHT / _PLANT_MAX_DIM

_KEEP_PLANT_MESHES = {
    "ficus_lyrata_054", "ficus_lyrata_053", "ficus_lyrata_052",
    "ficus_lyrata_051", "ficus_lyrata_05", "ficus_lyrata_061",
    "ficus_lyrata_06", "ficus_lyrata_07",
}

# ===== Strawberry helper functions =====

def _get_all_descendants(prim):
    result = []
    for child in prim.GetAllChildren():
        result.append(child)
        result.extend(_get_all_descendants(child))
    return result

def _apply_strawberry_material(prim, color):
    """Apply a solid color material to strawberry meshes."""
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

def _find_meshes_recursive(prim):
    meshes = []
    if prim.IsA(UsdGeom.Mesh):
        meshes.append(prim)
    for child in prim.GetAllChildren():
        meshes.extend(_find_meshes_recursive(child))
    return meshes

# ===== Measure strawberry template (once) =====
print("\n--- MEASURING STRAWBERRY TEMPLATE ---")
_template_path = "/World/_StrawberryTemplate"
_tp = stage.GetPrimAtPath(_template_path)
if _tp.IsValid():
    stage.RemovePrim(_template_path)

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

if _max_extent > 0:
    _scale_factor = STRAWBERRY_TARGET_DIAMETER / _max_extent
else:
    _scale_factor = 1.0

_template_has_mat = _has_any_material(_template_prim)
print(f"  Model bbox: {np.round(_bb_size, 4)}m, max_extent={_max_extent:.4f}m")
print(f"  Scale={_scale_factor:.6f}, Material={'YES' if _template_has_mat else 'NO (fallback)'}")
stage.RemovePrim(_template_path)


# ===== Stem helper functions =====

def _get_branch_world_points(stage):
    branch_prim = None
    for prim in stage.Traverse():
        if prim.GetName() == "ficus_lyrata_05" and prim.GetTypeName() == "Mesh":
            branch_prim = prim
            break
    if branch_prim is None:
        return None
    mesh = UsdGeom.Mesh(branch_prim)
    points_attr = mesh.GetPointsAttr()
    if not points_attr:
        return None
    local_pts = np.array(points_attr.Get(), dtype=np.float64)
    xf_cache = UsdGeom.XformCache(0)
    world_xf = xf_cache.GetLocalToWorldTransform(branch_prim)
    mat = np.array(world_xf, dtype=np.float64)
    ones = np.ones((local_pts.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([local_pts, ones])
    pts_world = pts_h @ mat
    return pts_world[:, :3]

def _find_nearest_branch_point(branch_pts, target, min_z_above=0.20):
    z_threshold = target[2] + min_z_above
    mask = branch_pts[:, 2] >= z_threshold
    if np.any(mask):
        candidates = branch_pts[mask]
    else:
        z_sorted_idx = np.argsort(branch_pts[:, 2])[::-1]
        top_n = max(1, len(branch_pts) // 10)
        candidates = branch_pts[z_sorted_idx[:top_n]]
    diffs = candidates - target
    dists = np.linalg.norm(diffs, axis=1)
    idx = np.argmin(dists)
    return candidates[idx], dists[idx]

def _create_stem_curve(stage, stem_path, start_pos, end_pos, num_segments=12, radius=0.0015):
    horiz_dist = np.linalg.norm(end_pos[:2] - start_pos[:2])
    sag = horiz_dist * 0.35 + 0.02
    ctrl1 = np.array([
        end_pos[0] + (start_pos[0] - end_pos[0]) * 0.3,
        end_pos[1] + (start_pos[1] - end_pos[1]) * 0.3,
        end_pos[2] - sag * 0.5,
    ])
    ctrl2 = np.array([
        start_pos[0] + (end_pos[0] - start_pos[0]) * 0.3,
        start_pos[1] + (end_pos[1] - start_pos[1]) * 0.3,
        start_pos[2] + 0.01,
    ])
    curve_points = []
    widths = []
    for i in range(num_segments + 1):
        t = i / num_segments
        b0 = (1 - t) ** 3
        b1 = 3 * (1 - t) ** 2 * t
        b2 = 3 * (1 - t) * t ** 2
        b3 = t ** 3
        pt = b0 * start_pos + b1 * ctrl2 + b2 * ctrl1 + b3 * end_pos
        curve_points.append(Gf.Vec3f(*pt.astype(float)))
        widths.append(radius * (1.0 + 0.6 * t))

    curves_prim = UsdGeom.BasisCurves.Define(stage, stem_path)
    curves_prim.CreateTypeAttr("cubic")
    curves_prim.CreateBasisAttr("catmullRom")
    curves_prim.CreateWrapAttr("nonperiodic")
    curves_prim.CreatePointsAttr(curve_points)
    curves_prim.CreateCurveVertexCountsAttr([len(curve_points)])
    curves_prim.CreateWidthsAttr(widths)
    curves_prim.SetWidthsInterpolation(UsdGeom.Tokens.vertex)

    mat_path = stem_path + "/_StemMat"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.18, 0.55, 0.12))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(curves_prim.GetPrim()).Bind(mat)


# ===== Collection box (created once) =====
print("\n--- CREATING PLACE BOX ---")
box_x, box_y, box_z_base = 0.95, 0.25, 0.80
box_w, box_d, box_h = 0.15, 0.15, 0.10
wall_t = 0.005

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
    mat_path = path + "/_Mat"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)

make_box_part("PlaceBox_floor", (box_x, box_y, box_z_base + wall_t/2), (box_w, box_d, wall_t))
make_box_part("PlaceBox_wall_L", (box_x, box_y - box_d/2 + wall_t/2, box_z_base + box_h/2), (box_w, wall_t, box_h))
make_box_part("PlaceBox_wall_R", (box_x, box_y + box_d/2 - wall_t/2, box_z_base + box_h/2), (box_w, wall_t, box_h))
make_box_part("PlaceBox_wall_F", (box_x - box_w/2 + wall_t/2, box_y, box_z_base + box_h/2), (wall_t, box_d, box_h))
make_box_part("PlaceBox_wall_B", (box_x + box_w/2 - wall_t/2, box_y, box_z_base + box_h/2), (wall_t, box_d, box_h))

# Box pedestal
_box_ped = UsdGeom.Cube.Define(stage, "/World/PlaceBox_pedestal")
_box_ped.GetSizeAttr().Set(1.0)
_bp_xf = UsdGeom.Xformable(_box_ped.GetPrim())
_bp_xf.ClearXformOpOrder()
_bp_xf.AddTranslateOp().Set(Gf.Vec3d(box_x, box_y, box_z_base - 0.03))
_bp_xf.AddScaleOp().Set(Gf.Vec3d(box_w * 0.9, box_d * 0.9, 0.06))
_box_ped.CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.35, 0.25)])

for _ in range(30): world.step(render=True)
print(f"  Box ready")

STEPS_CLOSE = 150
RX = 1.2
BOX_Y = 0.25
BOX_Z = 0.95
BOX_X = 1.1
ABOVE_BOX = np.array([BOX_X, BOX_Y, BOX_Z])


# ============================================================
# Scene generation / cleanup
# ============================================================

def cleanup_round(stage, n_berries):
    """Remove current round strawberries, stems, and plant."""
    for i in range(max(n_berries, 20)):
        for path in [f"/World/Strawberry_{i:02d}", f"/World/Stem_{i:02d}"]:
            p = stage.GetPrimAtPath(path)
            if p.IsValid():
                stage.RemovePrim(path)
    for path in ["/World/PlantAssembly", "/World/PlantPedestal"]:
        p = stage.GetPrimAtPath(path)
        if p.IsValid():
            stage.RemovePrim(path)
    for _ in range(20): world.step(render=True)


def generate_scene(stage, round_idx):
    """Generate new scene: plant (random Z rotation) + strawberries + stems + pedestal."""
    _seed = np.random.randint(0, 100000)
    np.random.seed(_seed)
    print(f"\n  Seed: {_seed}")

    # ── Plant ──
    _pa_prim = stage.DefinePrim("/World/PlantAssembly", "Xform")
    _pa_xf = UsdGeom.Xformable(_pa_prim)
    _pa_xf.ClearXformOpOrder()
    _plant_rotate_z = float(np.random.uniform(0, 360))
    print(f"  Plant Z rotation: {_plant_rotate_z:.1f}")
    _pa_xf.AddTranslateOp().Set(PLANT_TRANSLATE)
    _pa_xf.AddRotateZOp().Set(_plant_rotate_z)
    _pa_xf.AddRotateXOp().Set(PLANT_ROTATE_X)
    _pa_xf.AddScaleOp().Set(Gf.Vec3f(_PLANT_SCALE, _PLANT_SCALE, _PLANT_SCALE))

    _ficus_prim = stage.DefinePrim("/World/PlantAssembly/ficus", "Xform")
    _ficus_prim.GetReferences().AddReference(PLANT_USD_PATH)
    _ficus_xf = UsdGeom.Xformable(_ficus_prim)
    _ficus_xf.ClearXformOpOrder()
    _ficus_xf.AddTranslateOp().Set(_PLANT_CENTER)
    for _ in range(60): world.step(render=True)

    # Hide unnecessary meshes
    for _prim in stage.Traverse():
        if _prim.GetTypeName() == "Mesh" and "ficus_lyrata" in _prim.GetName():
            if _prim.GetName() not in _KEEP_PLANT_MESHES:
                UsdGeom.Imageable(_prim).MakeInvisible()

    # Remove all physics from plant
    for _prim in stage.Traverse():
        if not str(_prim.GetPath()).startswith("/World/PlantAssembly"):
            continue
        for api in [UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI,
                    UsdPhysics.RigidBodyAPI, PhysxSchema.PhysxCollisionAPI,
                    PhysxSchema.PhysxRigidBodyAPI]:
            if _prim.HasAPI(api):
                _prim.RemoveAPI(api)
        for attr_name in ["physics:collisionEnabled", "physics:rigidBodyEnabled"]:
            attr = _prim.GetAttribute(attr_name)
            if attr.IsValid(): attr.Clear()
    for _ in range(10): world.step(render=True)

    # ── Pedestal ──
    _ped_cylinder = UsdGeom.Cylinder.Define(stage, "/World/PlantPedestal")
    _ped_cylinder.CreateRadiusAttr(0.08)
    _ped_cylinder.CreateHeightAttr(0.13)
    _ped_cylinder.CreateAxisAttr("Z")
    _ped_xf = UsdGeom.Xformable(_ped_cylinder.GetPrim())
    _ped_xf.ClearXformOpOrder()
    _ped_xf.AddTranslateOp().Set(Gf.Vec3d(
        float(PLANT_TRANSLATE[0]), float(PLANT_TRANSLATE[1]), 0.83 + 0.065))
    _ped_cylinder.CreateDisplayColorAttr([Gf.Vec3f(0.30, 0.18, 0.08)])

    # ── Randomize strawberries ──
    strawberry_configs = []
    attempts = 0
    while len(strawberry_configs) < 15 and attempts < 2000:
        attempts += 1
        sx = np.random.uniform(0.76, 0.81)
        sy = np.random.uniform(-0.20, 0.10)
        sz = np.random.uniform(1.03, 1.33)
        for offset in [0.05, GRASP_OFFSET, PRE_GRASP_OFFSET]:
            pt = np.array([sx + offset, sy, sz])
            if np.linalg.norm(pt - base_pos) > 0.92 * UR5E_REACH:
                break
        else:
            pos = np.array([sx, sy, sz])
            too_close = any(np.linalg.norm(pos - np.array(e)) < 0.075
                           for e in strawberry_configs)
            if not too_close:
                strawberry_configs.append((sx, sy, sz))

    print(f"  Generated {len(strawberry_configs)} strawberries")

    # ── Assign maturity (ripe/yellow/green) ──
    n_total = len(strawberry_configs)
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

    # ── Create strawberry prims ──
    strawberry_positions = []
    for i, (sx, sy, sz) in enumerate(strawberry_configs):
        path = f"/World/Strawberry_{i:02d}"
        prim = stage.DefinePrim(path, "Xform")
        prim.GetReferences().AddReference(STRAWBERRY_MODEL_PATH)
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        for _child in _get_all_descendants(prim):
            try: UsdGeom.Xformable(_child).ClearXformOpOrder()
            except: pass
        xf.AddTranslateOp().Set(Gf.Vec3d(sx, sy, sz))
        xf.AddOrientOp().Set(Gf.Quatf(0.7071068, Gf.Vec3f(0.0, -0.7071068, 0.0)))
        if maturity_map.get(i, "ripe") in ("yellow", "green"):
            _s = _scale_factor * np.random.uniform(0.9, 1.3)  # unripe = smaller
        else:
            _s = _scale_factor * np.random.uniform(1.4, 1.8)  # ripe = normal
        xf.AddScaleOp().Set(Gf.Vec3d(_s, _s, _s))

        UsdPhysics.RigidBodyAPI.Apply(prim)
        prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)
        for _m in _find_meshes_recursive(prim):
            UsdPhysics.CollisionAPI.Apply(_m)
            UsdPhysics.MeshCollisionAPI.Apply(_m).CreateApproximationAttr("convexDecomposition")
        if not _find_meshes_recursive(prim):
            UsdPhysics.CollisionAPI.Apply(prim)
        strawberry_positions.append(np.array([sx, sy, sz]))

    for _ in range(30): world.step(render=True)

    # Materials
    for i in range(len(strawberry_configs)):
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

    # ── Stems ──
    _branch_pts = _get_branch_world_points(stage)
    if _branch_pts is not None and len(_branch_pts) > 0:
        for i, (sx, sy, sz) in enumerate(strawberry_configs):
            berry_top = np.array([sx, sy, sz + STRAWBERRY_TARGET_DIAMETER * 0.5])
            branch_pt, _ = _find_nearest_branch_point(_branch_pts, berry_top)
            _create_stem_curve(stage, f"/World/Stem_{i:02d}", berry_top, branch_pt)
        for i in range(len(strawberry_configs)):
            _sp = stage.GetPrimAtPath(f"/World/Stem_{i:02d}")
            if _sp.IsValid():
                if _sp.HasAPI(UsdPhysics.CollisionAPI): _sp.RemoveAPI(UsdPhysics.CollisionAPI)
                if _sp.HasAPI(UsdPhysics.RigidBodyAPI): _sp.RemoveAPI(UsdPhysics.RigidBodyAPI)
        print(f"  Created {len(strawberry_configs)} stems")

    for _ in range(10): world.step(render=True)
    return strawberry_positions, strawberry_configs, _seed, _plant_rotate_z, maturity_map


# ============================================================
# Initialize CSV log
# ============================================================
print(f"\n Log file: {LOG_FILE}")
with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "round", "seed", "plant_rotate_z", "n_berries",
        "pick_idx", "prim_idx", "berry_x", "berry_y", "berry_z",
        "advance_err", "status", "in_box",
        "round_success_rate", "timestamp"
    ])


# ============================================================
# Main collection loop
# ============================================================
all_round_stats = []

for round_idx in range(NUM_ROUNDS):
    round_t0 = time.time()
    print("\n" + "#"*65)
    print(f"#  ROUND {round_idx + 1} / {NUM_ROUNDS}")
    print("#"*65)

    # Return to HOME
    move_rmpflow(HOME_POS, HOME_QUAT, label="HOME-reset", max_steps=400)
    for _ in range(30): world.step(render=True)

    # Generate new scene
    strawberry_positions, strawberry_configs, round_seed, plant_rz, maturity_map = \
        generate_scene(stage, round_idx + 1)

    n_berries = len(strawberry_positions)
    print(f"\n  Scene ready: {n_berries} berries")

    # Sort by pick order
    # Only pick ripe (red) strawberries, skip yellow/green
    _indexed = [(i, pos) for i, pos in enumerate(strawberry_positions)
                if maturity_map.get(i, "ripe") == "ripe"]
    _indexed.sort(key=lambda t: (-t[1][1], -t[1][2]))
    pick_order = [(pick_i, orig_i, pos) for pick_i, (orig_i, pos) in enumerate(_indexed)]
    n_ripe = len(pick_order)
    print(f"  Picking {n_ripe} ripe strawberries (skipping unripe)")

    for pick_i, orig_i, pos in pick_order:
        print(f"  Pick#{pick_i}: S{orig_i:02d} ({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")

    # Warmup motion
    if len(pick_order) > 0:
        _, _, _first_pos = pick_order[0]
        _first_pre = np.array([_first_pos[0] + PRE_GRASP_OFFSET, _first_pos[1], _first_pos[2]])
        move_rmpflow(_first_pre, HOME_QUAT, label="warmup")
        for _ in range(30): world.step(render=True)

    # ── Pick loop ──
    results = []

    for pick_idx, (_, orig_idx, berry_pos) in enumerate(pick_order):
        bx, by, bz = berry_pos
        prim_idx = orig_idx
        print(f"\n{'='*55}")
        print(f"R{round_idx+1} Pick#{pick_idx} (S{prim_idx:02d}) ({bx:.3f},{by:.3f},{bz:.3f})")
        print(f"{'='*55}")

        # Reset arm to fixed HOME joint angles before each episode
        # This prevents wrist joint drift across episodes
        # v3: smooth blend instead of instant teleport to avoid jump
        _cur_js = robot.get_joint_positions().copy()
        _target_js = _cur_js.copy()
        _target_js[ur5e_idx] = HOME_JOINT_ANGLES[:6]
        _target_js[finger_idx] = HOME_JOINT_ANGLES[6]
        _target_js[rok_idx] = HOME_JOINT_ANGLES[7]
        _RESET_BLEND_STEPS = 40
        for _bi in range(_RESET_BLEND_STEPS):
            _alpha = (_bi + 1) / _RESET_BLEND_STEPS
            _blended = _cur_js * (1 - _alpha) + _target_js * _alpha
            robot.set_joint_positions(_blended)
            set_target_deg(finger_joint_prim, GRIPPER_OPEN_DEG)
            set_target_deg(right_outer_knuckle_prim, GRIPPER_OPEN_DEG)
            world.step(render=(_bi % 4 == 0))
        # settle
        for _ in range(20): world.step(render=True)

        # [VLA] Start recording this episode
        recorder.start_episode(
            round_idx + 1, pick_idx,
            prompt="pick the ripe strawberry and place it in the box"
        )

        # [1] open gripper
        set_gripper_for_steps(GRIPPER_OPEN_DEG, 60)

        # [2] pre-grasp
        pre_pos = np.array([bx + PRE_GRASP_OFFSET, by, bz])
        move_rmpflow(pre_pos, HOME_QUAT, label="pre-grasp")

        # [3] advance
        grasp_pos = np.array([bx + GRASP_OFFSET, by, bz])
        final = move_rmpflow(grasp_pos, HOME_QUAT, label="advance", tol=0.005)
        err_adv = np.linalg.norm(grasp_pos - final)
        for _ in range(30): world.step(render=True)

        capture_images(cam1, cam2, cam3, world, round_idx * 100 + pick_idx * 10)

        # Advance failure check
        if err_adv > ADVANCE_ERR_THRESHOLD:
            print(f"  ADVANCE ERR {err_adv:.4f}m -> FAIL")
            recorder.mark_anomaly(f"advance err={err_adv:.4f}m")  # [VLA]
            berry_path = f"/World/Strawberry_{prim_idx:02d}"
            berry_prim = stage.GetPrimAtPath(berry_path)
            ka = berry_prim.GetAttribute("physics:kinematicEnabled")
            if ka.IsValid(): ka.Set(False)
            for _ in range(90): world.step(render=True)
            hard_reset_gripper()
            if pick_idx + 1 < len(pick_order):
                _, _, npos = pick_order[pick_idx + 1]
                move_rmpflow(np.array([npos[0]+PRE_GRASP_OFFSET, npos[1], npos[2]]),
                             HOME_QUAT, label="R1-pregrasp")
            results.append((prim_idx, bx, by, bz, err_adv, "FAIL", False))
            recorder.end_episode(success=False)  # [VLA]
            continue

        # [4] close + attach
        f_deg, r_deg = set_gripper_for_steps(GRIPPER_CLOSE_DEG, STEPS_CLOSE)
        print(f"      finger={f_deg:.1f}, rok={r_deg:.1f}")

        berry_path = f"/World/Strawberry_{prim_idx:02d}"
        berry_xf = XFormPrim(berry_path)
        berry_prim = stage.GetPrimAtPath(berry_path)
        berry_pos_now, _ = berry_xf.get_world_pose()
        w3p, w3q = [np.array(x, np.float64) for x in XFormPrim(WRIST_3_LINK).get_world_pose()]
        berry_offset = rotate_vec_by_quat(np.array(berry_pos_now, np.float64) - w3p, quat_inverse(w3q))

        # Disable collision
        for _bm in _find_meshes_recursive(berry_prim):
            if _bm.HasAPI(UsdPhysics.CollisionAPI):
                _bm.GetAttribute("physics:collisionEnabled").Set(False)
        if berry_prim.HasAPI(UsdPhysics.CollisionAPI):
            berry_prim.GetAttribute("physics:collisionEnabled").Set(False)

        # [5] retract
        move_rmpflow_with_berry(np.array([RX, by, bz]), HOME_QUAT,
                                berry_xf, berry_offset, label="retract")

        # [6] above box
        move_rmpflow_with_berry(ABOVE_BOX, HOME_QUAT,
                                berry_xf, berry_offset, label="above-box")

        # [7] release — re-enable collision
        for _bm in _find_meshes_recursive(berry_prim):
            if _bm.HasAPI(UsdPhysics.CollisionAPI):
                _bm.GetAttribute("physics:collisionEnabled").Set(True)
        if berry_prim.HasAPI(UsdPhysics.CollisionAPI):
            berry_prim.GetAttribute("physics:collisionEnabled").Set(True)

        set_gripper_for_steps(GRIPPER_OPEN_DEG, 15)
        ka = berry_prim.GetAttribute("physics:kinematicEnabled")
        if ka.IsValid(): ka.Set(False)

        move_rmpflow(np.array([BOX_X, BOX_Y, BOX_Z + 0.15]), HOME_QUAT, label="lift-off")

        for _ in range(60): world.step(render=True)
        bpos_final, _ = berry_xf.get_world_pose()
        bpos_final = np.array(bpos_final, np.float64)

        in_x = (box_x - box_w/2 - 0.02) <= bpos_final[0] <= (box_x + box_w/2 + 0.02)
        in_y = (box_y - box_d/2 - 0.02) <= bpos_final[1] <= (box_y + box_d/2 + 0.02)
        in_z = box_z_base - 0.01 <= bpos_final[2] <= box_z_base + box_h + 0.05
        in_box = in_x and in_y and in_z
        print(f"      berry final: {np.round(bpos_final,3)} -> {'IN BOX' if in_box else 'MISSED'}")

        # [8] return to HOME (always, so every episode ends at HOME)
        hard_reset_gripper()
        move_rmpflow(HOME_POS, HOME_QUAT, label="return-HOME")

        results.append((prim_idx, bx, by, bz, err_adv, "DONE", in_box))
        recorder.end_episode(success=in_box)  # [VLA]
        print(f"  S{prim_idx:02d}: DONE")

    # ── Round statistics ──
    n_total = len(results)
    n_inbox = sum(1 for r in results if r[6])
    n_fail = sum(1 for r in results if r[5] == "FAIL")
    success_rate = n_inbox / n_total if n_total > 0 else 0.0
    round_time = time.time() - round_t0

    print(f"\n{'='*55}")
    print(f"ROUND {round_idx+1}: {n_inbox}/{n_total} in box, "
          f"{n_fail} fail, rate={success_rate*100:.1f}%, time={round_time:.0f}s")
    print(f"{'='*55}")
    for pidx, bx, by, bz, err, status, inbox in results:
        tag = "BOX" if inbox else ("FAIL" if status == "FAIL" else "MISS")
        print(f"  S{pidx:02d}: {status:4s} err={err:.4f}m {tag}")

    all_round_stats.append({
        "round": round_idx + 1, "seed": round_seed, "plant_rz": plant_rz,
        "n_berries": n_total, "n_inbox": n_inbox, "n_fail": n_fail,
        "success_rate": success_rate, "time": round_time,
    })

    # Write to CSV
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        for pi, (pidx, bx, by, bz, err, status, inbox) in enumerate(results):
            writer.writerow([
                round_idx+1, round_seed, f"{plant_rz:.1f}", n_total,
                pi, pidx, f"{bx:.4f}", f"{by:.4f}", f"{bz:.4f}",
                f"{err:.4f}", status, inbox, f"{success_rate:.3f}", ts
            ])

    # Cleanup
    cleanup_round(stage, n_total)
    print(f"  Scene cleaned")


# ============================================================
# Global summary
# ============================================================
print("\n" + "="*65)
print("GLOBAL SUMMARY")
print("="*65)
total_attempted = 0
total_inbox = 0
total_fail = 0
for rs in all_round_stats:
    r = rs["round"]
    print(f"  Round {r}: seed={rs['seed']}, rz={rs['plant_rz']:.1f}, "
          f"{rs['n_inbox']}/{rs['n_berries']} inbox, "
          f"fail={rs['n_fail']}, rate={rs['success_rate']*100:.1f}%, "
          f"time={rs['time']:.0f}s")
    total_attempted += rs["n_berries"]
    total_inbox += rs["n_inbox"]
    total_fail += rs["n_fail"]

overall_rate = total_inbox / total_attempted if total_attempted > 0 else 0
print(f"\n  TOTAL: {total_attempted} attempted, {total_inbox} inbox, "
      f"{total_fail} failed, overall={overall_rate*100:.1f}%")
print(f"  Log: {LOG_FILE}")
print("="*65)

# [VLA] Save global metadata
recorder.save_global_metadata()

while simulation_app.is_running():
    world.step(render=True)
simulation_app.close()
