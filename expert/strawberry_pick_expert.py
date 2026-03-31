"""
Strawberry Picking — RMPFlow Expert Controller
================================================
Rule-based expert that picks strawberries one-by-one using RMPFlow
motion planning with IK fallback and pseudo-grasp mechanics.
Generates visual stems (BasisCurves) for each strawberry.

Usage:
  $ISAAC_SIM_DIR/python.sh expert/strawberry_pick_expert.py
"""

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False})

import numpy as np
import omni
import omni.kit.app
from PIL import Image
import os

from pxr import UsdPhysics, PhysxSchema, Sdf, Gf, UsdGeom, UsdShade
from omni.isaac.core import World
from omni.isaac.core.prims import XFormPrim
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.stage import open_stage, get_current_stage
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.sensor import Camera

# ===== Configuration =====
# ===== PATHS (auto-resolved from project structure) =====
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(_THIS_DIR)  # strawberry_vla_pro/
ISAAC_SIM_DIR = os.environ.get("ISAAC_SIM_DIR", "/path/to/isaac-sim")

SCENE_PATH = os.path.join(BASE_DIR, "scene", "scene_assembled_manual.usd")
FIXED_SCENE_PATH = os.path.join(_THIS_DIR, "scene_assembled_fixed.usd")
SAVE_DIR = os.path.join(_THIS_DIR, "captures")
STRAWBERRY_MODEL_PATH = os.path.join(BASE_DIR, "strawberry", "Strawberry_gltf.gltf")
STRAWBERRY_TARGET_DIAMETER = 0.03

# Plant model
PLANT_USD_PATH = os.path.join(BASE_DIR, "plant", "ficus_obj.usd")
PLANT_TRANSLATE = Gf.Vec3d(0.65, -0.1, 1.35) # v9.84: (x, y, z)
PLANT_ROTATE_X = 90.0                           # AlongXaxis rotationangle
PLANT_TARGET_HEIGHT = 0.5 # degrees (meters)

ROBOT_PRIM_PATH = "/World/UR5e"
GRIPPER_ROOT = "/World/Robotiq_2F_140_physics_edit"
GRIPPER_BASE_LINK = f"{GRIPPER_ROOT}/robotiq_base_link"
WRIST_3_LINK = "/World/UR5e/wrist_3_link"
FINGER_JOINT = f"{GRIPPER_ROOT}/finger_joint"
RIGHT_OUTER_KNUCKLE_JOINT = f"{GRIPPER_ROOT}/right_outer_knuckle_joint"

CAM1_PATH = "/World/Cam1"
CAM2_PATH = "/World/Cam2"
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480

GRIPPER_OPEN_DEG = 30.0
GRIPPER_CLOSE_DEG = 45.0

# HOME
# Ry(-90deg) * Rz(90deg) -> gripper facing -X
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

#
PRE_GRASP_OFFSET = 0.25   # v9.5: x+0.20
GRASP_OFFSET = 0.20       # v9.84: x+0.20

# Place box — v8.2
# : x>=1.1 (dx>=-0.4) ; x=1.0 + y>0
# : dx=-0.5 + dy>0 UR5e wrist singularity
# : (1.1, 0.25) → dx=-0.4, dy=+0.25,
BOX_CENTER = np.array([1.1, 0.25, 0.85])

# ── Motion control parameters ── v9.8:
POS_TOLERANCE    = 0.008   # position convergence threshold (m)，below this = reached target
STALL_TOLERANCE  = 0.0002  # per-step displacement below this means RMPFlow stalled
MAX_MOVE_STEPS   = 400     # max steps per motion segment（prevent infinite loop）
MIN_MOVE_STEPS   = 5       # v9.8: 30→5，avoid overshooting after reaching target
WAYPOINT_SPACING = 0.15    # v9.8: 0.08→0.15m，reduce waypoint count
RENDER_EVERY     = 4       # v9.8: render every N steps，physicsphysics runs normally（3-4xspeedup）
IK_MAX_JOINT_DELTA = 0.5 # v9.8: IK fallback disable (rad)
ADVANCE_ERR_THRESHOLD = 0.01  # advance error exceeds this → grasp failure


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
    """v9.8: high stiffness drive for UR5e - reduce oscillation, faster response."""
    d = ensure_drive(jp)
    d.CreateTypeAttr("force")
    d.CreateStiffnessAttr(stiff)
    d.CreateDampingAttr(damp)
    d.CreateMaxForceAttr(mf)
    PhysxSchema.PhysxJointAPI.Apply(jp).CreateMaxJointVelocityAttr(3.14)  # ~180 deg/s speed limit

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


# ===== =====
def setup_cameras():
    c1=Camera(prim_path=CAM1_PATH, resolution=(IMAGE_WIDTH,IMAGE_HEIGHT))
    c2=Camera(prim_path=CAM2_PATH, resolution=(IMAGE_WIDTH,IMAGE_HEIGHT))
    c1.initialize(); c2.initialize()
    return c1, c2

def capture_images(c1, c2, world, step_id=0):
    for _ in range(3): world.step(render=True)
    r1,r2 = c1.get_rgba(), c2.get_rgba()
    i1 = r1[:,:,:3] if r1 is not None else None
    i2 = r2[:,:,:3] if r2 is not None else None
    if i1 is not None and i2 is not None:
        os.makedirs(SAVE_DIR, exist_ok=True)
        Image.fromarray(i1).save(f"{SAVE_DIR}/step{step_id:04d}_cam1.png")
        Image.fromarray(i2).save(f"{SAVE_DIR}/step{step_id:04d}_cam2.png")
    return i1, i2


# ============================================================
# PASS 1: Fix joint（same as v4/v7）
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

# ── v9: Remove pre-existing strawberries from scene，ensure clean scene for PASS2 ──
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

# v9.8: Set high stiffness for UR5e 6 joints drive（reduce oscillation + faster response）
print("--- Configuring UR5e arm drives (high stiffness) ---")
for jname in UR5E_JOINTS:
    jp_path = f"{ROBOT_PRIM_PATH}/{jname}"
    jp = stage.GetPrimAtPath(jp_path)
    if jp.IsValid():
        configure_arm_drive(jp)
        print(f"  ✅ {jname}: stiff=1e4, damp=1e2")
    else:
        # Joint may be nested under link，try searching all links
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

# ── v8: ──
UR5E_REACH = 0.85
SAFE_RATIO = 0.82 # 82% reach →
def check_reach(name, pos):
    d = np.linalg.norm(np.array(pos) - base_pos)
    ratio = d / UR5E_REACH
 tag = "✅" if ratio < SAFE_RATIO else ("⚠️ " if ratio < 0.92 else "❌ ")
    print(f"  {tag} {name}: dist={d:.3f}m ({ratio*100:.1f}% reach)")
    return ratio < 0.92

print("--- Workspace safety check ---")
# disable
check_reach("HOME", HOME_POS)
check_reach("ABOVE_BOX", [1.1, 0.25, 0.95])
check_reach("LIFT_OFF(1.1,0.25,1.1)", [1.1, 0.25, 1.1])
check_reach("RETRACT(1.2,0.25,0.95)", [1.2, 0.25, 0.95])

cam1, cam2 = setup_cameras()
for _ in range(20): world.step(render=True)
print("✅ Cameras ready")


# ============================================================
# — v8
# ============================================================

def get_ee_pos():
    """Multi-step motion."""
    pos, _ = art_kin.compute_end_effector_pose()
    return np.array(pos, np.float64)


def generate_waypoints(start, end, spacing=WAYPOINT_SPACING):
    """
 start→end spacing 。
 <= spacing， [end]。
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
 RMPFlow UR5e 6 disable。
 v9.8: render GPU （physics ）
 True RMPFlow 。
    """
    actions = art_rmpflow.get_next_articulation_action()
    valid = False
    if actions.joint_positions is not None:
        raw = np.array(actions.joint_positions, np.float64)
        arm = raw[:6].copy()
        if not np.any(np.isnan(arm)):
            robot.apply_action(ArticulationAction(
                joint_positions=arm, joint_indices=ur5e_idx))
            valid = True
    world.step(render=render)
    return valid


def move_rmpflow(target_pos, target_quat, label="",
                 tol=POS_TOLERANCE, max_steps=MAX_MOVE_STEPS,
                 min_steps=MIN_MOVE_STEPS):
    """
 RMPFlow Smooth motion。v9.8 ：
 1. （0.15m），
 2. RENDER_EVERY (physics steps all run)
 3. min_steps=5，，
 4. IK fallback ：disable>0.5rad
 : EE
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
 # v9.8: N ， physics
            do_render = (step % RENDER_EVERY == 0)
            rmpflow_step_arm(render=do_render)
            steps_this_wp += 1
            total_steps += 1

 # 5 （ min_steps ）
            if steps_this_wp >= min_steps and step % 5 == 0:
                now_pos = get_ee_pos()
                err = np.linalg.norm(wp - now_pos)

 # → under
                if err < tol:
                    break

 # stalled
                movement = np.linalg.norm(now_pos - prev_pos)
                if movement < STALL_TOLERANCE:
                    stall_count += 1
                else:
                    stall_count = 0
                prev_pos = now_pos

 # stalled → IK fallback（）
                if stall_count >= 6:
                    action, success = art_kin.compute_inverse_kinematics(wp, target_quat)
                    if success and action.joint_positions is not None:
                        arm_pos = np.array(action.joint_positions[:6], np.float64)
                        cur_joints = robot.get_joint_positions()[:6]
                        max_delta = np.max(np.abs(arm_pos - cur_joints))

 # v9.8: — disable IK same as
                        if max_delta <= IK_MAX_JOINT_DELTA:
 # IK ， blend
                            for blend_i in range(20):
                                alpha = (blend_i + 1) / 20.0
                                blended = cur_joints * (1 - alpha) + arm_pos * alpha
                                robot.apply_action(ArticulationAction(
                                    joint_positions=blended, joint_indices=ur5e_idx))
                                world.step(render=(blend_i % 4 == 0))
                                total_steps += 1
                        else:
 pass # IK ， RMPFlow

                    stall_count = 0
                    now_pos = get_ee_pos()
                    if np.linalg.norm(wp - now_pos) < tol:
                        break

 # same as
    world.step(render=True)

    final_pos = get_ee_pos()
    err = np.linalg.norm(target_pos - final_pos)
    if label:
        status = "✅" if err < tol * 2 else "⚠️"
        print(f"      {status} {label}: err={err:.4f}m ({total_steps} steps)")
    return final_pos


def move_rmpflow_with_berry(target_pos, target_quat, berry_xf, berry_offset,
                            label="", tol=POS_TOLERANCE, max_steps=MAX_MOVE_STEPS):
    """
 RMPFlow Smooth motion + follow。v9.8: same as move_rmpflow optimization。
    Berry follows every step wrist_3_link（including non-render steps，ensures physics consistency）。
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

 # Update（）
            w3p, w3q = [np.array(x, np.float64) for x in XFormPrim(WRIST_3_LINK).get_world_pose()]
            berry_xf.set_world_pose(
                position=w3p + rotate_vec_by_quat(berry_offset, w3q),
                orientation=np.array([1,0,0,0], dtype=np.float64))

 # 5
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

 #
    world.step(render=True)

    final_pos = get_ee_pos()
    err = np.linalg.norm(target_pos - final_pos)
    if label:
        status = "✅" if err < tol * 2 else "⚠️"
        print(f"      {status} {label}: err={err:.4f}m ({total_steps} steps)")
    return final_pos


def set_gripper_for_steps(deg, n_steps):
    """
 gripper (v7 dual enforcement).
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
        world.step(render=True)

    js = robot.get_joint_positions()
    f = np.degrees(js[finger_idx])
    r = np.degrees(js[rok_idx])
    delta = abs(f - r)
    if delta > 3.0:
        print(f"    ⚠️ Gripper asymmetry Δ={delta:.1f}°, re-enforcing...")
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
    """Set gripper position with enforcement (v7 dual-check)."""
    open_rad = np.radians(GRIPPER_OPEN_DEG)
    gripper_indices = np.array([finger_idx, rok_idx], dtype=int)
    gripper_targets = np.array([open_rad, open_rad], dtype=np.float64)

    for attempt in range(5):
        all_joints = robot.get_joint_positions().copy()
        all_joints[finger_idx] = open_rad
        all_joints[rok_idx] = open_rad
        for gi in range(6, n_dofs):
            if gi != finger_idx and gi != rok_idx:
                all_joints[gi] = 0.0
        robot.set_joint_positions(all_joints)
        set_target_deg(finger_joint_prim, GRIPPER_OPEN_DEG)
        set_target_deg(right_outer_knuckle_prim, GRIPPER_OPEN_DEG)
        robot.apply_action(ArticulationAction(
            joint_positions=gripper_targets, joint_indices=gripper_indices))
        for _ in range(20): world.step(render=True)

        js = robot.get_joint_positions()
        f = np.degrees(js[finger_idx])
        r = np.degrees(js[rok_idx])
        delta = abs(f - r)
        if delta < 5.0 and abs(f - GRIPPER_OPEN_DEG) < 10.0:
            print(f"      gripper reset OK: f={f:.1f}°, r={r:.1f}° (attempt {attempt+1})")
            return
    print(f"      ⚠️ gripper reset incomplete: f={f:.1f}°, r={r:.1f}°")


# ===== HOME =====
print("\n--- MOVE TO HOME ---")
# RMPFlow HOME
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
# v9.83: Import plant model
# ============================================================
print("\n--- IMPORTING PLANT ---")

# （ check ）
_PLANT_CENTER = Gf.Vec3d(311.5, 2252.9, 2504.4) # bbox，
_PLANT_MAX_DIM = 854.2
_PLANT_SCALE = PLANT_TARGET_HEIGHT / _PLANT_MAX_DIM

# mesh
_KEEP_PLANT_MESHES = {
    "ficus_lyrata_054", "ficus_lyrata_053", "ficus_lyrata_052",
    "ficus_lyrata_051", "ficus_lyrata_05", "ficus_lyrata_061",
    "ficus_lyrata_06", "ficus_lyrata_07",
}

# /World/PlantAssembly
_pa_prim = stage.DefinePrim("/World/PlantAssembly", "Xform")
_pa_xf = UsdGeom.Xformable(_pa_prim)
_pa_xf.ClearXformOpOrder()

# Z axis rotation
_plant_rotate_z = float(np.random.uniform(0, 360))
print(f"  Plant random Z rotation: {_plant_rotate_z:.1f}°")

# : translate → rotateZ(random) → rotateX → scale
_pa_xf.AddTranslateOp().Set(PLANT_TRANSLATE)
_pa_xf.AddRotateZOp().Set(_plant_rotate_z)
_pa_xf.AddRotateXOp().Set(PLANT_ROTATE_X)
_pa_xf.AddScaleOp().Set(Gf.Vec3f(_PLANT_SCALE, _PLANT_SCALE, _PLANT_SCALE))

#
_ficus_prim = stage.DefinePrim("/World/PlantAssembly/ficus", "Xform")
_ficus_prim.GetReferences().AddReference(PLANT_USD_PATH)

# :
_ficus_xf = UsdGeom.Xformable(_ficus_prim)
_ficus_xf.ClearXformOpOrder()
_ficus_xf.AddTranslateOp().Set(_PLANT_CENTER)

#
for _ in range(60): world.step(render=True)

# mesh（ stage ficus under Mesh）
_hidden = 0
_kept = 0
for _prim in stage.Traverse():
    if _prim.GetTypeName() == "Mesh" and "ficus_lyrata" in _prim.GetName():
        if _prim.GetName() not in _KEEP_PLANT_MESHES:
            UsdGeom.Imageable(_prim).MakeInvisible()
            _hidden += 1
        else:
            _kept += 1

for _ in range(30): world.step(render=True)
print(f"  Plant loaded: kept {_kept} meshes, hidden {_hidden}")
print(f"  Transform: translate{PLANT_TRANSLATE} rotateZ({_plant_rotate_z:.1f}°) rotateX({PLANT_ROTATE_X}°) scale({_PLANT_SCALE:.6f})")

# ── v9.84: Physics properties， ──
# USD CollisionAPI / RigidBodyAPI，
# /
print("  Stripping physics from plant...")
_stripped = 0
for _prim in stage.Traverse():
    _prim_path_str = str(_prim.GetPath())
    if not _prim_path_str.startswith("/World/PlantAssembly"):
        continue

 # CollisionAPI
    if _prim.HasAPI(UsdPhysics.CollisionAPI):
        _prim.RemoveAPI(UsdPhysics.CollisionAPI)
        _stripped += 1

 # MeshCollisionAPI
    if _prim.HasAPI(UsdPhysics.MeshCollisionAPI):
        _prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)

 # RigidBodyAPI
    if _prim.HasAPI(UsdPhysics.RigidBodyAPI):
        _prim.RemoveAPI(UsdPhysics.RigidBodyAPI)

 # PhysxCollisionAPI ()
    if _prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
        _prim.RemoveAPI(PhysxSchema.PhysxCollisionAPI)

 # PhysxRigidBodyAPI ()
    if _prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        _prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)

 # physics
    for attr_name in ["physics:collisionEnabled", "physics:rigidBodyEnabled"]:
        attr = _prim.GetAttribute(attr_name)
        if attr.IsValid():
            attr.Clear()

print(f"  Stripped collision from {_stripped} plant prims")
for _ in range(10): world.step(render=True)

# ── v9.84: underup（）──
_pedestal_path = "/World/PlantPedestal"
_old_ped = stage.GetPrimAtPath(_pedestal_path)
if _old_ped.IsValid():
    stage.RemovePrim(_pedestal_path)

_ped_radius = 0.08
_ped_height = 0.13
# PLANT_TRANSLATE xy ，z (0.80)
_ped_x = float(PLANT_TRANSLATE[0])
_ped_y = float(PLANT_TRANSLATE[1])
_ped_z_base = 0.80 # degrees

_ped_cylinder = UsdGeom.Cylinder.Define(stage, _pedestal_path)
_ped_cylinder.CreateRadiusAttr(_ped_radius)
_ped_cylinder.CreateHeightAttr(_ped_height)
_ped_cylinder.CreateAxisAttr("Z")
_ped_xf = UsdGeom.Xformable(_ped_cylinder.GetPrim())
_ped_xf.ClearXformOpOrder()
# z_base + height/2
_ped_xf.AddTranslateOp().Set(Gf.Vec3d(_ped_x, _ped_y, _ped_z_base + _ped_height / 2.0))
# （）
_ped_cylinder.CreateDisplayColorAttr([Gf.Vec3f(0.30, 0.18, 0.08)])
# ，collision（）
for _ in range(5): world.step(render=True)
print(f"  🪴 Pedestal at ({_ped_x:.2f}, {_ped_y:.2f}, {_ped_z_base:.2f}), r={_ped_radius}m, h={_ped_height}m")

print("✅ Plant imported (no physics, robot-transparent)\n")

# ── v9: Robot arm ready，spawning strawberries now ──
print("\n✅ Robot at HOME. Spawning strawberries now...")

# ===== Creating strawberries =====
print("\n--- CREATING STRAWBERRIES ---")
for i in range(20):
    sp = f"/World/Strawberry_{i:02d}"
    p = stage.GetPrimAtPath(sp)
    if p.IsValid(): stage.RemovePrim(sp)

# ── v9.2: Randomly generate strawberries，ensure each is within robot safe workspace ──
# : x∈[0.65,0.85], y∈[0,0.4], z∈[1.0,1.3]
# : (x+0.05, y, z) UR5e reach 92%
# >= 0.07m
# < 16
_seed = np.random.randint(0, 100000)
np.random.seed(_seed)
print(f"  Random seed: {_seed}")
MAX_BERRIES = 15
MIN_DIST = 0.075
REACH_CHECK_RATIO = 0.92 # (x+0.05,y,z) base < 0.92*0.85

strawberry_configs = []
attempts = 0
while len(strawberry_configs) < MAX_BERRIES and attempts < 2000:
    attempts += 1
    sx = np.random.uniform(0.74, 0.80)
    sy = np.random.uniform(-0.25, 0.05)
    sz = np.random.uniform(1.0, 1.3)

 # (x+0.05, y, z)
    check_pt = np.array([sx + 0.05, sy, sz])
    dist_to_base = np.linalg.norm(check_pt - base_pos)
    if dist_to_base > REACH_CHECK_RATIO * UR5E_REACH:
        continue

 # grasp (x+0.15, y, z)
    grasp_pt = np.array([sx + GRASP_OFFSET, sy, sz])
    if np.linalg.norm(grasp_pt - base_pos) > REACH_CHECK_RATIO * UR5E_REACH:
        continue

 # pre-grasp (x+0.25, y, z)
    pre_pt = np.array([sx + PRE_GRASP_OFFSET, sy, sz])
    if np.linalg.norm(pre_pt - base_pos) > REACH_CHECK_RATIO * UR5E_REACH:
        continue

 #
    pos = np.array([sx, sy, sz])
    too_close = False
    for existing in strawberry_configs:
        if np.linalg.norm(pos - np.array(existing)) < MIN_DIST:
            too_close = True
            break
    if too_close:
        continue

    strawberry_configs.append((sx, sy, sz))

print(f"  Generated {len(strawberry_configs)} strawberries ({attempts} attempts)")

strawberry_positions = []

# ── v9.8: GLTF Strawberry model + texture ──
# : GLTF > FBX，GLTF texture，
# : ， apply UsdPreviewSurface

def _get_all_descendants(prim):
    """Recursively get all descendants of a prim."""
    result = []
    for child in prim.GetAllChildren():
        result.append(child)
        result.extend(_get_all_descendants(child))
    return result

def _apply_fallback_strawberry_material(prim):
    """Apply red fallback material to strawberry meshes without material."""
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
 # does not have mesh，
    if bound_count == 0 and prim.IsA(UsdGeom.Mesh):
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
        bound_count = 1
    return bound_count

def _has_any_material(prim):
    """Recursively check prim and descendants for any material binding"""
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
    """Recursively find all Mesh prim"""
    meshes = []
    if prim.IsA(UsdGeom.Mesh):
        meshes.append(prim)
    for child in prim.GetAllChildren():
        meshes.extend(_find_meshes_recursive(child))
    return meshes

# Step 1: Load template and measure bounding box
_template_path = "/World/_StrawberryTemplate"
_tp = stage.GetPrimAtPath(_template_path)
if _tp.IsValid():
    stage.RemovePrim(_template_path)

_template_prim = stage.DefinePrim(_template_path, "Xform")
_template_prim.GetReferences().AddReference(STRAWBERRY_MODEL_PATH)
for _ in range(20): world.step(render=True) # GLTF

# Compute bounding box → scale factor
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
    print("  ⚠️ Could not compute bounding box, using scale=1.0")

# Check if template has material
_template_has_mat = _has_any_material(_template_prim)
print(f"  Model bbox: {np.round(_bb_size, 4)}m, max_extent={_max_extent:.4f}m")
print(f"  Target diameter: {STRAWBERRY_TARGET_DIAMETER}m → scale={_scale_factor:.6f}")
print(f"  Material detected in GLTF: {'✅ YES' if _template_has_mat else '❌ NO → will apply red fallback'}")

# List found mesh （for debugging）
_meshes = _find_meshes_recursive(_template_prim)
print(f"  Meshes in model: {len(_meshes)}")
for _m in _meshes[:5]:
    print(f"    {_m.GetPath()}")

stage.RemovePrim(_template_path)

# Step 2: Create each strawberry
for i, (sx, sy, sz) in enumerate(strawberry_configs):
    path = f"/World/Strawberry_{i:02d}"

    prim = stage.DefinePrim(path, "Xform")
    prim.GetReferences().AddReference(STRAWBERRY_MODEL_PATH)

 # Clear model built-in xformOps（ + ），reset
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    # Also clear all children xformOps，prevent GLTF internal transforms from overriding our scale
    for _child in _get_all_descendants(prim):
        _child_xf = UsdGeom.Xformable(_child)
        if _child_xf:
            try:
                _child_xf.ClearXformOpOrder()
            except:
                pass

    # Re-apply translate + rotateY(-90deg) + scale (random 1x-1.4x enlargement)
    xf.AddTranslateOp().Set(Gf.Vec3d(sx, sy, sz))
    # Y-axis rotation -90deg: quat = (cos(-45deg), 0, sin(-45deg), 0) = (sqrt(2)/2, 0, -sqrt(2)/2, 0)
    xf.AddOrientOp().Set(Gf.Quatf(0.7071068, Gf.Vec3f(0.0, -0.7071068, 0.0)))
    _random_scale_multiplier = np.random.uniform(1.4, 1.8)
    _final_scale = _scale_factor * _random_scale_multiplier
    xf.AddScaleOp().Set(Gf.Vec3d(_final_scale, _final_scale, _final_scale))
    if i == 0:
        print(f"  📐 Base scale factor: {_scale_factor:.6f} (target {STRAWBERRY_TARGET_DIAMETER}m from {_max_extent:.4f}m)")
        print(f"      + Y-axis rotation: -90°, random size multiplier: {_random_scale_multiplier:.3f}x")
    else:
        print(f"  📐 S{i}: scale_mult={_random_scale_multiplier:.3f}x → final_scale={_final_scale:.6f}")

    # Physics properties
    UsdPhysics.RigidBodyAPI.Apply(prim)
    prim.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)

 # mesh collision
    _found_mesh = False
    for _m in _find_meshes_recursive(prim):
        UsdPhysics.CollisionAPI.Apply(_m)
        mesh_col = UsdPhysics.MeshCollisionAPI.Apply(_m)
        mesh_col.CreateApproximationAttr("convexDecomposition")
        _found_mesh = True

    if not _found_mesh:
        UsdPhysics.CollisionAPI.Apply(prim)
        print(f"    ⚠️ S{i}: no mesh found, collision on root prim")

    spos = np.array([sx, sy, sz])
    grasp_dist = np.linalg.norm(np.array([sx + GRASP_OFFSET, sy, sz]) - base_pos)
    strawberry_positions.append(spos)
    print(f"  🍓 S{i} ({sx:.3f},{sy:.3f},{sz:.3f}) grasp_dist={grasp_dist:.3f}m ✅")

#
for _ in range(30): world.step(render=True)

# v9.8: ，does not haveup
if not _template_has_mat:
    print("\n--- Applying fallback red material to all strawberries ---")
    for i in range(len(strawberry_configs)):
        path = f"/World/Strawberry_{i:02d}"
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            n_bound = _apply_fallback_strawberry_material(prim)
            print(f"  🎨 S{i}: fallback material bound to {n_bound} mesh(es)")
    for _ in range(10): world.step(render=True)
else:
 # GLTF ，
    print("\n--- Verifying strawberry materials ---")
    _n_missing = 0
    for i in range(len(strawberry_configs)):
        path = f"/World/Strawberry_{i:02d}"
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid() and not _has_any_material(prim):
            n_bound = _apply_fallback_strawberry_material(prim)
            print(f"  🎨 S{i}: material missing, applied fallback ({n_bound} meshes)")
            _n_missing += 1
    if _n_missing == 0:
        print("  ✅ All strawberries have materials from GLTF")
    for _ in range(10): world.step(render=True)

# ============================================================
# v9.84: Generate strawberry stems (stem) — green curve from strawberry top to nearest branch point
# ============================================================
print("\n--- CREATING STRAWBERRY STEMS ---")

# ── Get branch ficus_lyrata_05 world spacevertices ──
def _get_branch_world_points(stage):
    """
 ficus_lyrata_05 mesh vertices，Transform to world coordinates。
 Nx3 numpy array。
    """
    branch_prim = None
    for prim in stage.Traverse():
        if prim.GetName() == "ficus_lyrata_05" and prim.GetTypeName() == "Mesh":
            branch_prim = prim
            break

    if branch_prim is None:
        print("  ⚠️ ficus_lyrata_05 not found!")
        return None

    mesh = UsdGeom.Mesh(branch_prim)
    points_attr = mesh.GetPointsAttr()
    if not points_attr:
        print("  ⚠️ No points attribute on branch mesh!")
        return None

    local_pts = np.array(points_attr.Get(), dtype=np.float64)  # Nx3

 # world transform
    xf_cache = UsdGeom.XformCache(0)
    world_xf = xf_cache.GetLocalToWorldTransform(branch_prim)
    mat = np.array(world_xf, dtype=np.float64)  # 4x4 row-major

    # Transform to world coordinates: p_world = p_local * mat (USD convention: row vectors)
    ones = np.ones((local_pts.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([local_pts, ones])  # Nx4
    pts_world = pts_h @ mat  # Nx4
    pts_world = pts_world[:, :3]  # Nx3

    print(f"  Branch '{branch_prim.GetPath()}': {len(pts_world)} vertices")
    print(f"    World bbox: min={np.round(pts_world.min(axis=0), 3)}, max={np.round(pts_world.max(axis=0), 3)}")
    return pts_world


def _find_nearest_branch_point(branch_pts, target, min_z_above=0.20):
    """
 vertices target ，
 z >= target[2] + min_z_above（stemupunder）。
 does not have， z 。
    """
 # z >= target_z + min_z_above
    z_threshold = target[2] + min_z_above
    mask = branch_pts[:, 2] >= z_threshold
    if np.any(mask):
        candidates = branch_pts[mask]
    else:
 # does not have， z 10%
        z_sorted_idx = np.argsort(branch_pts[:, 2])[::-1]
        top_n = max(1, len(branch_pts) // 10)
        candidates = branch_pts[z_sorted_idx[:top_n]]

    diffs = candidates - target
    dists = np.linalg.norm(diffs, axis=1)
    idx = np.argmin(dists)
    return candidates[idx], dists[idx]


def _create_stem_curve(stage, stem_path, start_pos, end_pos, num_segments=12, radius=0.0015):
    """
 start_pos () end_pos () understem。
 BasisCurves prim，visual only, no physics。

 （under）:
 - end_pos () up，start_pos () under
 - stem，under/
 - ctrl1: under+
 - ctrl2: up, curves down to strawberry
    """
 # Horizontal distanceVertical distance
    horiz_dist = np.linalg.norm(end_pos[:2] - start_pos[:2])
 vert_dist = end_pos[2] - start_pos[2] # :

 # Sag amount: Horizontal distanceunder，
 sag = horiz_dist * 0.35 + 0.02 # Sag amount

 # Midpoint position（，z under）
    mid_xy = (start_pos[:2] + end_pos[:2]) / 2.0
 mid_z = min(start_pos[2], end_pos[2]) - sag # → under

 # ctrl1: ， + under
    ctrl1 = np.array([
        end_pos[0] + (start_pos[0] - end_pos[0]) * 0.3,
        end_pos[1] + (start_pos[1] - end_pos[1]) * 0.3,
 end_pos[2] - sag * 0.5, # under
    ])

 # ctrl2: from strawberry， + （under）
    ctrl2 = np.array([
        start_pos[0] + (end_pos[0] - start_pos[0]) * 0.3,
        start_pos[1] + (end_pos[1] - start_pos[1]) * 0.3,
 start_pos[2] + 0.01, # up，under
    ])

 # Bezier interpolation (from strawberry start → end)
    curve_points = []
    widths = []
    for i in range(num_segments + 1):
        t = i / num_segments
        # Cubic Bezier: B(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3
        b0 = (1 - t) ** 3
        b1 = 3 * (1 - t) ** 2 * t
        b2 = 3 * (1 - t) * t ** 2
        b3 = t ** 3
        pt = b0 * start_pos + b1 * ctrl2 + b2 * ctrl1 + b3 * end_pos
        curve_points.append(Gf.Vec3f(*pt.astype(float)))

 # degrees: （t=1），（t=0）
        w = radius * (1.0 + 0.6 * t)
        widths.append(w)

 # BasisCurves prim
    curves_prim = UsdGeom.BasisCurves.Define(stage, stem_path)
    curves_prim.CreateTypeAttr("cubic")
    curves_prim.CreateBasisAttr("catmullRom")
    curves_prim.CreateWrapAttr("nonperiodic")

    curves_prim.CreatePointsAttr(curve_points)
    curves_prim.CreateCurveVertexCountsAttr([len(curve_points)])
    curves_prim.CreateWidthsAttr(widths)
    curves_prim.SetWidthsInterpolation(UsdGeom.Tokens.vertex)

 # stem
    mat_path = stem_path + "/_StemMat"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
 shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.18, 0.55, 0.12)) #
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(curves_prim.GetPrim()).Bind(mat)

 # — RigidBody Collision
    return curves_prim


# Get branchvertices
_branch_pts = _get_branch_world_points(stage)

if _branch_pts is not None and len(_branch_pts) > 0:
    _stems_created = 0
    for i, (sx, sy, sz) in enumerate(strawberry_configs):
        stem_path = f"/World/Stem_{i:02d}"

 # stem
        _old = stage.GetPrimAtPath(stem_path)
        if _old.IsValid():
            stage.RemovePrim(stem_path)

 # （up）
        # Strawberry is rotated -90deg around Y, so the model's "up" direction is approximately +Z in world coords
        berry_top = np.array([sx, sy, sz + STRAWBERRY_TARGET_DIAMETER * 0.5])

 #
        branch_pt, dist = _find_nearest_branch_point(_branch_pts, berry_top)

        _create_stem_curve(stage, stem_path, berry_top, branch_pt)
        _stems_created += 1

 if i < 3: #
            print(f"  🌿 S{i}: berry_top=({sx:.3f},{sy:.3f},{sz+0.015:.3f}) → "
                  f"branch=({branch_pt[0]:.3f},{branch_pt[1]:.3f},{branch_pt[2]:.3f}) "
                  f"dist={dist:.3f}m")

    for _ in range(10): world.step(render=True)
    print(f"  ✅ Created {_stems_created} stems (visual only, no physics)")

 # : Stem prim does not havecollision
    for i in range(len(strawberry_configs)):
        _sp = stage.GetPrimAtPath(f"/World/Stem_{i:02d}")
        if _sp.IsValid():
            if _sp.HasAPI(UsdPhysics.CollisionAPI):
                _sp.RemoveAPI(UsdPhysics.CollisionAPI)
            if _sp.HasAPI(UsdPhysics.RigidBodyAPI):
                _sp.RemoveAPI(UsdPhysics.RigidBodyAPI)
else:
    print("  ⚠️ Skipping stem creation: no branch vertices found")

print("")

# ===== Place box =====
print("\n--- CREATING PLACE BOX (open top) ---")
box_x, box_y, box_z_base = 0.95, 0.25, 0.80 # v9.1: x-0.1
box_w, box_d, box_h = 0.15, 0.15, 0.10 # v9: 0.15
wall_t = 0.005

for p in ["/World/PlaceBox", "/World/PlaceBox_floor",
          "/World/PlaceBox_wall_L", "/World/PlaceBox_wall_R",
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

 #
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

make_box_part("PlaceBox_floor",
    (box_x, box_y, box_z_base + wall_t/2),
    (box_w, box_d, wall_t))
make_box_part("PlaceBox_wall_L",
    (box_x, box_y - box_d/2 + wall_t/2, box_z_base + box_h/2),
    (box_w, wall_t, box_h))
make_box_part("PlaceBox_wall_R",
    (box_x, box_y + box_d/2 - wall_t/2, box_z_base + box_h/2),
    (box_w, wall_t, box_h))
make_box_part("PlaceBox_wall_F",
    (box_x - box_w/2 + wall_t/2, box_y, box_z_base + box_h/2),
    (wall_t, box_d, box_h))
make_box_part("PlaceBox_wall_B",
    (box_x + box_w/2 - wall_t/2, box_y, box_z_base + box_h/2),
    (wall_t, box_d, box_h))

print(f"  📦 Open box at ({box_x},{box_y}), base z={box_z_base}, size {box_w}x{box_d}x{box_h}")

# ── v9.84: ，up ──
_box_ped_path = "/World/PlaceBox_pedestal"
_old_bp = stage.GetPrimAtPath(_box_ped_path)
if _old_bp.IsValid():
    stage.RemovePrim(_box_ped_path)

_box_ped_h = 0.06 # degrees
_box_ped = UsdGeom.Cube.Define(stage, _box_ped_path)
_box_ped.GetSizeAttr().Set(1.0)
_bp_xf = UsdGeom.Xformable(_box_ped.GetPrim())
_bp_xf.ClearXformOpOrder()
# (0.80 - _box_ped_h) box_z_base，under
_bp_xf.AddTranslateOp().Set(Gf.Vec3d(box_x, box_y, box_z_base - _box_ped_h / 2.0))
_bp_xf.AddScaleOp().Set(Gf.Vec3d(box_w * 0.9, box_d * 0.9, _box_ped_h))
_box_ped.CreateDisplayColorAttr([Gf.Vec3f(0.45, 0.35, 0.25)]) #
# ，collision

PLACE_POS = np.array([1.1, 0.25, 0.95]) # v8.2: 0.90+5cm
RETRACT_X = 1.1

for _ in range(30): world.step(render=True)
print(f"✅ Scene ready: {len(strawberry_positions)} berries + place box")


# ============================================================
# — v9.5
# ============================================================

# ── y 、z Sort（y ）──
# strawberry_positions[i] prim /World/Strawberry_{i:02d}
# Sort pick_order: [(picking_idx, original_prim_idx, pos), ...]
_indexed = list(enumerate(strawberry_positions))  # [(orig_idx, pos), ...]
_indexed.sort(key=lambda t: (-t[1][1], -t[1][2]))  # y desc, then z desc
pick_order = [(pick_i, orig_i, pos) for pick_i, (orig_i, pos) in enumerate(_indexed)]

print("\n--- PICKING ORDER (sorted by y↓, z↓) ---")
for pick_i, orig_i, pos in pick_order:
    print(f"  Pick #{pick_i}: Prim S{orig_i:02d} at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

print("\n"+"="*60)
print(f"PICKING {len(pick_order)} STRAWBERRIES")
print("="*60)

STEPS_CLOSE = 150
RX = 1.2
BOX_Y = 0.25
BOX_Z = 0.95
BOX_X = 1.1
ABOVE_BOX = np.array([BOX_X, BOX_Y, BOX_Z])

# ── Warmup： HOME pre-grasp ──
# HOME→pre-grasp disable advance
if len(pick_order) > 0:
    _, _first_orig, _first_pos = pick_order[0]
    _first_pre = np.array([_first_pos[0] + PRE_GRASP_OFFSET, _first_pos[1], _first_pos[2]])
    print(f"\n--- WARM-UP: HOME → first pre-grasp ({np.round(_first_pre, 3)}) ---")
    move_rmpflow(_first_pre, HOME_QUAT, label="warmup-pregrasp")
    for _ in range(30): world.step(render=True)
    print("  ✅ Arm warmed up at first pre-grasp position")

results = []

for pick_idx, (_, orig_idx, berry_pos) in enumerate(pick_order):
    bx, by, bz = berry_pos
 prim_idx = orig_idx # prim ， /World/Strawberry_{prim_idx:02d}
    print(f"\n{'='*55}")
    print(f"🍓 Pick#{pick_idx} (Prim S{prim_idx:02d}) at ({bx:.3f}, {by:.3f}, {bz:.3f})")
    print(f"{'='*55}")

    # ─── PICK ───

    # [1] Open gripper
    print("  [1] OPEN GRIPPER")
    set_gripper_for_steps(GRIPPER_OPEN_DEG, 60)

    # [2] → pre-grasp (bx+0.2, by, bz)
    pre_pos = np.array([bx + PRE_GRASP_OFFSET, by, bz])
    print(f"  [2] PRE-GRASP → {np.round(pre_pos,3)}")
    move_rmpflow(pre_pos, HOME_QUAT, label="pre-grasp")

    # [3] → advance (bx+0.1, by, bz)
    grasp_pos = np.array([bx + GRASP_OFFSET, by, bz])
    print(f"  [3] ADVANCE → {np.round(grasp_pos,3)}")
    final = move_rmpflow(grasp_pos, HOME_QUAT, label="advance", tol=0.005)
    err_adv = np.linalg.norm(grasp_pos - final)
    for _ in range(30): world.step(render=True)

    # 📷
    capture_images(cam1, cam2, world, pick_idx*10)

 # ── v9.7: advance ， ──
    if err_adv > ADVANCE_ERR_THRESHOLD:
        print(f"  ❌ ADVANCE ERR {err_adv:.4f}m > {ADVANCE_ERR_THRESHOLD}m → GRASP FAIL")

 # ，natural drop
        berry_path = f"/World/Strawberry_{prim_idx:02d}"
        berry_prim = stage.GetPrimAtPath(berry_path)
        ka = berry_prim.GetAttribute("physics:kinematicEnabled")
        if ka.IsValid():
            ka.Set(False)
        print(f"      🍓 Restored gravity for S{prim_idx:02d} — dropping...")

        # Wait for strawberry to settle
        for _ in range(90): world.step(render=True)

 #
        berry_xf_tmp = XFormPrim(berry_path)
        bpos_final, _ = berry_xf_tmp.get_world_pose()
        bpos_final = np.array(bpos_final, np.float64)
        print(f"      berry final: {np.round(bpos_final,3)} → 🔻 DROPPED")

 # RETURN
        print(f"  [8] RESET + RETURN (skip grasp)")
        hard_reset_gripper()

        if pick_idx + 1 < len(pick_order):
            _, next_orig_idx, next_pos = pick_order[pick_idx + 1]
            nbx, nby, nbz = next_pos
            next_pre = np.array([nbx + PRE_GRASP_OFFSET, nby, nbz])
            print(f"  [R1] → PRE-GRASP ({nbx+PRE_GRASP_OFFSET:.2f}, {nby:.2f}, {nbz:.2f})")
            move_rmpflow(next_pre, HOME_QUAT, label="R1-pregrasp")
            print(f"  ✅ Ready for Pick#{pick_idx+1} (S{next_orig_idx:02d})")
        else:
            move_rmpflow(HOME_POS, HOME_QUAT, label="HOME")
            print(f"  ✅ Back to HOME")

        results.append((prim_idx, "FAIL", err_adv, False))
        print(f"  ❌ S{prim_idx:02d}: FAIL (advance err too large)")
        continue

    # [4] Close gripper + Attach
    print(f"  [4] CLOSE + ATTACH")
    f_deg, r_deg = set_gripper_for_steps(GRIPPER_CLOSE_DEG, STEPS_CLOSE)
    print(f"      finger={f_deg:.1f}°, rok={r_deg:.1f}°, Δ={abs(f_deg-r_deg):.1f}°")

    berry_path = f"/World/Strawberry_{prim_idx:02d}"
    berry_xf = XFormPrim(berry_path)
    berry_prim = stage.GetPrimAtPath(berry_path)
    berry_pos_now, _ = berry_xf.get_world_pose()
    w3p, w3q = [np.array(x, np.float64) for x in XFormPrim(WRIST_3_LINK).get_world_pose()]
    berry_offset = rotate_vec_by_quat(np.array(berry_pos_now, np.float64) - w3p, quat_inverse(w3q))

 # v9.84: Transportdisablecollision — preventcollision/or arm causing physics issues
    for _bm in _find_meshes_recursive(berry_prim):
        if _bm.HasAPI(UsdPhysics.CollisionAPI):
            _bm.GetAttribute("physics:collisionEnabled").Set(False)
    if berry_prim.HasAPI(UsdPhysics.CollisionAPI):
        berry_prim.GetAttribute("physics:collisionEnabled").Set(False)

 # ─── TRANSPORT（：retract → Move to above box）───

    # [5] Retract → (RX, by, bz)  only change x
    wp1 = np.array([RX, by, bz])
    print(f"  [5] RETRACT → ({RX}, {by:.2f}, {bz:.2f})")
    move_rmpflow_with_berry(wp1, HOME_QUAT, berry_xf, berry_offset, label="retract")

    # [6] Move to ABOVE BOX (1.1, 0.25, 0.95) — RMPFlow handles diagonal path automatically
    print(f"  [6] → ABOVE BOX ({BOX_X}, {BOX_Y}, {BOX_Z})")
    move_rmpflow_with_berry(ABOVE_BOX, HOME_QUAT, berry_xf, berry_offset, label="above-box")

 # [7] Release — collision，Open gripper，disable kinematic，natural drop
    print(f"  [7] RELEASE")

 # v9.84: collision（so it can fall into box）
    for _bm in _find_meshes_recursive(berry_prim):
        if _bm.HasAPI(UsdPhysics.CollisionAPI):
            _bm.GetAttribute("physics:collisionEnabled").Set(True)
    if berry_prim.HasAPI(UsdPhysics.CollisionAPI):
        berry_prim.GetAttribute("physics:collisionEnabled").Set(True)

    set_gripper_for_steps(GRIPPER_OPEN_DEG, 15)
    ka = berry_prim.GetAttribute("physics:kinematicEnabled")
    if ka.IsValid(): ka.Set(False)

    # Lift arm（strawberry in free fall）
    above_pos = np.array([BOX_X, BOX_Y, BOX_Z + 0.15])
    move_rmpflow(above_pos, HOME_QUAT, label="lift-off")

    # Wait for strawberry to settle
    for _ in range(60): world.step(render=True)
    bpos_final, _ = berry_xf.get_world_pose()
    bpos_final = np.array(bpos_final, np.float64)

    # Check if strawberry fell into box
    in_x = (box_x - box_w/2 - 0.02) <= bpos_final[0] <= (box_x + box_w/2 + 0.02)
    in_y = (box_y - box_d/2 - 0.02) <= bpos_final[1] <= (box_y + box_d/2 + 0.02)
    in_z = box_z_base - 0.01 <= bpos_final[2] <= box_z_base + box_h + 0.05
    in_box = in_x and in_y and in_z
    tag = "📦 IN BOX" if in_box else "❌ MISSED"
    print(f"      berry final: {np.round(bpos_final,3)} → {tag}")

 # ─── RETURN（v9.6: lift-off under pre-grasp）───
    print(f"  [8] RESET + RETURN")
    hard_reset_gripper()

    if pick_idx + 1 < len(pick_order):
        _, next_orig_idx, next_pos = pick_order[pick_idx + 1]
        nbx, nby, nbz = next_pos
        next_pre = np.array([nbx + PRE_GRASP_OFFSET, nby, nbz])

 # under pre-grasp
        print(f"  [R1] → PRE-GRASP ({nbx+PRE_GRASP_OFFSET:.2f}, {nby:.2f}, {nbz:.2f})")
        move_rmpflow(next_pre, HOME_QUAT, label="R1-pregrasp")

        print(f"  ✅ Ready for Pick#{pick_idx+1} (S{next_orig_idx:02d})")
    else:
 # ， HOME
        move_rmpflow(HOME_POS, HOME_QUAT, label="HOME")
        print(f"  ✅ Back to HOME")

    results.append((prim_idx, "DONE", err_adv, in_box))
    print(f"  🎉 S{prim_idx:02d}: DONE")

# ===== =====
print("\n"+"="*60)
print("SUMMARY")
print("="*60)
for pidx, status, err, inbox in results:
    if status == "FAIL":
        print(f"  S{pidx:02d}: {status:6s} advance_err={err:.4f}m 🔻 DROPPED")
    else:
        box_tag = "📦" if inbox else "❌"
        print(f"  S{pidx:02d}: {status:6s} advance_err={err:.4f}m {box_tag}")
n_inbox = sum(1 for _,_,_,ib in results if ib)
n_fail = sum(1 for _,s,_,_ in results if s == "FAIL")
print(f"\n  {len(results)} attempted, {n_inbox}/{len(results)} in box, {n_fail} failed (advance err)")
print("="*60)

while simulation_app.is_running():
    world.step(render=True)
simulation_app.close()
