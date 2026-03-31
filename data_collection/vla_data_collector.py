"""
VLA Data Collector for Isaac Sim Strawberry Picking (3-Camera) v4
==================================================================
Based on vla_data_collector_3cam2.py. Changes in v4:
  - [FPS] record_every default changed from 6 to 1 (no second-layer filtering)
    Actual recording rate now controlled solely by caller's render frequency
  - [FIX] Inherits _next_episode_id fix and discard_log from v2

Usage:
  from vla_data_collector import EpisodeRecorder
  recorder = EpisodeRecorder(save_dir, cam1, cam2, cam3, robot, world,
                             ur5e_idx, finger_idx, rok_idx)
"""

import numpy as np
import os
import json
import csv
import h5py
import time
from PIL import Image
from datetime import datetime


# ── Configuration ──
RECORD_EVERY = 1        # v4: no second-layer filtering; rate controlled by RENDER_EVERY in caller
IMAGE_SIZE = (224, 224)  # Resize for VLA input (pi0.5 expects 224x224)
MAX_STEPS_PER_EPISODE = 2000  # Safety cap


class EpisodeRecorder:
    """
    Records expert demonstration data for VLA training.

    Each episode = one strawberry pick-and-place attempt.
    Data is stored in HDF5 + image folders, ready for LeRobot conversion.
    """

    def __init__(self, save_dir, cam1, cam2, cam3, robot, world,
                 ur5e_idx, finger_idx, rok_idx,
                 art_kin=None, record_every=RECORD_EVERY):
        """
        Args:
            save_dir:    Root directory for all episode data
            cam1, cam2, cam3:  Isaac Sim Camera objects
            robot:       Articulation object for UR5e
            world:       Isaac Sim World
            ur5e_idx:    np.array of 6 UR5e joint indices in the articulation
            finger_idx:  int, index of finger_joint
            rok_idx:     int, index of right_outer_knuckle_joint
            art_kin:     ArticulationKinematicsSolver (optional, for EE pose)
            record_every: Record every N physics steps
        """
        self.save_dir = save_dir
        self.cam1 = cam1
        self.cam2 = cam2
        self.cam3 = cam3
        self.robot = robot
        self.world = world
        self.ur5e_idx = ur5e_idx
        self.finger_idx = finger_idx
        self.rok_idx = rok_idx
        self.art_kin = art_kin
        self.record_every = record_every

        # Episode state
        self._recording = False
        self._episode_id = None
        self._episode_dir = None
        self._step_count = 0          # Total physics steps in this episode
        self._record_count = 0        # Recorded frames in this episode
        self._anomalies = []           # Quality issues detected during episode
        self._data_buffer = {
            "state": [],              # (N, 8) float32
            "action": [],             # (N, 8) float32
            "ee_pos": [],             # (N, 3) float32 — auxiliary
            "ee_quat": [],            # (N, 4) float32 — auxiliary
            "timestamp": [],          # (N,) float64
        }
        self._prompt = ""
        self._round_idx = 0
        self._pick_idx = 0

        # Global counters
        self._next_episode_id = 0     # monotonic, always increments
        self._total_episodes = 0      # only counts saved (non-discarded) episodes
        self._discarded_episodes = 0
        self._metadata_log = []

        os.makedirs(save_dir, exist_ok=True)

        # ── Discard log: persistent CSV recording every discard reason ──
        self._discard_log_path = os.path.join(save_dir, "discard_log.csv")
        with open(self._discard_log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode_id", "round_idx", "pick_idx", "n_frames",
                "reason", "anomalies", "timestamp"
            ])

        print(f"[VLA Recorder] Save dir: {save_dir}")
        print(f"[VLA Recorder] Discard log: {self._discard_log_path}")

    # ────────────────────────────────────────────
    # Episode lifecycle
    # ────────────────────────────────────────────

    def start_episode(self, round_idx, pick_idx, prompt="pick the strawberry and place it in the box"):
        """Call BEFORE the pick cycle begins (before open gripper)."""
        self._episode_id = f"episode_{self._next_episode_id:06d}"
        self._next_episode_id += 1   # always increment, even if this episode gets discarded
        self._episode_dir = os.path.join(self.save_dir, self._episode_id)
        os.makedirs(self._episode_dir, exist_ok=True)
        os.makedirs(os.path.join(self._episode_dir, "cam1"), exist_ok=True)
        os.makedirs(os.path.join(self._episode_dir, "cam2"), exist_ok=True)
        os.makedirs(os.path.join(self._episode_dir, "cam3"), exist_ok=True)

        self._recording = True
        self._step_count = 0
        self._record_count = 0
        self._anomalies = []
        self._prompt = prompt
        self._round_idx = round_idx
        self._pick_idx = pick_idx
        self._t0 = time.time()

        for key in self._data_buffer:
            self._data_buffer[key] = []

        print(f"[VLA Recorder] START {self._episode_id} "
              f"(R{round_idx}, P{pick_idx}, prompt='{prompt[:40]}...')")

    def mark_anomaly(self, reason):
        """
        Mark current episode as having a quality issue.
        Episode will be discarded at end_episode().

        Call from the main script when you detect:
          - motion error > threshold (e.g. err > 0.05m)
          - severe gripper asymmetry (> 10°)
          - advance failure
          - any other abnormal behavior
        """
        if not self._recording:
            return
        self._anomalies.append(reason)
        print(f"[VLA Recorder] ANOMALY in {self._episode_id}: {reason}")

    def _log_discard(self, reason, n_frames):
        """Append one row to the persistent discard log CSV."""
        anomalies_str = "; ".join(self._anomalies) if self._anomalies else ""
        try:
            with open(self._discard_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self._episode_id,
                    int(self._round_idx),
                    int(self._pick_idx),
                    int(n_frames),
                    reason,
                    anomalies_str,
                    datetime.now().isoformat(),
                ])
        except Exception as e:
            print(f"[VLA Recorder] WARNING: failed to write discard log: {e}")

    def end_episode(self, success=False):
        """Call AFTER the pick cycle ends (after place / after failure)."""
        if not self._recording:
            return

        self._recording = False
        n_frames = self._record_count

        # ── Quality gate: discard bad episodes ──
        if n_frames < 5:
            print(f"[VLA Recorder] DISCARD {self._episode_id} — only {n_frames} frames")
            self._log_discard("too_few_frames", n_frames)
            import shutil
            shutil.rmtree(self._episode_dir, ignore_errors=True)
            self._discarded_episodes += 1
            return

        if self._anomalies:
            reasons = "; ".join(self._anomalies)
            print(f"[VLA Recorder] DISCARD {self._episode_id} — anomaly: {reasons}")
            self._log_discard("anomaly", n_frames)
            import shutil
            shutil.rmtree(self._episode_dir, ignore_errors=True)
            self._discarded_episodes += 1
            return

        # Save HDF5
        h5_path = os.path.join(self._episode_dir, "data.hdf5")
        with h5py.File(h5_path, "w") as f:
            for key, vals in self._data_buffer.items():
                arr = np.array(vals)
                f.create_dataset(key, data=arr, compression="gzip", compression_opts=4)

            # Metadata as HDF5 attributes
            f.attrs["episode_id"] = self._episode_id
            f.attrs["prompt"] = self._prompt
            f.attrs["round_idx"] = int(self._round_idx)
            f.attrs["pick_idx"] = int(self._pick_idx)
            f.attrs["success"] = bool(success)
            f.attrs["n_frames"] = int(n_frames)
            f.attrs["n_physics_steps"] = int(self._step_count)
            f.attrs["fps_effective"] = float(n_frames / max(time.time() - self._t0, 0.01))
            f.attrs["record_every"] = int(self.record_every)
            f.attrs["timestamp"] = datetime.now().isoformat()
            f.attrs["state_dim"] = 8
            f.attrs["action_dim"] = 8
            f.attrs["image_size"] = list(IMAGE_SIZE)

        # Also save metadata JSON (easier to parse later)
        meta = {
            "episode_id": self._episode_id,
            "prompt": self._prompt,
            "round_idx": int(self._round_idx),
            "pick_idx": int(self._pick_idx),
            "success": bool(success),
            "n_frames": int(n_frames),
            "state_dim": 8,
            "action_dim": 8,
        }
        with open(os.path.join(self._episode_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        self._metadata_log.append(meta)
        self._total_episodes += 1
        tag = "SUCCESS" if success else "FAIL"
        print(f"[VLA Recorder] END {self._episode_id}: {n_frames} frames, {tag}")

    # ────────────────────────────────────────────
    # Per-step recording
    # ────────────────────────────────────────────

    def record_step(self, action_8dim):
        """
        Call this AFTER every world.step() that you want recorded.

        Args:
            action_8dim: np.array shape (8,) — the joint position targets
                         that were sent to the robot at this step.
                         [ur5e_0..5, finger, rok] in radians.
        """
        if not self._recording:
            return

        self._step_count += 1

        # Subsample to ~10 Hz
        if self._step_count % self.record_every != 0:
            return

        if self._record_count >= MAX_STEPS_PER_EPISODE:
            return

        # ── Get current state ──
        all_joints = self.robot.get_joint_positions()
        state = np.zeros(8, dtype=np.float32)
        # UR5e 6 joints
        for i, idx in enumerate(self.ur5e_idx):
            state[i] = float(all_joints[idx])
        # Gripper 2 joints
        state[6] = float(all_joints[self.finger_idx])
        state[7] = float(all_joints[self.rok_idx])

        # ── Get action (already provided) ──
        action = np.array(action_8dim, dtype=np.float32)

        # ── Get EE pose (optional) ──
        ee_pos = np.zeros(3, dtype=np.float32)
        ee_quat = np.array([1, 0, 0, 0], dtype=np.float32)
        if self.art_kin is not None:
            try:
                pos, rot = self.art_kin.compute_end_effector_pose()
                ee_pos = np.array(pos, dtype=np.float32)
                if hasattr(rot, 'shape') and rot.shape == (3, 3):
                    # Convert rotation matrix to quaternion if needed
                    pass  # Keep default quat
                else:
                    ee_quat = np.array(rot, dtype=np.float32)
            except:
                pass

        # ── Capture images (only on render frames) ──
        img1 = self.cam1.get_rgba()
        img2 = self.cam2.get_rgba()
        img3 = self.cam3.get_rgba()

        if img1 is not None and img2 is not None and img3 is not None:
            # Resize to 224x224 for VLA
            pil1 = Image.fromarray(img1[:, :, :3]).resize(IMAGE_SIZE, Image.LANCZOS)
            pil2 = Image.fromarray(img2[:, :, :3]).resize(IMAGE_SIZE, Image.LANCZOS)
            pil3 = Image.fromarray(img3[:, :, :3]).resize(IMAGE_SIZE, Image.LANCZOS)

            frame_id = self._record_count
            pil1.save(os.path.join(self._episode_dir, "cam1", f"{frame_id:04d}.png"))
            pil2.save(os.path.join(self._episode_dir, "cam2", f"{frame_id:04d}.png"))
            pil3.save(os.path.join(self._episode_dir, "cam3", f"{frame_id:04d}.png"))

        # ── Buffer data ──
        self._data_buffer["state"].append(state)
        self._data_buffer["action"].append(action)
        self._data_buffer["ee_pos"].append(ee_pos)
        self._data_buffer["ee_quat"].append(ee_quat)
        self._data_buffer["timestamp"].append(time.time())

        self._record_count += 1

    # ────────────────────────────────────────────
    # Convenience: get current action vector
    # ────────────────────────────────────────────

    def get_current_arm_action(self):
        """
        Helper to read the current joint targets (what RMPFlow just computed)
        and format as 8-dim action vector.
        Returns np.array shape (8,) in radians.
        """
        all_joints = self.robot.get_joint_positions()
        action = np.zeros(8, dtype=np.float32)
        for i, idx in enumerate(self.ur5e_idx):
            action[i] = float(all_joints[idx])
        action[6] = float(all_joints[self.finger_idx])
        action[7] = float(all_joints[self.rok_idx])
        return action

    # ────────────────────────────────────────────
    # Final summary
    # ────────────────────────────────────────────

    def save_global_metadata(self):
        """Call once at the end of all rounds."""
        meta_path = os.path.join(self.save_dir, "dataset_meta.json")
        summary = {
            "total_episodes": self._total_episodes,
            "discarded_episodes": self._discarded_episodes,
            "success_episodes": sum(1 for m in self._metadata_log if m["success"]),
            "failed_episodes": sum(1 for m in self._metadata_log if not m["success"]),
            "state_dim": 8,
            "action_dim": 8,
            "image_size": list(IMAGE_SIZE),
            "record_fps": 15.0,  # v4: controlled by RENDER_EVERY=4 in caller (60/4=15)
            "robot_type": "ur5e_robotiq2f140",
            "camera_names": ["cam1", "cam2", "cam3"],
            "joint_names": [
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
                "finger_joint", "right_outer_knuckle_joint",
            ],
            "created_at": datetime.now().isoformat(),
            "episodes": self._metadata_log,
        }
        with open(meta_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[VLA Recorder] GLOBAL SUMMARY")
        print(f"  Total saved: {self._total_episodes}")
        print(f"  Discarded: {self._discarded_episodes}")
        print(f"  Successful: {summary['success_episodes']}")
        print(f"  Failed: {summary['failed_episodes']}")
        print(f"  Metadata: {meta_path}")
