"""
Convert Isaac Sim Strawberry Picking Data (3-Camera) → LeRobot v2 Format
==========================================================================
Based on convert_to_lerobot.py. Changes:
  - Added cam3 image feature and loading
  - repo_id changed to local/strawberry_picking_3c

"""

import os
import json
import shutil
import numpy as np
import h5py
from PIL import Image

try:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.constants import HF_LEROBOT_HOME
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

import tyro

REPO_NAME = "local/strawberry_picking_3c"
FPS = 10
IMAGE_SHAPE = (224, 224, 3)
STATE_DIM = 8
ACTION_DIM = 8


def main(data_dir: str, *, push_to_hub: bool = False, only_success: bool = False):
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="ur5e",
        fps=FPS,
        features={
            "observation.images.cam1": {
                "dtype": "image",
                "shape": IMAGE_SHAPE,
                "names": ["height", "width", "channel"],
            },
            "observation.images.cam2": {
                "dtype": "image",
                "shape": IMAGE_SHAPE,
                "names": ["height", "width", "channel"],
            },
            "observation.images.cam3": {
                "dtype": "image",
                "shape": IMAGE_SHAPE,
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (STATE_DIM,),
                "names": [
                    "shoulder_pan", "shoulder_lift", "elbow",
                    "wrist_1", "wrist_2", "wrist_3",
                    "finger", "right_outer_knuckle",
                ],
            },
            "action": {
                "dtype": "float32",
                "shape": (ACTION_DIM,),
                "names": [
                    "shoulder_pan", "shoulder_lift", "elbow",
                    "wrist_1", "wrist_2", "wrist_3",
                    "finger", "right_outer_knuckle",
                ],
            },
        },
        image_writer_threads=4,
        image_writer_processes=2,
    )

    episode_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("episode_")
    ])
    print(f"Found {len(episode_dirs)} episode directories")

    n_converted = 0
    n_skipped = 0
    total_frames = 0

    for ep_dir_name in episode_dirs:
        ep_dir = os.path.join(data_dir, ep_dir_name)
        meta_path = os.path.join(ep_dir, "meta.json")
        h5_path = os.path.join(ep_dir, "data.hdf5")

        if not os.path.exists(h5_path) or not os.path.exists(meta_path):
            n_skipped += 1
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        if only_success and not meta.get("success", False):
            n_skipped += 1
            continue

        with h5py.File(h5_path, "r") as f:
            states = f["state"][:]
            states_raw = f["state"][:]
            actions = np.zeros_like(states_raw)
            actions[:-1] = states_raw[1:]
            actions[-1] = states_raw[-1]
            n_frames = len(states)

        if n_frames < 5:
            n_skipped += 1
            continue

        prompt = meta.get("prompt", "pick the ripe strawberry and place it in the box")

        for i in range(n_frames):
            cam1_path = os.path.join(ep_dir, "cam1", f"{i:04d}.png")
            cam2_path = os.path.join(ep_dir, "cam2", f"{i:04d}.png")
            cam3_path = os.path.join(ep_dir, "cam3", f"{i:04d}.png")

            if not (os.path.exists(cam1_path) and os.path.exists(cam2_path) and os.path.exists(cam3_path)):
                continue

            img1 = np.array(Image.open(cam1_path))
            img2 = np.array(Image.open(cam2_path))
            img3 = np.array(Image.open(cam3_path))

            dataset.add_frame({
                "observation.images.cam1": img1,
                "observation.images.cam2": img2,
                "observation.images.cam3": img3,
                "observation.state": states[i],
                "action": actions[i],
                "task": prompt,
            })

        dataset.save_episode()
        n_converted += 1
        total_frames += n_frames

        if n_converted % 50 == 0:
            print(f"  Converted {n_converted} episodes ({total_frames} frames)...")

    print(f"\nConversion complete:")
    print(f"  Episodes converted: {n_converted}")
    print(f"  Episodes skipped:   {n_skipped}")
    print(f"  Total frames:       {total_frames}")
    print(f"  Dataset saved to:   {output_path}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["strawberry", "ur5e", "isaac-sim", "picking", "3cam"],
            private=False, push_videos=True, license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
