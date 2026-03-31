"""
Merge worker episode data into a single episodes/ directory.
==============================================================
Usage:
    python3 merge_episodes.py
    python3 merge_episodes.py --base_dir /path/to/data_collection --output_dir /path/to/episodes
"""

import os
import json
import shutil
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    base_dir = args.base_dir
    output_dir = args.output_dir or os.path.join(base_dir, "episodes")

    # Auto-detect episodes_workerN directories
    worker_dirs = sorted([
        os.path.join(base_dir, d) for d in os.listdir(base_dir)
        if d.startswith("episodes_worker") and os.path.isdir(os.path.join(base_dir, d))
    ])

    if not worker_dirs:
        print("No episodes_worker* directories found!")
        return

    print(f"Found {len(worker_dirs)} worker directories:")
    for wd in worker_dirs:
        n = len([d for d in os.listdir(wd) if d.startswith("episode_")]) if os.path.exists(wd) else 0
        print(f"  {wd}: {n} episodes")

    if os.path.exists(output_dir):
        existing = [d for d in os.listdir(output_dir) if d.startswith("episode_")]
        if existing:
            print(f"\nWARNING: {output_dir} already has {len(existing)} episodes.")
            resp = input("Delete and recreate? [y/N]: ").strip().lower()
            if resp != 'y':
                print("Aborted.")
                return
            for d in existing:
                shutil.rmtree(os.path.join(output_dir, d))

    os.makedirs(output_dir, exist_ok=True)

    all_episodes = []
    for wd in worker_dirs:
        if not os.path.exists(wd):
            continue
        episodes = sorted([
            d for d in os.listdir(wd)
            if d.startswith("episode_") and os.path.isdir(os.path.join(wd, d))
        ])
        for ep in episodes:
            ep_path = os.path.join(wd, ep)
            if os.path.exists(os.path.join(ep_path, "data.hdf5")) and \
               os.path.exists(os.path.join(ep_path, "meta.json")):
                # Check cam3 folder exists (3-camera setup)
                if os.path.isdir(os.path.join(ep_path, "cam3")):
                    all_episodes.append(ep_path)
                else:
                    print(f"  SKIP {ep_path} (no cam3 folder)")
            else:
                print(f"  SKIP {ep_path} (incomplete)")

    print(f"\nTotal valid 3-camera episodes: {len(all_episodes)}")
    if not all_episodes:
        print("No episodes to merge!")
        return

    all_meta = []
    n_success = 0
    n_failed = 0

    for new_idx, old_path in enumerate(all_episodes):
        new_name = f"episode_{new_idx:06d}"
        new_path = os.path.join(output_dir, new_name)
        shutil.copytree(old_path, new_path)

        meta_path = os.path.join(new_path, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        old_id = meta.get("episode_id", "")
        meta["episode_id"] = new_name
        meta["original_id"] = old_id
        meta["original_path"] = old_path
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        all_meta.append(meta)
        if meta.get("success", False):
            n_success += 1
        else:
            n_failed += 1

        if (new_idx + 1) % 100 == 0:
            print(f"  Merged {new_idx + 1}/{len(all_episodes)} episodes...")

    global_meta = {
        "total_episodes": len(all_meta),
        "success_episodes": n_success,
        "failed_episodes": n_failed,
        "state_dim": 8,
        "action_dim": 8,
        "image_size": [224, 224],
        "record_fps": 10.0,
        "robot_type": "ur5e_robotiq2f140",
        "camera_names": ["cam1", "cam2", "cam3"],
        "joint_names": [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
            "finger_joint", "right_outer_knuckle_joint",
        ],
        "merged_from": [os.path.basename(wd) for wd in worker_dirs],
        "episodes": all_meta,
    }
    with open(os.path.join(output_dir, "dataset_meta.json"), "w") as f:
        json.dump(global_meta, f, indent=2)

    print(f"\n{'='*50}")
    print(f"MERGE COMPLETE")
    print(f"  Output: {output_dir}")
    print(f"  Total episodes: {len(all_meta)}")
    print(f"  Successful: {n_success}")
    print(f"  Failed: {n_failed}")
    print(f"{'='*50}")
    print(f"\nNext step:")
    print(f"  cd openpi")
    print(f"  uv run convert_to_lerobot.py --data_dir {output_dir} --only_success")


if __name__ == "__main__":
    main()
