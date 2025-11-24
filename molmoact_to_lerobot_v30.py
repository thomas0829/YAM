#!/usr/bin/env python
"""
Convert MolmoAct-style dataset to LeRobot v3.0 format in one shot.

This follows the same high-level pipeline and logic as molmoact_to_lerobot_v21.py:
  1. Episode-first layout on disk:
        data_dir/
        ├── 000001/
        │   ├── 000001.json
        │   ├── left_rgb/
        │   ├── right_rgb/
        │   └── front_rgb/
        ├── 000002/
        │   ├── 000002.json
        │   ├── left_rgb/
        │   ├── right_rgb/
        │   └── front_rgb/
        └── ...

  2. Load all episodes into memory (qpos, actions, images).
  3. Stream frames into a LeRobotDataset via `add_frame` + `save_episode`.

Differences vs the v2.1 script:
  - Uses the v3.0 LeRobotDataset API from `lerobot.datasets.lerobot_dataset`.
  - Creates a v3.0 dataset directly (no v2.1→v3.0 conversion step).
  - Calls `dataset.finalize()` at the end to produce a valid v3.0 dataset.
  - Does NOT support resume into an existing dataset directory; the output_dir must
    be new or empty.

Usage example:

    python molmoact_to_lerobot_v30.py \
        --data_dir /path/to/molmoact \
        --output_dir /path/to/molmoact_lerobot_v30 \
        --repo_id your-user/molmoact_v30 \
        --fps 10

You can then train with:

    python lerobot/scripts/train.py \
        --dataset.repo_id=/path/to/molmoact_lerobot_v30 \
        --policy.type=diffusion

Note: For local training, `--dataset.repo_id` can be the absolute path to the dataset directory.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

# LeRobot v3.0 API
from lerobot_v30.src.lerobot.datasets.lerobot_dataset import LeRobotDataset


def load_molmoact_data(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load MolmoAct episodes from an episode-first layout:

    data_dir/
    ├── 000001/
    │   ├── 000001.json
    │   ├── left_rgb/
    │   ├── right_rgb/
    │   └── front_rgb/
    ├── 000002/
    │   ├── 000002.json
    │   ├── left_rgb/
    │   ├── right_rgb/
    │   └── front_rgb/
    └── ...

    Expected JSON format per episode_id (list of frames):
      [
        {
          "left_joint": "[...]",          # JSON-encoded list of floats
          "right_joint": "[...]",
          # optionally:
          "left_delta_action": "[...]",
          "right_delta_action": "[...]",
          "task": "...",
          "language_instruction": ["..."]
        },
        ...
      ]

    This function:
      - builds qpos = concat(left_joint, right_joint) -> shape (T, 14)
      - builds actions = concat(left_actions, right_actions) -> shape (T, 14)
        (currently using joint positions as actions, as in the v2.1 script)
      - loads images from left_rgb/, right_rgb/, front_rgb/
    """
    episodes: List[Dict[str, Any]] = []
    data_path = Path(data_dir)

    # Episode folders named like "000001", "000002", ...
    episode_dirs = sorted(
        [d for d in data_path.iterdir() if d.is_dir() and d.name.isdigit()]
    )
    print(f"Found {len(episode_dirs)} episodes under {data_dir}")

    for ep_dir in episode_dirs:
        episode_id = ep_dir.name
        json_path = ep_dir / f"{episode_id}.json"

        if not json_path.exists():
            print(f"Skipping {episode_id}: missing JSON file {json_path}")
            continue

        try:
            with open(json_path, "r") as f:
                episode_data = json.load(f)
        except Exception as e:
            print(f"Skipping {episode_id}: failed to load JSON ({e})")
            continue

        if not episode_data:
            print(f"Skipping {episode_id}: empty JSON data")
            continue

        # Task description and language instruction (if present)
        first_frame = episode_data[0]
        task_description = first_frame.get("task", f"task_{episode_id}")
        language_instruction = first_frame.get("language_instruction", ["no_instruction"])

        try:
            # Joint positions
            left_joint = np.array(
                [json.loads(frame["left_joint"]) for frame in episode_data],
                dtype=np.float32,
            )
            right_joint = np.array(
                [json.loads(frame["right_joint"]) for frame in episode_data],
                dtype=np.float32,
            )

            # Actions: here we mirror what the v2.1 script did — using joints as actions.
            left_actions = np.array(
                [json.loads(frame["left_joint"]) for frame in episode_data],
                dtype=np.float32,
            )
            right_actions = np.array(
                [json.loads(frame["right_joint"]) for frame in episode_data],
                dtype=np.float32,
            )

            qpos = np.concatenate([left_joint, right_joint], axis=1)   # (T, 14)
            actions = np.concatenate([left_actions, right_actions], axis=1)  # (T, 14)

        except Exception as e:
            print(f"Error parsing episode {episode_id}: {e}")
            continue

        episode_info: Dict[str, Any] = {
            "task_name": "default_task",  # can be customized if you have multiple tasks
            "episode_id": episode_id,
            "task_description": task_description,
            "language_instruction": language_instruction,
            "qpos": qpos,
            "actions": actions,
            "episode_length": len(actions),
            "images": [],
        }

        # Load camera images: left_rgb, right_rgb, front_rgb
        for camera_dir in [
            ep_dir / "left_rgb",
            ep_dir / "right_rgb",
            ep_dir / "front_rgb",
        ]:
            if not camera_dir.exists():
                continue

            camera_name = camera_dir.name.replace("_rgb", "")
            image_files = sorted(
                [
                    f
                    for f in camera_dir.iterdir()
                    if f.suffix.lower() in [".png", ".jpg", ".jpeg"]
                ]
            )

            camera_images: List[Image.Image] = []
            for img_file in image_files:
                try:
                    img = Image.open(img_file)
                    camera_images.append(img)
                except Exception as e:
                    print(f"  Warning: Failed to load image {img_file}: {e}")

            episode_info["images"].append(
                {
                    "camera_name": camera_name,  # "left", "right", "front"
                    "images": camera_images,
                }
            )

        episodes.append(episode_info)

    print(f"Loaded {len(episodes)} total episodes")
    return episodes


def infer_camera_shapes(
    episodes: List[Dict[str, Any]],
    expected_cameras: Tuple[str, ...] = ("left", "right", "front"),
) -> Dict[str, Tuple[int, int, int]]:
    """
    Infer (height, width, channels) for each camera by inspecting the first available frame.
    Falls back to (480, 640, 3) if a camera has no images at all.
    """
    inferred: Dict[str, Tuple[int, int, int]] = {}
    default_shape = (480, 640, 3)

    for episode in episodes:
        for cam_data in episode["images"]:
            cam_name = cam_data["camera_name"]
            if cam_name in inferred:
                continue
            for img in cam_data["images"]:
                if img is None:
                    continue
                width, height = img.size
                channels = len(img.getbands())
                inferred[cam_name] = (height, width, channels)
                break
        if all(cam in inferred for cam in expected_cameras):
            break

    for cam in expected_cameras:
        if cam not in inferred:
            inferred[cam] = default_shape
            print(
                f"[warn] Could not infer resolution for camera '{cam}'. "
                f"Defaulting to {default_shape} (HxWxC)."
            )
    return inferred


def create_lerobot_dataset_v30(
    episodes: List[Dict[str, Any]],
    output_dir: str,
    repo_id: str,
    fps: int = 30,
    robot_type: str = "molmoact_dual_arm",
) -> None:
    """
    Convert MolmoAct episodes into a LeRobot v3.0 dataset.

    Args:
        episodes: List of dicts from `load_molmoact_data`.
        output_dir: Target directory for the LeRobot dataset.
        repo_id: Dataset identifier (used in meta; can be HF hub id or any string).
        fps: Frame rate of the data.
        robot_type: Optional robot_type string recorded in the dataset metadata.

    Notes:
        - This function assumes `output_dir` is either non-existent or empty.
        - It creates a v3.0 dataset via `LeRobotDataset.create(...)` and calls
          `dataset.finalize()` at the end.
    """
    output_path = Path(output_dir)

    if output_path.exists() and any(output_path.iterdir()):
        raise RuntimeError(
            f"Output directory '{output_dir}' already exists and is not empty.\n"
            "For a clean v3.0 dataset, please either:\n"
            "  * remove that directory, or\n"
            "  * choose a different --output_dir."
        )

    # output_path.mkdir(parents=True, exist_ok=True)

    camera_shapes = infer_camera_shapes(episodes, expected_cameras=("left", "right", "front"))
    image_dim_names = ["height", "width", "channels"]

    # Feature schema (same keys and shapes as the v2.1 script; v3.0 layout is handled by LeRobot).
    features: Dict[str, Dict[str, Any]] = {
        # Robot joint positions
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": [
                "left_joint1",
                "left_joint2",
                "left_joint3",
                "left_joint4",
                "left_joint5",
                "left_joint6",
                "left_gripper",
                "right_joint1",
                "right_joint2",
                "right_joint3",
                "right_joint4",
                "right_joint5",
                "right_joint6",
                "right_gripper",
            ],
        },
        # Actions (joint-space)
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": [
                "left_m1",
                "left_m2",
                "left_m3",
                "left_m4",
                "left_m5",
                "left_m6",
                "left_m7",
                "right_m8",
                "right_m9",
                "right_m3",
                "right_m4",
                "right_m5",
                "right_m6",
                "right_m7",
            ],
        },
        # Image observations
        "observation.images.camera_left": {
            "dtype": "image",
            "shape": camera_shapes["left"],
            "names": image_dim_names,
        },
        "observation.images.camera_right": {
            "dtype": "image",
            "shape": camera_shapes["right"],
            "names": image_dim_names,
        },
        "observation.images.camera_front": {
            "dtype": "image",
            "shape": camera_shapes["front"],
            "names": image_dim_names,
        },
    }

    print(f"Creating LeRobot v3.0 dataset in: {output_dir}")
    print(f"Repo ID (for metadata/hub): {repo_id}")
    print(f"FPS: {fps}")
    print(f"Robot type: {robot_type}")
    print(f"Features: {list(features.keys())}")

    # Create a fresh v3.0 dataset
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output_dir,
        robot_type=robot_type,
        features=features,
        use_videos=True,  # Encode camera streams as videos on disk
    )

    import tqdm
    from tqdm import trange

    total_episodes = len(episodes)
    print(f"Converting {total_episodes} episodes...")

    for episode_idx, episode_data in enumerate(tqdm.tqdm(episodes, desc="Episodes")):
        qpos = episode_data["qpos"]            # (T, 14)
        actions = episode_data["actions"]      # (T, 14)
        task_description = episode_data["task_description"]
        episode_length = episode_data["episode_length"]
        camera_images_map: Dict[str, List[Image.Image]] = {}

        for cam_data in episode_data["images"]:
            cam_name = cam_data["camera_name"]  # "left", "right", "front"
            camera_images_map[cam_name] = cam_data["images"]

        # Stream frames into the dataset
        # Note: we skip first 5 frames as in the v2.1 script.
        for frame_idx in trange(episode_length, leave=False, desc=f"Frames (ep {episode_idx})"):
            if frame_idx < 5:
                continue

            frame_data: Dict[str, Any] = {
                # low-dimensional state
                "observation.state": qpos[frame_idx].astype(np.float32),
                # action
                "action": actions[frame_idx].astype(np.float32),
                # per-frame task string (LeRobot will map to task_index)
                "task": task_description,
                # optional timestamp: if not provided, LeRobot uses frame_index / fps
                # "timestamp": frame_idx / fps,
            }

            # Attach images if present
            for cam_name, images in camera_images_map.items():
                if frame_idx < len(images):
                    key = f"observation.images.camera_{cam_name}"
                    frame_data[key] = images[frame_idx]

            dataset.add_frame(frame_data)

        dataset.save_episode()

    # Finalize v3.0 dataset (very important!)
    print("Finalizing dataset (writing metadata, closing parquet writers)...")
    dataset.finalize()

    # Reload metadata for summary (optional)
    ds_loaded = LeRobotDataset(repo_id=repo_id, root=output_dir)
    meta = ds_loaded.meta

    print("Dataset creation completed!")
    print(f"Total episodes: {meta.total_episodes}")
    print(f"Total frames: {meta.total_frames}")
    print(f"Dataset root: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MolmoAct data to LeRobot v3.0 dataset format."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Input MolmoAct data directory (episode-first layout).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for the LeRobot v3.0 dataset.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="molmoact_dataset_v30",
        help=(
            "Dataset identifier stored in metadata and used when pushing to the Hub. "
            "For local-only use, this can be any string."
        ),
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Data collection frame rate (Hz). Used for timestamps and metadata.",
    )
    parser.add_argument(
        "--robot_type",
        type=str,
        default="molmoact_dual_arm",
        help="Robot type string recorded in the dataset metadata.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Input directory does not exist: {args.data_dir}")

    print("Loading MolmoAct data...")
    episodes = load_molmoact_data(args.data_dir)
    if len(episodes) == 0:
        raise RuntimeError("No valid episodes found in the input directory.")

    total_frames = sum(ep["episode_length"] for ep in episodes)
    tasks = {ep["task_name"] for ep in episodes}
    print(f"Total episodes: {len(episodes)}")
    print(f"Total frames: {total_frames}")
    print(f"Task types: {tasks}")

    print("Converting to LeRobot v3.0 dataset...")
    create_lerobot_dataset_v30(
        episodes=episodes,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
    )

    print("Conversion completed.")


if __name__ == "__main__":
    main()