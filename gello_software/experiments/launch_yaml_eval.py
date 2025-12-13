import atexit
import signal
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Optional

from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_policy, make_pre_post_processors, get_policy_class
from lerobot.policies.utils import make_robot_action
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
import torch
import tyro
from omegaconf import OmegaConf

from gello.utils.launch_utils import instantiate_from_dict, move_to_start_position
import numpy as np

from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids
from gello.env import RobotEnv
from gello.utils.logging_utils import log_collect_demos
import logging
import os
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
DEVICE = os.environ.get("LEROBOT_TEST_DEVICE", "cuda") if torch.cuda.is_available() else "cpu"

# Global variables for cleanup
cleanup_in_progress = False

_env = None
_bimanual = False
_left_cfg = None
_right_cfg = None


def cleanup():
    """Clean up resources before exit."""
    global cleanup_in_progress
    if cleanup_in_progress:
        return
    cleanup_in_progress = True

    print("Cleaning up resources...")
    if _bimanual:
        move_to_start_position(_env, _bimanual, _left_cfg, _right_cfg)
    else:
        move_to_start_position(_env, _bimanual, _left_cfg)
    print("Cleanup completed.")


@dataclass
class Args:
    left_config_path: str
    """Path to the left arm configuration YAML file."""

    right_config_path: Optional[str] = None
    """Path to the right arm configuration YAML file (for bimanual operation)."""

    # use_save_interface: bool = False
    # """Enable saving data with keyboard interface."""


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    cleanup()
    import os

    os._exit(0)


def main():
    # Register cleanup handlers
    # If terminated without cleanup, can leave ZMQ sockets bound causing "address in use" errors or resource leaks

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = tyro.cli(Args)

    # left, right front camera (the device id order is based on the plugged in order on the adapter)
    ids = get_device_ids()
    print(f"Found {len(ids)} camera devices")
    print(ids)
    cameras = {
        "left_camera": RealSenseCamera(ids[0]),
        "front_camera": RealSenseCamera(ids[1]),
        "right_camera": RealSenseCamera(ids[2]),
    }

    bimanual = args.right_config_path is not None

    # Load configs
    left_cfg = OmegaConf.to_container(
        OmegaConf.load(args.left_config_path), resolve=True
    )

    if bimanual:
        right_cfg = OmegaConf.to_container(
            OmegaConf.load(args.right_config_path), resolve=True
        )

    # Initialize policy
    model_id = left_cfg["policy"]["checkpoint_path"]
    dataset_id = left_cfg["policy"]["repo_id"]
    ds_meta = LeRobotDatasetMetadata(dataset_id)
    
    # ==================== Option 1: Diffusion/DiT Policy ====================
    # policy = DiffusionPolicy.from_pretrained(model_id)
    # ds_meta = LeRobotDatasetMetadata(dataset_id)
    # preprocess, postprocess = make_pre_post_processors(
    #     policy.config, model_id, dataset_stats=ds_meta.stats
    # )
    
    # ==================== Option 2: PI05 Policy ====================
    policy = PI05Policy.from_pretrained(model_id)
    # policy.dataset_stats = ds_meta.stats
    preprocess, postprocess = make_pre_post_processors(
        policy.config, model_id, 
        # dataset_stats=ds_meta.stats
    )
    
    # Debug: Check dataset stats
    print(f"\n[DEBUG] Dataset ID: {dataset_id}")
    print(f"[DEBUG] Action stats q01: {ds_meta.stats['action']['q01'][:3]}")
    print(f"[DEBUG] Action stats q99: {ds_meta.stats['action']['q99'][:3]}")

    # Create robot(s)
    left_robot_cfg = left_cfg["robot"]
    if isinstance(left_robot_cfg.get("config"), str):
        left_robot_cfg["config"] = OmegaConf.to_container(
            OmegaConf.load(left_robot_cfg["config"]), resolve=True
        )

    left_robot = instantiate_from_dict(left_robot_cfg)

    if bimanual:
        from gello.robots.robot import BimanualRobot

        right_robot_cfg = right_cfg["robot"]
        if isinstance(right_robot_cfg.get("config"), str):
            right_robot_cfg["config"] = OmegaConf.to_container(
                OmegaConf.load(right_robot_cfg["config"]), resolve=True
            )

        right_robot = instantiate_from_dict(right_robot_cfg)
        robot = BimanualRobot(left_robot, right_robot)

        # For bimanual, use the left config for general settings (hz, etc.)
        cfg = left_cfg
    else:
        robot = left_robot
        cfg = left_cfg

    from gello.env import RobotEnv
    env = RobotEnv(robot, control_rate_hz=cfg.get("hz", 30), camera_dict=cameras)

    # Store global variables for cleanup
    global _env, _bimanual, _left_cfg, _right_cfg
    _env = env
    _bimanual = bimanual
    _left_cfg = left_cfg
    _right_cfg = right_cfg if bimanual else None

    # Move robot to start_joints position if specified in config
    from gello.utils.launch_utils import move_to_start_position

    if bimanual:
        move_to_start_position(env, bimanual, left_cfg, right_cfg)
    else:
        move_to_start_position(env, bimanual, left_cfg)

    print(
        f"Launching robot: {robot.__class__.__name__}"
    )
    print(f"Control loop: {cfg.get('hz', 30)} Hz")

    # ==================== Option 1: Run with Diffusion/DiT (no task needed) ====================
    # run_control_loop_eval_dit(env, policy=policy, preprocessor=preprocess, postprocessor=postprocess, ds_meta=ds_meta)

    # ==================== Option 2: Run with PI05 (requires task) ====================
    task_instruction = left_cfg.get("storage", {}).get("language_instruction", None)
    run_control_loop_eval_dit(env, policy=policy, preprocessor=preprocess, postprocessor=postprocess, ds_meta=ds_meta, task=task_instruction)


def run_control_loop_eval_dit(
    env: RobotEnv,
    policy: DiffusionPolicy = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] = None,
    ds_meta: LeRobotDatasetMetadata = None,
    task: str = None,
) -> None:
    """Run the main control loop.

    Args:
        env: Robot environment
        agent: Agent for control
        save_interface: Optional save interface for data collection
        print_timing: Whether to print timing information
        use_colors: Whether to use colored terminal output
        task: Language instruction for the task
    """
    start_time = time.time()
    obs = env.get_obs()
    policy.reset()
    logger.info("Starting policy inference...")

    while True:
        observation = preprocess_observation(obs)
        # observation = {key: observation[key].to(DEVICE, non_blocking=True) for key in observation}
        observation = {key: observation[key].to(DEVICE, non_blocking=True) for key in observation}
        
        # Add task to observation for PI05
        if task is not None:
            observation["task"] = [task]
        
        log_collect_demos("Running policy inference...", "info")
        start_time = time.time()
        
        # Debug: Check observation before preprocessing
        # print(f"\n[DEBUG] Before preprocess - keys: {observation.keys()}")
        
        observation = preprocessor(observation)
        
        # Debug: Check observation after preprocessing
        # print(f"[DEBUG] After preprocess - keys: {observation.keys()}")
        # if 'action' in observation:
        #     print(f"[DEBUG] Has 'action' key in observation")
        
        # Debug: Check if preprocessor/postprocessor have stats
        # print(f"[DEBUG] Postprocessor type: {type(postprocessor)}")
        
        actions = policy.select_action(observation)
        
        # Debug: Check action before postprocessing
        actions_before = actions.clone()
        # print(f"[DEBUG] Before postprocess - range: [{actions.min():.4f}, {actions.max():.4f}]")
        
        actions = postprocessor(actions)
        
        # Debug: Check action after postprocessing
        # print(f"[DEBUG] After postprocess - range: [{actions.min():.4f}, {actions.max():.4f}]")
        
        actions = actions.squeeze(0).detach().cpu().numpy()
        
        # Debug: Print action values
        # print(f"[DEBUG] Final action shape: {actions.shape}")
        # print(f"[DEBUG] Final action range: [{actions.min():.4f}, {actions.max():.4f}]")
        # print(f"[DEBUG] First 7 action dims: {actions[:7]}\n")
        
        inference_time = time.time() - start_time
        log_collect_demos(f"Policy inference completed in {inference_time:.3f}s", "success")
        log_collect_demos(f"Generated {len(actions)} action(s)", "data_info")
        obs = smooth_move_while_inference_envstep(env, actions)

def run_control_loop_eval(
    env: RobotEnv,
    policy: DiffusionPolicy = None,
) -> None:
    """Run the main control loop.

    Args:
        env: Robot environment
        agent: Agent for control
        save_interface: Optional save interface for data collection
        print_timing: Whether to print timing information
        use_colors: Whether to use colored terminal output
    """
    start_time = time.time()
    obs = env.get_obs()
    policy.reset()
    logger.info("Starting policy inference...")

    while True:
        observation = preprocess_observation(obs)
        # observation = {key: observation[key].to(DEVICE, non_blocking=True) for key in observation}
        observation = {key: observation[key].to(DEVICE, non_blocking=True) for key in observation}
        log_collect_demos("Running policy inference...", "info")
        start_time = time.time()
        actions = policy.select_action(observation)
        inference_time = time.time() - start_time
        log_collect_demos(f"Policy inference completed in {inference_time:.3f}s", "success")
        log_collect_demos(f"Generated {len(actions)} action(s)", "data_info")
        actions = actions.squeeze(0).detach().cpu().numpy()
        obs = smooth_move_while_inference_envstep(env, actions)

def smooth_move_while_inference_envstep(env: RobotEnv, action):
    current_joint = env.get_obs()["joint_positions"]
    target_joint = action

    steps = 5
    obs = None
    for i in range(steps + 1):
        alpha = i / steps  # Interpolation factor
        interpolated_joint = (1 - alpha) * current_joint + alpha * target_joint  # Linear interpolation
        obs = env.step(interpolated_joint)
        time.sleep(0.5 / steps)

    return obs

def preprocess_observation(observations: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return_observations = {}
    
    # Define the target size
    TARGET_HEIGHT = 360
    TARGET_WIDTH = 640

    # ==================== Option 1: Camera mapping for DiT/Diffusion Policy ====================
    # camera_mapping = {
    #     "left_camera_rgb": '1',    # Left camera -> observation.images.1
    #     "right_camera_rgb": '0',   # Right camera -> observation.images.0  
    #     "front_camera_rgb": '2'    # Front camera -> observation.images.2
    # }
    
    # ==================== Option 2: Camera mapping for PI05 Policy ====================
    camera_mapping = {
        "left_camera_rgb": 'left',    # Left camera -> observation.images.left
        "right_camera_rgb": 'right',  # Right camera -> observation.images.right  
        "front_camera_rgb": 'top'     # Front camera -> observation.images.top
    }
    
    for cam_name, cam_key in camera_mapping.items():
        if cam_name in observations:
            img_np = observations[cam_name]
            
            # 1. Convert NumPy array to PIL Image
            img_pil = Image.fromarray(img_np)
            
            # 2. Resize the image
            # The order in PIL.Image.resize is (width, height)
            img_resized_pil = img_pil.resize((TARGET_WIDTH, TARGET_HEIGHT))
            
            # 3. Convert resized PIL Image back to NumPy array
            img_resized_np = np.array(img_resized_pil)
            
            # 4. Convert to Tensor, permute, add batch dim, and normalize
            img_tensor = torch.from_numpy(img_resized_np).float().permute(2, 0, 1).cuda().unsqueeze(0)
            return_observations[f"observation.images.{cam_key}"] = img_tensor

    # Concatenate robot state
    state = observations["joint_positions"]
    state_tensor = torch.from_numpy(state).float().cuda().unsqueeze(0)  # 1,N
    return_observations["observation.state"] = state_tensor

    # # for dit only
    # return_observations["task"] = "fold towel"

    return return_observations


if __name__ == "__main__":
    main()
