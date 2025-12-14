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
    policy.dataset_stats = ds_meta.stats
    preprocess, postprocess = make_pre_post_processors(
        policy.config, model_id, 
        dataset_stats=ds_meta.stats
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
    run_control_loop_eval_processor(env, policy=policy, preprocessor=preprocess, postprocessor=postprocess, ds_meta=ds_meta, task=task_instruction)


def run_control_loop_eval_dit(
    env: RobotEnv,
    policy: DiffusionPolicy = None,
) -> None:
    """Run the main control loop for DiT/Diffusion using manual preprocessing.

    Args:
        env: Robot environment
        policy: Policy for inference
    """
    start_time = time.time()
    obs = env.get_obs()
    policy.reset()
    logger.info("Starting policy inference...")

    while True:
        observation = preprocess_observation(obs)
        observation = {key: observation[key].to(DEVICE, non_blocking=True) for key in observation}
        log_collect_demos("Running policy inference...", "info")
        start_time = time.time()
        actions = policy.select_action(observation)
        inference_time = time.time() - start_time
        log_collect_demos(f"Policy inference completed in {inference_time:.3f}s", "success")
        log_collect_demos(f"Generated {len(actions)} action(s)", "data_info")
        actions = actions.squeeze(0).detach().cpu().numpy()
        obs = smooth_move_while_inference_envstep(env, actions)


def run_control_loop_eval_processor(
    env: RobotEnv,
    policy: DiffusionPolicy = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] = None,
    ds_meta: LeRobotDatasetMetadata = None,
    task: str = None,
) -> None:
    """Run the main control loop using preprocessor pipeline (for PI05) with action chunking.

    Args:
        env: Robot environment
        policy: Policy for inference
        preprocessor: Preprocessor pipeline
        postprocessor: Postprocessor pipeline
        ds_meta: Dataset metadata
        task: Language instruction for the task
    """
    from collections import deque
    
    # Action queue configuration (matching PI05 async inference)
    actions_per_chunk = 30  # PI05 outputs 30 actions per inference
    chunk_size_threshold = 0.0  # Request new inference when queue is empty
    action_queue = deque()  # Queue to store future actions
    current_timestep = 0
    
    # Control frequency configuration
    control_dt = 1.0 / 30.0  # 30Hz control loop
    
    policy.reset()
    logger.info("Starting policy inference with action chunking...")
    log_collect_demos(f"Action chunking: {actions_per_chunk} actions/chunk, threshold: {chunk_size_threshold}", "info")
    log_collect_demos(f"Control frequency: {1.0/control_dt:.0f} Hz", "info")

    while True:
        cycle_start = time.time()
        
        # Step 1: Execute one action from queue (if available) - do this FIRST like PI05
        if len(action_queue) > 0:
            action = action_queue.popleft()
            current_timestep += 1
            env.step(action)  # Direct step, no smooth_move interpolation
        else:
            if current_timestep > 0:  # Only warn after first inference
                log_collect_demos("WARNING: Action queue empty, waiting for inference...", "warning")
        
        # Step 2: Get current observation (after executing action, to get latest state)
        obs = env.get_obs()
        
        # Step 3: Check if we need new inference (queue below threshold)
        queue_size = len(action_queue)
        need_inference = queue_size <= int(chunk_size_threshold * actions_per_chunk)
        
        if need_inference:
            log_collect_demos(f"Queue size: {queue_size}/{actions_per_chunk}, requesting new inference...", "info")
            
            # Prepare observation following lerobot_pi05 official format
            observation = {}
            
            # Process images: resize → tensor → /255 → permute → unsqueeze → GPU
            TARGET_HEIGHT = 360
            TARGET_WIDTH = 640
            camera_mapping = {
                "left_camera_rgb": 'left',
                "right_camera_rgb": 'right',  
                "front_camera_rgb": 'top'
            }
            
            for cam_name, cam_key in camera_mapping.items():
                if cam_name in obs:
                    img_np = obs[cam_name]
                    # Resize using PIL to match policy config
                    img_pil = Image.fromarray(img_np)
                    img_resized_pil = img_pil.resize((TARGET_WIDTH, TARGET_HEIGHT))
                    img_resized_np = np.array(img_resized_pil)
                    
                    # Convert to tensor, normalize, permute, add batch dim, move to GPU
                    img_tensor = torch.from_numpy(img_resized_np).float() / 255.0
                    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                    observation[f"observation.images.{cam_key}"] = img_tensor
            
            # Process state: tensor → unsqueeze → GPU
            state = obs["joint_positions"]
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            observation["observation.state"] = state_tensor
            
            # Add task to observation for PI05
            if task is not None:
                observation["task"] = [task]
            
            inference_start = time.time()
            
            # Preprocess and run inference
            observation = preprocessor(observation)
            actions_chunk = policy.predict_action_chunk(observation)  # Returns [1, 50, 14] (full chunk)
            actions_chunk = actions_chunk[:, :actions_per_chunk, :]  # Take only first 30 actions
            
            # Apply postprocessor to each action in the chunk (postprocessor expects (B, action_dim))
            _, chunk_size, _ = actions_chunk.shape
            processed_actions = []
            for i in range(chunk_size):
                single_action = actions_chunk[:, i, :]  # (B, action_dim)
                processed_action = postprocessor(single_action)  # Denormalize
                processed_actions.append(processed_action)
            
            # Stack back and convert to numpy
            actions_chunk = torch.stack(processed_actions, dim=1).squeeze(0).detach().cpu().numpy()  # [30, 14]
            
            inference_time = time.time() - inference_start
            log_collect_demos(f"Policy inference completed in {inference_time:.3f}s, generated {len(actions_chunk)} actions", "success")
            
            # Add all actions to queue
            for action in actions_chunk:
                action_queue.append(action)
        
        # Step 4: Maintain control frequency (like PI05's dynamic sleep)
        cycle_duration = time.time() - cycle_start
        sleep_time = max(0, control_dt - cycle_duration)
        if sleep_time > 0:
            time.sleep(sleep_time)


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
            
            # 4. Convert to Tensor, permute, add batch dim, and normalize to [0,1]
            img_tensor = torch.from_numpy(img_resized_np).float().permute(2, 0, 1).cuda().unsqueeze(0) / 255.0
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
