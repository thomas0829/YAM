"""Shared utilities for robot control loops."""

from calendar import c
from copy import deepcopy
import datetime
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
import numpy as np
import torch
import tqdm

from gello.agents.agent import Agent
from gello.env import RobotEnv
import logging

from gello.data_utils.data_saver import DataSaver
from gello.data_utils.keyboard_interface import KBReset
from gello.data_utils.data_saver_thread import EpisodeSaverThread
from gello.utils.launch_utils import instantiate_from_dict
from gello.dynamixel.driver import DynamixelDriver
from gello.utils.logging_utils import log_collect_demos
from gello.data_utils.data_replay import DataReplayer
DEFAULT_MAX_JOINT_DELTA = 1.0
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def move_to_start_position(
    env: RobotEnv, agent: Agent, max_delta: float = 1.0, steps: int = 25
) -> bool:
    """Move robot to start position gradually.

    Args:
        env: Robot environment
        agent: Agent that provides target position
        max_delta: Maximum joint delta per step
        steps: Number of steps for gradual movement

    Returns:
        bool: True if successful, False if position too far
    """
    print("Going to start position")
    start_pos = agent.act(env.get_obs())
    obs = env.get_obs()
    joints = obs["joint_positions"]

    abs_deltas = np.abs(start_pos - joints)
    id_max_joint_delta = np.argmax(abs_deltas)

    max_joint_delta = DEFAULT_MAX_JOINT_DELTA
    if abs_deltas[id_max_joint_delta] > max_joint_delta:
        id_mask = abs_deltas > max_joint_delta
        print()
        ids = np.arange(len(id_mask))[id_mask]
        for i, delta, joint, current_j in zip(
            ids,
            abs_deltas[id_mask],
            start_pos[id_mask],
            joints[id_mask],
        ):
            print(
                f"joint[{i}]: \t delta: {delta:4.3f} , leader: \t{joint:4.3f} , follower: \t{current_j:4.3f}"
            )
        return False

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    for _ in range(steps):
        obs = env.get_obs()
        command_joints = agent.act(obs)
        current_joints = obs["joint_positions"]
        delta = command_joints - current_joints
        max_joint_delta = np.abs(delta).max()
        if max_joint_delta > max_delta:
            delta = delta / max_joint_delta * max_delta
        env.step(current_joints + delta)

    return True


class SaveInterface:
    """Handles keyboard-based data saving interface."""

    def __init__(
        self,
        data_dir: str = "data",
        agent_name: str = "Agent",
        expand_user: bool = False,
    ):
        """Initialize save interface.

        Args:
            data_dir: Base directory for saving data
            agent_name: Name of agent (used for subdirectory)
            expand_user: Whether to expand ~ in data_dir path
        """
        from gello.data_utils.keyboard_interface import KBReset

        self.kb_interface = KBReset()
        self.data_dir = Path(data_dir).expanduser() if expand_user else Path(data_dir)
        self.agent_name = agent_name
        self.save_path: Optional[Path] = None

        print("Save interface enabled. Use keyboard controls:")
        print("  S: Start recording")
        print("  Q: Stop recording")

    def update(self, obs: Dict[str, Any], action: np.ndarray) -> Optional[str]:
        """Update save interface and handle saving.

        Args:
            obs: Current observations
            action: Current action

        Returns:
            Optional[str]: "quit" if user wants to exit, None otherwise
        """
        from gello.data_utils.format_obs import save_frame

        dt = datetime.datetime.now()
        state = self.kb_interface.update()

        if state == "start":
            dt_time = datetime.datetime.now()
            self.save_path = (
                self.data_dir / self.agent_name / dt_time.strftime("%m%d_%H%M%S")
            )
            self.save_path.mkdir(parents=True, exist_ok=True)
            print(f"Saving to {self.save_path}")
        elif state == "save":
            if self.save_path is not None:
                save_frame(self.save_path, dt, obs, action)
        elif state == "normal":
            self.save_path = None
        elif state == "quit":
            print("\nExiting.")
            return "quit"
        else:
            raise ValueError(f"Invalid state {state}")

        return None


def run_control_loop(
    env: RobotEnv,
    agent: Agent,
    save_interface: Optional[SaveInterface] = None,
    print_timing: bool = True,
    use_colors: bool = False,
) -> None:
    """Run the main control loop.

    Args:
        env: Robot environment
        agent: Agent for control
        save_interface: Optional save interface for data collection
        print_timing: Whether to print timing information
        use_colors: Whether to use colored terminal output
    """
    # Check if we can use colors
    colors_available = False
    if use_colors:
        try:
            from termcolor import colored

            colors_available = True
            start_msg = colored("\nStart 🚀🚀🚀", color="green", attrs=["bold"])
        except ImportError:
            start_msg = "\nStart 🚀🚀🚀"
    else:
        start_msg = "\nStart 🚀🚀🚀"

    print(start_msg)

    start_time = time.time()
    obs = env.get_obs()

    while True:
        if print_timing:
            num = time.time() - start_time
            message = f"\rTime passed: {round(num, 2)}          "

            if colors_available:
                print(
                    colored(message, color="white", attrs=["bold"]), end="", flush=True
                )
            else:
                print(message, end="", flush=True)

        action = agent.act(obs)

        # Handle save interface
        if save_interface is not None:
            result = save_interface.update(obs, action)
            if result == "quit":
                break

        obs = env.step(action)

def run_control_loop_prior(
    env: RobotEnv,
    agent: Agent,
    left_cfg: dict = None,
    right_cfg: Optional[dict] = None,
    print_timing: bool = True,
    use_colors: bool = False,
    data_saver: DataSaver = None,
    kb_interface: KBReset = None,
) -> None:
    """Run the main control loop.

    Args:
        env: Robot environment
        agent: Agent for control
        save_interface: Optional save interface for data collection
        print_timing: Whether to print timing information
        use_colors: Whether to use colored terminal output
    """
    # Check if we can use colors
    colors_available = False
    if use_colors:
        try:
            from termcolor import colored

            colors_available = True
            start_msg = colored("\nStart 🚀🚀🚀", color="green", attrs=["bold"])
        except ImportError:
            start_msg = "\nStart 🚀🚀🚀"
    else:
        start_msg = "\nStart 🚀🚀🚀"

    print(start_msg)

    # for data collection
    saver_thread = EpisodeSaverThread(data_saver)
    saver_thread.start()
    logger = logging.getLogger(__name__)
    num_traj = 1
    hz = 10

    start_time = time.time()
    last_save_time = time.time()

    while num_traj <= left_cfg['storage']['episodes']:
        obs = env.get_obs()
        data_saver.reset_buffer()

        if print_timing:
            num = time.time() - start_time
            message = f"\rTime passed: {round(num, 2)}     "
            print(message, end="", flush=True)

        logger.info(f"Press 's' to start collecting data: ")
        while True:
            result = kb_interface.update()
            if result == "start":
                logger.info(f"Successfully pressed 's', starting to collect data")
                time.sleep(1)
                obs = env.get_obs()
                break
        logger.info(f"Press 'a' to save the data, press 'b' to discard the data")

        for _ in tqdm.tqdm(range(left_cfg['collection']['max_episode_length']), desc=f"Collecting data {num_traj}/{left_cfg['storage']['episodes']}"):
            result = kb_interface.update()
            if result == "save" or result == "discard":
                break
            else:
                action = agent.act(obs)
                next_obs = env.step(action)
                if time.time() - last_save_time > (1/hz):
                    obs["next_joint"] = next_obs["joint_positions"]
                    data_saver.add_observation(obs)
                    last_save_time = time.time()
                obs = next_obs.copy()

        if result == "save":
            if data_saver.buffer == []:
                logger.info(f"No data collected, skipping save")
                continue
            saver_thread.save_episode(data_saver.buffer.copy())
            num_traj += 1
            logger.info(f"Successfully collected data")
        else:
            logger.info(f"Failure")
        from gello.utils.launch_utils import move_to_start_position
        if right_cfg is not None:
            move_to_start_position(env, agent, left_cfg=left_cfg, right_cfg=right_cfg)
        else:
            move_to_start_position(env, agent, left_cfg=left_cfg)

    
    saver_thread.stop()
    saver_thread.join()
    logger.info(f"Finished collecting data")

DEVICE = os.environ.get("LEROBOT_TEST_DEVICE", "cuda") if torch.cuda.is_available() else "cpu"
def run_control_loop_eval(
    env: RobotEnv,
    agent: Agent,
    left_cfg: dict = None,
    right_cfg: Optional[dict] = None,
    policy: DiffusionPolicy = None,
    print_timing: bool = True,
    use_colors: bool = False,
) -> None:
    """Run the main control loop.

    Args:
        env: Robot environment
        agent: Agent for control
        save_interface: Optional save interface for data collection
        print_timing: Whether to print timing information
        use_colors: Whether to use colored terminal output
    """
    # Check if we can use colors
    colors_available = False
    if use_colors:
        try:
            from termcolor import colored

            colors_available = True
            start_msg = colored("\nStart 🚀🚀🚀", color="green", attrs=["bold"])
        except ImportError:
            start_msg = "\nStart 🚀🚀🚀"
    else:
        start_msg = "\nStart 🚀🚀🚀"

    print(start_msg)

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
        obs = env.step(actions)

def preprocess_observation(observations: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return_observations = {}
    
    # Define the target size
    TARGET_HEIGHT = 256
    TARGET_WIDTH = 342

    # Map cameras
    camera_mapping = {"image_left_rgb": 'left', "image_right_rgb": 'right', "image_front_rgb": 'front'}
    for cam_name, cam_idx in camera_mapping.items():
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
            return_observations[f"observation.images.camera_{cam_idx}"] = img_tensor

    # Concatenate robot state
    state = np.concatenate([
        observations["left_joint"],
        observations["right_joint"],
    ])
    state_tensor = torch.from_numpy(state).float().cuda().unsqueeze(0)  # 1,N
    return_observations["observation.state"] = state_tensor

    return return_observations


def run_control_loop_eval_open_loop(
    env: RobotEnv,
    agent: Agent,
    left_cfg: dict = None,
    right_cfg: Optional[dict] = None,
    policy: DiffusionPolicy = None,
    data_replayer: DataReplayer = None,
) -> None:
    """Run the main control loop.
    """
    logger.info("Starting policy inference...")
    # Init environment and warm up agent
    policy.reset()
    obs = env.get_obs()

    # Main control loop
    demo_length = data_replayer.get_demo_length()
    obs_index = 0
    while obs_index < demo_length:
        obs = data_replayer.get_observation(obs_index)
        input_dict = preprocess_observation(obs)
        log_collect_demos("Running policy inference...", "info")

        start_time = time.time()
        actions = policy.select_action(input_dict)
        inference_time = time.time() - start_time
        log_collect_demos(f"Policy inference completed in {inference_time:.3f}s", "success")
        log_collect_demos(f"Generated {len(actions)} action(s)", "data_info")
        actions = actions.squeeze(0).detach().cpu().numpy()
        obs = env.step(actions)
        obs_index += 1
    logger.info("Finished policy inference")
