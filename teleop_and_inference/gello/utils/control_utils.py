"""Shared utilities for robot control loops."""

import datetime
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tqdm
import pygame

from gello.agents.agent import Agent
from gello.env import RobotEnv
import logging

from gello.data_utils.data_saver import DataSaver
from gello.data_utils.keyboard_interface import KBReset
from gello.data_utils.data_saver_thread import EpisodeSaverThread

# Import LeRobot data saver thread if available
try:
    from gello.data_utils.lerobot_data_saver_thread import LeRobotDataSaverThread
    LEROBOT_THREAD_AVAILABLE = True
except ImportError:
    LEROBOT_THREAD_AVAILABLE = False
DEFAULT_MAX_JOINT_DELTA = 1.0
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_obs_to_lerobot_format(obs, next_obs):
    """
    Convert YAM observation format to LeRobot format.
    
    Args:
        obs: YAM observation dictionary with keys:
            - 'joint_positions': array(14,) [left_joints(6), left_gripper(1), right_joints(6), right_gripper(1)]
            - 'left_camera_rgb': array(H, W, 3)
            - 'front_camera_rgb': array(H, W, 3)
            - 'right_camera_rgb': array(H, W, 3)
        next_obs: Next observation for computing gripper positions
    
    Returns:
        Dictionary in LeRobot format with keys:
            - 'left': {'joint_pos': array(6,), 'gripper_pos': array(1,)}
            - 'right': {'joint_pos': array(6,), 'gripper_pos': array(1,)}
            - 'left_camera': array(H, W, 3)
            - 'front_camera': array(H, W, 3)
            - 'right_camera': array(H, W, 3)
    """
    joint_positions = obs['joint_positions']
    
    # Split joint positions: [left(7), right(7)]
    left_joints = joint_positions[:6]
    left_gripper = joint_positions[6:7]  # Keep as array
    right_joints = joint_positions[7:13]
    right_gripper = joint_positions[13:14]  # Keep as array
    
    lerobot_obs = {
        'left': {
            'joint_pos': left_joints,
            'gripper_pos': left_gripper,
        },
        'right': {
            'joint_pos': right_joints,
            'gripper_pos': right_gripper,
        },
        'left_camera': obs.get('left_camera_rgb'),
        'front_camera': obs.get('front_camera_rgb'),
        'right_camera': obs.get('right_camera_rgb'),
    }
    
    return lerobot_obs


def convert_action_to_lerobot_format(action):
    """
    Convert YAM action format to LeRobot format.
    
    Args:
        action: array(14,) [left(7), right(7)] where each arm has [joints(6), gripper(1)]
    
    Returns:
        Dictionary with keys:
            - 'left': {'pos': array(7,)}
            - 'right': {'pos': array(7,)}
    """
    left_action = action[:7]
    right_action = action[7:14]
    
    lerobot_action = {
        'left': {'pos': left_action},
        'right': {'pos': right_action},
    }
    
    return lerobot_action


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
    
    # kb_interface already has pygame window with status display
    # Update initial status
    kb_interface.update_status(
        total_episodes=left_cfg['storage']['episodes'],
        state='Waiting',
        message='Press S to start collecting'
    )

    # Check if using LeRobot data saver
    is_lerobot = hasattr(data_saver, 'dataset')  # LeRobotDataSaver has 'dataset' attribute
    
    # for data collection
    if is_lerobot:
        # LeRobot format - no separate thread needed, save synchronously
        # Video encoding happens in LeRobotDataset's internal background thread
        saver_thread = None
        logger.info("Using LeRobot format (direct save, async video encoding)")
    else:
        saver_thread = EpisodeSaverThread(data_saver)
        saver_thread.start()
    num_traj = 1
    
    # Use the actual data collection frequency from config
    hz = left_cfg.get('hz', 30)
    logger.info(f"Data collection frequency: {hz} Hz")

    start_time = time.time()
    last_save_time = time.time()
    frame_count = 0

    # Set FIXED reference position for relative control
    # Use start_joints as the reference, not current Gello position
    left_start = np.array(left_cfg["agent"]["start_joints"])
    if right_cfg is not None:
        right_start = np.array(right_cfg["agent"]["start_joints"])
        reference_joints = np.concatenate([left_start, right_start])
    else:
        reference_joints = left_start
    
    logger.info(f"Using fixed reference position: {reference_joints}")
    # Don't set offset yet - we'll do it when user presses 's'
    # This prevents robot from moving when the script starts

    while num_traj <= left_cfg['storage']['episodes']:
        obs = env.get_obs()
        
        # Update status display in kb_interface
        left_joints = obs.get('joint_positions', np.zeros(14))[:7]
        right_joints = obs.get('joint_positions', np.zeros(14))[7:14]
        kb_interface.update_status(
            episode=num_traj - 1,
            left_arm=left_joints.tolist(),
            right_arm=right_joints.tolist(),
        )
        
        # Reset buffer based on data saver type
        if is_lerobot:
            # LeRobot doesn't use buffer pattern, just clear episode_started flag
            data_saver.episode_started = False
        else:
            data_saver.reset_buffer()
        
        frame_count = 0

        if print_timing:
            num = time.time() - start_time
            message = f"\rTime passed: {round(num, 2)}     "
            print(message, end="", flush=True)

        logger.info(f"Press 's' to start collecting data: ")
        while True:
            # Update pygame display while waiting
            obs = env.get_obs()
            left_joints = obs.get('joint_positions', np.zeros(14))[:7]
            right_joints = obs.get('joint_positions', np.zeros(14))[7:14]
            kb_interface.update_status(
                episode=num_traj,
                state='Waiting',
                message='Press S to start collecting',
                left_arm=left_joints.tolist(),
                right_arm=right_joints.tolist(),
            )
            
            result = kb_interface.update()
            if result == "start":
                logger.info(f"Successfully pressed 's', starting to collect data")
                # Set the offset NOW when user presses 's'
                # Calculate offset so that current Gello reading maps to reference position
                current_gello = agent.act(env.get_obs())
                env.set_original_offset(current_gello)  # This sets offset = gello - robot_state
                # Now robot will follow Gello movements relative to current position
                time.sleep(0.5)
                obs = env.get_obs()
                break
            time.sleep(0.033)  # ~30Hz update rate
        logger.info(f"Press 'a' to save the data, press 'b' to discard the data")
        
        # Update status display for collecting state
        kb_interface.update_status(
            state='Collecting',
            episode=num_traj,
            frames_collected=0,
            message='Collecting... Press A to save, B to discard'
        )
        
        fps_start_time = time.time()

        for _ in tqdm.tqdm(range(left_cfg['collection']['max_episode_length']), desc=f"Collecting data {num_traj}/{left_cfg['storage']['episodes']}"):
            result = kb_interface.update()
            if result == "save" or result == "discard":
                break
            else:
                action = agent.act(obs)
                next_obs = env.step(action)
                if time.time() - last_save_time > (1/hz):
                    if is_lerobot:
                        # Convert observation to LeRobot format
                        lerobot_obs = convert_obs_to_lerobot_format(obs, next_obs)
                        lerobot_action = convert_action_to_lerobot_format(action)
                        data_saver.add_frame(lerobot_obs, lerobot_action)
                    else:
                        obs["next_joint"] = next_obs["joint_positions"]
                        data_saver.add_observation(obs)
                    last_save_time = time.time()
                    frame_count += 1
                    
                    # Update status data (without drawing to avoid blocking)
                    if frame_count % 10 == 0:  # Update every 10 frames
                        elapsed = time.time() - fps_start_time
                        current_fps = frame_count / elapsed if elapsed > 0 else 0
                        left_joints = next_obs.get('joint_positions', np.zeros(14))[:7]
                        right_joints = next_obs.get('joint_positions', np.zeros(14))[7:14]
                        kb_interface.update_status(
                            frames_collected=frame_count,
                            fps=current_fps,
                            left_arm=left_joints.tolist(),
                            right_arm=right_joints.tolist(),
                        )
                        # Drawing happens in kb_interface.update() call above
                obs = next_obs.copy()

        if result == "save":
            # Update status for saving
            kb_interface.update_status(
                state='Saving',
                message='Saving episode...'
            )
            
            if is_lerobot:
                # LeRobot format - save directly (video encoding happens in background automatically)
                if data_saver.episode_started:
                    logger.info(f"Saving LeRobot episode {num_traj}...")
                    data_saver.save_episode()
                    num_traj += 1
                    logger.info(f"Episode saved! (Video encoding continues in background)")
                else:
                    logger.info(f"No data collected, skipping save")
            else:
                # For legacy format
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
    
    # Stop saver thread if using legacy format
    if saver_thread is not None:
        saver_thread.stop()
    
    # For LeRobot format, return data_saver so caller can finalize after cleanup
    # For legacy format, finalization is already done
    logger.info(f"Finished collecting data")
    return data_saver if is_lerobot else None