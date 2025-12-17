# Collect data for a single task. Pair with oculus viser agent.
import os
import sys
import cv2

import numpy as np
from scipy.spatial.transform import Slerp, Rotation as R

from yam_realtime.agents.agent import Agent
from yam_realtime.envs.configs.instantiate import instantiate
from yam_realtime.envs.configs.loader import DictLoader
from yam_realtime.robots.utils import Rate, Timeout
from yam_realtime.utils.launch_utils import cleanup_processes, initialize_agent, initialize_robots, initialize_sensors, setup_can_interfaces, setup_logging
from yam_realtime.utils.camera_thread import EpisodeSaverThread
from yam_realtime.utils.safety_checker import SafetyChecker

import tqdm
from yam_realtime.robots.inverse_kinematics.yam_pyroki import YamPyroki
from yam_realtime.agents.teleoperation.oculus_viser_agent import OculusViserAgent
from yam_realtime.envs.robot_env import RobotEnv
from yam_realtime.utils.data_saver import DataSaver

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
from yam_realtime.sensors.cameras.camera import CameraDriver
from yam_realtime.robots.robot import Robot
import logging
import time
import tyro
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class LaunchConfig:
    hz: float = 30.0
    cameras: Dict[str, Tuple[CameraDriver, int]] = field(default_factory=dict)
    robots: Dict[str, Union[str, Robot]] = field(default_factory=dict)
    max_steps: Optional[int] = None  # this is for testing
    save_path: Optional[str] = None
    station_metadata: Dict[str, str] = field(default_factory=dict)
    storage: Dict[str, Any] = field(default_factory=dict)
    collection: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Args:
    config_path: Tuple[str, ...] = ("~/thomas/YAM/yam_realtime/configs/yam_record_replay.yaml",)
    

def main(args: Args):
    logger = setup_logging()
    logger.info("Starting YAM realtime control system...")

    server_processes = []
    try:
        logger.info("Loading configuration...")
        configs_dict = DictLoader.load([os.path.expanduser(x) for x in args.config_path])

        agent_cfg = configs_dict.pop("agent")
        sensors_cfg = configs_dict.pop("sensors", None)
        safety_cfg = configs_dict.pop("safety", {})  # Extract safety config before instantiate
        main_config = instantiate(configs_dict)

        logger.info("Initializing sensors...")
        camera_dict, camera_info = initialize_sensors(sensors_cfg, server_processes)

        setup_can_interfaces()

        logger.info("Initializing robots...")
        robots = initialize_robots(main_config.robots, server_processes)

        agent = initialize_agent(agent_cfg, server_processes)

        logger.info("Creating robot environment...")
        frequency = main_config.hz
        rate = Rate(frequency, rate_name="control_loop")
        logger.info(f"camera_dict: {camera_dict}")

        env = RobotEnv(
            robot_dict=robots,
            camera_dict=camera_dict,
            control_rate_hz=rate,
        )

        # Camera preview
        logger.info("=" * 50)
        logger.info("Camera Preview - Press ENTER to start recording")
        logger.info("=" * 50)
        preview_camera(env)
        logger.info("Starting data collection...")

        reset_robot(agent, env, 'left')

        logger.info("Starting control loop...")
        data_saver = DataSaver(task_directory=configs_dict['storage']['task_directory'], language_instruction=configs_dict['storage']['language_instruction'])
        _run_control_loop(env, agent, main_config, configs_dict, data_saver, safety_cfg)


    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise e
    finally:
        # Cleanup
        logger.info("Shutting down...")
        if "env" in locals():
            env.close()
        if "agent" in locals():
            cleanup_processes(agent, server_processes)


def preview_camera(env: RobotEnv):
    """Preview camera feed until user presses Enter in the preview window"""
    
    logger.info("Showing camera preview. Press ENTER key in the preview window to start recording...")
    
    while True:
        obs = env.get_obs()
        
        # Find all camera feeds
        camera_feeds = {}
        for key in obs.keys():
            if 'camera' in key and 'images' in obs[key] and 'rgb' in obs[key]['images']:
                camera_feeds[key] = obs[key]['images']['rgb']
        
        if not camera_feeds:
            logger.warning("No camera feeds found!")
            break
        
        # Display each camera feed
        for cam_name, frame in camera_feeds.items():
            # Convert RGB to BGR for OpenCV display
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # Add text instruction on the frame
            cv2.putText(frame_bgr, 'Press ENTER to start recording', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(f'{cam_name}', frame_bgr)
        
        # Wait for Enter key (13) or Return key
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 10:  # Enter key
            break
    
    cv2.destroyAllWindows()
    logger.info("Camera preview closed. Starting recording...")

# slowly move the robot back to original position
def reset_robot(agent: Agent, env: RobotEnv, side: str):
    agent.act({})
    current_pos = env.robot(side).get_joint_pos()
    target_joint_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    steps = 50
    for i in range(steps + 1):
        alpha = i / steps  # Interpolation factor
        target_pos = (1 - alpha) * current_pos + alpha * target_joint_positions  # Linear interpolation
        env.robot(side).command_joint_pos(target_pos)
        time.sleep(2 / steps)

def _run_control_loop(env: RobotEnv, agent: Agent, config: LaunchConfig, configs_dict: Dict, data_saver: DataSaver, safety_cfg: Dict) -> None:
    """
    Run the main control loop.

    Args:
        env: Robot environment
        agent: Agent instance
        config: Configuration object
    """
    saver_thread = EpisodeSaverThread(data_saver)
    saver_thread.start()

    logger = logging.getLogger(__name__)
    num_traj = 0

    # Initialize safety checker
    safety_checker = SafetyChecker(
        max_joint_delta=safety_cfg.get('max_joint_delta', 0.2),
        max_velocity=safety_cfg.get('max_velocity', 2.0),
        max_gripper_delta=safety_cfg.get('max_gripper_delta', 1.0),
        enable_safety=safety_cfg.get('enable_safety', True),
        dt=safety_cfg.get('dt', 0.1),
        max_violations=safety_cfg.get('max_violations', 100),
        clamp_actions=safety_cfg.get('clamp_actions', False),
        warmup_steps=safety_cfg.get('warmup_steps', 20)
    )
    
    # Init environment and warm up agent
    obs = env.reset()
    logger.info(f"Action spec: {env.action_spec()}")
    
    # Reset safety checker for first episode
    safety_checker.reset()

    # Main control loop
    try:
        while num_traj < configs_dict['storage']['episodes']:
            obs = env.reset()
            data_saver.reset_buffer()
            # Safety checker is already reset after previous episode's reset_robot

            logger.info(f"Press 'A' to start collecting data: ")

            while True:
                info = agent.get_info()
                if info["success"]:
                    logger.info(f"Successfully pressed 'A', starting to collect data")
                    time.sleep(2)
                    break
            logger.info(f"Press 'A' to save the data, press 'B' to discard the data")

            for _ in tqdm.tqdm(range(configs_dict['collection']['max_episode_length']), desc=f"Collecting data {num_traj}/{configs_dict['storage']['episodes']}"):
                info = agent.get_info()
                # In single-arm mode with right controller, check 'right' movement_enabled
                while (not info["success"] and not info["failure"]) and not info["movement_enabled"]['right']:
                    info = agent.get_info()
                
                save = False
                if info["success"]:
                    save = True
                    break
                elif info["failure"]:
                    break

                if info["movement_enabled"]['right']:
                    act = agent.act(obs)
                    action = {'left': {'pos':act['left']['pos']}}
                    
                    # Safety check - get current joint positions
                    current_joint_pos = None
                    if 'left' in obs and 'joint_pos' in obs['left']:
                        # Combine joint_pos and gripper_pos to match action format
                        joint_pos = obs['left']['joint_pos']
                        gripper_pos = obs['left'].get('gripper_pos', np.array([0.0]))
                        current_joint_pos = {'left': np.concatenate([joint_pos, gripper_pos])}
                    
                    is_safe, reason, safe_action = safety_checker.check_action(action, current_joint_pos)
                    
                    if not is_safe and safe_action is None:
                        # Critical safety violation - stop episode
                        logger.error(f"Critical safety violation: {reason}")
                        logger.error("Stopping episode for safety. Press 'B' to discard.")
                        break
                    
                    # Use the safe action (either original if safe, or clamped version)
                    if safe_action is not None:
                        action = safe_action
                    
                    data_saver.add_observation(obs, act)
                    next_obs = env.step(action)
                    obs = next_obs.copy()
            
            if save:
                if data_saver.buffer == []:
                    logger.info(f"No data collected, skipping save")
                    continue
                saver_thread.save_episode(data_saver.buffer.copy())
                num_traj +=1
                logger.info(f"Successfully collected data")
            else:
                logger.info(f"Failure")
            
            # Reset safety checker BEFORE resetting robot to avoid counting reset movements
            safety_checker.reset()
            reset_robot(agent, env, 'left')
    finally:
        logger.info("Waiting for all episodes to finish saving...")
        saver_thread.stop()
        saver_thread.join()
        logger.info("All episodes saved.")

    env.reset()
    logger.info(f"Finished collecting data")

if __name__ == "__main__":
    main(tyro.cli(Args))
