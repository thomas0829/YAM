# Collect data for a single task. Pair with oculus viser agent.
import os
import sys
from tkinter import Y

# Set robot_descriptions commit to avoid git checkout issues
os.environ['ROBOT_DESCRIPTION_COMMIT'] = '7809b5b'

sys.path.append('/home/sean/Desktop/YAM')
sys.path.append('/home/sean/Desktop/YAM/yam_realtime')
import numpy as np
from scipy.spatial.transform import Slerp, Rotation as R

from yam_realtime.agents.agent import Agent
from yam_realtime.envs.configs.instantiate import instantiate
from yam_realtime.envs.configs.loader import DictLoader
from yam_realtime.robots.utils import Rate, Timeout
from yam_realtime.utils.launch_utils import cleanup_processes, initialize_agent, initialize_robots, initialize_sensors, setup_can_interfaces, setup_logging

import tqdm
from yam_realtime.robots.inverse_kinematics.yam_pyroki import YamPyroki
from yam_realtime.agents.teleoperation.oculus_viser_agent import OculusViserAgent
from yam_realtime.envs.robot_env import RobotEnv

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union
from yam_realtime.sensors.cameras.camera import CameraDriver
from yam_realtime.robots.robot import Robot
import logging
import time
import tyro

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
    config_path: Tuple[str, ...] = ("../../configs/yam_record_replay.yaml",)
    

def main(args: Args):
    logger = setup_logging()
    logger.info("Starting YAM realtime control system...")

    server_processes = []
    try:
        logger.info("Loading configuration...")
        configs_dict = DictLoader.load([os.path.expanduser(x) for x in args.config_path])

        agent_cfg = configs_dict.pop("agent")
        sensors_cfg = configs_dict.pop("sensors", None)
        main_config = instantiate(configs_dict)

        # Camera is handled by separate process, no need to initialize here
        # logger.info("Initializing sensors...")
        # camera_dict, camera_info = initialize_sensors(sensors_cfg, server_processes)
        camera_dict = {}

        # No need to reset CAN interfaces every time
        # setup_can_interfaces()

        logger.info("Initializing robots...")
        robots = initialize_robots(main_config.robots, server_processes)

        agent = initialize_agent(agent_cfg, server_processes)

        logger.info("Creating robot environment...")
        frequency = main_config.hz
        rate = Rate(frequency, rate_name="control_loop")

        env = RobotEnv(
            robot_dict=robots,
            camera_dict={},  # Camera handled separately
            control_rate_hz=rate,
        )

        reset_robot(agent, env, 'left')
        reset_robot(agent, env, 'right')
        

        logger.info("Starting pure teleop mode (no data saving)...")
        _run_control_loop(env, agent, main_config, configs_dict)


    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise e
    finally:
        # Cleanup - MUST close in correct order
        logger.info("Shutting down...")
        
        # First close all robots to disable motors
        if "robots" in locals():
            try:
                logger.info("Disabling all motors...")
                for side, robot in robots.items():
                    try:
                        logger.info(f"Closing {side} robot...")
                        robot.close()
                    except Exception as e:
                        logger.error(f"Error closing {side} robot: {e}")
            except Exception as e:
                logger.error(f"Error during robot cleanup: {e}")
        
        # Then close environment (cameras)
        if "env" in locals():
            try:
                logger.info("Closing environment...")
                env.close()
            except Exception as e:
                logger.error(f"Error closing environment: {e}")
        
        # Finally stop agent and all server processes
        if "agent" in locals():
            logger.info("Stopping agent and servers...")
            cleanup_processes(agent, server_processes)
        
        logger.info("Cleanup complete")


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



def _run_control_loop(env: RobotEnv, agent: Agent, config: LaunchConfig, configs_dict: Dict) -> None:
    """
    Run the main control loop in pure teleop mode (no data saving).

    Args:
        env: Robot environment
        agent: Agent instance
        config: Configuration object
    """
    logger = logging.getLogger(__name__)

    # Init environment and warm up agent
    obs = env.reset()
    logger.info(f"Action spec: {env.action_spec()}")
    
    logger.info("Pure teleop mode - press 'A' to enable movement, 'B' to stop")

    # Main control loop
    while True:
        obs = env.reset()

        logger.info("Waiting for 'A' button to start teleop...")

        while True:
            info = agent.get_info()
            if info["success"]:
                logger.info("Movement enabled - starting teleop")
                time.sleep(0.5)
                break
        
        logger.info("Teleop active - controlling robot")

        while True:
            info = agent.get_info()
            
            # Check for exit condition
            if info.get("failure"):
                logger.info("Stopping teleop (B pressed)")
                break

            if info["movement_enabled"]['left'] or info["movement_enabled"]['right']:
                act = agent.act(obs)
                action = {'left': {'pos':act['left']['pos']}, 'right': {'pos':act['right']['pos']}}
                next_obs = env.step(action)
                obs = next_obs.copy()

    env.reset()
    logger.info("Teleop finished")

if __name__ == "__main__":
    main(tyro.cli(Args))
