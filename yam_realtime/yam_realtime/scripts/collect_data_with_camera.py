# Collect data for a single task. Pair with oculus viser agent.
import os
import sys
from tkinter import Y

# Add paths relative to script location
script_dir = os.path.dirname(os.path.abspath(__file__))
yam_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
yam_realtime_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(yam_root)
sys.path.append(yam_realtime_root)

import numpy as np
from scipy.spatial.transform import Slerp, Rotation as R

from yam_realtime.agents.agent import Agent
from yam_realtime.envs.configs.instantiate import instantiate
from yam_realtime.envs.configs.loader import DictLoader
from yam_realtime.robots.utils import Rate, Timeout
from yam_realtime.utils.launch_utils import cleanup_processes, initialize_agent, initialize_robots, initialize_sensors, setup_can_interfaces, setup_logging
from yam_realtime.utils.camera_thread import EpisodeSaverThread

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
    # Change to yam_realtime directory for relative paths to work
    script_dir = os.path.dirname(os.path.abspath(__file__))
    yam_realtime_dir = os.path.abspath(os.path.join(script_dir, '..', '..'))
    os.chdir(yam_realtime_dir)
    
    logger = setup_logging()
    logger.info("Starting YAM realtime control system...")
    logger.info(f"Working directory: {os.getcwd()}")

    server_processes = []
    try:
        logger.info("Loading configuration...")
        configs_dict = DictLoader.load([os.path.expanduser(x) for x in args.config_path])

        agent_cfg = configs_dict.pop("agent")
        sensors_cfg = configs_dict.pop("sensors", None)
        main_config = instantiate(configs_dict)

        logger.info("Initializing sensors...")
        camera_dict, camera_info = initialize_sensors(sensors_cfg, server_processes)

        # setup_can_interfaces()

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

        reset_robot(agent, env, 'left')
        reset_robot(agent, env, 'right')
        

        logger.info("Starting control loop...")
        data_saver = DataSaver(task_directory=configs_dict['storage']['task_directory'], language_instruction=configs_dict['storage']['language_instruction'])
        _run_control_loop(env, agent, main_config, configs_dict, data_saver)


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

def sum_delta_action(prev_delta, curr_delta):
    prev_delta_pos_left = prev_delta["left"]["delta"][:3]
    prev_delta_quat_left = np.concatenate([prev_delta["left"]["delta"][4:], [prev_delta["left"]["delta"][3]]])

    curr_delta_pos_left = curr_delta["left"]["delta"][:3]
    curr_delta_quat_left = np.concatenate([curr_delta["left"]["delta"][4:], [curr_delta["left"]["delta"][3]]])

    sum_delta_pos_left = curr_delta_pos_left + prev_delta_pos_left
    sum_delta_quat_left = (R.from_quat(curr_delta_quat_left) * R.from_quat(prev_delta_quat_left)).as_quat()

    prev_delta_pos_right = prev_delta["right"]["delta"][:3]
    prev_delta_quat_right = np.concatenate([prev_delta["right"]["delta"][4:], [prev_delta["right"]["delta"][3]]])

    curr_delta_pos_right = curr_delta["right"]["delta"][:3]
    curr_delta_quat_right = np.concatenate([curr_delta["right"]["delta"][4:], [curr_delta["right"]["delta"][3]]])

    sum_delta_pos_right = curr_delta_pos_right + prev_delta_pos_right
    sum_delta_quat_right = (R.from_quat(curr_delta_quat_right) * R.from_quat(prev_delta_quat_right)).as_quat()

    delta_sum = {
        "left": {"delta" : np.concatenate([sum_delta_pos_left, [sum_delta_quat_left[3]], sum_delta_quat_left[:3]])},
        "right": {"delta" : np.concatenate([sum_delta_pos_right, [sum_delta_quat_right[3]], sum_delta_quat_right[:3]])}
    }
    return delta_sum

def _run_control_loop(env: RobotEnv, agent: Agent, config: LaunchConfig, configs_dict: Dict, data_saver: DataSaver) -> None:
    """
    Run the main control loop.

    Args:
        env: Robot environment
        agent: Agent instance
        config: Configuration object
    """
    previous_action = agent.act({})
    saver_thread = EpisodeSaverThread(data_saver)
    saver_thread.start()

    logger = logging.getLogger(__name__)
    num_traj = 1

    # Init environment and warm up agent
    obs = env.reset()
    logger.info(f"Action spec: {env.action_spec()}")
    last_save_time = time.time()
    hz = 10
    delta_cumulative = {
        "left": {"delta" : np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])},
        "right": {"delta" : np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])}
    }

    # Main control loop
    while num_traj <= configs_dict['storage']['episodes']:
        obs = env.reset()
        data_saver.reset_buffer()
        

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
            while (not info["success"] and not info["failure"]) and not info["movement_enabled"]['left'] and not info["movement_enabled"]['right']:
                info = agent.get_info()
            
            save = False
            if info["success"]:
                save = True
                break
            elif info["failure"]:
                break

            # within agent, we use 7 point representation for action (x, y, z, w, x, y, z) for both left and right arm
            # but for robot to actually move, we use 6 value joints position instead. So the act function in agent converts 
            # the 7 points action. Also, there are some nasty conversion for quat from (w, x, y, z) to (x, y, z, w),
            # the robot and viser uses (w, x, y, z) for quat. but python scipy uses (x, y, z, w) for quat.

            # currently in the json for traning, the delta acton is actually a 7 point action without gripper. (from calc_delta_action)
            # Also make sure that gripper is either 0 or 1. 


            # note that we didn't really use obs (this is the outside world joints state containing pos, gripper, eff, vel)
            # we use viser agent to compute the action in 3D viser space, which is then solved by ik to get the joints position.
            # The action that the env.step() uses is the viser joints (the action returned by agent.act()).
            if info["movement_enabled"]['left'] or info["movement_enabled"]['right']:
                act = agent.act(obs)
                action = {'left': {'pos':act['left']['pos']}, 'right': {'pos':act['right']['pos']}}
                delta_cumulative = sum_delta_action(delta_cumulative, act)
                if time.time() - last_save_time > (1/hz):
                    act["left"]["delta"] = delta_cumulative["left"]["delta"]
                    act["right"]["delta"] = delta_cumulative["right"]["delta"]
                    data_saver.add_observation(obs, act)
                    delta_cumulative = {
                            "left": {"delta" : np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])},
                            "right": {"delta" : np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])}
                        }
                    last_save_time = time.time()
                next_obs = env.step(action)
                obs = next_obs.copy()
        
        if save:
            if data_saver.buffer == []:
                logger.info(f"No data collected, skipping save")
                continue
            # data_saver.save_episode_json()
            saver_thread.save_episode(data_saver.buffer.copy())
            num_traj +=1
            logger.info(f"Successfully collected data")
        else:
            logger.info(f"Failure")
        reset_robot(agent, env, 'left')
        reset_robot(agent, env, 'right')

    saver_thread.stop()
    saver_thread.join()

    env.reset()
    logger.info(f"Finished collecting data")

if __name__ == "__main__":
    main(tyro.cli(Args))
