import atexit
from math import inf
from multiprocessing import Process
import signal
import threading
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
import tyro
import zmq.error
from omegaconf import OmegaConf

from gello.utils.launch_utils import instantiate_from_dict, move_to_start_position
from gello.dynamixel.driver import DynamixelDriver
import numpy as np

from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids
from gello.data_utils.data_saver import DataSaver
from gello.data_utils.keyboard_interface import KBReset
from gello.utils.control_utils import run_control_loop_eval, run_control_loop_prior
from gello.zmq_core.camera_node import ZMQClientCamera, ZMQServerCamera

# Global variables for cleanup
active_threads = []
active_servers = []
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
    for server in active_servers:
        try:
            if hasattr(server, "close"):
                server.close()
        except Exception as e:
            print(f"Error closing server: {e}")

    for thread in active_threads:
        if thread.is_alive():
            thread.join(timeout=2)

    print("Cleanup completed.")


def wait_for_server_ready(port, host="127.0.0.1", timeout_seconds=5):
    """Wait for ZMQ server to be ready with retry logic."""
    from gello.zmq_core.robot_node import ZMQClientRobot

    attempts = int(timeout_seconds * 10)  # 0.1s intervals
    for attempt in range(attempts):
        try:
            client = ZMQClientRobot(port=port, host=host)
            time.sleep(0.1)
            return True
        except (zmq.error.ZMQError, Exception):
            time.sleep(0.1)
        finally:
            if "client" in locals():
                client.close()
            time.sleep(0.1)
            if attempt == attempts - 1:
                raise RuntimeError(
                    f"Server failed to start on {host}:{port} within {timeout_seconds} seconds"
                )
    return False


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

def get_joint_offsets(
    cfg: dict, port: str
):
    """Get joint offsets using the same logic as gello_get_offset.py."""
    joint_ids = list(cfg["agent"]["dynamixel_config"]["joint_ids"])
    driver = DynamixelDriver(joint_ids, port=port, baudrate=57600)

    def get_error(offset: float, index: int, joint_state: np.ndarray) -> float:
        joint_sign_i = cfg["agent"]["dynamixel_config"]["joint_signs"][index]
        joint_i = joint_sign_i * (joint_state[index] - offset)
        start_i = cfg["agent"]["start_joints"][index]
        return np.abs(joint_i - start_i)

    # Warmup
    for _ in range(10):
        driver.get_joints()

    best_offsets = []
    curr_joints = driver.get_joints()

    for i in range(len(joint_ids)):
        best_offset = 0
        best_error = float('inf')
        for offset in np.linspace(-8 * np.pi, 8 * np.pi, 500):
            error = get_error(offset, i, curr_joints)
            if error < best_error:
                best_error = error
                best_offset = offset
        best_offsets.append(best_offset)

    driver.close()
    return best_offsets

def update_offsets(cfg):
    joint_offsets = get_joint_offsets(cfg, cfg["agent"]["port"])
    cfg["agent"]["dynamixel_config"]["joint_offsets"] = joint_offsets
    return cfg

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
    left_cfg = update_offsets(left_cfg)
    if bimanual:
        right_cfg = OmegaConf.to_container(
            OmegaConf.load(args.right_config_path), resolve=True
        )
        right_cfg = update_offsets(right_cfg)

    # Initialize policy
    ds_meta = LeRobotDatasetMetadata(
        repo_id=left_cfg["policy"]["repo_id"]
    )
    policy = DiffusionPolicy.from_pretrained(left_cfg["policy"]["checkpoint_path"], dataset_stats=ds_meta.stats)
    policy.to('cuda')
    policy.eval()

    # Create agent
    if bimanual:
        from gello.agents.agent import BimanualAgent

        agent = BimanualAgent(
            agent_left=instantiate_from_dict(left_cfg["agent"]),
            agent_right=instantiate_from_dict(right_cfg["agent"]),
        )
    else:
        agent = instantiate_from_dict(left_cfg["agent"])

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

    # Handle different robot types
    if hasattr(robot, "serve"):  # MujocoRobotServer or ZMQServerRobot
        print("Starting robot server...")
        from gello.env import RobotEnv
        from gello.zmq_core.robot_node import ZMQClientRobot

        # Get server configuration
        server_port = cfg["robot"].get("port", 5556)
        server_host = cfg["robot"].get("host", "127.0.0.1")

        # Start server in background (non-daemon for proper cleanup)
        server_thread = threading.Thread(target=robot.serve, daemon=False)
        server_thread.start()

        # Track for cleanup
        active_threads.append(server_thread)
        active_servers.append(robot)

        # Wait for server to be ready
        print(f"Waiting for server to start on {server_host}:{server_port}...")
        wait_for_server_ready(server_port, server_host)
        print("Server ready!")

        # Create client to communicate with server using port and host from config
        robot_client = ZMQClientRobot(port=server_port, host=server_host)
    else:  # Direct robot (hardware)
        from gello.env import RobotEnv
        from gello.zmq_core.robot_node import ZMQClientRobot, ZMQServerRobot

        # Get server configuration (use a different default port for hardware)
        hardware_port = cfg.get("hardware_server_port", 6001)
        hardware_host = "127.0.0.1"

        # Create ZMQ server for the hardware robot
        server = ZMQServerRobot(robot, port=hardware_port, host=hardware_host)
        server_thread = threading.Thread(target=server.serve, daemon=False)
        server_thread.start()

        # Track for cleanup
        active_threads.append(server_thread)
        active_servers.append(server)

        # Wait for server to be ready
        print(
            f"Waiting for hardware server to start on {hardware_host}:{hardware_port}..."
        )
        wait_for_server_ready(hardware_port, hardware_host)
        print("Hardware server ready!")

        # Create client to communicate with hardware
        robot_client = ZMQClientRobot(port=hardware_port, host=hardware_host)

    env = RobotEnv(robot_client, control_rate_hz=cfg.get("hz", 30), camera_dict=cameras)

    # Store global variables for cleanup
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
        f"Launching robot: {robot.__class__.__name__}, agent: {agent.__class__.__name__}"
    )
    print(f"Control loop: {cfg.get('hz', 30)} Hz")

    # from gello.utils.control_utils import SaveInterface, run_control_loop

    # Initialize save interface if requested
    # save_interface = None
    # if args.use_save_interface:
    #     save_interface = SaveInterface(
    #         data_dir=Path(args.left_config_path).parents[1] / "data",
    #         agent_name=agent.__class__.__name__,
    #         expand_user=True,
    #     )

    # # Run main control loop
    # run_control_loop(env, agent, save_interface)

    # Run main control loop
    if bimanual:
        run_control_loop_eval(env, agent, left_cfg=left_cfg, right_cfg=right_cfg, policy=policy)
    else:
        run_control_loop_eval(env, agent, left_cfg=left_cfg, policy=policy)



if __name__ == "__main__":
    main()
