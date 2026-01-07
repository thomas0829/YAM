import atexit
from math import inf
from multiprocessing import Process
import signal
import threading
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional

import tyro
import zmq.error
from omegaconf import OmegaConf

from gello.utils.launch_utils import instantiate_from_dict, move_to_start_position
from gello.dynamixel.driver import DynamixelDriver
import numpy as np

from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids
from gello.data_utils.data_saver import DataSaver
from gello.data_utils.keyboard_interface import KBReset
from gello.utils.control_utils import run_control_loop_prior
from gello.zmq_core.camera_node import ZMQClientCamera, ZMQServerCamera

# Import LeRobot data saver if available
try:
    import sys
    from pathlib import Path
    # Add YAM root directory to path (contains src/ and yam_realtime/)
    # teleop_and_inference/experiments/launch_yaml_collect_data.py -> go up 2 levels to YAM root
    yam_root = Path(__file__).resolve().parents[2]
    if str(yam_root) not in sys.path:
        sys.path.insert(0, str(yam_root))
    from yam_realtime.yam_realtime.utils.lerobot_data_saver import LeRobotDataSaver
    LEROBOT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LeRobotDataSaver not available: {e}")
    LEROBOT_AVAILABLE = False

# Global variables for cleanup
active_threads = []
active_servers = []
cleanup_in_progress = False

_env = None
_bimanual = False
_left_cfg = None
_right_cfg = None
_kb_interface = None
_robot_client = None
_cameras = None
_agent = None
_robot = None


def cleanup():
    """Clean up resources before exit."""
    global cleanup_in_progress
    if cleanup_in_progress:
        return
    cleanup_in_progress = True

    print("Cleaning up resources...")
    
    # Move robot to start position
    try:
        if _env is not None and _left_cfg is not None:
            if _bimanual and _right_cfg is not None:
                move_to_start_position(_env, _bimanual, _left_cfg, _right_cfg)
            else:
                move_to_start_position(_env, _bimanual, _left_cfg)
    except Exception as e:
        print(f"Error moving to start position: {e}")
    
    # Close agent (this closes Dynamixel drivers)
    if _agent is not None:
        try:
            if hasattr(_agent, "close"):
                _agent.close()
                print("Agent closed")
        except Exception as e:
            print(f"Error closing agent: {e}")
    
    # Close robot (this closes motor drivers and CAN interfaces)
    if _robot is not None:
        try:
            if hasattr(_robot, "close"):
                _robot.close()
                print("Robot closed")
        except Exception as e:
            print(f"Error closing robot: {e}")
    
    # Stop all ZMQ servers
    for server in active_servers:
        try:
            if hasattr(server, "stop"):
                server.stop()
            if hasattr(server, "close"):
                server.close()
        except Exception as e:
            print(f"Error stopping/closing server: {e}")

    # Wait for threads to finish with timeout
    for thread in active_threads:
        if thread.is_alive():
            thread.join(timeout=2)
    
    # Close robot client
    if _robot_client is not None:
        try:
            if hasattr(_robot_client, "close"):
                _robot_client.close()
        except Exception as e:
            print(f"Error closing robot client: {e}")
    
    # Close cameras
    if _cameras is not None:
        try:
            for camera in _cameras.values():
                if hasattr(camera, "close"):
                    camera.close()
        except Exception as e:
            print(f"Error closing cameras: {e}")
    
    # Close environment
    if _env is not None:
        try:
            if hasattr(_env, "close"):
                _env.close()
        except Exception as e:
            print(f"Error closing environment: {e}")
    
    # Close pygame if keyboard interface was used
    if _kb_interface is not None:
        try:
            import pygame
            pygame.quit()
        except Exception as e:
            print(f"Error closing pygame: {e}")

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
    
    # Save to global for cleanup
    global _cameras
    _cameras = cameras

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

    # Initialize data saver and keyboard interface
    save_format = left_cfg['storage'].get('save_format', 'json')
    
    # Create keyboard interface FIRST (initializes pygame window)
    kb_interface = KBReset()
    
    if save_format == 'lerobot':
        if not LEROBOT_AVAILABLE:
            raise ImportError("LeRobotDataSaver is not available. Please install lerobot package.")
        
        # Create LeRobot data saver
        task_directory = left_cfg['storage']['task_directory']
        repo_id = task_directory.replace(' ', '_').lower()  # For HF repo name and folder name
        base_dir = Path(left_cfg['storage']['base_dir'])
        
        # Dataset path: data/repo_id (use underscore version)
        dataset_path = base_dir / repo_id
        
        # Check if dataset already exists
        if dataset_path.exists() and (dataset_path / "meta" / "info.json").exists():
            # Use pygame dialog for selection
            choice = kb_interface.show_options(
                f"Dataset exists: {repo_id}",
                [
                    "1. Continue - Append new episodes",
                    "2. Delete - Remove and start fresh",
                    "3. Rename - Backup and create new",
                    "4. Exit"
                ]
            )
            
            # Process choice
            if choice == 0:  # Continue
                kb_interface.update_status(message='Appending to existing dataset')
                print(f"✓ Will append to existing dataset")
            elif choice == 1:  # Delete
                import shutil
                kb_interface.update_status(message=f'Deleting {dataset_path.name}...')
                print(f"Deleting {dataset_path}...")
                shutil.rmtree(dataset_path)
                kb_interface.update_status(message='Deleted existing dataset')
                print(f"✓ Deleted existing dataset")
            elif choice == 2:  # Rename
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                new_name = dataset_path.parent / f"{dataset_path.name}_backup_{timestamp}"
                kb_interface.update_status(message=f'Renaming to backup_{timestamp}...')
                print(f"Renaming {dataset_path} -> {new_name}")
                dataset_path.rename(new_name)
                kb_interface.update_status(message=f'Renamed to backup')
                print(f"✓ Renamed to {new_name}")
            else:  # Exit
                print("Exiting...")
                import sys
                sys.exit(0)
        
        print(f"Creating LeRobot dataset:")
        print(f"  Task: {task_directory}")
        print(f"  Repo ID: {repo_id}")
        print(f"  Dataset path: {dataset_path}")
        
        # Get total episodes for batch encoding
        total_episodes = left_cfg['storage'].get('episodes', 1)
        batch_encoding = left_cfg['storage'].get('batch_encoding', False)
        
        # Set batch encoding size based on config
        if batch_encoding:
            batch_encoding_size = total_episodes  # Encode all at once at the end
            print(f"  Batch encoding: Enabled (will encode all {total_episodes} episodes at the end)")
        else:
            batch_encoding_size = None  # Encode immediately after each episode
            print(f"  Batch encoding: Disabled (encoding after each episode)")
        
        data_saver = LeRobotDataSaver(
            repo_id=repo_id,  # Used for HF repo name
            root=str(dataset_path),  # Full path to dataset directory
            fps=left_cfg.get('hz', 30),
            task_name=left_cfg['storage']['language_instruction'],
            robot_type="yam",
            camera_names=["left_camera", "front_camera", "right_camera"],
            use_videos=True,
            image_writer_processes=4,
            image_writer_threads=4,
            hf_user=left_cfg['storage'].get('hf_user'),  # Read from config
            auto_upload=left_cfg['storage'].get('auto_upload', False),  # Read from config
            batch_encoding_size=batch_encoding_size,  # Batch encode for better performance
        )
        print(f"Using LeRobot data saver for dataset: {task_directory}")
        print(f"Dataset will be saved to: {dataset_path}")
    else:
        # Use legacy DataSaver for json/npy formats
        data_saver = DataSaver(
            save_dir=left_cfg['storage']['base_dir'],
            task_directory=left_cfg['storage']['task_directory'],
            language_instruction=left_cfg['storage']['language_instruction']
        )
        print(f"Using legacy DataSaver with format: {save_format}")
    
    # Save to global for cleanup
    global _kb_interface, _left_cfg, _right_cfg, _bimanual
    _kb_interface = kb_interface
    _left_cfg = left_cfg
    _right_cfg = right_cfg if bimanual else None
    _bimanual = bimanual

    # Create agent
    if bimanual:
        from gello.agents.agent import BimanualAgent

        agent = BimanualAgent(
            agent_left=instantiate_from_dict(left_cfg["agent"]),
            agent_right=instantiate_from_dict(right_cfg["agent"]),
        )
    else:
        agent = instantiate_from_dict(left_cfg["agent"])
    
    # Save to global for cleanup
    global _agent
    _agent = agent

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
    
    # Save to global for cleanup
    global _robot
    _robot = robot

    # Handle different robot types
    if hasattr(robot, "serve"):  # MujocoRobotServer or ZMQServerRobot
        print("Starting robot server...")
        from gello.env import RobotEnv
        from gello.zmq_core.robot_node import ZMQClientRobot

        # Get server configuration
        server_port = cfg["robot"].get("port", 5556)
        server_host = cfg["robot"].get("host", "127.0.0.1")

        # Start server in background
        server_thread = threading.Thread(target=robot.serve, daemon=True)
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
        server_thread = threading.Thread(target=server.serve, daemon=True)
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

    # Store remaining global variables for cleanup
    global _env, _robot_client
    _env = env
    _robot_client = robot_client

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
        data_saver = run_control_loop_prior(env, agent, left_cfg=left_cfg, right_cfg=right_cfg, data_saver=data_saver, kb_interface=kb_interface)
    else:
        data_saver = run_control_loop_prior(env, agent, left_cfg=left_cfg, data_saver=data_saver, kb_interface=kb_interface)
    
    # Data collection complete - now cleanup resources BEFORE video encoding
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("Cleaning up robot and camera resources...")
    print("=" * 60)
    cleanup()
    
    # Now do video encoding with maximum resources available
    if data_saver is not None and save_format == 'lerobot':
        print("\n" + "=" * 60)
        print("Starting video encoding (this may take several minutes)...")
        print("All robot and camera resources have been released.")
        print("=" * 60)
        
        kb_interface.update_status_and_draw(
            state='Finalizing',
            message='Encoding videos... Please wait'
        )
        
        # Keep updating display during finalization
        import threading
        import time as time_module
        
        finalize_done = threading.Event()
        
        def finalize_thread():
            data_saver.finalize()
            finalize_done.set()
        
        # Start finalization in background
        thread = threading.Thread(target=finalize_thread, daemon=True)
        thread.start()
        
        # Update display while waiting
        while not finalize_done.is_set():
            kb_interface.update_status_and_draw(
                state='Finalizing',
                message='Encoding videos... Please wait'
            )
            # Process pygame events to keep window responsive
            pygame.event.pump()
            time_module.sleep(0.5)
        
        thread.join()
        print("\n" + "=" * 60)
        print("Video encoding complete!")
        print("=" * 60)
        
        kb_interface.update_status_and_draw(state='Done', message='All done!')
        time_module.sleep(2)  # Show final message
    
    # Explicitly exit the program - use os._exit to force immediate termination
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
