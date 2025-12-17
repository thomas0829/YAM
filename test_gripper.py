#!/usr/bin/env python3
"""Test gripper control and read position."""

import sys
import time
import os
import numpy as np

# Set working directory to YAM root
os.chdir('/home/prior/thomas/YAM')

sys.path.append('/home/prior/thomas/YAM/i2rt')
sys.path.append('/home/prior/thomas/YAM/yam_realtime')

from yam_realtime.envs.configs.loader import DictLoader
from yam_realtime.envs.configs.instantiate import instantiate

def main():
    print("=== Gripper Test ===")
    print("Loading robot configuration...")
    
    # Load robot config with absolute path
    config_path = "/home/prior/thomas/YAM/yam_realtime/robot_configs/yam/left.yaml"
    config_dict = DictLoader.load(config_path)
    
    # Fix XML path to absolute
    config_dict['xml_path'] = "/home/prior/thomas/YAM/i2rt/i2rt/robot_models/yam/yam.xml"
    
    robot = instantiate(config_dict)
    
    print(f"Robot initialized: {robot}")
    print(f"Gripper index: {robot._gripper_index}")
    print(f"Gripper limits: {robot._gripper_limits}")
    print(f"Gripper kp: {robot._kp[robot._gripper_index]}, kd: {robot._kd[robot._gripper_index]}")
    
    time.sleep(1)
    
    # Get initial state
    obs = robot.get_observations()
    print(f"\nInitial gripper position: {obs['gripper_pos'][0]:.4f}")
    
    # Test sequence
    test_positions = [
        (0.0, "Closed (0.0)"),
        (0.5, "Half (0.5)"),
        (1.0, "Open (1.0)"),
        (0.5, "Half (0.5)"),
        (0.0, "Closed (0.0)"),
    ]
    
    print("\n=== Starting Gripper Movement Test ===")
    print("Command is in normalized space [0.0, 1.0]")
    print("Will be mapped to robot space via JointMapper")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        for cmd_pos, description in test_positions:
            # Create full joint position command
            # Use current arm positions + gripper command
            current_joint_pos = obs['joint_pos']
            full_cmd = np.append(current_joint_pos, cmd_pos)
            
            print(f"\nCommanding gripper to: {description} (cmd={cmd_pos:.1f})")
            robot.command_joint_pos(full_cmd)
            
            # Wait and monitor
            for i in range(10):
                time.sleep(0.2)
                obs = robot.get_observations()
                gripper_pos = obs['gripper_pos'][0]
                # Get actual robot space position
                robot_space_pos = robot.remapper.to_robot_joint_pos_space(
                    np.array([0, 0, 0, 0, 0, 0, gripper_pos])
                )[6]
                print(f"  t={i*0.2:.1f}s: cmd_space={gripper_pos:.4f}, robot_space={robot_space_pos:.4f}", end='\r')
            
            print(f"\n  Final: cmd_space={gripper_pos:.4f}, robot_space={robot_space_pos:.4f}")
        
        print("\n=== Test Complete ===")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    
    finally:
        print("\nClosing robot...")
        robot.close()
        print("Done!")

if __name__ == "__main__":
    main()
