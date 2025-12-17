#!/usr/bin/env python3
"""Read current gripper position."""

import sys
import os

os.chdir('/home/prior/thomas/YAM')
sys.path.append('/home/prior/thomas/YAM/i2rt')
sys.path.append('/home/prior/thomas/YAM/yam_realtime')

from yam_realtime.envs.configs.loader import DictLoader
from yam_realtime.envs.configs.instantiate import instantiate
import numpy as np

def main():
    print("Reading current gripper position...")
    
    # Load robot config
    config_path = "/home/prior/thomas/YAM/yam_realtime/robot_configs/yam/left.yaml"
    config_dict = DictLoader.load(config_path)
    config_dict['xml_path'] = "/home/prior/thomas/YAM/i2rt/i2rt/robot_models/yam/yam.xml"
    
    robot = instantiate(config_dict)
    
    # Read current position
    obs = robot.get_observations()
    gripper_cmd_space = obs['gripper_pos'][0]
    
    # Convert to robot space
    gripper_robot_space = robot.remapper.to_robot_joint_pos_space(
        np.array([0, 0, 0, 0, 0, 0, gripper_cmd_space])
    )[6]
    
    print(f"\nCurrent gripper position:")
    print(f"  Command space: {gripper_cmd_space:.4f}")
    print(f"  Robot space:   {gripper_robot_space:.4f}")
    print(f"\nThis is the actual motor position.")
    print(f"Update gripper_limits to: [0.0, {gripper_robot_space:.1f}] for full range\n")
    
    robot.close()

if __name__ == "__main__":
    main()
