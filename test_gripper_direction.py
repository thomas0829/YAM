#!/usr/bin/env python3
"""Quick test to determine if gripper needs positive or negative values."""

import sys
import time
sys.path.append('/home/prior/thomas/YAM/i2rt')

from i2rt.motor_drivers.dm_driver import DMChainCanInterface

# Initialize motor chain on can0
motor_chain = DMChainCanInterface(
    motor_list=[
        [0x01, "DM4340"],
        [0x02, "DM4340"],
        [0x03, "DM4340"],
        [0x04, "DM4310"],
        [0x05, "DM4310"],
        [0x06, "DM4310"],
        [0x07, "DM4310"],
    ],
    motor_offset=[0, 0, 0, 0, 0, 0, 0],
    motor_direction=[1, 1, 1, 1, 1, 1, 1],
    channel="can0",
    motor_chain_name="test_gripper",
)

# Get current state (this will automatically initialize)
motor_chain.get_state()
time.sleep(0.5)

current_state = motor_chain.get_state()
print(f"Current gripper (motor 7) position: {current_state.pos[6]:.3f}")

# Test positive value (e.g., 0.5)
print("\nTesting gripper position = 0.5 (positive)")
motor_chain.set_commands(
    [0, 0, 0, 0, 0, 0, 0],
    pos=[current_state.pos[0], current_state.pos[1], current_state.pos[2],
         current_state.pos[3], current_state.pos[4], current_state.pos[5],
         0.5],  # Motor 7 (gripper)
    vel=[0, 0, 0, 0, 0, 0, 0],
    kp=[80, 80, 80, 10, 10, 10, 20],
    kd=[5, 5, 5, 1.5, 1.5, 1.5, 0.5]
)
time.sleep(2)
state = motor_chain.get_state()
print(f"Gripper position after 0.5 command: {state.pos[6]:.3f}")

# Test negative value (e.g., -0.5)
print("\nTesting gripper position = -0.5 (negative)")
motor_chain.set_commands(
    [0, 0, 0, 0, 0, 0, 0],
    pos=[current_state.pos[0], current_state.pos[1], current_state.pos[2],
         current_state.pos[3], current_state.pos[4], current_state.pos[5],
         -0.5],  # Motor 7 (gripper)
    vel=[0, 0, 0, 0, 0, 0, 0],
    kp=[80, 80, 80, 10, 10, 10, 20],
    kd=[5, 5, 5, 1.5, 1.5, 1.5, 0.5]
)
time.sleep(2)
state = motor_chain.get_state()
print(f"Gripper position after -0.5 command: {state.pos[6]:.3f}")

# Return to initial
print("\nReturning to initial position")
motor_chain.set_commands(
    [0, 0, 0, 0, 0, 0, 0],
    pos=[current_state.pos[0], current_state.pos[1], current_state.pos[2],
         current_state.pos[3], current_state.pos[4], current_state.pos[5],
         current_state.pos[6]],
    vel=[0, 0, 0, 0, 0, 0, 0],
    kp=[80, 80, 80, 10, 10, 10, 20],
    kd=[5, 5, 5, 1.5, 1.5, 1.5, 0.5]
)
time.sleep(1)

print("\nTest complete!")
