#!/usr/bin/env python3
"""Test script to initialize motors slowly and diagnose communication issues."""

import sys
import time
sys.path.append('/home/prior/thomas/YAM/i2rt')

from i2rt.motor_drivers.dm_driver import DMChainCanInterface, ReceiveMode
import logging

logging.basicConfig(level=logging.INFO)

# Configuration matching left.yaml
motor_list = [
    (0x01, "DM4340"),
    (0x02, "DM4340"),
    (0x03, "DM4340"),
    (0x04, "DM4310"),
    (0x05, "DM4310"),
    (0x06, "DM4310"),
    (0x07, "DM4310"),
]

print("Creating motor chain interface...")
motor_chain = DMChainCanInterface(
    motor_list=motor_list,
    motor_offset=[0, 0, 0, 0, 0, 0, 0],
    motor_direction=[1, 1, 1, 1, 1, 1, 1],
    channel="can0",
    motor_chain_name="yam_left",
    receive_mode=ReceiveMode.p16,
)

print("Motor chain interface created successfully!")
print("Motors are initialized and ready.")

# Keep running for a bit
time.sleep(2)

print("Shutting down...")
motor_chain.close()
print("Done.")
