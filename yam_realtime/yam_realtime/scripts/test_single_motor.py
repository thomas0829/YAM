#!/usr/bin/env python3
"""Test single motor communication with detailed debugging."""

import can
import time
import logging

logging.basicConfig(level=logging.DEBUG)

print("Testing direct CAN communication...")

# Create CAN bus
bus = can.interface.Bus(channel='can0', interface='socketcan', bitrate=1000000)
print(f"CAN bus created: {bus}")

# Test motor 0x01
motor_id = 0x01
expected_response_id = 0x10 + motor_id  # 0x11

print(f"\nTesting motor 0x{motor_id:02X}, expecting response on 0x{expected_response_id:02X}")

# Send motor_on command
data = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC]
msg = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)

print(f"Sending: {msg}")
bus.send(msg)

# Wait for response
print("Waiting for response...")
start_time = time.time()
response = None
while time.time() - start_time < 1.0:
    recv_msg = bus.recv(timeout=0.1)
    if recv_msg:
        print(f"Received: ID=0x{recv_msg.arbitration_id:02X}, Data={recv_msg.data.hex()}")
        if recv_msg.arbitration_id == expected_response_id:
            response = recv_msg
            break

if response:
    print(f"\n✓ Successfully received response from motor 0x{motor_id:02X}")
else:
    print(f"\n✗ No response received on expected ID 0x{expected_response_id:02X}")

bus.shutdown()
