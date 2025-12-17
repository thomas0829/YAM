#!/usr/bin/env python3
"""Test CanInterface directly to debug mailbox listener."""

import sys
import time
sys.path.append('/home/prior/thomas/YAM/i2rt')

from i2rt.motor_drivers.can_interface import CanInterface
import logging

logging.basicConfig(level=logging.DEBUG)

print("Creating CanInterface...")
can_interface = CanInterface(
    channel="can0",
    bustype="socketcan",
    bitrate=1000000,
    name="test_interface"
)

print("CanInterface created successfully!")
print(f"Bus: {can_interface.bus}")
print(f"Notifier: {can_interface.notifier}")
print(f"Mailbox listener: {can_interface._mailbox_listener}")

# Wait a bit for listener to initialize
time.sleep(0.1)

# Try to send a message and get response
motor_id = 0x01
expected_id = 0x10 + motor_id

print(f"\nSending message to motor {motor_id:#x}, expecting response on {expected_id:#x}...")

try:
    response = can_interface._send_message_get_response(
        id=motor_id,
        motor_id=motor_id,
        data=[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC]
    )
    print(f"✓ Got response: {response}")
except Exception as e:
    print(f"✗ Error: {e}")
    # Check if there are any messages in the mailboxes
    print(f"\nMailboxes state:")
    for arb_id, mailbox in can_interface._mailboxes.items():
        if mailbox:
            print(f"  ID {arb_id:#x}: {len(mailbox)} messages")

can_interface.close()
print("\nDone.")
