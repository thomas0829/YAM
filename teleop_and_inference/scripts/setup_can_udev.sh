#!/bin/bash

# Script to install udev rules for persistent CAN device naming

if [ "$(id -u)" != "0" ]; then
    SUDO="sudo"
else
    SUDO=""
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UDEV_RULES_FILE="70-persistent-can.rules"
UDEV_RULES_PATH="/etc/udev/rules.d/$UDEV_RULES_FILE"

echo "Installing CAN udev rules..."

# Copy the rules file
$SUDO cp "$SCRIPT_DIR/$UDEV_RULES_FILE" "$UDEV_RULES_PATH"

if [ $? -ne 0 ]; then
    echo "Error: Failed to copy udev rules file"
    exit 1
fi

echo "udev rules file installed to $UDEV_RULES_PATH"

# Reload udev rules
echo "Reloading udev rules..."
$SUDO udevadm control --reload-rules
$SUDO udevadm trigger

echo "Done! Please unplug and replug your CAN devices for the changes to take effect."
echo ""
echo "After replugging, your CAN devices will be named:"
echo "  - can_right (序列號: 0056002F594E501820313332)"
echo "  - can_left  (序列號: 00220036594E501820313332)"
echo ""
echo "You can verify with: ip link show | grep can"
