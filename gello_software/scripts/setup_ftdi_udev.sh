#!/bin/bash

# Script to install udev rules for persistent FTDI device naming

if [ "$(id -u)" != "0" ]; then
    SUDO="sudo"
else
    SUDO=""
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UDEV_RULES_FILE="70-persistent-ftdi.rules"
UDEV_RULES_PATH="/etc/udev/rules.d/$UDEV_RULES_FILE"

echo "Installing FTDI udev rules..."

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
$SUDO udevadm trigger --subsystem-match=tty

echo "Done! Please unplug and replug your FTDI devices for the changes to take effect."
echo ""
echo "After replugging, your FTDI devices will be accessible as:"
echo "  - /dev/ttyUSB_left  (序列號: FTAO9WPU) - 左臂"
echo "  - /dev/ttyUSB_right (序列號: FTAO9WCV) - 右臂"
echo ""
echo "You can verify with: ls -la /dev/ttyUSB*"
