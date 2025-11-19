import argparse
import time
import numpy as np
import curses
import os
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import GripperType

def main(stdscr):
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel_1", type=str, default="can0")
    parser.add_argument("--channel_2", type=str, default="can1")
    parser.add_argument("--gripper", type=str, default="linear_3507")
    parser.add_argument("--output", type=str, default="./example_trajectory.npy")
    parser.add_argument("--load", type=str, help="Load trajectory from file")
    args, _ = parser.parse_known_args()

    # Initialize robot
    gripper_type = GripperType.from_string_name(args.gripper)
    robot_1 = get_yam_robot(channel=args.channel_1, gripper_type=gripper_type)
    robot_2 = get_yam_robot(channel=args.channel_2, gripper_type=gripper_type)


    # Curses setup
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(0)

    trajectory_1 = []
    trajectory_2 = []
    timestamps = []
    recording = False
    replaying = False
    replay_idx = 0
    target_freq = 60.0  # 30 Hz
    dt = 1.0 / target_freq

    # Load trajectory if specified
    if args.load and os.path.exists(args.load):
        try:
            data = np.load(args.load, allow_pickle=True).item()
            trajectory_1 = data['trajectory_1'].tolist()
            trajectory_2 = data['trajectory_2'].tolist()
            timestamps = data['timestamps'].tolist()
            if 'frequency' in data:
                target_freq = data['frequency']
                dt = 1.0 / target_freq
        except Exception as e:
            print(f"Error loading trajectory: {e}")
            trajectory_1 = []
            trajectory_2 = []
            timestamps = []

    instructions = [
        "Controls:",
        "  r : Start/stop recording",
        "  p : Start replay",
        "  s : Save trajectory",
        "  l : Load trajectory from file",
        "  q : Quit",
        "",
        "Status:"
    ]

    last_record_time = time.monotonic()
    last_replay_time = time.monotonic()

    while True:
        current_time = time.monotonic()
        key = stdscr.getch()

        if key != -1:
            if key == ord('q'):
                break
            elif key == ord('r'):
                recording = not recording
                replaying = False
                if recording:
                    trajectory_1 = []
                    trajectory_2 = []
                    timestamps = []
                    last_record_time = current_time
                stdscr.addstr(len(instructions)+2, 0, f"Recording: {recording}         ")
            elif key == ord('p'):
                if len(trajectory_1) > 0 and len(trajectory_2) > 0:
                    replaying = True
                    recording = False
                    replay_idx = 0
                    last_replay_time = current_time
                else:
                    stdscr.addstr(len(instructions)+2, 0, "No trajectory to replay.      ")
            elif key == ord('s'):
                if len(trajectory_1) > 0 and len(trajectory_2) > 0:
                    # Save both trajectory and timestamps
                    data = {
                        'trajectory_1': np.array(trajectory_1),
                        'trajectory_2': np.array(trajectory_2),
                        'timestamps': np.array(timestamps),
                        'frequency': target_freq
                    }
                    np.save(args.output, data)
                    stdscr.addstr(len(instructions)+2, 0, f"Saved to {args.output}        ")
                else:
                    stdscr.addstr(len(instructions)+2, 0, "No trajectory to save.        ")
            elif key == ord('l'):
                # Load trajectory from file
                stdscr.addstr(len(instructions)+2, 0, "Enter filename to load: ")
                stdscr.refresh()

                # Simple filename input (basic implementation)
                filename = ""
                while True:
                    key = stdscr.getch()
                    if key == ord('\n'):  # Enter key
                        break
                    elif key == ord('\x1b'):  # Escape key
                        filename = ""
                        break
                    elif key == ord('\x7f'):  # Backspace
                        if filename:
                            filename = filename[:-1]
                    elif 32 <= key <= 126:  # Printable characters
                        filename += chr(key)

                    stdscr.addstr(len(instructions)+2, 0, f"Enter filename to load: {filename}")
                    stdscr.refresh()

                if filename and os.path.exists(filename):
                    try:
                        data = np.load(filename, allow_pickle=True).item()
                        trajectory_1 = data['trajectory_1'].tolist()
                        trajectory_2 = data['trajectory_2'].tolist()
                        timestamps = data['timestamps'].tolist()
                        if 'frequency' in data:
                            target_freq = data['frequency']
                            dt = 1.0 / target_freq
                        stdscr.addstr(len(instructions)+2, 0, f"Loaded {filename} successfully    ")
                    except Exception as e:
                        stdscr.addstr(len(instructions)+2, 0, f"Error loading {filename}      ")
                elif filename:
                    stdscr.addstr(len(instructions)+2, 0, f"File {filename} not found      ")

        # UI
        stdscr.erase()
        for i, line in enumerate(instructions):
            stdscr.addstr(i, 0, line)
        stdscr.addstr(len(instructions), 0, f"Recording: {recording}  Replaying: {replaying}")
        stdscr.addstr(len(instructions)+1, 0, f"Trajectory_1 length: {len(trajectory_1)} samples")
        stdscr.addstr(len(instructions)+1, 0, f"Trajectory_2 length: {len(trajectory_2)} samples")
        stdscr.addstr(len(instructions)+3, 0, "Press 'q' to quit.")

        # Record trajectory at 30Hz
        if recording and (current_time - last_record_time) >= dt:
            qpos = robot_1.get_joint_pos()
            trajectory_1.append(np.copy(qpos))
            qpos = robot_2.get_joint_pos()
            trajectory_2.append(np.copy(qpos))
            timestamps.append(current_time)
            last_record_time = current_time

        # Replay trajectory at 30Hz
        if replaying and len(trajectory_1) > 0 and len(trajectory_2) > 0:
            if replay_idx == 0:
                # slowly move the arm to fist way point
                robot_1.move_joints(np.array(trajectory_1[replay_idx]), time_interval_s=1.5)
                robot_2.move_joints(np.array(trajectory_2[replay_idx]), time_interval_s=1.5)
            if replay_idx < len(trajectory_1) and (current_time - last_replay_time) >= dt:
                robot_1.command_joint_pos(trajectory_1[replay_idx])
                robot_2.command_joint_pos(trajectory_2[replay_idx])
                replay_idx += 1
                last_replay_time = current_time
                stdscr.addstr(len(instructions)+4, 0, f"Replaying: {replay_idx}/{len(trajectory_1)}")
            elif replay_idx >= len(trajectory_1):
                replaying = False
                stdscr.addstr(len(instructions)+4, 0, "Replay finished.              ")

        stdscr.refresh()
        time.sleep(0.02)  # Higher refresh rate for smoother operation

if __name__ == "__main__":
    curses.wrapper(main)
