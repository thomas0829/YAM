"""Test script to demonstrate safety checker functionality."""
import numpy as np
from yam_realtime.utils.safety_checker import SafetyChecker

def test_safety_checker():
    """Test the safety checker with various scenarios."""
    
    print("=" * 60)
    print("Testing Safety Checker")
    print("=" * 60)
    
    # Initialize safety checker
    safety = SafetyChecker(
        max_joint_delta=0.1,  # 0.1 rad = ~5.7 degrees
        max_velocity=0.5,
        max_gripper_delta=0.5,
        enable_safety=True,
        dt=0.1
    )
    
    print("\n1. Testing normal safe action:")
    action1 = {'left': {'pos': np.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.0])}}
    is_safe, reason, safe_action = safety.check_action(action1)
    print(f"   Is safe: {is_safe}, Reason: {reason}")
    
    print("\n2. Testing small change (should be safe):")
    action2 = {'left': {'pos': np.array([0.12, 0.21, 0.31, 0.11, 0.11, 0.11, 0.0])}}
    is_safe, reason, safe_action = safety.check_action(action2)
    print(f"   Is safe: {is_safe}, Reason: {reason}")
    print(f"   Delta: {action2['left']['pos'] - action1['left']['pos']}")
    
    print("\n3. Testing large joint change (should be clamped):")
    action3 = {'left': {'pos': np.array([0.5, 0.21, 0.31, 0.11, 0.11, 0.11, 0.0])}}
    is_safe, reason, safe_action = safety.check_action(action3)
    print(f"   Is safe: {is_safe}, Reason: {reason}")
    print(f"   Original action[0]: {action3['left']['pos'][0]:.3f}")
    print(f"   Safe action[0]: {safe_action['left']['pos'][0]:.3f}")
    print(f"   Delta: {action3['left']['pos'][0] - action2['left']['pos'][0]:.3f} rad (max allowed: 0.1)")
    
    print("\n4. Testing large gripper change (should be clamped):")
    action4 = {'left': {'pos': np.array([0.22, 0.21, 0.31, 0.11, 0.11, 0.11, -2.0])}}
    is_safe, reason, safe_action = safety.check_action(action4)
    print(f"   Is safe: {is_safe}, Reason: {reason}")
    print(f"   Original gripper: {action4['left']['pos'][-1]:.3f}")
    print(f"   Safe gripper: {safe_action['left']['pos'][-1]:.3f}")
    
    print("\n5. Testing velocity limit (with current joint position):")
    current_pos = {'left': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}
    action5 = {'left': {'pos': np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])}}
    is_safe, reason, safe_action = safety.check_action(action5, current_pos)
    print(f"   Is safe: {is_safe}, Reason: {reason}")
    print(f"   Requested velocity: {(action5['left']['pos'][0] - current_pos['left'][0]) / 0.1:.3f} rad/s")
    print(f"   Max velocity: 0.5 rad/s")
    print(f"   Safe position: {safe_action['left']['pos'][0]:.3f}")
    
    print("\n6. Checking violation count:")
    print(f"   Total violations: {safety.get_violation_count()}")
    
    print("\n7. Testing with safety disabled:")
    safety_disabled = SafetyChecker(enable_safety=False)
    dangerous_action = {'left': {'pos': np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])}}
    is_safe, reason, safe_action = safety_disabled.check_action(dangerous_action)
    print(f"   Is safe: {is_safe}, Reason: {reason}")
    
    print("\n" + "=" * 60)
    print("Safety Checker Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_safety_checker()
