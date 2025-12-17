"""Safety checker for robot actions to prevent dangerous movements."""
import numpy as np
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class SafetyChecker:
    """Check if robot actions are safe before execution."""
    
    def __init__(
        self,
        max_joint_delta: float = 0.1,  # Maximum change per joint per step (radians)
        max_velocity: float = 0.5,  # Maximum joint velocity (rad/s)
        max_gripper_delta: float = 0.5,  # Maximum gripper change per step
        enable_safety: bool = True,
        dt: float = 0.1,  # Time step (seconds)
        max_violations: int = 100,  # Maximum violations before stopping
        clamp_actions: bool = False,  # Whether to clamp actions or just warn
        warmup_steps: int = 10,  # Number of initial steps to skip safety checks
    ):
        """
        Initialize safety checker.
        
        Args:
            max_joint_delta: Maximum allowed change in joint position per step (radians)
            max_velocity: Maximum allowed joint velocity (rad/s)
            max_gripper_delta: Maximum allowed gripper change per step
            enable_safety: Enable or disable safety checks
            dt: Time step duration in seconds
            max_violations: Maximum violations before critical stop
            clamp_actions: If True, clamp unsafe actions; if False, only warn
            warmup_steps: Number of initial steps to skip safety checks (for startup)
        """
        self.max_joint_delta = max_joint_delta
        self.max_velocity = max_velocity
        self.max_gripper_delta = max_gripper_delta
        self.enable_safety = enable_safety
        self.dt = dt
        
        self.last_action: Optional[Dict] = None
        self.violation_count = 0
        self.max_violations = max_violations
        self.clamp_actions = clamp_actions
        self.warmup_steps = warmup_steps
        self.step_count = 0
        
        logger.info(f"SafetyChecker initialized: max_joint_delta={max_joint_delta}, "
                   f"max_velocity={max_velocity}, max_gripper_delta={max_gripper_delta}, "
                   f"enabled={enable_safety}, clamp_actions={clamp_actions}, warmup_steps={warmup_steps}")
    
    def check_action(
        self, 
        current_action: Dict[str, Dict[str, np.ndarray]],
        current_joint_pos: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[bool, str, Optional[Dict]]:
        """
        Check if an action is safe to execute.
        
        Args:
            current_action: Action dictionary with structure {'left': {'pos': array}}
            current_joint_pos: Current joint positions (optional, for additional checks)
            
        Returns:
            Tuple of (is_safe, reason, safe_action)
            - is_safe: True if action passes all safety checks
            - reason: Explanation if action is unsafe
            - safe_action: Clamped safe version of the action (or None if disabled)
        """
        if not self.enable_safety:
            return True, "Safety checks disabled", current_action
        
        # Skip safety checks during warmup period
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # Still update last_action for future checks
            self.last_action = {arm: {'pos': current_action[arm]['pos'].copy()} 
                               for arm in current_action if 'pos' in current_action[arm]}
            return True, f"Warmup step {self.step_count}/{self.warmup_steps}", current_action
        
        # Check for too many violations
        if self.violation_count >= self.max_violations:
            return False, f"Too many safety violations ({self.violation_count}). Stopping for safety.", None
        
        safe_action = {}
        violation_in_this_action = False
        
        for arm in current_action:
            if 'pos' not in current_action[arm]:
                continue
                
            pos = current_action[arm]['pos']
            safe_pos = pos.copy()
            
            # Check 1: Maximum joint delta (if we have previous action)
            if self.last_action is not None and arm in self.last_action:
                last_pos = self.last_action[arm]['pos']
                delta = pos - last_pos
                
                # Check each joint (excluding gripper which is last)
                for i in range(len(delta) - 1):  # Exclude gripper
                    if abs(delta[i]) > self.max_joint_delta:
                        logger.warning(f"Joint {i} delta too large: {delta[i]:.4f} rad "
                                     f"(max: {self.max_joint_delta})")
                        violation_in_this_action = True
                        
                        # Only clamp if enabled
                        if self.clamp_actions:
                            safe_pos[i] = last_pos[i] + np.clip(delta[i], 
                                                                -self.max_joint_delta, 
                                                                self.max_joint_delta)
                
                # Check gripper separately (last element)
                gripper_delta = abs(delta[-1])
                if gripper_delta > self.max_gripper_delta:
                    logger.warning(f"Gripper delta too large: {gripper_delta:.4f} "
                                 f"(max: {self.max_gripper_delta})")
                    violation_in_this_action = True
                    
                    # Only clamp if enabled
                    if self.clamp_actions:
                        safe_pos[-1] = last_pos[-1] + np.clip(delta[-1],
                                                              -self.max_gripper_delta,
                                                              self.max_gripper_delta)
            
            # Check 2: Velocity limit (requires current_joint_pos)
            if current_joint_pos is not None and arm in current_joint_pos and current_joint_pos[arm] is not None:
                curr_pos = current_joint_pos[arm]
                velocity = (safe_pos - curr_pos) / self.dt
                
                for i in range(len(velocity) - 1):  # Exclude gripper
                    if abs(velocity[i]) > self.max_velocity:
                        logger.warning(f"Joint {i} velocity too high: {velocity[i]:.4f} rad/s "
                                     f"(max: {self.max_velocity})")
                        violation_in_this_action = True
                        
                        # Only clamp if enabled
                        if self.clamp_actions:
                            max_delta = self.max_velocity * self.dt
                            safe_pos[i] = curr_pos[i] + np.clip(safe_pos[i] - curr_pos[i],
                                                                -max_delta, max_delta)
            
            safe_action[arm] = {'pos': safe_pos}
        
        # Count violations
        if violation_in_this_action:
            self.violation_count += 1
        
        # Update last action
        self.last_action = {arm: {'pos': safe_action[arm]['pos'].copy()} 
                           for arm in safe_action}
        
        # Check if we had to modify the action
        is_safe = True
        reason = "Action is safe"
        
        # If clamping is disabled, always return original action
        if not self.clamp_actions:
            if violation_in_this_action:
                is_safe = False
                reason = "Action has violations but clamping is disabled - using original action"
            return is_safe, reason, current_action
        
        # If clamping is enabled, check if action was modified
        for arm in current_action:
            if arm in safe_action:
                if not np.allclose(current_action[arm]['pos'], safe_action[arm]['pos'], atol=1e-6):
                    is_safe = False
                    reason = "Action was clamped to safe limits"
                    logger.info("Action modified for safety")
                    break
        
        return is_safe, reason, safe_action
    
    def reset(self):
        """Reset the safety checker state."""
        self.last_action = None
        self.violation_count = 0
        self.step_count = 0
        logger.info("SafetyChecker reset")
    
    def get_violation_count(self) -> int:
        """Get the current violation count."""
        return self.violation_count
