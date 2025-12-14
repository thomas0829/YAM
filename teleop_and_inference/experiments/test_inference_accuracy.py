#!/usr/bin/env python
"""
Test script to compare inference predictions with ground truth actions from dataset.
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
import tyro
from dataclasses import dataclass
from typing import Optional

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TestArgs:
    model_id: str = "Jiafei1224/pi05-fold-clothes"
    """Model checkpoint path"""
    
    dataset_id: str = "Jiafei1224/d"
    """Dataset ID to load ground truth from"""
    
    num_samples: int = 100
    """Number of samples to test"""
    
    start_index: int = 0
    """Start index in dataset"""


def prepare_observation_for_inference(obs_dict, device):
    """Prepare observation following lerobot_pi05 official format"""
    observation = {}
    
    # Process images: resize → tensor → /255 → permute → unsqueeze → GPU
    TARGET_HEIGHT = 360
    TARGET_WIDTH = 640
    
    camera_keys = ['left', 'right', 'top']
    for cam_key in camera_keys:
        img_key = f"observation.images.{cam_key}"
        if img_key in obs_dict:
            img_np = obs_dict[img_key]
            
            # If already tensor, convert to numpy
            if torch.is_tensor(img_np):
                img_np = img_np.cpu().numpy()
            
            # Handle channel dimension
            if img_np.shape[0] == 3:  # (C, H, W)
                img_np = img_np.transpose(1, 2, 0)  # (H, W, C)
            
            # Resize if needed
            if img_np.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH):
                img_pil = Image.fromarray(img_np.astype(np.uint8))
                img_resized_pil = img_pil.resize((TARGET_WIDTH, TARGET_HEIGHT))
                img_np = np.array(img_resized_pil)
            
            # Convert to tensor, normalize, permute, add batch dim, move to GPU
            img_tensor = torch.from_numpy(img_np).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
            observation[img_key] = img_tensor
    
    # Process state: tensor → unsqueeze → GPU
    if "observation.state" in obs_dict:
        state = obs_dict["observation.state"]
        if torch.is_tensor(state):
            state = state.cpu().numpy()
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        observation["observation.state"] = state_tensor
    
    return observation


def compute_metrics(predicted_actions, ground_truth_actions):
    """Compute various metrics between predicted and ground truth actions"""
    # Convert to numpy if needed
    if torch.is_tensor(predicted_actions):
        predicted_actions = predicted_actions.cpu().numpy()
    if torch.is_tensor(ground_truth_actions):
        ground_truth_actions = ground_truth_actions.cpu().numpy()
    
    # Mean Absolute Error
    mae = np.abs(predicted_actions - ground_truth_actions).mean()
    
    # Mean Squared Error
    mse = ((predicted_actions - ground_truth_actions) ** 2).mean()
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Per-dimension MAE
    per_dim_mae = np.abs(predicted_actions - ground_truth_actions).mean(axis=0)
    
    # Max error
    max_error = np.abs(predicted_actions - ground_truth_actions).max()
    
    # Correlation coefficient
    correlation = np.corrcoef(predicted_actions.flatten(), ground_truth_actions.flatten())[0, 1]
    
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "max_error": max_error,
        "correlation": correlation,
        "per_dim_mae": per_dim_mae
    }


def main():
    args = tyro.cli(TestArgs)
    
    print(f"\n{'='*80}")
    print("Testing Inference Accuracy")
    print(f"{'='*80}")
    print(f"Model: {args.model_id}")
    print(f"Dataset: {args.dataset_id}")
    print(f"Samples: {args.num_samples} (starting from index {args.start_index})")
    print(f"Device: {DEVICE}")
    print(f"{'='*80}\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset = LeRobotDataset(args.dataset_id)
    ds_meta = LeRobotDatasetMetadata(args.dataset_id)
    print(f"Dataset loaded: {len(dataset)} frames")
    
    # Get task from dataset
    task = None
    try:
        if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'tasks'):
            tasks = dataset.meta.tasks
            if hasattr(tasks, 'values'):
                task = list(tasks.values())[0]
            elif isinstance(tasks, dict):
                task = list(tasks.values())[0]
    except:
        pass
    
    if task is None:
        task = "fold the cloth"  # Default task
    print(f"Task: {task}\n")
    
    # Load policy
    print("Loading policy...")
    policy = PI05Policy.from_pretrained(args.model_id)
    policy.dataset_stats = ds_meta.stats
    policy.to(DEVICE)
    policy.eval()
    print("Policy loaded successfully\n")
    
    # Load preprocessor and postprocessor
    print("Loading preprocessor/postprocessor...")
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, args.model_id, 
        dataset_stats=ds_meta.stats
    )
    print("Preprocessors loaded successfully\n")
    
    # Test on samples
    print(f"Running inference on {args.num_samples} samples...\n")
    
    all_predicted_actions = []
    all_ground_truth_actions = []
    
    policy.reset()
    
    for i in range(args.start_index, args.start_index + args.num_samples):
        if i >= len(dataset):
            print(f"Reached end of dataset at index {i}")
            break
        
        # Get sample from dataset
        sample = dataset[i]
        
        # Prepare observation
        observation = prepare_observation_for_inference(sample, DEVICE)
        
        # Add task
        observation["task"] = [task]
        
        # Run preprocessing
        observation = preprocessor(observation)
        
        # Run inference
        with torch.no_grad():
            predicted_action = policy.select_action(observation)
        
        # Run postprocessing
        predicted_action = postprocessor(predicted_action)
        predicted_action = predicted_action.squeeze(0).cpu().numpy()
        
        # Get ground truth action
        gt_action = sample["action"]
        if torch.is_tensor(gt_action):
            gt_action = gt_action.cpu().numpy()
        
        # Store results
        all_predicted_actions.append(predicted_action)
        all_ground_truth_actions.append(gt_action)
        
        if (i - args.start_index + 1) % 10 == 0:
            print(f"  Processed {i - args.start_index + 1}/{args.num_samples} samples")
    
    # Convert to arrays
    all_predicted_actions = np.array(all_predicted_actions)
    all_ground_truth_actions = np.array(all_ground_truth_actions)
    
    print(f"\nCompleted inference on {len(all_predicted_actions)} samples\n")
    
    # Compute metrics
    print(f"{'='*80}")
    print("Results")
    print(f"{'='*80}\n")
    
    metrics = compute_metrics(all_predicted_actions, all_ground_truth_actions)
    
    print(f"Overall Metrics:")
    print(f"  Mean Absolute Error (MAE):  {metrics['mae']:.6f}")
    print(f"  Root Mean Squared Error:     {metrics['rmse']:.6f}")
    print(f"  Max Error:                   {metrics['max_error']:.6f}")
    print(f"  Correlation:                 {metrics['correlation']:.6f}")
    
    print(f"\nPer-Dimension MAE:")
    action_dim = len(metrics['per_dim_mae'])
    for dim in range(action_dim):
        print(f"  Dim {dim:2d}: {metrics['per_dim_mae'][dim]:.6f}")
    
    print(f"\nAction Statistics:")
    print(f"  Predicted - Mean: {all_predicted_actions.mean():.6f}, Std: {all_predicted_actions.std():.6f}")
    print(f"  Predicted - Min:  {all_predicted_actions.min():.6f}, Max: {all_predicted_actions.max():.6f}")
    print(f"  Ground Truth - Mean: {all_ground_truth_actions.mean():.6f}, Std: {all_ground_truth_actions.std():.6f}")
    print(f"  Ground Truth - Min:  {all_ground_truth_actions.min():.6f}, Max: {all_ground_truth_actions.max():.6f}")
    
    # Save results
    output_file = Path("inference_accuracy_results.npz")
    np.savez(
        output_file,
        predicted_actions=all_predicted_actions,
        ground_truth_actions=all_ground_truth_actions,
        metrics=metrics
    )
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
