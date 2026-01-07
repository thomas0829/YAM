#!/usr/bin/env python3
"""
Load pi05_base Orbax checkpoint and save as PyTorch safetensors.
Automatically generates processor configs from dataset statistics.
"""

from pathlib import Path
from safetensors.torch import save_file
import torch
import orbax.checkpoint as ocp
import json
import numpy as np
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Convert pi05 JAX checkpoint to PyTorch with processor configs')
parser.add_argument('--checkpoint-path', type=str, default='/home/prior/thomas/YAM/pi05_base/params',
                    help='Path to Orbax checkpoint directory')
parser.add_argument('--dataset-path', type=str, 
                    default='/home/prior/thomas/YAM/datasets/pick_up_four_cubes_and_stack_them_in_the_middle-v3.0',
                    help='Path to dataset directory (for reading stats.json)')
parser.add_argument('--output-dir', type=str, default='pi05_pytorch',
                    help='Output directory for converted model')
args = parser.parse_args()

print("Loading Orbax checkpoint from", args.checkpoint_path)
checkpoint_path = Path(args.checkpoint_path)
checkpointer = ocp.PyTreeCheckpointer()
restored = checkpointer.restore(checkpoint_path)

print(f"Checkpoint type: {type(restored)}")
print(f"Top-level keys: {list(restored.keys())}")

# Get params
jax_params = restored['params'] if 'params' in restored else restored
print(f"Starting conversion...")

state_dict = {}
param_count = 0

def convert_jax_to_pytorch(jax_tree, prefix=''):
    """Recursively convert JAX pytree to PyTorch state dict."""
    global param_count
    
    if isinstance(jax_tree, dict):
        for k, v in jax_tree.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            convert_jax_to_pytorch(v, new_prefix)
    elif hasattr(jax_tree, 'shape') and hasattr(jax_tree, 'dtype'):
        # This is an array-like object (ArrayImpl, np.ndarray, etc.)
        try:
            arr = np.asarray(jax_tree)
            tensor = torch.from_numpy(arr)
        
            # Apply key renaming (order matters to avoid double replacement)
            key = prefix
            if key.endswith('.kernel'):
                key = key[:-7] + '.weight'
            elif key.endswith('.scale'):
                key = key[:-6] + '.weight'
            elif key.endswith('.w'):
                key = key[:-2] + '.weight'
            
            # Check if this is a stacked layer parameter (first dim = num_layers)
            # Common patterns: encoderblock, layers, etc.
            needs_unstacking = any(pattern in key for pattern in [
                'encoderblock', '.layers.', 'decoder_block'
            ])
            
            if needs_unstacking and len(tensor.shape) >= 2:
                # First dimension is number of layers, unstack them
                num_layers = tensor.shape[0]
                print(f"  Unstacking {key}: {tensor.shape} into {num_layers} layers")
                
                for layer_idx in range(num_layers):
                    layer_tensor = tensor[layer_idx]
                    # Insert layer index into key
                    if 'encoderblock' in key:
                        layer_key = key.replace('encoderblock', f'encoderblock_{layer_idx}')
                    elif '.layers.' in key:
                        layer_key = key.replace('.layers.', f'.layers.{layer_idx}.')
                    else:
                        layer_key = f"{key}.{layer_idx}"
                    
                    # Add model. prefix if not present
                    if not layer_key.startswith('model.'):
                        layer_key = f"model.{layer_key}"
                    
                    # Fix PaliGemma paths
                    layer_key = layer_key.replace('model.PaliGemma.', 'model.paligemma_with_expert.paligemma.model.')
                    
                    state_dict[layer_key] = layer_tensor
                    param_count += layer_tensor.numel()
            else:
                # Single tensor, no unstacking needed
                # Add model. prefix if not present
                if not key.startswith('model.'):
                    key = f"model.{key}"
                
                # Fix PaliGemma paths
                key = key.replace('model.PaliGemma.', 'model.paligemma_with_expert.paligemma.model.')
                
                # Transpose action projection layers (JAX uses different convention)
                if 'action_in_proj.weight' in key or 'action_out_proj.weight' in key:
                    tensor = tensor.T.contiguous()
                    print(f"  Transposed {key}: {list(reversed(list(arr.shape)))} -> {list(tensor.shape)}")
                
                state_dict[key] = tensor
                param_count += tensor.numel()
            
            if len(state_dict) % 50 == 0:  # Print every 50 params
                print(f"  Converted {len(state_dict)} params...")
        except Exception as e:
            print(f"Warning: Could not convert {prefix}: {e}")
    elif hasattr(jax_tree, '__dict__'):
        for k, v in jax_tree.__dict__.items():
            new_prefix = f"{prefix}.{k}" if prefix else k
            convert_jax_to_pytorch(v, new_prefix)
    else:
        # Try to convert as array (fallback)
        try:
            arr = np.asarray(jax_tree)
            if arr.size == 0:  # Skip empty arrays
                return
            tensor = torch.from_numpy(arr)
        
            # Apply key renaming (order matters to avoid double replacement)
            key = prefix
            if key.endswith('.kernel'):
                key = key[:-7] + '.weight'
            elif key.endswith('.scale'):
                key = key[:-6] + '.weight'
            elif key.endswith('.w'):
                key = key[:-2] + '.weight'
            
            # Add model. prefix if not present
            if not key.startswith('model.'):
                key = f"model.{key}"
            
            # Fix PaliGemma paths
            key = key.replace('model.PaliGemma.', 'model.paligemma_with_expert.paligemma.model.')
            
            # Transpose action projection layers (JAX uses different convention)
            if 'action_in_proj.weight' in key or 'action_out_proj.weight' in key:
                tensor = tensor.T.contiguous()
                print(f"  Transposed {key}: {list(reversed(list(arr.shape)))} -> {list(tensor.shape)}")
            
            state_dict[key] = tensor
            param_count += tensor.numel()
            
            if len(state_dict) % 50 == 0:
                print(f"  Converted {len(state_dict)} params...")
        except Exception:
            pass  # Not an array, skip

convert_jax_to_pytorch(jax_params)

print(f"\nConverted {len(state_dict)} parameters ({param_count:,} total elements)")

# Load dataset statistics to create processor configs
print("\n" + "="*60)
print("Creating processor configurations from dataset stats...")
print("="*60)

dataset_path = Path(args.dataset_path)
stats_path = dataset_path / "meta" / "stats.json"

if not stats_path.exists():
    print(f"Warning: Dataset stats not found at {stats_path}")
    print("Processor configs will not be created. You'll need to create them manually.")
    stats = None
else:
    with open(stats_path) as f:
        stats = json.load(f)
    print(f"Loaded stats from {stats_path}")

# Save as safetensors
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True)

print(f"Saving to {output_dir}/model.safetensors...")
save_file(state_dict, str(output_dir / "model.safetensors"))

# Detect features from dataset stats
if stats:
    # Find all observation.state and observation.images.* keys
    state_features = {}
    image_features = {}
    action_feature = None
    
    for key in stats.keys():
        if key == "action":
            action_shape = len(stats[key]["mean"])
            action_feature = {
                "type": "ACTION",
                "shape": [action_shape],
                "normalization": {
                    "mode": "MEAN_STD",
                    "mean": stats[key]["mean"],
                    "std": stats[key]["std"]
                }
            }
        elif key == "observation.state":
            state_shape = len(stats[key]["mean"])
            state_features[key] = {
                "type": "STATE",
                "shape": [state_shape],
                "normalization": {
                    "mode": "MIN_MAX",
                    "min": stats[key]["min"],
                    "max": stats[key]["max"]
                }
            }
        elif key.startswith("observation.images."):
            # Flatten nested list structure for mean/std
            mean = [stats[key]["mean"][i][0][0] for i in range(3)]
            std = [stats[key]["std"][i][0][0] for i in range(3)]
            image_features[key] = {
                "type": "VISUAL",
                "shape": [3, 224, 224],
                "normalization": {
                    "mode": "MEAN_STD",
                    "mean": mean,
                    "std": std
                }
            }
    
    # Build config with detected features
    input_features = {**state_features, **image_features}
    output_features = {"action": action_feature} if action_feature else {}
    
    config = {
        "type": "pi05",
        "n_obs_steps": 1,
        "input_features": {k: {tk: tv for tk, tv in v.items() if tk != "normalization"} 
                           for k, v in input_features.items()},
        "output_features": {k: {tk: tv for tk, tv in v.items() if tk != "normalization"} 
                            for k, v in output_features.items()},
        "device": "cuda",
        "use_amp": False,
        "push_to_hub": True,
        "repo_id": None,
        "private": None,
        "tags": None,
        "license": None,
        "pretrained_path": None,
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "dtype": "bfloat16",
        "chunk_size": 50,
        "n_action_steps": 50,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "num_inference_steps": 10,
        "time_sampling_beta_alpha": 1.5,
        "time_sampling_beta_beta": 1.0,
        "time_sampling_scale": 0.999,
        "time_sampling_offset": 0.001,
        "min_period": 0.004,
        "max_period": 4.0,
        "rtc_config": None,
        "image_resolution": [224, 224],
        "empty_cameras": 0,
        "tokenizer_max_length": 200,
        "normalization_mapping": {
            "VISUAL": "MEAN_STD",
            "STATE": "MIN_MAX",
            "ACTION": "MEAN_STD"
        },
        "gradient_checkpointing": True,
        "compile_model": False,
        "compile_mode": "max-autotune",
        "optimizer_lr": 2.5e-05,
        "optimizer_betas": [0.9, 0.95],
        "optimizer_eps": 1e-08,
        "optimizer_weight_decay": 0.01,
        "optimizer_grad_clip_norm": 1.0,
        "scheduler_warmup_steps": 1000,
        "scheduler_decay_steps": 30000,
        "scheduler_decay_lr": 2.5e-06
    }
    
    # Create processor configurations
    all_features = {**input_features, **output_features}
    
    preprocessor = {
        "name": "policy_preprocessor",
        "steps": [
            {
                "registry_name": "rename_observations_processor",
                "config": {"rename_map": {}}
            },
            {
                "registry_name": "to_batch_processor",
                "config": {}
            },
            {
                "registry_name": "normalizer_processor",
                "config": {
                    "eps": 1e-08,
                    "features": all_features,
                    "norm_map": {
                        "VISUAL": "MEAN_STD",
                        "STATE": "MIN_MAX",
                        "ACTION": "MEAN_STD"
                    }
                }
            },
            {
                "registry_name": "pi05_prepare_state_tokenizer_processor_step",
                "config": {}
            },
            {
                "registry_name": "tokenizer_processor",
                "config": {
                    "max_length": 200,
                    "task_key": "task",
                    "padding_side": "right",
                    "padding": "max_length",
                    "truncation": True,
                    "tokenizer_name": "google/paligemma-3b-pt-224"
                }
            },
            {
                "registry_name": "device_processor",
                "config": {
                    "device": "cuda",
                    "float_dtype": None
                }
            }
        ]
    }
    
    postprocessor = {
        "name": "policy_postprocessor",
        "steps": [
            {
                "registry_name": "device_processor",
                "config": {
                    "device": "cpu",
                    "float_dtype": None
                }
            },
            {
                "registry_name": "unnormalizer_processor",
                "config": {
                    "eps": 1e-08,
                    "features": output_features,
                    "norm_map": {
                        "ACTION": "MEAN_STD"
                    }
                }
            }
        ]
    }
    
    with open(output_dir / "policy_preprocessor.json", "w") as f:
        json.dump(preprocessor, f, indent=2)
    
    with open(output_dir / "policy_postprocessor.json", "w") as f:
        json.dump(postprocessor, f, indent=2)
    
    print("✓ Created processor configurations:")
    print(f"  - {output_dir / 'policy_preprocessor.json'}")
    print(f"  - {output_dir / 'policy_postprocessor.json'}")
    print("\nDetected features:")
    for name, feature in input_features.items():
        print(f"  - {name}: {feature['type']} {feature['shape']}")
    for name, feature in output_features.items():
        print(f"  - {name}: {feature['type']} {feature['shape']}")
else:
    # No stats available, create minimal config
    config = {
        "type": "pi05",
        "n_obs_steps": 1,
        "device": "cuda",
        "dtype": "bfloat16",
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "chunk_size": 50,
        "n_action_steps": 50,
        "gradient_checkpointing": True
    }

# Save config

# Save config
with open(output_dir / "config.json", 'w') as f:
    json.dump(config, f, indent=2)

print(f"✓ Created config.json")
print(f"\n{'='*60}")
print(f"✓ Conversion complete!")
print(f"✓ Model ready at: {output_dir.absolute()}")
print(f"{'='*60}")

