#!/usr/bin/env python
"""
Create processor configuration files for pi05_pytorch model using dataset statistics.
"""

import json
from pathlib import Path

# Load dataset statistics
dataset_path = Path("datasets/pick_up_four_cubes_and_stack_them_in_the_middle-v3.0")
stats_path = dataset_path / "meta" / "stats.json"

with open(stats_path) as f:
    stats = json.load(f)

# Extract statistics for each feature
features = {}

# State feature (MIN_MAX normalization)
features["observation.state"] = {
    "type": "STATE",
    "shape": [6],
    "normalization": {
        "mode": "MIN_MAX",
        "min": stats["observation.state"]["min"],
        "max": stats["observation.state"]["max"]
    }
}

# Image features (MEAN_STD normalization)
for camera in ["top", "wrist"]:
    key = f"observation.images.{camera}"
    # Flatten nested lists to simple arrays
    mean = [stats[key]["mean"][i][0][0] for i in range(3)]
    std = [stats[key]["std"][i][0][0] for i in range(3)]
    
    features[key] = {
        "type": "VISUAL",
        "shape": [3, 224, 224],
        "normalization": {
            "mode": "MEAN_STD",
            "mean": mean,
            "std": std
        }
    }

# Action feature (MEAN_STD normalization)
features["action"] = {
    "type": "ACTION",
    "shape": [6],
    "normalization": {
        "mode": "MEAN_STD",
        "mean": stats["action"]["mean"],
        "std": stats["action"]["std"]
    }
}

# Create preprocessor configuration
preprocessor = {
    "name": "policy_preprocessor",
    "steps": [
        {
            "registry_name": "rename_observations_processor",
            "config": {
                "rename_map": {}
            }
        },
        {
            "registry_name": "to_batch_processor",
            "config": {}
        },
        {
            "registry_name": "normalizer_processor",
            "config": {
                "eps": 1e-08,
                "features": features,
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

# Create postprocessor configuration (for unnormalizing actions)
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
                "features": {
                    "action": features["action"]
                },
                "norm_map": {
                    "ACTION": "MEAN_STD"
                }
            }
        }
    ]
}

# Save configurations
output_dir = Path("pi05_pytorch")
with open(output_dir / "policy_preprocessor.json", "w") as f:
    json.dump(preprocessor, f, indent=2)

with open(output_dir / "policy_postprocessor.json", "w") as f:
    json.dump(postprocessor, f, indent=2)

print("✓ Created processor configurations:")
print(f"  - {output_dir / 'policy_preprocessor.json'}")
print(f"  - {output_dir / 'policy_postprocessor.json'}")
print("\nFeatures configured:")
for name, feature in features.items():
    print(f"  - {name}: {feature['type']} {feature['shape']}")
