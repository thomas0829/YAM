# YAM 
Teleoperation, Data Collection, and model evaluation on Bimanual YAM.  

## Gello Configuration
Everything for gello is located in gello_software.
```
cd gello_software
```

Upon reconnecting the YAM to the PC, make sure to reset the CAN.
```
bash scripts/reset_all_can.sh
```
Configuration of the left arm is in ```configs/yam_left.yaml``` and configuration of the right arm is in ```configs/yam_right.yaml```.


### Teleoperation
To perform teleoperation, simply run
```
python experiments/launch_yaml.py --left_config_path=configs/yam_left.yaml --right_config_path=configs/yam_right.yaml
```

### Data Collection
To perform data collection, we need to change the content of the configuration file ```configs/yam_left.yaml```.

Goto ```yam_left.yaml``` and you will see a section called ```storage```:
```bash
# Data storage configuration
storage:
  episodes: 50
  base_dir: "/home/sean/Desktop/YAM/gello_software/data"
  task_directory: "stack_cubes"
  language_instruction: "stack cubes"
  teleop_device: "oculus" # ["oculus", "keyboard", "gello", "none"]
  save_format: "json" # ["json", "npy"]
  old_format: false
```
```episode``` is the maximum episode you can collect. The program will be killed when it reaches this number.

```base_dir``` is the location to store all the collected data.

```task_directory``` is the name of the directory you want to store all the data.

```language_instruction``` is the instruction of the task.

Don't need to change the other parameters. 

To perform data collection after configured the data storage, simply run:
```bash
python experiments/launch_yaml_collect_data.py --left_config_path=configs/yam_left.yaml --right_config_path=configs/yam_right.yaml
```

The program will launch a color pad to take keyboard input. 

Press ```s``` to start collecting 1 episode of data and the color pad will turn blue.

Press ```a``` to end and save collected episode and the color pad will turn green.

Press ```b``` to end and delete collected episode and the color pad will turn red.

Note: make sure you are on the color pad so it can take in the keyboard input. (don't put it in the background). But to kill the program with ```ctrl+c```, you will need to be on cursor.
If not, the color pad will obsorb the keyboard command and the program will not be killed. 

### Model Evaluation
To perform evaluation, we need to change the content of the configuration file ```configs/yam_left.yaml```.

Goto ```yam_left.yaml``` and you will see a section called ```policy```:
```bash
# Policy configuration
policy:
  _target_: lerobot.common.policies.diffusion.configuration_diffusion.DiffusionConfig
  repo_id: "/home/sean/Desktop/YAM/lerobot_v30/src/lerobot/datasets/yam_fold_towel_dp_dataset_v3_test"
  checkpoint_path: "hqfang/nov22-yam-fold_towel-ditx-vit-clip-flow_matching"
```

```_target_``` is the path to the model configuration. Don't need to change if evaluating diffusion or dit model.

```repo_id``` is the absolute path to the dataset used to train the model. You will need to download the dataset from huggingface if it's not in local.

```checkpoint_path``` is the name of the model directory on huggingface or the absolute path to the model in local directory.

To perform evaluation after configured the policy, simply run:
```bash
python experiments/launch_yaml_eval.py --left_config_path=configs/yam_left.yaml --right_config_path=configs/yam_right.yaml
```

If you are not evaluating diffusion or dit, you will need to also update ```experiments/launch_yaml_eval.py``` after you updated the ```_target_``` in ```yam_left.yaml```. Find this line that 
creates the policy and change it to the model you are evaluating:
```bash
policy = DiffusionPolicy.from_pretrained(model_id)
```

Notes: this function ```preprocess_observation``` in ```experiments/launch_yaml_eval.py``` is used to convert the robot observation into model input format. Make sure its output matches the 
model desired input. Currently, it doesn't passed in the language instruction. You can add it in here if needed. 
```bash
    # Define the target image size
    TARGET_HEIGHT = 256
    TARGET_WIDTH = 342

    # Map cameras "observation : model"
    camera_mapping = {"left_camera_rgb": 'left', "right_camera_rgb": 'right', "front_camera_rgb": 'front'}
```


