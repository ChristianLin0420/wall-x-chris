# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wall-X is a Vision-Language-Action (VLA) foundation model for robotics, built on Qwen2.5-VL. It combines multimodal understanding (vision + language) with precise robotic action prediction using flow matching and discrete action tokenization.

## Build & Installation

```bash
# Create conda environment
conda create --name wallx python=3.10
conda activate wallx

# Install dependencies
pip install -r requirements.txt
MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation

# Install LeRobot (data loading)
git clone https://github.com/huggingface/lerobot.git
cd lerobot && pip install -e . && cd ..

# Install Wall-X with CUDA extensions
git submodule update --init --recursive
MAX_JOBS=4 pip install --no-build-isolation --verbose .
```

## Training

Training uses Accelerate for distributed multi-GPU training with BF16 mixed precision.

```bash
# Run training (8 GPUs by default)
bash ./workspace/lerobot_example/run.sh

# Or manually with accelerate
accelerate launch --num_processes=8 train_qact.py --config workspace/lerobot_example/config_qact.yml --seed 42
```

Configuration is in `workspace/lerobot_example/config_qact.yml`. Key paths to configure:
- `processor_path`: Path to model processor
- `pretrained_qwen_vl_path`: Pretrained Qwen VL model
- `save_path`: Output directory

## Inference

```bash
python ./scripts/fake_inference.py  # Test model loading and inference
python ./scripts/draw_openloop_plot.py  # Generate open-loop comparison plots
```

## Architecture

### Core Components

- **wall_x/model/qwen2_5_based/**: Main VLA model implementation
  - `modeling_qwen2_5_vl_act.py`: `Qwen2_5_VLMoEForAction` - the full VLA model with MoE transformer
  - `modeling_qwen2_5_vl.py`: Base vision-language model
  - `configuration_qwen2_5_vl.py`: Model configuration

- **wall_x/model/action_head.py**: Action prediction head using flow matching

- **wall_x/trainer/qwen_vl_act_trainer.py**: `QwenVlAct_Trainer` - main training loop with distributed training support

- **wall_x/data/**: Data loading pipeline
  - `load_lerobot_dataset.py`: LeRobot dataset adapter
  - `config.py`: Data configuration
  - `utils.py`: Data utilities

- **wall_x/fusions/**: Custom CUDA operations
  - `ops.py`: Python bindings for CUDA kernels
  - `backend.py`: Backend selection

- **csrc/**: Custom CUDA kernels
  - `dual_asym_grouped_gemm.cu/h`: Grouped GEMM for MoE
  - `rope.cu/h`: 3D RoPE implementation
  - `permute.cu/h`: Tensor permutation

### Model Architecture (from docs/ARCHITECTURE.md)

```
Input → Vision Encoder → Embedding Fusion → MoE Transformer → Output Heads → Loss
```

- **Vision Encoder**: Qwen2.5 ViT with 3D RoPE for spatial-temporal understanding
- **MoE Transformer**: 32 layers, 8 experts per layer
  - Expert 0: General language
  - Expert 1: Action prediction
  - Experts 2-7: Specialized multimodal processing
- **Action Head**: Flow matching with beta distribution noise scheduling

### Key Tensor Shapes

- `pixel_values`: `[B, N, C, H, W]` - multi-camera images
- `input_ids`: `[B, S]` - tokenized text with special tokens
- `action_chunk`: `[B, T, A]` - action sequences (T=32 horizon, A=DOF)
- `proprioception`: `[B, T, P]` - robot state (joint positions, gripper state)

### Special Tokens

- `<|propri|>`, `<|propri_start|>`, `<|propri_end|>`: Proprioceptive data
- `<|action|>`, `<|action_start|>`, `<|action_end|>`: Action sequences
- `<|image|>`, `<|video|>`: Vision tokens
- `<|camera_face|>`, `<|camera_wrist|>`, `<|camera_top|>`: Camera-specific tokens

## Robot Configuration

DOF configuration in `config_qact.yml` defines the robot's action space:
```yaml
dof_config:
  follow_left_ee_cartesian_pos: 3   # Left arm position (x,y,z)
  follow_left_ee_rotation: 3        # Left arm rotation (roll,pitch,yaw)
  follow_left_gripper: 1            # Left gripper
  follow_right_ee_cartesian_pos: 3  # Right arm position
  follow_right_ee_rotation: 3       # Right arm rotation
  follow_right_gripper: 1           # Right gripper
  head_actions: 2                   # Head/camera (pan,tilt)
  height: 1                         # Mobile base height
  car_pose: 3                       # Mobile base (x,y,theta)
```

## Key Dependencies

- PyTorch 2.6.0 with CUDA 12.x
- Flash Attention 2.7.4.post1
- Transformers 4.49.0
- Accelerate 1.10.1
- LeRobot (for dataset loading)
- CUTLASS (submodule at 3rdparty/cutlass)
