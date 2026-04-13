#!/bin/bash
# Training script for Wall-X on LIBERO benchmark dataset
# LIBERO: Lifelong Benchmark for Robot Learning (NeurIPS 2023)

# Configure GPUs (default: use all 8 GPUs)
# export CUDA_VISIBLE_DEVICES=4,5,6,7  # Use subset of GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Print current time and configuration
echo "[current time: $(date +'%Y-%m-%d %H:%M:%S')]"
echo "Number of GPUs: $NUM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Update these paths to match your environment
code_dir="/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/wall-x-chris"
config_path="${code_dir}/workspace/lerobot_example"

# Generate a random port for distributed training
export PORT=$((21000 + $RANDOM % 30000))
MASTER_PORT=10239  # Use 5-digit port

# Configure accelerate launcher
export LAUNCHER="accelerate launch --num_processes=$NUM_GPUS --main_process_port=$PORT"

# Set training script and configuration
export SCRIPT="${code_dir}/train_qact.py"
export SCRIPT_ARGS="--config ${config_path}/config_libero.yml --seed $MASTER_PORT"

echo "=" * 80
echo "Wall-X Training on LIBERO Dataset"
echo "=" * 80
echo "Configuration: config_libero.yml"
echo "Dataset: LIBERO Spatial (7-DOF Franka Panda)"
echo "GPUs: $NUM_GPUS"
echo "Port: $PORT"
echo "=" * 80

echo "Running command: $LAUNCHER $SCRIPT $SCRIPT_ARGS"
$LAUNCHER $SCRIPT $SCRIPT_ARGS
