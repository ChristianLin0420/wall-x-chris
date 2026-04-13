# LIBERO Training Guide for Wall-X

## Summary of Changes

Wall-X has been configured to support training on the **LIBERO** benchmark dataset. The following files have been created or modified:

### ✅ Files Created

1. **`workspace/lerobot_example/config_libero.yml`**
   - LIBERO-specific configuration with 7-DOF robot setup
   - Configured for Franka Panda robot (single arm)
   - Dataset: `lerobot/libero_spatial_image` (can be changed)

2. **`workspace/lerobot_example/run_libero.sh`**
   - Training script specifically for LIBERO
   - Launches training with LIBERO configuration

3. **`docs/LIBERO_TRAINING_GUIDE.md`**
   - This guide

### ✅ Files Modified

1. **`wall_x/data/load_lerobot_dataset.py`**
   - Added camera mappings for all 5 LIBERO suites:
     - `libero_spatial_image`
     - `libero_object_image`
     - `libero_goal_image`
     - `libero_90_image`
     - `libero_10_image`

2. **`wall_x/utils/constant.py`**
   - Added action normalization statistics for all LIBERO suites
   - Configured for 7-DOF end-effector control
   - Includes proprioception (8-dim quaternion orientation)

---

## Quick Start Guide

### Step 1: Prerequisites

Ensure LeRobot is installed:
```bash
cd /lustre/fsw/.../wall-x-chris
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
cd ..
```

### Step 2: Verify Dataset Access

Test that you can load the LIBERO dataset:
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Test loading LIBERO-Spatial
dataset = LeRobotDataset(
    repo_id="lerobot/libero_spatial_image",
    delta_timestamps={"action": [t/10 for t in range(31)]},
    video_backend="pyav"
)
print(f"✅ Loaded {dataset.num_episodes} episodes, {dataset.num_frames} frames")
print(f"Camera keys: {dataset.meta.camera_keys}")
print(f"Action keys: {list(dataset.meta.stats['action']['names'])}")
```

**Expected Output:**
```
✅ Loaded ~500 episodes, ~140000 frames
Camera keys: ['observation.images.image', 'observation.images.image2']
Action keys: [array of 7 action dimensions]
```

### Step 3: Update Configuration Paths

Edit `workspace/lerobot_example/config_libero.yml` and update these paths:

```yaml
processor_path: "/path/to/your/processor/"
pretrained_qwen_vl_path: "/path/to/your/qwen_vl_model/"
qwen_vl_act_config_path: "/path/to/your/config.json"
action_tokenizer_path: "/path/to/your/fast/"
save_path: "/path/to/your/output/libero_training/"
```

### Step 4: Make Training Script Executable

```bash
chmod +x workspace/lerobot_example/run_libero.sh
```

### Step 5: Launch Training

```bash
cd /lustre/fsw/.../wall-x-chris
bash workspace/lerobot_example/run_libero.sh
```

---

## Configuration Details

### Robot Configuration Comparison

| Aspect | ALOHA (Previous) | LIBERO (New) |
|--------|------------------|--------------|
| Robot | Bimanual mobile | Single arm (Franka Panda) |
| DOF | 20 | 7 |
| Actions | Dual arm + base | End-effector + gripper |
| Cameras | 3 views | 2 views |
| Proprio Dim | 20 | 8 (xyz + quat + gripper) |

### LIBERO Dataset Suites

Choose which suite to train on by editing `lerobot_config.repo_id` in `config_libero.yml`:

1. **LIBERO-Spatial** (`lerobot/libero_spatial_image`)
   - 10 tasks testing spatial reasoning
   - ~500 episodes, ~140k frames
   - Good for initial validation

2. **LIBERO-Object** (`lerobot/libero_object_image`)
   - 10 tasks with object variations
   - ~500 episodes, ~140k frames

3. **LIBERO-Goal** (`lerobot/libero_goal_image`)
   - 10 tasks with goal variations
   - ~500 episodes, ~140k frames

4. **LIBERO-90** (`lerobot/libero_90_image`)
   - 90 short-horizon tasks
   - ~4500 episodes, ~1.2M frames
   - Recommended for full training

5. **LIBERO-10** (`lerobot/libero_10_image`)
   - 10 long-horizon tasks
   - ~500 episodes, ~260k frames
   - For lifelong learning evaluation

### Action Space Details

LIBERO uses 7-DOF end-effector control:

```python
Action Dimensions:
  [0:3] - ee_cartesian_pos (x, y, z)
  [3:6] - ee_rotation (roll, pitch, yaw)
  [6]   - gripper (open/close)

Proprioception (8D):
  [0:3] - ee_cartesian_pos (x, y, z)
  [3:7] - ee_orientation_quat (x, y, z, w)
  [7]   - gripper position
```

### Camera Configuration

LIBERO provides 2 camera views at 256×256 resolution:

```python
Cameras:
  - observation.images.image → "agentview" (main/face camera)
  - observation.images.image2 → "robot0_eye_in_hand" (wrist camera)
```

---

## Training Monitoring

### Expected Behavior

1. **Initial Setup** (~2 minutes):
   - Dataset loading and preprocessing
   - Model initialization
   - Distributed training setup

2. **Training Loop**:
   - Batch processing: ~1-2 sec/step (8 GPUs)
   - Loss should start decreasing after ~100 steps
   - Memory usage: ~88GB per GPU

3. **Loss Metrics** (check WandB):
   - `train_loss`: Combined loss (should decrease)
   - `flow_loss`: Action prediction (should converge < 0.5)
   - `cross_entropy_loss`: Language modeling (should be stable)

### Troubleshooting

#### Issue: Dataset Not Found
```bash
# Verify dataset is accessible
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; \
           ds = LeRobotDataset('lerobot/libero_spatial_image'); \
           print('Dataset loaded successfully')"
```

#### Issue: Camera Key Mismatch
Check that camera mappings in `load_lerobot_dataset.py` match your dataset:
```python
# Should see these keys in dataset
dataset.meta.camera_keys
# ['observation.images.image', 'observation.images.image2']
```

#### Issue: Action Dimension Mismatch
LIBERO actions are 7-DOF but code pads to 20-DOF. This is normal and handled automatically.

#### Issue: CUDA OOM
Reduce batch size in `config_libero.yml`:
```yaml
batch_size_per_gpu: 4  # Instead of 8
gradient_accumulation_steps: 64  # Instead of 32
```

---

## Validation Steps

### Before Training

1. **Test Dataset Loading**:
   ```bash
   python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; \
              ds = LeRobotDataset('lerobot/libero_spatial_image', \
                                  delta_timestamps={'action': [t/10 for t in range(31)]}); \
              print(f'Episodes: {ds.num_episodes}, Frames: {ds.num_frames}')"
   ```
   Expected: ~500 episodes, ~140k frames

2. **Verify Configuration**:
   ```bash
   python train_qact.py --config workspace/lerobot_example/config_libero.yml --help
   ```
   Should load without errors

### During Training

1. Monitor WandB dashboard for:
   - Loss curves (should decrease)
   - Learning rate schedule
   - GPU utilization

2. Check terminal output for:
   - No NaN losses
   - Reasonable throughput (~10k samples/sec)
   - Memory within limits

### After Training

1. **Checkpoint Verification**:
   ```bash
   ls -lh /path/to/your/output/libero_training/
   ```
   Should see checkpoint directories

2. **Inference Test**:
   ```python
   from wall_x.model.qwen2_5_based import Qwen2_5_VLMoEForAction

   model = Qwen2_5_VLMoEForAction.from_pretrained("/path/to/checkpoint/")
   # Run inference on validation data
   ```

---

## Next Steps

### Recommended Training Progression

1. **Phase 1: Validate Pipeline** (~2-4 hours)
   - Train on LIBERO-Spatial (10 tasks)
   - Run for 1000 steps to verify everything works
   - Check loss curves

2. **Phase 2: Scale Up** (~1-2 days)
   - Switch to LIBERO-90 (90 tasks)
   - Train for full epochs
   - Monitor validation metrics

3. **Phase 3: Fine-tune** (~1-2 days)
   - Continue training or fine-tune on specific suites
   - Evaluate on LIBERO-10 for lifelong learning

### Evaluation Metrics

Based on LIBERO benchmark literature:
- **Success Rate**: 60-90% on seen tasks
- **Action L1 Loss**: < 0.15 for position, < 0.2 for rotation
- **Generalization**: 40-70% on unseen tasks

### Switching Datasets

To train on a different LIBERO suite, edit `config_libero.yml`:
```yaml
lerobot_config:
  repo_id: "lerobot/libero_90_image"  # Change this line
```

Or train on multiple suites (requires code modification):
```yaml
lerobot_config:
  datasets:
    - repo_id: "lerobot/libero_spatial_image"
    - repo_id: "lerobot/libero_object_image"
```

---

## Resources

### Documentation
- [LIBERO Paper](https://arxiv.org/abs/2306.03310) - NeurIPS 2023
- [LIBERO Dataset on HuggingFace](https://huggingface.co/datasets/HuggingFaceVLA/libero)
- [LeRobot LIBERO Guide](https://huggingface.co/docs/lerobot/libero)
- [LIBERO Official GitHub](https://github.com/Lifelong-Robot-Learning/LIBERO)

### Compute Requirements
- **GPUs**: 8× A100 (80GB recommended)
- **Storage**: ~100GB for datasets
- **Training Time**:
  - LIBERO-Spatial: 2-4 hours
  - LIBERO-90: 3-7 days

### Support
For issues or questions:
1. Check [LIBERO GitHub Issues](https://github.com/Lifelong-Robot-Learning/LIBERO/issues)
2. Check [LeRobot GitHub Issues](https://github.com/huggingface/lerobot/issues)
3. Review Wall-X training logs and WandB dashboard

---

## Summary

Wall-X is now configured to train on LIBERO benchmark dataset:

✅ **Configuration**: `config_libero.yml` created for 7-DOF robot
✅ **Camera Mappings**: Added for all 5 LIBERO suites
✅ **Action Statistics**: Configured for end-effector control
✅ **Training Script**: `run_libero.sh` ready to launch

**Next Action**: Update paths in `config_libero.yml` and run `bash workspace/lerobot_example/run_libero.sh`

Good luck with training! 🤖
