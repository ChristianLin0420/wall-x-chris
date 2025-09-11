# WALL-X VLA Model Training Guide

## Table of Contents
1. [Overview](#overview)
2. [Model Architecture](#model-architecture)
3. [Data Preprocessing](#data-preprocessing)
4. [Training Configuration](#training-configuration)
5. [Model Training Process](#model-training-process)
6. [Advanced Features](#advanced-features)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

## Overview

WALL-X is a Vision-Language-Action (VLA) foundation model that represents a breakthrough in robotic AI by seamlessly integrating multimodal understanding with precise robotic control capabilities. Built upon the robust Qwen2.5-VL architecture, WALL-X extends the traditional vision-language paradigm to include sophisticated action prediction and robotic manipulation, making it a comprehensive solution for embodied AI applications.

### Key Features

#### Multi-modal Processing
- **Vision Processing**: Handles multiple camera views (face, wrist, top, etc.) with 3D RoPE for spatial-temporal understanding
- **Language Understanding**: Processes natural language instructions with contextual grounding and task-specific reasoning
- **Action Prediction**: Generates precise robotic actions using both continuous flow matching and discrete token prediction
- **Proprioceptive Integration**: Incorporates joint positions, orientations, and sensor data for comprehensive state awareness

#### Mixture of Experts (MoE) Architecture
- **8 Specialized Experts**: Each expert network handles specific aspects of the multimodal task
- **Dynamic Routing**: Intelligent token routing based on content type and task requirements
- **Efficient Scaling**: Reduces computational overhead while maintaining model capacity
- **Expert Specialization**: 
  - Expert 0: General language understanding and reasoning
  - Expert 1: Action prediction and robotic control
  - Experts 2-7: Specialized multimodal processing (vision-language fusion, temporal modeling, etc.)

#### Flow Matching for Action Prediction
- **Continuous Diffusion**: Advanced action prediction using continuous diffusion processes
- **Beta Distribution Noise**: Sophisticated noise scheduling for stable training
- **Multi-robot Support**: Configurable for different robot configurations and degrees of freedom
- **Temporal Consistency**: Maintains smooth action sequences across time steps

#### Multi-robot Support
- **Configurable DOF**: Supports various robot configurations (bimanual, mobile, etc.)
- **Action Tokenization**: Discrete action tokens for fast inference mode
- **Robot-specific Adaptation**: Customizable for different robotic platforms
- **Scalable Architecture**: Easily extensible to new robot types and configurations

#### Distributed Training
- **Multi-GPU Support**: Efficient training across multiple GPUs using Accelerate
- **Mixed Precision**: BF16 training for memory efficiency and speed
- **Gradient Accumulation**: Large effective batch sizes with limited GPU memory
- **Checkpoint Management**: Robust checkpointing and resumption capabilities

## Model Architecture

### 1. Core Components

#### Vision Encoder
The vision encoder is based on the Qwen2.5 Vision Transformer, specifically designed for multimodal robotic applications.

**Architecture Details:**
- **Base Model**: Qwen2.5 Vision Transformer with 32 layers
- **Input Processing**: Multi-camera views (face, wrist, top, etc.) with smart resizing
- **Resolution**: Configurable (default 256x256, supports up to 1024x1024)
- **Patch Size**: 16x16 patches for optimal balance between detail and efficiency
- **Hidden Dimension**: 4096 for rich feature representation

**3D RoPE (Rotary Position Embedding):**
```python
# 3D RoPE Implementation
class RotaryPositionEmbedding3D:
    def __init__(self, dim, max_freq=10000):
        self.dim = dim
        self.max_freq = max_freq
        
    def get_3d_rotary_emb(self, x, t, h, w):
        # Temporal dimension (t): Video frame timing
        # Height dimension (h): Vertical spatial position  
        # Width dimension (w): Horizontal spatial position
        freq_t = 1.0 / (self.max_freq ** (torch.arange(0, self.dim//3, 2) / (self.dim//3)))
        freq_h = 1.0 / (self.max_freq ** (torch.arange(0, self.dim//3, 2) / (self.dim//3)))
        freq_w = 1.0 / (self.max_freq ** (torch.arange(0, self.dim//3, 2) / (self.dim//3)))
        
        # Apply rotary embeddings
        return self.apply_rotary_emb(x, t, h, w, freq_t, freq_h, freq_w)
```

**Multi-Camera Processing:**
```python
def process_multi_camera_views(frames, camera_configs):
    """Process multiple camera views with consistent formatting"""
    processed_views = {}
    for camera_name, config in camera_configs.items():
        # Smart resizing maintaining aspect ratio
        resized_frame = smart_resize(frames[camera_name], config['target_size'])
        # Apply camera-specific preprocessing
        processed_views[camera_name] = apply_camera_preprocessing(resized_frame, config)
    return processed_views
```

#### Language Model
The language model extends Qwen2.5-VL with specialized tokens for robotic applications.

**Architecture Details:**
- **Base Architecture**: Qwen2.5-VL Transformer with 32 decoder layers
- **Hidden Size**: 4096 dimensions
- **Vocabulary Size**: 151,936 tokens (includes special robotic tokens)
- **Context Length**: Up to 32,768 tokens for long sequences

**Special Tokens:**
```python
# Special Token Definitions
SPECIAL_TOKENS = {
    # Proprioceptive data tokens
    '<|propri|>': 'Proprioceptive sensor data (joint positions, orientations)',
    '<|propri_start|>': 'Start of proprioceptive data sequence',
    '<|propri_end|>': 'End of proprioceptive data sequence',
    
    # Action sequence tokens
    '<|action|>': 'Action sequence data (continuous values)',
    '<|action_start|>': 'Start of action sequence',
    '<|action_end|>': 'End of action sequence',
    
    # Discrete action tokens (fast mode)
    '<|action_token_0|>': 'Discrete action token 0',
    '<|action_token_1|>': 'Discrete action token 1',
    # ... up to action_token_255 for 256 discrete actions
    
    # Vision tokens
    '<|image|>': 'Image data token',
    '<|video|>': 'Video data token',
    '<|camera_face|>': 'Face camera view',
    '<|camera_wrist|>': 'Wrist camera view',
    '<|camera_top|>': 'Top camera view',
    
    # Task-specific tokens
    '<|task_start|>': 'Task execution start',
    '<|task_end|>': 'Task execution end',
    '<|error|>': 'Error state token',
    '<|success|>': 'Success state token'
}
```

**Token Processing Pipeline:**
```python
def process_instruction_tokens(instruction, context):
    """Process natural language instructions with robotic context"""
    # Add task context
    if context.get('task_type'):
        instruction = f"Task: {context['task_type']}. {instruction}"
    
    # Add camera context
    if context.get('camera_views'):
        camera_info = " ".join([f"<|camera_{view}|>" for view in context['camera_views']])
        instruction = f"{camera_info} {instruction}"
    
    # Add proprioceptive context
    if context.get('proprioception'):
        instruction = f"<|propri_start|> {context['proprioception']} <|propri_end|> {instruction}"
    
    return instruction
```

#### Action Processing
The action processing component handles both continuous and discrete action prediction.

**Flow Matching Implementation:**
```python
class FlowMatchingActionHead(nn.Module):
    def __init__(self, hidden_size, action_dim, num_experts=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.num_experts = num_experts
        
        # Action projection layers
        self.action_proj = nn.Linear(hidden_size, action_dim)
        self.action_proj_back = nn.Linear(action_dim, hidden_size)
        
        # Flow matching parameters
        self.beta_alpha = 1.5
        self.beta_beta = 1.0
        self.s = 0.999
        
    def forward(self, hidden_states, action_targets=None, training=True):
        if training and action_targets is not None:
            # Flow matching training
            return self.flow_matching_loss(hidden_states, action_targets)
        else:
            # Inference
            return self.action_proj(hidden_states)
    
    def flow_matching_loss(self, hidden_states, action_targets):
        """Compute flow matching loss for action prediction"""
        # Sample noise from beta distribution
        noise = torch.distributions.Beta(self.beta_alpha, self.beta_beta).sample(action_targets.shape)
        noise = noise.to(action_targets.device)
        
        # Interpolate between noise and targets
        t = torch.rand(action_targets.shape[0], 1, device=action_targets.device)
        interpolated = (1 - t) * noise + t * action_targets
        
        # Predict flow
        flow_pred = self.action_proj(hidden_states)
        flow_target = action_targets - noise
        
        # Compute loss
        loss = F.mse_loss(flow_pred, flow_target, reduction='none')
        return loss.mean()
```

**Discrete Action Tokenization:**
```python
class DiscreteActionTokenizer:
    def __init__(self, action_dim, num_tokens=256):
        self.action_dim = action_dim
        self.num_tokens = num_tokens
        self.tokenizer = nn.Linear(action_dim, num_tokens)
        
    def encode_actions(self, actions):
        """Convert continuous actions to discrete tokens"""
        logits = self.tokenizer(actions)
        tokens = torch.argmax(logits, dim=-1)
        return tokens
    
    def decode_actions(self, tokens):
        """Convert discrete tokens back to continuous actions"""
        # Use learned embeddings for decoding
        action_embeddings = self.tokenizer.weight[tokens]
        return action_embeddings
```

### 2. Mixture of Experts (MoE)

The MoE architecture enables efficient scaling and specialized processing for different aspects of the multimodal task.

**MoE Configuration:**
```python
# MoE Configuration
class MoEConfig:
    num_experts = 8  # Number of expert networks
    expert_capacity = 64  # Tokens per expert
    routing_strategy = "random"  # Token routing method
    expert_dropout = 0.1  # Dropout for expert networks
    load_balancing_loss_weight = 0.01  # Load balancing loss weight
    
    # Expert specialization
    expert_specializations = {
        0: "general_language",  # General language understanding
        1: "action_prediction",  # Action prediction and robotic control
        2: "vision_language_fusion",  # Vision-language integration
        3: "temporal_modeling",  # Temporal sequence modeling
        4: "proprioceptive_processing",  # Proprioceptive data processing
        5: "task_planning",  # High-level task planning
        6: "error_handling",  # Error detection and recovery
        7: "multimodal_reasoning"  # Complex multimodal reasoning
    }
```

**Expert Network Implementation:**
```python
class ExpertNetwork(nn.Module):
    def __init__(self, hidden_size, expert_type, dropout=0.1):
        super().__init__()
        self.expert_type = expert_type
        self.hidden_size = hidden_size
        
        # Expert-specific architecture
        if expert_type == "action_prediction":
            self.layers = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Dropout(dropout)
            )
        elif expert_type == "vision_language_fusion":
            self.layers = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.LayerNorm(hidden_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Dropout(dropout)
            )
        # ... other expert types
        
    def forward(self, x):
        return self.layers(x)
```

**Token Routing Strategy:**
```python
class TokenRouter(nn.Module):
    def __init__(self, hidden_size, num_experts, routing_strategy="random"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.routing_strategy = routing_strategy
        
        # Routing network
        self.router = nn.Linear(hidden_size, num_experts)
        
    def forward(self, hidden_states):
        # Compute routing scores
        routing_scores = self.router(hidden_states)
        
        if self.routing_strategy == "random":
            # Random routing for load balancing
            expert_indices = torch.randint(0, self.num_experts, (hidden_states.size(0),))
        elif self.routing_strategy == "top_k":
            # Top-k routing
            top_k = 2
            _, expert_indices = torch.topk(routing_scores, top_k, dim=-1)
        else:
            # Greedy routing
            expert_indices = torch.argmax(routing_scores, dim=-1)
            
        return expert_indices, routing_scores
```

### 3. Attention Mechanisms

**Multi-head Attention:**
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        # Attention projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_size)
        
    def forward(self, query, key, value, attention_mask=None):
        batch_size, seq_len, _ = query.size()
        
        # Project to Q, K, V
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, -1e9)
            
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        context = context.contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)
        return self.out_proj(context)
```

**3D RoPE Implementation:**
```python
class RotaryPositionEmbedding3D(nn.Module):
    def __init__(self, dim, max_freq=10000):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq
        
    def get_3d_rotary_emb(self, x, t, h, w):
        """Apply 3D rotary position embedding"""
        # Temporal dimension (t): Video frame timing
        # Height dimension (h): Vertical spatial position  
        # Width dimension (w): Horizontal spatial position
        
        # Compute frequencies for each dimension
        freq_t = 1.0 / (self.max_freq ** (torch.arange(0, self.dim//3, 2) / (self.dim//3)))
        freq_h = 1.0 / (self.max_freq ** (torch.arange(0, self.dim//3, 2) / (self.dim//3)))
        freq_w = 1.0 / (self.max_freq ** (torch.arange(0, self.dim//3, 2) / (self.dim//3)))
        
        # Apply rotary embeddings
        return self.apply_rotary_emb(x, t, h, w, freq_t, freq_h, freq_w)
    
    def apply_rotary_emb(self, x, t, h, w, freq_t, freq_h, freq_w):
        """Apply rotary position embedding to input tensor"""
        # Implementation details for 3D RoPE
        # This is a simplified version - full implementation would be more complex
        pass
```

## Data Preprocessing

### 1. Dataset Structure

#### LeRobot Format
The WALL-X model is designed to work with the LeRobot dataset format, which provides a standardized way to handle multimodal robotic data.

**Dataset Configuration:**
```yaml
# Dataset Configuration
repo_id: "lerobot/aloha_mobile_cabinet"
action_horizon: 32  # Future action steps to predict
train_test_split: 0.95  # 95% training, 5% validation
resolution:
  face_view: 256  # Face camera resolution
  left_wrist_view: 256  # Left wrist camera resolution
  right_wrist_view: 256  # Right wrist camera resolution
  top_view: 256  # Top-down camera resolution

# Additional configuration
video_backend: "pyav"  # Video processing backend
delta_timestamps:
  action: [t/50 for t in range(31)]  # Action timestamps
  observation: [t/50 for t in range(31)]  # Observation timestamps

# Data augmentation settings
augmentation:
  color_jitter: 0.1  # Color jittering strength
  temporal_shift: 0.1  # Temporal shifting probability
  spatial_rotation: 5  # Maximum rotation in degrees
```

**Supported Datasets:**
- `lerobot/aloha_mobile_cabinet`: Mobile manipulation tasks
- `lerobot/pusht`: Push manipulation tasks
- `lerobot/xarm_lift_medium`: Lift manipulation tasks
- `lerobot/berkeley_autolab_ur5`: UR5 manipulation tasks

#### Data Types and Formats

**Images:**
- **Format**: Multi-camera views with consistent preprocessing
- **Resolution**: Configurable (default 256x256, supports up to 1024x1024)
- **Channels**: RGB (3 channels)
- **Normalization**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

**Actions:**
- **Format**: Continuous values normalized to [-1, 1] range
- **Dimensions**: Configurable based on robot DOF
- **Temporal**: Action sequences with configurable horizon
- **Normalization**: Min-max normalization with clipping

**Proprioception:**
- **Joint Positions**: 7-DOF arm joint angles
- **Joint Velocities**: Joint angular velocities
- **Gripper State**: Gripper open/close state
- **End-Effector Pose**: 6-DOF end-effector position and orientation

**Text:**
- **Format**: Natural language instructions
- **Encoding**: Tokenized using Qwen2.5 tokenizer
- **Context**: Enhanced with task-specific and camera-specific information

### 2. Preprocessing Pipeline

#### Vision Processing
The vision processing pipeline handles multiple camera views with consistent formatting and augmentation.

**Smart Resizing Implementation:**
```python
def smart_resize(image, target_size, maintain_aspect_ratio=True):
    """Resize image while maintaining aspect ratio"""
    h, w = image.shape[:2]
    
    if maintain_aspect_ratio:
        # Calculate scaling factor
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2
        
        padded = cv2.copyMakeBorder(
            resized, pad_h, target_size - new_h - pad_h,
            pad_w, target_size - new_w - pad_w,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        
        return padded
    else:
        return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

def preprocess_images(frames, target_size=256, augmentation_config=None):
    """Process multi-camera images with smart resizing and augmentation"""
    processed_frames = {}
    
    for camera_key, frame in frames.items():
        # Convert to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Smart resize
        frame = smart_resize(frame, target_size)
        
        # Apply augmentation if configured
        if augmentation_config:
            frame = apply_vision_augmentation(frame, augmentation_config)
        
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame = (frame - mean) / std
        
        processed_frames[camera_key] = frame
    
    return processed_frames
```

**Multi-Camera Processing:**
```python
def process_multi_camera_views(frames, camera_configs):
    """Process multiple camera views with consistent formatting"""
    processed_views = {}
    
    for camera_name, config in camera_configs.items():
        if camera_name in frames:
            # Apply camera-specific preprocessing
            frame = frames[camera_name]
            
            # Camera-specific transformations
            if config.get('flip_horizontal', False):
                frame = cv2.flip(frame, 1)
            
            if config.get('crop_region'):
                x1, y1, x2, y2 = config['crop_region']
                frame = frame[y1:y2, x1:x2]
            
            # Smart resize
            frame = smart_resize(frame, config['target_size'])
            
            # Apply camera-specific normalization
            if config.get('normalization'):
                frame = apply_camera_normalization(frame, config['normalization'])
            
            processed_views[camera_name] = frame
    
    return processed_views
```

#### Action Normalization
Action normalization ensures consistent action values across different robot configurations.

**Action Normalization Implementation:**
```python
class ActionNormalizer:
    def __init__(self, action_bounds, normalization_type='min_max'):
        self.action_bounds = action_bounds
        self.normalization_type = normalization_type
        
    def normalize_actions(self, actions, action_keys=None):
        """Normalize actions to [-1, 1] range"""
        if action_keys is None:
            action_keys = list(self.action_bounds.keys())
        
        normalized_actions = {}
        
        for key in action_keys:
            if key in actions and key in self.action_bounds:
                min_vals = self.action_bounds[key]['min']
                max_vals = self.action_bounds[key]['max']
                
                if self.normalization_type == 'min_max':
                    # Min-max normalization
                    delta = max_vals - min_vals
                    normalized = (actions[key] - min_vals) / delta
                    normalized = normalized * 2 - 1  # Scale to [-1, 1]
                elif self.normalization_type == 'z_score':
                    # Z-score normalization
                    mean = self.action_bounds[key]['mean']
                    std = self.action_bounds[key]['std']
                    normalized = (actions[key] - mean) / std
                    normalized = torch.clamp(normalized, -3, 3)  # Clip outliers
                    normalized = normalized / 3  # Scale to [-1, 1]
                
                # Clamp to valid range
                normalized = torch.clamp(normalized, -1, 1)
                normalized_actions[key] = normalized
        
        return normalized_actions
    
    def denormalize_actions(self, normalized_actions, action_keys=None):
        """Convert normalized actions back to original scale"""
        if action_keys is None:
            action_keys = list(self.action_bounds.keys())
        
        denormalized_actions = {}
        
        for key in action_keys:
            if key in normalized_actions and key in self.action_bounds:
                min_vals = self.action_bounds[key]['min']
                max_vals = self.action_bounds[key]['max']
                
                if self.normalization_type == 'min_max':
                    # Reverse min-max normalization
                    normalized = normalized_actions[key]
                    normalized = (normalized + 1) / 2  # Scale from [-1, 1] to [0, 1]
                    delta = max_vals - min_vals
                    denormalized = normalized * delta + min_vals
                elif self.normalization_type == 'z_score':
                    # Reverse z-score normalization
                    mean = self.action_bounds[key]['mean']
                    std = self.action_bounds[key]['std']
                    normalized = normalized_actions[key] * 3  # Scale from [-1, 1] to [-3, 3]
                    denormalized = normalized * std + mean
                
                denormalized_actions[key] = denormalized
        
        return denormalized_actions
```

**Action Sequence Processing:**
```python
def process_action_sequences(actions, action_horizon=32, action_keys=None):
    """Process action sequences for training"""
    if action_keys is None:
        action_keys = list(actions.keys())
    
    processed_sequences = {}
    
    for key in action_keys:
        if key in actions:
            action_seq = actions[key]
            
            # Ensure sequence length
            if len(action_seq) < action_horizon:
                # Pad with last action
                last_action = action_seq[-1]
                padding = [last_action] * (action_horizon - len(action_seq))
                action_seq = action_seq + padding
            elif len(action_seq) > action_horizon:
                # Truncate to action_horizon
                action_seq = action_seq[:action_horizon]
            
            processed_sequences[key] = torch.tensor(action_seq, dtype=torch.float32)
    
    return processed_sequences
```

#### Text Processing
Text processing enhances natural language instructions with contextual information.

**Instruction Processing:**
```python
def process_instructions(instruction, context, tokenizer):
    """Generate contextual instructions with grounding"""
    # Add task context
    if context.get('task_type'):
        instruction = f"Task: {context['task_type']}. {instruction}"
    
    # Add frame context
    if context.get('frame_index') is not None:
        instruction = f"Frame {context['frame_index']}: {instruction}"
    
    # Add camera context
    if context.get('camera_views'):
        camera_info = " ".join([f"<|camera_{view}|>" for view in context['camera_views']])
        instruction = f"{camera_info} {instruction}"
    
    # Add proprioceptive context
    if context.get('proprioception'):
        proprio_info = format_proprioception(context['proprioception'])
        instruction = f"<|propri_start|> {proprio_info} <|propri_end|> {instruction}"
    
    # Add action context
    if context.get('previous_actions'):
        action_info = format_actions(context['previous_actions'])
        instruction = f"<|action_start|> {action_info} <|action_end|> {instruction}"
    
    # Tokenize instruction
    tokens = tokenizer.encode(instruction, return_tensors='pt')
    
    return tokens, instruction

def format_proprioception(proprioception):
    """Format proprioceptive data for text processing"""
    formatted = []
    
    if 'joint_positions' in proprioception:
        joint_pos = proprioception['joint_positions']
        formatted.append(f"Joint positions: {', '.join([f'{pos:.3f}' for pos in joint_pos])}")
    
    if 'joint_velocities' in proprioception:
        joint_vel = proprioception['joint_velocities']
        formatted.append(f"Joint velocities: {', '.join([f'{vel:.3f}' for vel in joint_vel])}")
    
    if 'gripper_state' in proprioception:
        gripper = proprioception['gripper_state']
        formatted.append(f"Gripper state: {gripper:.3f}")
    
    return " | ".join(formatted)

def format_actions(actions):
    """Format action data for text processing"""
    formatted = []
    
    for key, values in actions.items():
        if isinstance(values, (list, tuple)):
            values_str = ', '.join([f'{v:.3f}' for v in values])
        else:
            values_str = f'{values:.3f}'
        formatted.append(f"{key}: {values_str}")
    
    return " | ".join(formatted)
```

### 3. Data Augmentation

#### Vision Augmentation
Vision augmentation improves model robustness and generalization.

**Vision Augmentation Implementation:**
```python
class VisionAugmentation:
    def __init__(self, config):
        self.config = config
        
    def apply_augmentation(self, image, camera_type=None):
        """Apply vision augmentation to image"""
        augmented = image.copy()
        
        # Color jittering
        if self.config.get('color_jitter', 0) > 0:
            augmented = self.color_jitter(augmented, self.config['color_jitter'])
        
        # Spatial augmentation
        if self.config.get('spatial_rotation', 0) > 0:
            augmented = self.random_rotation(augmented, self.config['spatial_rotation'])
        
        if self.config.get('spatial_translation', 0) > 0:
            augmented = self.random_translation(augmented, self.config['spatial_translation'])
        
        # Temporal augmentation
        if self.config.get('temporal_shift', 0) > 0:
            augmented = self.temporal_shift(augmented, self.config['temporal_shift'])
        
        return augmented
    
    def color_jitter(self, image, strength=0.1):
        """Apply color jittering"""
        # Brightness
        brightness = 1 + np.random.uniform(-strength, strength)
        image = image * brightness
        
        # Contrast
        contrast = 1 + np.random.uniform(-strength, strength)
        image = (image - 0.5) * contrast + 0.5
        
        # Saturation
        saturation = 1 + np.random.uniform(-strength, strength)
        image = image * saturation
        
        return np.clip(image, 0, 1)
    
    def random_rotation(self, image, max_angle=5):
        """Apply random rotation"""
        angle = np.random.uniform(-max_angle, max_angle)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        
        return rotated
    
    def random_translation(self, image, max_shift=10):
        """Apply random translation"""
        h, w = image.shape[:2]
        shift_x = np.random.uniform(-max_shift, max_shift)
        shift_y = np.random.uniform(-max_shift, max_shift)
        
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        translated = cv2.warpAffine(image, M, (w, h))
        
        return translated
```

#### Action Augmentation
Action augmentation improves action prediction robustness.

**Action Augmentation Implementation:**
```python
class ActionAugmentation:
    def __init__(self, config):
        self.config = config
        
    def apply_augmentation(self, actions, action_keys=None):
        """Apply action augmentation"""
        if action_keys is None:
            action_keys = list(actions.keys())
        
        augmented_actions = {}
        
        for key in action_keys:
            if key in actions:
                action = actions[key].copy()
                
                # Noise injection
                if self.config.get('noise_injection', 0) > 0:
                    action = self.add_noise(action, self.config['noise_injection'])
                
                # Temporal shifting
                if self.config.get('temporal_shift', 0) > 0:
                    action = self.temporal_shift(action, self.config['temporal_shift'])
                
                # DOF masking
                if self.config.get('dof_masking', 0) > 0:
                    action = self.mask_dof(action, self.config['dof_masking'])
                
                augmented_actions[key] = action
        
        return augmented_actions
    
    def add_noise(self, action, noise_std=0.01):
        """Add Gaussian noise to actions"""
        noise = np.random.normal(0, noise_std, action.shape)
        return action + noise
    
    def temporal_shift(self, action, shift_prob=0.1):
        """Apply temporal shifting to action sequences"""
        if np.random.random() < shift_prob:
            shift = np.random.randint(-2, 3)  # Shift by -2 to +2 steps
            if shift > 0:
                # Shift right, pad with last action
                action = np.concatenate([action[-shift:], action[:-shift]])
            elif shift < 0:
                # Shift left, pad with first action
                action = np.concatenate([action[-shift:], action[:shift]])
        
        return action
    
    def mask_dof(self, action, mask_prob=0.1):
        """Randomly mask degrees of freedom"""
        if np.random.random() < mask_prob:
            # Randomly select DOF to mask
            mask_indices = np.random.choice(
                action.shape[-1], 
                size=int(action.shape[-1] * 0.1), 
                replace=False
            )
            action[..., mask_indices] = 0
        
        return action
```

#### Temporal Augmentation
Temporal augmentation improves temporal sequence modeling.

**Temporal Augmentation Implementation:**
```python
class TemporalAugmentation:
    def __init__(self, config):
        self.config = config
        
    def apply_temporal_augmentation(self, sequence, sequence_type='action'):
        """Apply temporal augmentation to sequences"""
        augmented = sequence.copy()
        
        # Temporal sampling
        if self.config.get('temporal_sampling', 0) > 0:
            augmented = self.temporal_sampling(augmented, self.config['temporal_sampling'])
        
        # Temporal masking
        if self.config.get('temporal_masking', 0) > 0:
            augmented = self.temporal_masking(augmented, self.config['temporal_masking'])
        
        # Temporal interpolation
        if self.config.get('temporal_interpolation', 0) > 0:
            augmented = self.temporal_interpolation(augmented, self.config['temporal_interpolation'])
        
        return augmented
    
    def temporal_sampling(self, sequence, sample_prob=0.1):
        """Randomly sample from temporal sequence"""
        if np.random.random() < sample_prob:
            # Randomly select subset of sequence
            seq_len = len(sequence)
            sample_len = int(seq_len * np.random.uniform(0.8, 1.0))
            start_idx = np.random.randint(0, seq_len - sample_len + 1)
            sequence = sequence[start_idx:start_idx + sample_len]
        
        return sequence
    
    def temporal_masking(self, sequence, mask_prob=0.1):
        """Randomly mask temporal steps"""
        if np.random.random() < mask_prob:
            # Randomly select steps to mask
            mask_indices = np.random.choice(
                len(sequence), 
                size=int(len(sequence) * 0.1), 
                replace=False
            )
            sequence[mask_indices] = 0
        
        return sequence
    
    def temporal_interpolation(self, sequence, interp_prob=0.1):
        """Apply temporal interpolation"""
        if np.random.random() < interp_prob:
            # Interpolate between adjacent steps
            for i in range(1, len(sequence) - 1):
                if np.random.random() < 0.5:
                    sequence[i] = (sequence[i-1] + sequence[i+1]) / 2
        
        return sequence
```

## Training Configuration

### 1. Basic Configuration

#### Model Parameters
The model configuration defines the core architecture parameters for the WALL-X VLA model.

**Model Architecture Configuration:**
```yaml
# Model Configuration
model_type: qwen2_5  # Base model type
hidden_size: 4096  # Hidden dimension size
num_hidden_layers: 32  # Number of transformer layers
num_attention_heads: 32  # Number of attention heads
head_dim: 128  # Dimension per attention head
num_experts: 8  # Number of MoE experts
expert_capacity: 64  # Tokens per expert
vocab_size: 151936  # Vocabulary size including special tokens
max_position_embeddings: 32768  # Maximum sequence length
intermediate_size: 14336  # FFN intermediate size
activation_function: "silu"  # Activation function
rms_norm_eps: 1e-6  # RMS normalization epsilon
use_cache: true  # Enable KV cache for inference
```

**Vision Encoder Configuration:**
```yaml
# Vision Encoder Settings
vision_config:
  hidden_size: 4096  # Vision hidden size
  intermediate_size: 14336  # Vision FFN size
  num_hidden_layers: 32  # Vision transformer layers
  num_attention_heads: 32  # Vision attention heads
  num_channels: 3  # Input channels (RGB)
  patch_size: 16  # Patch size for image patches
  image_size: 256  # Input image size
  num_frames: 8  # Number of video frames
  use_3d_rope: true  # Use 3D RoPE for spatial-temporal understanding
  rope_theta: 10000  # RoPE theta parameter
```

#### Training Hyperparameters
Training hyperparameters control the learning process and optimization.

**Learning Rate Configuration:**
```yaml
# Learning Rate Settings
learning_rate: 0.00009  # Base learning rate
min_lr: 0.00005  # Minimum learning rate
num_warmup_steps: 100  # Warmup steps
num_training_steps: 64000000  # Total training steps
lr_scheduler_type: "cosine_with_min_lr"  # Scheduler type
warmup_ratio: 0.01  # Warmup ratio
```

**Optimization Configuration:**
```yaml
# Optimization Settings
optimizer_type: "adamw"  # Optimizer type
weight_decay: 0.1  # Weight decay
beta1: 0.9  # Adam beta1
beta2: 0.95  # Adam beta2
eps: 1e-8  # Adam epsilon
max_grad_norm: 1.0  # Gradient clipping norm
```

**Training Configuration:**
```yaml
# Training Settings
batch_size_per_gpu: 8  # Batch size per GPU
gradient_accumulation_steps: 32  # Gradient accumulation steps
effective_batch_size: 256  # Effective batch size (8 * 32)
num_epochs: 100  # Number of training epochs
save_steps: 1000  # Save checkpoint every N steps
eval_steps: 500  # Evaluate every N steps
logging_steps: 100  # Log every N steps
```

**Memory and Performance:**
```yaml
# Memory and Performance Settings
mixed_precision: "bf16"  # Mixed precision training
gradient_checkpointing: true  # Enable gradient checkpointing
dataloader_num_workers: 4  # DataLoader workers
dataloader_pin_memory: true  # Pin memory for DataLoader
dataloader_persistent_workers: true  # Persistent workers
```

### 2. Robot Configuration

#### DOF (Degrees of Freedom) Setup
The robot configuration defines the degrees of freedom for different robot types and configurations.

**Bimanual Mobile Robot Configuration:**
```yaml
# Bimanual Mobile Robot DOF Configuration
dof_config:
  # Left arm (7 DOF)
  follow_left_ee_cartesian_pos: 3    # Left end-effector position (x, y, z)
  follow_left_ee_rotation: 3         # Left end-effector rotation (roll, pitch, yaw)
  follow_left_gripper: 1             # Left gripper control (open/close)
  
  # Right arm (7 DOF)
  follow_right_ee_cartesian_pos: 3   # Right end-effector position (x, y, z)
  follow_right_ee_rotation: 3        # Right end-effector rotation (roll, pitch, yaw)
  follow_right_gripper: 1            # Right gripper control (open/close)
  
  # Mobile base (3 DOF)
  car_pose: 3                        # Mobile base pose (x, y, theta)
  height: 1                          # Mobile base height
  
  # Head/camera (2 DOF)
  head_actions: 2                    # Head/camera movement (pan, tilt)
  
  # Total DOF: 20 (7 + 7 + 3 + 1 + 2)
  total_dof: 20
```

**Single Arm Robot Configuration:**
```yaml
# Single Arm Robot DOF Configuration
dof_config:
  # Single arm (7 DOF)
  follow_ee_cartesian_pos: 3         # End-effector position (x, y, z)
  follow_ee_rotation: 3              # End-effector rotation (roll, pitch, yaw)
  follow_gripper: 1                  # Gripper control (open/close)
  
  # Total DOF: 7
  total_dof: 7
```

**Action Bounds Configuration:**
```yaml
# Action Bounds for Normalization
action_bounds:
  follow_left_ee_cartesian_pos:
    min: [-0.5, -0.5, 0.0]  # Minimum position (x, y, z)
    max: [0.5, 0.5, 0.8]    # Maximum position (x, y, z)
    mean: [0.0, 0.0, 0.4]   # Mean position for z-score normalization
    std: [0.2, 0.2, 0.2]    # Standard deviation for z-score normalization
  
  follow_left_ee_rotation:
    min: [-3.14, -3.14, -3.14]  # Minimum rotation (roll, pitch, yaw)
    max: [3.14, 3.14, 3.14]     # Maximum rotation (roll, pitch, yaw)
    mean: [0.0, 0.0, 0.0]       # Mean rotation
    std: [1.0, 1.0, 1.0]        # Standard deviation
  
  follow_left_gripper:
    min: [0.0]    # Minimum gripper position (closed)
    max: [1.0]    # Maximum gripper position (open)
    mean: [0.5]   # Mean gripper position
    std: [0.3]    # Standard deviation
  
  # ... similar configuration for other DOF
```

#### Action Token Configuration
Action token configuration defines which actions are used for observation and prediction.

**Action Token Settings:**
```yaml
# Action Token Configuration
obs_action_keys:  # Actions used as observation context
  - follow_left_ee_cartesian_pos
  - follow_left_ee_rotation
  - follow_left_gripper
  - follow_right_ee_cartesian_pos
  - follow_right_ee_rotation
  - follow_right_gripper
  - car_pose
  - height
  - head_actions

predict_action_keys:  # Actions to predict
  - follow_left_ee_cartesian_pos
  - follow_left_ee_rotation
  - follow_left_gripper
  - follow_right_ee_cartesian_pos
  - follow_right_ee_rotation
  - follow_right_gripper
  - car_pose
  - height
  - head_actions

# Action tokenization settings
action_tokenization:
  enabled: true  # Enable discrete action tokens
  num_tokens: 256  # Number of discrete action tokens
  token_dim: 64  # Dimension of action token embeddings
  use_continuous: true  # Use continuous actions alongside tokens
```

**Action Horizon Configuration:**
```yaml
# Action Horizon Settings
action_horizon: 32  # Number of future action steps to predict
action_sequence_length: 64  # Total action sequence length
action_padding: "last"  # Padding strategy for action sequences
action_interpolation: "linear"  # Interpolation method for action sequences
```

### 3. Advanced Configuration

#### MoE Training Strategies
Mixture of Experts training strategies for efficient and specialized learning.

**MoE Training Configuration:**
```yaml
# MoE Training Options
moe_config:
  freeze_vlm: false  # Freeze vision-language model
  action_expert_learning_rate: 0.0001  # Separate LR for action expert
  train_action_expert_only: false  # Train only action expert
  expert_dropout: 0.1  # Dropout for expert networks
  load_balancing_loss_weight: 0.01  # Load balancing loss weight
  expert_capacity_factor: 1.25  # Expert capacity factor
  router_aux_loss_coef: 0.001  # Router auxiliary loss coefficient
  
  # Expert specialization
  expert_specializations:
    0: "general_language"  # General language understanding
    1: "action_prediction"  # Action prediction and robotic control
    2: "vision_language_fusion"  # Vision-language integration
    3: "temporal_modeling"  # Temporal sequence modeling
    4: "proprioceptive_processing"  # Proprioceptive data processing
    5: "task_planning"  # High-level task planning
    6: "error_handling"  # Error detection and recovery
    7: "multimodal_reasoning"  # Complex multimodal reasoning
  
  # Routing strategies
  routing_strategy: "random"  # Token routing method
  top_k_experts: 2  # Number of top-k experts for routing
  expert_capacity: 64  # Tokens per expert
  min_expert_capacity: 4  # Minimum expert capacity
```

**Progressive Training Strategy:**
```yaml
# Progressive Training Configuration
progressive_training:
  enabled: true  # Enable progressive training
  phases:
    phase1:
      name: "vision_language_pretraining"
      duration: 0.3  # 30% of total training
      freeze_components: ["action_head"]
      learning_rate_multiplier: 1.0
    
    phase2:
      name: "action_head_training"
      duration: 0.4  # 40% of total training
      freeze_components: ["vision_encoder", "language_model"]
      learning_rate_multiplier: 2.0
    
    phase3:
      name: "end_to_end_finetuning"
      duration: 0.3  # 30% of total training
      freeze_components: []
      learning_rate_multiplier: 0.5
```

#### Flow Matching Configuration
Flow matching configuration for continuous action prediction.

**Flow Matching Settings:**
```yaml
# Flow Matching Configuration
flow_matching:
  enabled: true  # Enable flow matching
  flow_loss_weight: 1.0  # Weight for flow matching loss
  
  # Noise scheduler
  noise_scheduler:
    type: "beta"  # Noise distribution type
    beta_alpha: 1.5  # Beta distribution α parameter
    beta_beta: 1.0   # Beta distribution β parameter
    s: 0.999         # Scaling factor
    num_timesteps: 1000  # Number of noise timesteps
  
  # Flow matching parameters
  flow_matching_params:
    sigma_min: 0.01  # Minimum noise level
    sigma_max: 1.0   # Maximum noise level
    num_sampling_steps: 50  # Number of sampling steps
    sampling_method: "euler"  # Sampling method
  
  # Action prediction
  action_prediction:
    use_flow_matching: true  # Use flow matching for action prediction
    use_discrete_tokens: false  # Use discrete action tokens
    action_horizon: 32  # Action prediction horizon
    action_dim: 20  # Action dimension (total DOF)
```

**Discrete Action Token Configuration:**
```yaml
# Discrete Action Token Configuration
discrete_action_tokens:
  enabled: false  # Enable discrete action tokens
  num_tokens: 256  # Number of discrete action tokens
  token_dim: 64  # Dimension of action token embeddings
  quantization_method: "uniform"  # Quantization method
  quantization_bits: 8  # Number of quantization bits
  
  # Token learning
  token_learning:
    learning_rate: 0.001  # Learning rate for token embeddings
    update_frequency: 100  # Update frequency for token embeddings
    temperature: 1.0  # Temperature for token sampling
```

#### Loss Function Configuration
Loss function configuration for multi-task learning.

**Loss Function Settings:**
```yaml
# Loss Function Configuration
loss_config:
  # Language modeling loss
  language_loss:
    weight: 1.0  # Weight for language loss
    type: "cross_entropy"  # Loss type
    label_smoothing: 0.1  # Label smoothing factor
    ignore_index: -100  # Index to ignore in loss computation
  
  # Action prediction loss
  action_loss:
    weight: 1.0  # Weight for action loss
    type: "mse"  # Loss type (MSE for flow matching)
    reduction: "mean"  # Loss reduction method
    dof_weights:  # Per-DOF loss weights
      follow_left_ee_cartesian_pos: 1.0
      follow_left_ee_rotation: 1.0
      follow_left_gripper: 0.5
      # ... other DOF weights
  
  # Flow matching loss
  flow_matching_loss:
    weight: 1.0  # Weight for flow matching loss
    type: "flow_matching"  # Loss type
    noise_scheduler: "beta"  # Noise scheduler type
  
  # MoE load balancing loss
  load_balancing_loss:
    weight: 0.01  # Weight for load balancing loss
    type: "auxiliary"  # Loss type
    auxiliary_loss_coef: 0.001  # Auxiliary loss coefficient
  
  # Total loss
  total_loss:
    type: "weighted_sum"  # Total loss computation method
    normalize: true  # Normalize loss by number of tasks
```

#### Data Configuration
Data configuration for training and validation.

**Data Configuration:**
```yaml
# Data Configuration
data_config:
  # Dataset settings
  dataset_name: "lerobot/aloha_mobile_cabinet"
  train_split: 0.95  # Training split ratio
  val_split: 0.05  # Validation split ratio
  
  # Data loading
  batch_size: 8  # Batch size per GPU
  num_workers: 4  # Number of data loading workers
  pin_memory: true  # Pin memory for data loading
  persistent_workers: true  # Persistent workers
  
  # Data augmentation
  augmentation:
    vision:
      color_jitter: 0.1  # Color jittering strength
      spatial_rotation: 5  # Maximum rotation in degrees
      spatial_translation: 10  # Maximum translation in pixels
      temporal_shift: 0.1  # Temporal shifting probability
    
    action:
      noise_injection: 0.01  # Action noise injection
      temporal_shift: 0.1  # Action temporal shifting
      dof_masking: 0.1  # DOF masking probability
  
  # Data preprocessing
  preprocessing:
    image_size: 256  # Input image size
    action_horizon: 32  # Action prediction horizon
    normalize_actions: true  # Normalize actions to [-1, 1]
    action_normalization_type: "min_max"  # Action normalization type
```

## Model Training Process

### 1. Environment Setup

#### Prerequisites
```bash
# System Requirements
CUDA: 12.0+
Python: 3.10
PyTorch: 2.0+
Flash Attention: 2.7.4.post1

# Install Dependencies
conda create --name wallx python=3.10
conda activate wallx
pip install -r requirements.txt
MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation
```

#### LeRobot Installation
```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
pip install -e .
```

#### WALL-X Installation
```bash
git submodule update --init --recursive
MAX_JOBS=4 pip install --no-build-isolation --verbose .
```

### 2. Data Preparation

#### Download Dataset
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load LeRobot dataset
dataset = LeRobotDataset(
    repo_id="lerobot/aloha_mobile_cabinet",
    delta_timestamps={"action": [t/50 for t in range(31)]},
    video_backend="pyav"
)
```

#### Preprocess Data
```python
from wall_x.data.load_lerobot_dataset import PreprocessedDataset

# Create preprocessed dataset
preprocessed_dataset = PreprocessedDataset(
    dataset=dataset,
    config=config,
    dataload_config=dataload_config,
    seed=42,
    rank=0,
    world_size=1
)
```

### 3. Training Loop

#### Initialize Trainer
```python
from wall_x.trainer.qwen_vl_act_trainer import QwenVlAct_Trainer

trainer = QwenVlAct_Trainer(
    config=config,
    logger=wandb_logger,
    accelerator=accelerator,
    seed=42,
    data_config_path="config_qact.yml"
)
```

#### Training Process
```python
# Main training loop
for epoch in range(num_epochs):
    # Training phase
    trainer.train_loop(epoch)
    
    # Validation phase
    trainer.val_loop()
    
    # Save checkpoint
    trainer.save_checkpoint(epoch)
```

### 4. Loss Functions

#### Cross-Entropy Loss
```python
# Language modeling loss
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
ce_loss = CrossEntropyLoss()(shift_logits.view(-1, vocab_size), 
                           shift_labels.view(-1))
```

#### Flow Matching Loss
```python
# Action prediction loss
def flow_loss(action_hidden_states, flow_target, dof_mask):
    action_pred = action_proj_back(action_hidden_states)
    loss = MSELoss(reduction='none')(action_pred, flow_target)
    if dof_mask is not None:
        loss = loss * dof_mask
    return loss
```

#### Combined Loss
```python
total_loss = ce_loss + flow_loss_weight * flow_loss
```

### 5. Optimization

#### AdamW Optimizer
```python
optimizer = AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.1
)
```

#### Learning Rate Scheduler
```python
scheduler = get_cosine_with_min_lr_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
    min_lr=min_lr
)
```

#### Gradient Clipping
```python
total_norm = accelerator.clip_grad_norm_(
    model.parameters(), 
    max_grad_norm=1.0
)
```

## Advanced Features

### 1. Distributed Training

#### Multi-GPU Setup
```bash
# Single node, multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --num_processes=4 train_qact.py --config config_qact.yml
```

#### Multi-Node Training
```bash
# Multi-node setup
accelerate launch \
    --num_processes=8 \
    --num_machines=2 \
    --machine_rank=0 \
    --main_process_port=29500 \
    train_qact.py --config config_qact.yml
```

### 2. Mixed Precision Training

#### BF16 Configuration
```python
accelerator = Accelerator(
    mixed_precision="bf16",
    gradient_accumulation_steps=32
)
```

#### Memory Optimization
```python
# Gradient checkpointing
model.gradient_checkpointing_enable()

# CPU offloading
accelerator = Accelerator(
    device_placement=True,
    cpu_offload=True
)
```

### 3. Checkpoint Management

#### Save Checkpoints
```python
def save_checkpoint(epoch, step=0):
    save_path = f"{config['save_path']}/{epoch}_{step}"
    accelerator.save_state(save_path)
```

#### Resume Training
```python
def resume_from_checkpoint(checkpoint_path):
    accelerator.load_state(checkpoint_path)
    # Resume optimizer and scheduler states
```

### 4. Monitoring and Logging

#### Weights & Biases Integration
```python
import wandb

wandb.init(
    project="wall-x-training",
    name="experiment_name",
    config=config
)

# Log metrics
wandb.log({
    "train_loss": train_loss,
    "val_loss": val_loss,
    "learning_rate": lr,
    "gradient_norm": grad_norm
})
```

#### Performance Profiling
```python
# PyTorch profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, 
                torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=10, warmup=5, active=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./profiler")
) as prof:
    # Training step
    prof.step()
```

## Troubleshooting

### 1. Common Issues

#### CUDA Out of Memory
```python
# Solutions:
# 1. Reduce batch size
batch_size_per_gpu = 4  # Instead of 8

# 2. Increase gradient accumulation
gradient_accumulation_steps = 64  # Instead of 32

# 3. Enable gradient checkpointing
model.gradient_checkpointing_enable()

# 4. Use CPU offloading
accelerator = Accelerator(cpu_offload=True)
```

#### NaN Loss
```python
# Debug NaN loss
if torch.isnan(loss):
    print("NaN detected in loss")
    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")
```

#### Slow Training
```python
# Optimize data loading
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,  # Increase workers
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)
```

### 2. Debugging Tools

#### Memory Monitoring
```python
# Monitor GPU memory
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

#### Gradient Monitoring
```python
# Check gradient norms
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** (1. / 2)
print(f"Gradient norm: {total_norm}")
```

## Best Practices

### 1. Training Strategy

#### Progressive Training
1. **Phase 1**: Train vision-language components
2. **Phase 2**: Add action prediction head
3. **Phase 3**: Fine-tune entire model

#### Learning Rate Scheduling
```python
# Warmup + Cosine Annealing
warmup_steps = 1000
total_steps = 1000000
min_lr = 0.1 * max_lr

scheduler = get_cosine_with_min_lr_schedule_with_warmup(
    optimizer, warmup_steps, total_steps, min_lr
)
```

### 2. Data Quality

#### Data Validation
```python
def validate_batch(batch):
    """Validate batch data quality"""
    # Check for NaN values
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any():
                print(f"NaN detected in {key}")
                return False
    return True
```

#### Data Augmentation
```python
# Balanced augmentation
augmentation_prob = 0.5
if random.random() < augmentation_prob:
    # Apply augmentation
    batch = apply_augmentation(batch)
```

### 3. Model Architecture

#### Expert Routing
```python
# Monitor expert usage
expert_usage = torch.zeros(num_experts)
for expert_idx in expert_indices:
    expert_usage[expert_idx] += 1

# Log expert utilization
print(f"Expert usage: {expert_usage}")
```

#### Attention Patterns
```python
# Visualize attention weights
attention_weights = model.get_attention_weights()
# Plot attention heatmaps
plot_attention_heatmap(attention_weights)
```

### 4. Performance Optimization

#### Memory Efficiency
```python
# Use mixed precision
model = model.to(torch.bfloat16)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use Flash Attention
config._attn_implementation = "flash_attention_2"
```

#### Training Speed
```python
# Optimize data loading
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

# Use compiled model (PyTorch 2.0+)
model = torch.compile(model)
```

### 5. Evaluation

#### Action Prediction Metrics
```python
def evaluate_actions(predictions, targets, dof_mask):
    """Evaluate action prediction accuracy"""
    # L1 loss per DOF
    l1_loss = F.l1_loss(predictions, targets, reduction='none')
    l1_loss = l1_loss * dof_mask
    return l1_loss.mean()
```

#### Language Understanding Metrics
```python
def evaluate_language(predictions, targets):
    """Evaluate language understanding"""
    # Perplexity
    perplexity = torch.exp(F.cross_entropy(predictions, targets))
    return perplexity
```

## Conclusion

This comprehensive training guide covers all aspects of training the WALL-X VLA model, from data preprocessing to advanced optimization techniques. The model's unique combination of vision, language, and action processing makes it a powerful foundation for robotic applications.

For additional support and updates, refer to the official WALL-X repository and documentation.

---

**Note**: This guide is based on the WALL-X codebase and may need adjustments for specific use cases or hardware configurations. Always validate configurations on your specific setup before running large-scale training experiments.
