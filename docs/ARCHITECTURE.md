# WALL-X VLA Model Architecture Diagram

## Complete Model Architecture with Input/Output Shapes

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                    WALL-X VLA MODEL ARCHITECTURE                                │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

INPUT MODALITIES:
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   VISION DATA   │  │  LANGUAGE DATA  │  │  ACTION DATA    │  │ PROPRIOCEPTION  │  │   METADATA      │
│                 │  │                 │  │                 │  │                 │  │                 │
│ pixel_values    │  │ input_ids       │  │ action_chunk    │  │ proprioception  │  │ dataset_names   │
│ [B, N, C, H, W] │  │ [B, S]          │  │ [B, T, A]       │  │ [B, T, P]       │  │ [B]             │
│                 │  │                 │  │                 │  │                 │  │                 │
│ image_grid_thw  │  │ attention_mask  │  │ dof_mask        │  │ agent_pos_mask  │  │ moe_token_types │
│ [B, N, 3]       │  │ [B, S]          │  │ [B, T, A]       │  │ [B, T, P]       │  │ [B, S]          │
│                 │  │                 │  │                 │  │                 │  │                 │
│ video_grid_thw  │  │ position_ids    │  │                 │  │                 │  │                 │
│ [B, M, 3]       │  │ [3, B, S]       │  │                 │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
         │                     │                     │                     │                     │
         ▼                     ▼                     ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           INPUT PROCESSING LAYER                                                │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           VISION ENCODER                                                       │
│                                                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              Qwen2.5 Vision Transformer                                                 │   │
│  │                                                                                                         │   │
│  │  pixel_values [B, N, C, H, W] ──► Patch Embedding ──► [B, N, P, D_v]                                    │   │
│  │  image_grid_thw [B, N, 3] ──────► 3D RoPE ──────────► Position IDs [3, B, N*P]                          │   │
│  │                                                                                                         │   │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐    │   │
│  │  │                    Vision Transformer Blocks (L_v layers)                                       │    │   │
│  │  │                                                                                                 │    │   │
│  │  │  [B, N*P, D_v] ──► Multi-Head Attention ──► [B, N*P, D_v]                                       │    │   │
│  │  │       │                    │                       │                                            │    │   │
│  │  │       ▼                    ▼                       ▼                                            │    │   │
│  │  │  Layer Norm ──► MLP ──► Layer Norm ──► Residual ──► [B, N*P, D_v]                               │    │   │
│  │  │                                                                                                 │    │   │
│  │  │  Repeat for L_v layers                                                                          │    │   │
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────┘    │   │
│  │                                                                                                         │   │
│  │  Output: image_embeds [B, N*P, D_v]                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        EMBEDDING FUSION LAYER                                                  │
│                                                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                              Token Embedding Layer                                                      │   │
│  │                                                                                                         │   │
│  │  input_ids [B, S] ──► Embedding Lookup ──► [B, S, D]                                                    │   │
│  │                                                                                                         │   │
│  │  Special Token Processing:                                                                              │   │
│  │  • <|propri|> tokens ──► proprioception_proj ──► [B, S, D]                                              │   │
│  │  • <|action|> tokens ──► action_preprocessor ──► [B, S, D]                                              │   │
│  │  • <|image|> tokens ──► image_embeds (from vision encoder) ──► [B, S, D]                                │   │
│  │  • <|video|> tokens ──► video_embeds (from vision encoder) ──► [B, S, D]                                │   │
│  │                                                                                                         │   │
│  │  Output: inputs_embeds [B, S, D]                                                                        │   │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                        MAIN TRANSFORMER MODEL                                                   │
│                                                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                              Qwen2.5-VL MoE Transformer                                                 │    │
│  │                                                                                                         │    │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐    │    │
│  │  │                        Decoder Layer 1                                                          │    │    │
│  │  │                                                                                                 │    │    │
│  │  │  inputs_embeds [B, S, D] ──► Layer Norm ──► Multi-Head Attention ──► [B, S, D]                  │    │    │
│  │  │       │                           │                    │                       │                │    │    │
│  │  │       ▼                           ▼                    ▼                       ▼                │    │    │
│  │  │  Residual ──► Layer Norm ──► MoE Block ──► Residual ──► [B, S, D]                               │    │    │
│  │  │                                                                                                 │    │    │
│  │  │  ┌─────────────────────────────────────────────────────────────────────────────────────────┐    │    │    │
│  │  │  │                        MoE Block (Mixture of Experts)                                   │    │    │    │
│  │  │  │                                                                                         │    │    │    │
│  │  │  │  [B, S, D] ──► Token Routing ──► Expert Selection ──► [B, S, D]                         │    │    │    │
│  │  │  │       │              │                    │                       │                     │    │    │    │
│  │  │  │       ▼              ▼                    ▼                       ▼                     │    │    │    │
│  │  │  │  Expert 0: General Language ──► [B, S, D]                                               │    │    │    │
│  │  │  │  Expert 1: Action Prediction ──► [B, S, D]                                              │    │    │    │
│  │  │  │  Expert 2-7: Specialized Tasks ──► [B, S, D]                                            │    │    │    │
│  │  │  │                                                                                         │    │    │    │
│  │  │  │  Output: [B, S, D] (weighted combination)                                               │    │    │    │
│  │  │  └─────────────────────────────────────────────────────────────────────────────────────────┘    │    │    │
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                                                         │    │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐    │    │
│  │  │                        Decoder Layer 2 to L (L=32 layers)                                       │    │    │
│  │  │                                                                                                 │    │    │
│  │  │  [B, S, D] ──► Layer Norm ──► Multi-Head Attention ──► [B, S, D]                                │    │    │
│  │  │       │              │                    │                       │                             │    │    │
│  │  │       ▼              ▼                    ▼                       ▼                             │    │    │
│  │  │  Residual ──► Layer Norm ──► MoE Block ──► Residual ──► [B, S, D]                               │    │    │
│  │  │                                                                                                 │    │    │
│  │  │  Repeat for L-1 layers                                                                          │    │    │
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                                                         │    │
│  │  ┌─────────────────────────────────────────────────────────────────────────────────────────────────┐    │    │
│  │  │                        Final Layer Normalization                                                │    │    │
│  │  │                                                                                                 │    │    │
│  │  │  [B, S, D] ──► RMSNorm ──► [B, S, D]                                                            │    │    │
│  │  └─────────────────────────────────────────────────────────────────────────────────────────────────┘    │    │
│  │                                                                                                         │    │
│  │  Output: hidden_states [B, S, D]                                                                        │    │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           OUTPUT HEADS                                                          │
│                                                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                              Language Modeling Head                                                     │    │
│  │                                                                                                         │    │
│  │  hidden_states [B, S, D] ──► Linear(D, V) ──► logits [B, S, V]                                          │    │
│  │                                                                                                         │    │
│  │  V = vocab_size (151,936 tokens)                                                                        │    │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                              Action Prediction Head                                                     │    │
│  │                                                                                                         │    │
│  │  action_hidden_states [B, S, D] ──► action_proj_back ──► action_pred [B, S, A]                          │    │
│  │  (extracted from <|action|> tokens)                                                                     │    │
│  │                                                                                                         │    │
│  │  A = action_dim (sum of all DOF dimensions)                                                             │    │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           LOSS COMPUTATION                                                      │
│                                                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                              Cross-Entropy Loss (Language)                                              │    │
│  │                                                                                                         │    │
│  │  logits [B, S, V] ──► Shift ──► shift_logits [B, S-1, V]                                                │    │
│  │  labels [B, S] ──────► Shift ──► shift_labels [B, S-1]                                                  │    │
│  │                                                                                                         │    │
│  │  CrossEntropyLoss(shift_logits, shift_labels) ──► ce_loss [scalar]                                      │    │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                              Flow Matching Loss (Actions)                                               │    │
│  │                                                                                                         │    │
│  │  action_pred [B, S, A] ──► MSE Loss ──► flow_loss [B, S, A]                                             │    │
│  │  flow_target [B, S, A] ──► (action - noise)                                                             │    │
│  │                                                                                                         │    │
│  │  MSE Loss with DOF masking ──► flow_loss [scalar]                                                       │    │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                              Total Loss                                                                 │    │
│  │                                                                                                         │    │
│  │  total_loss = ce_loss + flow_loss_weight * flow_loss                                                    │    │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           MODEL OUTPUTS                                                         │
│                                                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                              Training Outputs                                                           │    │
│  │                                                                                                         │    │
│  │  • loss: torch.FloatTensor [scalar] - Total training loss                                               │    │
│  │  • cross_entropy_loss: torch.FloatTensor [scalar] - Language modeling loss                              │    │
│  │  • flow_loss: torch.FloatTensor [scalar] - Action prediction loss                                       │    │
│  │  • logits: torch.FloatTensor [B, S, V] - Language model predictions                                     │    │
│  │  • hidden_states: Tuple[torch.FloatTensor] - Hidden states from all layers                              │    │
│  │  • attentions: Tuple[torch.FloatTensor] - Attention weights from all layers                             │    │
│  │  • channel_loss_dict: Dict[str, torch.FloatTensor] - Per-dataset losses                                 │    │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                              Inference Outputs                                                          │    │
│  │                                                                                                         │    │
│  │  Text Mode:                                                                                             │    │
│  │  • predict_output_text: List[str] - Generated text responses                                            │    │
│  │  • input_text: List[str] - Input text prompts                                                           │    │
│  │                                                                                                         │    │
│  │  Fast Action Mode:                                                                                      │    │
│  │  • predict_action: torch.Tensor [B, T, A] - Predicted action sequences                                  │    │
│  │  • gt_action: torch.Tensor [B, T, A] - Ground truth actions (if available)                              │    │
│  │                                                                                                         │    │
│  │  Diffusion Action Mode:                                                                                 │    │
│  │  • predict_action: torch.Tensor [B, T, A] - Predicted action sequences                                  │    │
│  │  • gt_action: torch.Tensor [B, T, A] - Ground truth actions (if available)                              │    │
│  └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           DETAILED SHAPE DEFINITIONS                                            │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

B = batch_size (e.g., 8)
S = sequence_length (e.g., 768)
T = action_horizon (e.g., 32)
A = action_dim (sum of DOF dimensions, e.g., 20)
P = proprioception_dim (e.g., 20)
N = number_of_images (variable)
M = number_of_videos (variable)
C = image_channels (e.g., 3)
H = image_height (e.g., 256)
W = image_width (e.g., 256)
D = hidden_size (e.g., 4096)
D_v = vision_hidden_size (e.g., 1024)
V = vocab_size (e.g., 151,936)
L = num_hidden_layers (e.g., 32)
L_v = vision_layers (e.g., 24)

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           KEY COMPONENTS EXPLAINED                                              │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

1. VISION ENCODER:
   - Processes multi-camera images and videos
   - Uses 3D RoPE for spatial-temporal understanding
   - Outputs vision embeddings that replace <|image|> and <|video|> tokens

2. EMBEDDING FUSION:
   - Combines text, vision, action, and proprioception embeddings
   - Special tokens: <|propri|>, <|action|>, <|image|>, <|video|>
   - All modalities mapped to same hidden dimension D

3. MOE TRANSFORMER:
   - 32 decoder layers with Mixture of Experts
   - 8 experts per MoE layer
   - Expert 0: General language understanding
   - Expert 1: Action prediction and robotic control
   - Experts 2-7: Specialized multimodal processing

4. ATTENTION MECHANISMS:
   - Multi-head attention with 32 heads
   - Head dimension: 128
   - Flash Attention 2 for efficiency
   - 3D RoPE for vision tokens (temporal, height, width)

5. ACTION PROCESSING:
   - Flow matching for continuous action prediction
   - Beta distribution noise scheduling
   - DOF masking for multi-robot support
   - Proprioception integration

6. LOSS FUNCTIONS:
   - Cross-entropy loss for language modeling
   - Flow matching loss for action prediction
   - Per-dataset channel losses for monitoring

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           DATA FLOW SUMMARY                                                     │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

Input → Vision Encoder → Embedding Fusion → MoE Transformer → Output Heads → Loss Computation → Model Outputs

Multi-modal inputs are processed in parallel and fused into a unified representation that flows through the transformer
for joint understanding and prediction of both language and robotic actions.
