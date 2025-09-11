# WALL-X Dockerfile for Slurm Environment with 8 A100 GPUs
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.4 support
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install packaging and other build dependencies first
RUN pip install packaging wheel setuptools

# Install Flash Attention for CUDA 12.4
RUN MAX_JOBS=4 pip install flash-attn==2.7.4.post1 --no-build-isolation

# Install other requirements
RUN pip install \
    transformers==4.49.0 \
    accelerate==1.10.1 \
    peft==0.17.1 \
    scipy==1.15.3 \
    torchdiffeq==0.2.5 \
    qwen_vl_utils==0.0.11

# Install LeRobot
RUN git clone https://github.com/huggingface/lerobot.git /opt/lerobot && \
    cd /opt/lerobot && \
    pip install -e .

# Set working directory
WORKDIR /workspace

# Copy project files
COPY . /workspace/

# Install WALL-X
RUN git submodule update --init --recursive && \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" MAX_JOBS=4 pip install --no-build-isolation --verbose .

# Set default command
CMD ["/bin/bash"]
