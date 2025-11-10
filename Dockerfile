FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Set working directory
WORKDIR /workspace

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy training script
COPY gams_instruct_lora_train.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

# Default command shows help
CMD ["python3", "gams_instruct_lora_train.py", "--help"]
