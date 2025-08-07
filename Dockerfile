# Stage 1: Download models using Python 3.12
FROM python:3.12 as model-downloader
WORKDIR /tmp

# Install just enough to download models
RUN pip install huggingface_hub torch==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu123

# Download Qwen3 models
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen3-4B'); \
    snapshot_download('Qwen/Qwen3-Embedding-0.6B')"

# Stage 2: Final runtime image with CUDA 12.3 + Miniconda
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04

# Install system tools and Miniconda
RUN apt update && apt install -y wget build-essential git && apt clean && \
    mkdir -p ~/miniconda3 && \
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-py312_24.7.1-0-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm -rf ~/miniconda3/miniconda.sh

ENV PATH=/root/miniconda3/bin:$PATH

# Install pip-based tools first
RUN pip install --upgrade pip setuptools wheel

# Preinstall torch + torchvision + torchaudio
RUN pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Copy models from the model-downloader stage
COPY --from=model-downloader /root/.cache/huggingface /root/.cache/huggingface

# Set up project files
WORKDIR /data
COPY . .

# Install Python dependencies
RUN pip install -r requirements.txt

CMD ["bash"]


# RUN sed -i 's/cuda:0/cpu/g' opendu/core/config.py && \
#     pip install -r requirements.txt && python -m opendu && python -m opendu.inference.cache_model && \
#     sed -i 's/cpu/cuda:0/g' opendu/core/config.py
