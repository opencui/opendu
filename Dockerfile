FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel AS model-downloader

# Download models
RUN pip install --no-cache-dir huggingface_hub

ENV HF_HOME=/models
RUN mkdir -p /models

RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Qwen/Qwen3-4B', local_dir='/models/Qwen3-4B', local_dir_use_symlinks=False); \
snapshot_download('Qwen/Qwen3-Embedding-0.6B', local_dir='/models/Qwen3-Embedding-0.6B', local_dir_use_symlinks=False) \
"

# Production stage  
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Copy models
COPY --from=model-downloader /models /data/models

# Set environment
ENV VLLM_ATTENTION_BACKEND=XFORMERS
ENV HF_HOME=/data/models
ENV HF_HUB_CACHE=/data/models
ENV TRANSFORMERS_CACHE=/data/models
ENV HF_HUB_OFFLINE=1
ENV VLLM_DISABLE_TELEMETRY=1
ENV VLLM_USE_TRITON_FLASH_ATTN=0
ENV VLLM_DISABLE_FLASH_ATTN=1
ENV VLLM_USE_TRITON_ATTENTION=0
ENV TRITON_DISABLE_LINE_INFO=1
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTHONPATH=/data
ENV VLLM_USE_TRITON_AWQ=0
ENV VLLM_USE_TRITON_MARLIN=0
ENV VLLM_DISABLE_TRITON_AUTOTUNE=1
ENV TRITON_CACHE_DISABLED=1
ENV VLLM_DISABLE_TRITON_KERNEL=1
ENV VLLM_USE_TRITON_KERNEL=0
ENV VLLM_DISABLE_FLASH_ATTN=1


# Fix MKL threading conflict
ENV MKL_THREADING_LAYER=GNU
ENV MKL_SERVICE_FORCE_INTEL=1

# Force specific compute capability if needed
ENV CUDA_ARCH_LIST="7.5"
ENV TORCH_CUDA_ARCH_LIST="7.5"

WORKDIR /data

# Install dependencies layer (rarely changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code layer (changes frequently)
COPY . .

# Verify setup
RUN python -c "\
import torch; \
import os; \
print(f'CUDA available: {torch.cuda.is_available()}'); \
print(f'CUDA devices: {torch.cuda.device_count()}'); \
print(f'Device name: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No CUDA'); \
print(f'Compute capability: {torch.cuda.get_device_capability(0)}' if torch.cuda.is_available() else 'No CUDA'); \
print('Models directory contents:', os.listdir('/data/models')) \
"