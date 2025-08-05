FROM python:3.12 as model-downloader
WORKDIR /tmp
RUN pip install huggingface_hub torch

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B');snapshot_download('Qwen/Qwen3-Embedding-0.6B')"


FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04
RUN  apt update && apt install -y wget build-essential && apt clean && \
    mkdir -p ~/miniconda3 && \
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-py312_24.7.1-0-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm -rf ~/miniconda3/miniconda.sh
ENV  PATH=/root/miniconda3/bin:$PATH

COPY --from=model-downloader /root/.cache/huggingface /root/.cache/huggingface
WORKDIR /data
COPY . .
RUN pip install -r requirements.txt

# RUN sed -i 's/cuda:0/cpu/g' opendu/core/config.py && \
#     pip install -r requirements.txt && python -m opendu && python -m opendu.inference.cache_model && \
#     sed -i 's/cpu/cuda:0/g' opendu/core/config.py
