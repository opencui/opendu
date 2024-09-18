FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04
RUN  apt update && apt install -y wget build-essential && apt clean && \
     mkdir -p ~/miniconda3 && \
     wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
     bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
     rm -rf ~/miniconda3/miniconda.sh
ENV  PATH=/root/miniconda3/bin:$PATH

WORKDIR /data
COPY . .
RUN sed -i 's/cuda:0/cpu/g' opencui/core/config.py && \
    pip install -r requirements.txt && python -m opencui && python -m opencui.inference.cache_model && \
	sed -i 's/cpu/cuda:0/g' opencui/core/config.py
