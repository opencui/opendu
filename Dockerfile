FROM python:3.11.7-bullseye
WORKDIR /data
COPY . .
RUN pip install -r requirements.txt && python -m opencui && python -m opencui.inference.cache_model && rm -rf *
