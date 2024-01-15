FROM python:3.11.7-bullseye
RUN set -x && git clone https://github.com/opencui/dug.git && \
    (cd dug && pip install -r requirements.txt && python -m opencui) && \
    rm -rf dug
