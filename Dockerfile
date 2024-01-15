FROM python:3.11.7-bullseye
RUN set -x && git clone https://github.com/opencui/dug.git && \
    (cd dug && pip install -r requirements.txt && ls -lh opencui && touch opencui/__main__.py && ls -lh opencui && python -m opencui) && \
    rm -rf dug
