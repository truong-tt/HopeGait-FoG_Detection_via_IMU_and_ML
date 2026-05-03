# Cloud-GPU image for HopeGait training.
#
# Build:
#   docker build -t hopegait:latest .
#
# Run on a CUDA host (RunPod / Lambda / Vast / your own GPU box):
#   docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models \
#     hopegait:latest python src/main.py
#
# CPU-only smoke test on any host:
#   docker build -t hopegait:cpu --build-arg BASE=python:3.11-slim .
#   docker run --rm hopegait:cpu python scripts/smoke_train.py

ARG BASE=pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
FROM ${BASE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install OS deps that scipy / pandas wheels sometimes need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install all training deps. On the CUDA base image torch is already installed;
# pip will see the existing torch>=2.3 wheel and skip the reinstall, so we do
# not need --no-deps gymnastics (which silently skips transitive deps and can
# leave the image broken if the base bumps).
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/src

CMD ["python", "src/main.py"]
