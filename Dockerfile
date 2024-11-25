FROM nvcr.io/nvidia/pytorch:24.07-py3 AS base
ARG CUDA_VERSION=12.4.1
ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements.txt
