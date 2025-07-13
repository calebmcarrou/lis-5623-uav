# Start with PyTorch image
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# System deps & setups
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    unzip \
    && apt-get clean

WORKDIR /UAV

COPY . /UAV

RUN export POLARS_SKIP_CPU_CHECK=1

# Set up Python env
RUN pip install --upgrade pip \
    && pip install -r requirements.txt