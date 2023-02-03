FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install \
    -y \
    --no-install-recommends \
    apt-utils \
    binutils \
    build-essential \
    ca-certificates \
    curl \
    gcc \
    git \
    htop \
    less \
    libaio-dev \
    libxext6 \
    libx11-6 \
    libglib2.0-0 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    locales \
    nano \
    ninja-build \
    python3-dev \
    python3-setuptools \
    python3-venv \
    python3-pip \
    screen \
    ssh \
    sudo \
    tmux \
    unzip \
    vim \
    wget

RUN python3 -m \
    pip install \
    --no-cache-dir \
    --upgrade \
    pip

RUN python3 -m \
    pip install \
    --no-cache-dir \
    torch>=1.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# pip installs for which we want to specify package versions
RUN python3 -m \
    pip install \
    --no-cache-dir \
    captum>=0.5 \
    nilearn>=0.9.0 \
    ray[tune]>=1.11.0 \
    zennit>=0.4.5

# pip install where we do not care about specific packages versions
RUN python3 -m \
    pip install \
    --no-cache-dir \
    wandb \
    seaborn \
    gitdir

CMD bash