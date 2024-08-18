# Base image must at least have pytorch and CUDA installed.
# We are using NVIDIA NGC's PyTorch image here, see: https://ngc.nvidia.com/catalog/containers/nvidia:pytorch for latest version
# See https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html#framework-matrix-2021 for installed python, pytorch, etc. versions

# for LSV A100s server
FROM nvcr.io/nvidia/pytorch:22.02-py3

# for LSV V100 server
# FROM nvcr.io/nvidia/pytorch:21.07-py3

# Set path to CUDA
ENV CUDA_HOME=/usr/local/cuda

# Install additional programs
RUN apt-get update && apt-get install -y \
    build-essential \
    htop \
    gnupg \
    curl \
    ca-certificates \
    vim \
    tmux \
    git \
    ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Update pip
RUN python3 -m pip install --upgrade pip

# Environment settings
ENV LANG C.UTF-8

# Install core dependencies
RUN python3 -m pip install \
    accelerate \
    wandb \
    optuna \
    torch \
    pandas \
    scikit-learn \
    transformers \
    plotly \
    matplotlib \
    rdkit-pypi \
    datasets \
    ninja  

# System updates and Python 3.10 installation
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get install -y python3.10 python3.10-venv python3.10-dev \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && python3.10 -m pip install --upgrade pip setuptools wheel

# Download and install the specific wheel
ADD https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl /tmp
RUN python3.10 -m pip install /tmp/mamba_ssm-2.2.2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Ensure CUDA versions match between nvcc and PyTorch
RUN echo "PyTorch CUDA version:" && python3 -c "import torch; print(torch.version.cuda)" \
    && echo "nvcc CUDA version:" && nvcc --version
    
# Clone the specific version of causal_conv1d that is compatible
RUN git clone https://github.com/Dao-AILab/causal-conv1d.git /app/causal-conv1d \
    && cd /app/causal-conv1d \
    && git checkout v1.0.2 \
    && CAUSAL_CONV1D_FORCE_BUILD=TRUE python3 -m pip install .
    
# Clone and setup the Mamba repository
RUN git clone https://github.com/state-spaces/mamba.git /app/mamba
WORKDIR /app/mamba
ENV MAMBA_FORCE_BUILD=TRUE
RUN python3 -m pip install .

# Set main application directory
WORKDIR /app

# Specify a new user (USER_NAME and USER_UID are specified via --build-arg)
ARG USER_UID
ARG USER_NAME
ENV USER_GID=$USER_UID
ENV USER_GROUP="users"

# Create the user
RUN mkdir /home/$USER_NAME
RUN useradd -l -d /home/$USER_NAME -u $USER_UID -g $USER_GROUP $USER_NAME
# this will fix a wandb issue
RUN mkdir /home/$USER_NAME/.local

# Change owner of home dir (Note: this is not the lsv nethome)
RUN chown -R ${USER_UID}:${USER_GID} /home/$USER_NAME/

CMD ["/bin/bash"]