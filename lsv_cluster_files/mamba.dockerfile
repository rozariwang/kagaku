# For LSV A100s server
FROM nvcr.io/nvidia/pytorch:22.02-py3

# Set path to CUDA
ENV CUDA_HOME=/usr/local/cuda

# Install additional programs
# Install additional system utilities and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    htop \
    gnupg \
    curl \
    ca-certificates \
    vim \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Explicitly install Python packages and check CUDA
RUN pip install --upgrade pip && \
    pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install \
    accelerate \
    wandb \
    optuna \
    pandas \
    scikit-learn \
    transformers \
    plotly \
    matplotlib \
    rdkit-pypi \
    datasets \
    ninja && \
    pip install mamba-ssm[causal-conv1d]

# Create and run a Python script to check CUDA availability
RUN printf "import torch\nprint('CUDA Available:', torch.cuda.is_available())\nif torch.cuda.is_available():\n    print('CUDA Device Count:', torch.cuda.device_count())\n    print('CUDA Device Name:', torch.cuda.get_device_name(0))\n" > /check_cuda.py
RUN python /check_cuda.py

# Specify a new user (USER_NAME and USER_UID are specified via --build-arg)
ARG USER_UID
ARG USER_NAME
ENV USER_GID=$USER_UID
ENV USER_GROUP="users"

# Create the user
RUN mkdir /home/$USER_NAME && \
    useradd -l -d /home/$USER_NAME -u $USER_UID -g $USER_GROUP $USER_NAME && \
    mkdir /home/$USER_NAME/.local && \
    chown -R ${USER_UID}:${USER_GID} /home/$USER_NAME/

CMD ["/bin/bash"]