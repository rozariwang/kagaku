# For LSV A100s server
FROM nvcr.io/nvidia/pytorch:22.02-py3

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
    && rm -rf /var/lib/apt/lists/*

# Explicitly install Python packages and check CUDA

RUN python3 -m pip install \
    torch 
    #\
#    accelerate \
#    wandb \
#    optuna \
#    pandas \
#    scikit-learn \
#    transformers \
#    plotly \
#    matplotlib \
#    rdkit-pypi \
#    datasets \
#    ninja && \
#    pip install mamba-ssm[causal-conv1d]

#check CUDA availability
RUN python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); if torch.cuda.is_available(): print('CUDA Device Count:', torch.cuda.device_count()); print('CUDA Device Name:', torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'No CUDA Devices Found'"
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