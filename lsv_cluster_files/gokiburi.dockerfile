# Base image must at least have pytorch and CUDA installed.
# We are using NVIDIA NGC's PyTorch image here, see: https://ngc.nvidia.com/catalog/containers/nvidia:pytorch for latest version
# See https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html#framework-matrix-2021 for installed python, pytorch, etc. versions

# for LSV A100s server
FROM nvcr.io/nvidia/pytorch:24.01-py3

# for LSV V100 server
#FROM nvcr.io/nvidia/pytorch:21.07-py3

# SSH tunnel
#EXPOSE 8888

# Set path to CUDA
ENV CUDA_HOME=/usr/local/cuda

# Install additional programs
# Removing Post-Invoke scripts
RUN echo '' > /etc/apt/apt.conf.d/90invoke

# Cleaning up APT
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/*.bin

# Updating APT
RUN apt-get update -o APT::Update::Pre-Invoke::= -o APT::Update::Post-Invoke::=

# Installing packages
RUN apt-get install -y \
    build-essential \
    htop \
    gnupg \
    curl \
    ca-certificates \
    wget \
    vim \
    tmux

# Cleanup
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*.deb /var/cache/apt/archives/partial/*.deb /var/cache/apt/*.bin

# Update pip
RUN python3 -m pip install --upgrade pip

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

# Install dependencies (this is not necessary when using an *external* mini conda environment)
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
    datasets



# Add the user setup
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

# Set environment variable for Hugging Face cache directory
#ENV HF_HOME /projects/misinfo_sp/.cache/

# Print the Hugging Face cache directory
#RUN echo $HF_HOME

CMD ["/bin/bash"]
