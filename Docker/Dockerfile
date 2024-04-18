# Base image: Ubuntu 20.04 with CUDA runtime libraries.
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Update system. Install packages including ASAP 2.0 from deb package file.
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install --assume-yes \
    wget=1.20.3-* git=1:2.25.1-* git-lfs=2.9.2-* && \
    apt clean

# Setup a passwordless user called "aiosyn".
RUN useradd --create-home aiosyn && passwd --delete aiosyn

# Switch to the user called 'aiosyn' and move to its home directory for further installations.
USER aiosyn
WORKDIR /home/aiosyn

# Install Miniconda.
ARG MINICONDA3_VERSION=py39_23.1.0-1-Linux-x86_64
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA3_VERSION}.sh --output-document=miniconda3.sh && \
    /bin/bash ./miniconda3.sh -b && rm ./miniconda3.sh
ENV PATH=/home/aiosyn/miniconda3/bin:${PATH}
RUN conda install --quiet --name=base conda-libmamba-solver && \
    conda config --set solver libmamba && \
    conda clean --all --yes && \
    conda init bash

# Make the name of the conda environment as an environment variable. This will be present in the docker image.
ENV AIOSYN_CONDA_ENV=core

# Create conda environment.
COPY --chown=aiosyn requirements_18-4.yaml ./
RUN CONDA_OVERRIDE_CUDA=${CUDA_VERSION} conda env create --quiet --name=core --file=requirements_18-4.yaml && \
    conda clean --all --yes && \
    rm requirements_18-4.yaml

# Install AWS-CLI in a separate environment
RUN conda create --name=aws python=3.9 awscli=1.27.113 --channel=conda-forge

# Fit the path of libdevice.10.bc for ptxas (Nvidia PTX optimizing assembler) of cuda-nvcc conda package.
RUN mkdir --parents ./miniconda3/envs/core/lib/nvvm/libdevice/ && \
    ln --symbolic --relative ./miniconda3/envs/core/lib/libdevice.10.bc ./miniconda3/envs/core/lib/nvvm/libdevice/
ENV XLA_FLAGS="--xla_gpu_cuda_data_dir=/home/aiosyn/miniconda3/envs/core/lib"

# Install Cytomine client. All packages that we require are availabel in the environment hence the `--no-deps` option.
ARG CYTOMINE_VERSION=2.2.2
RUN wget --quiet https://github.com/cytomine/Cytomine-python-client/archive/refs/tags/v${CYTOMINE_VERSION}.tar.gz --output-document=cytomine.tar.gz && \
    tar --extract --file=cytomine.tar.gz && \
    rm cytomine.tar.gz && \
    /home/aiosyn/miniconda3/envs/core/bin/pip install ./Cytomine-python-client-${CYTOMINE_VERSION} --no-deps

# Configure the entrypoint.
COPY --chown=aiosyn entrypoint.sh /home/aiosyn/
ENTRYPOINT ["/bin/bash", "./entrypoint.sh"]