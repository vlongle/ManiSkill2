FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04
ENV NVIDIA_DRIVER_CAPABILITIES all

# Install os-level packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    htop \
    libegl1 \
    libxext6 \
    libjpeg-dev \
    libpng-dev  \
    libvulkan1 \
    rsync \
    tmux \
    unzip \
    vim \
    vulkan-utils \
    wget \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install (mini) conda
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda init && \
    /opt/conda/bin/conda install -y python=3.8 && \
    /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH
SHELL ["/bin/bash", "-c"]

# https://github.com/haosulab/ManiSkill/issues/9
COPY nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json

# https://github.com/haosulab/ManiSkill2/issues/89
RUN pip install setuptools==65.5.0
RUN pip install mani-skill2==0.4.2 && pip cache purge
# more dependencies
RUN pip install stable_baselines3==1.6.2
RUN pip install tensorboard
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# download data (cached)
ARG USER=vlongle
ARG REPO=ManiSkill2
ARG BRANCH=main
ARG GIT_HOME=/ManiSkill2
RUN git clone -b $BRANCH https://github.com/$USER/$REPO.git $GIT_HOME/

# NOTE: download data
# RUN cd /ManiSkill2/my_code && python -m mani_skill2.utils.download_asset all --non-interactive


#  TODO: a bit annoying to redownload data everytime maybe use data volume instead
# should probably do git clone (cached), download data (cached) 
# and then ADD and git pull (uncached to get the latest stuff)
# change working directory
# https://stackoverflow.com/questions/36996046/how-to-prevent-dockerfile-caching-git-clone/39278224#39278224
# uncache git pull
ADD https://api.github.com/repos/$USER/$REPO/git/refs/heads/$BRANCH version.json
RUN cd ${GIT_HOME} && git pull

WORKDIR /ManiSkill2/my_code