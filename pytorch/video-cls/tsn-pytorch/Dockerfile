FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
LABEL maintainer="baoyixin@qiniu.com"

# dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libiomp-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libopenmpi-dev \
    libprotobuf-dev \
    libsnappy-dev \
    openmpi-bin \
    openmpi-doc \
    protobuf-compiler \
    python-dev \
    python-numpy \
    python-pip \
    python-pydot \
    python-setuptools \
    python-scipy \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda install -y pytorch torchvision -c pytorch

# nvvl dependencies
ARG FFMPEG_VERSION=3.4.2

# nvcuvid deps
RUN apt-get update --fix-missing && \
    apt-get install -y libx11-6 libxext6
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility

# ffmpeg from source
RUN apt-get install -y yasm wget && \
    cd /tmp && wget -q http://ffmpeg.org/releases/ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    tar xf ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    rm ffmpeg-$FFMPEG_VERSION.tar.bz2 && \
    cd ffmpeg-$FFMPEG_VERSION && \
    ./configure \
      --prefix=/usr/local \
      --enable-shared && \
    make -j8 && make install && \
    cd /tmp && rm -rf ffmpeg-$FFMPEG_VERSION && \
    apt-get remove -y yasm

# install nvvl
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple cmake && \
    apt-get install -y pkg-config && \
    cd /tmp && \
    git clone https://github.com/pikerbright/nvvl.git && \
    cd nvvl && \
    cp examples/pytorch_superres/docker/libnvcuvid.so /usr/local/cuda/lib64/stubs && \
    cd pytorch && \
    python3 setup.py install && \
    pip uninstall -y cmake && \
    apt-get remove -y pkg-config && \
    apt-get autoremove -y