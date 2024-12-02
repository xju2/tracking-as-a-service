# syntax=docker/dockerfile:experimental

FROM nvcr.io/nvidia/tritonserver:24.05-py3
# nvcc version: 12.4 ## nvcc --version
# cudnn version: 9.1.0  ## find / -name "libcudnn*" 2>/dev/null

LABEL description="Triton Server backend with other dependencies for Tracking-as-a-Service"
LABEL version="1.0"

# Install dependencies
# Update the CUDA Linux GPG Repository Key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

# See also https://root.cern.ch/build-prerequisites
RUN apt-get update -y && apt-get install -y \
    build-essential curl git freeglut3-dev libfreetype6-dev libpcre3-dev\
    libboost-dev libboost-filesystem-dev libboost-program-options-dev libboost-test-dev \
    libtbb-dev ninja-build time tree \
    python3 python3-dev python3-pip python3-numpy \
    rsync zlib1g-dev ccache vim unzip libblas-dev liblapack-dev swig \
    rapidjson-dev \
    libexpat-dev libeigen3-dev libftgl-dev libgl2ps-dev libglew-dev libgsl-dev \
    liblz4-dev liblzma-dev libx11-dev libxext-dev libxft-dev libxpm-dev libxerces-c-dev \
    libzstd-dev ccache libb64-dev graphviz graphviz-dev \
  && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip3 install --upgrade pip
RUN pip3 install -U pandas matplotlib seaborn git+https://github.com/LAL/trackml-library.git \
pyyaml click pytest pytest-cov class-resolver scipy pandas matplotlib uproot tqdm \
ipykernel atlasify networkx seaborn wandb pygraphviz tritonclient[all]

# Environment variables
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib:/usr/local/lib"
ENV GET="curl --location --silent --create-dirs"
ENV UNPACK_TO_SRC="tar -xz --strip-components=1 --directory src"
ENV PREFIX="/usr/local"
ENV TORCH_CUDA_ARCH_LIST="80"
ENV PYTHONNOUSERSITE=True

# Manual builds for specific packages
# Install CMake v3.29.4
RUN cd /tmp && mkdir -p src \
  && ${GET} https://github.com/Kitware/CMake/releases/download/v3.29.4/cmake-3.29.4-Linux-x86_64.tar.gz \
    | ${UNPACK_TO_SRC} \
  && rsync -ru src/ ${PREFIX} \
  && cd /tmp && rm -rf /tmp/src

# Install xxHash v0.7.3
RUN cd /tmp && mkdir -p src \
  && ${GET} https://github.com/Cyan4973/xxHash/archive/v0.8.2.tar.gz \
    | ${UNPACK_TO_SRC} \
  && cmake -B build -S src/cmake_unofficial -GNinja\
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
  && cmake --build build -- install -j20\
  && cd /tmp && rm -rf src build

RUN pip3 install pyyaml astunparse expecttest!=0.2.0 hypothesis numpy psutil pyyaml requests setuptools types-dataclasses \
    typing-extensions>=4.8.0 sympy filelock networkx jinja2 fsspec lintrunner ninja packaging optree>=0.11.0 setuptools

RUN apt-get update -y && apt-get install -y gfortran && apt-get clean -y
# install magma
RUN cd /tmp && mkdir -p src \
  && ${GET} https://icl.utk.edu/projectsfiles/magma/downloads/magma-2.8.0.tar.gz \
    | ${UNPACK_TO_SRC} \
  && cmake -B build -S src -DGPU_TARGET="Ampere" -DCMAKE_INSTALL_PREFIX=${PREFIX} -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build -- install -j20 \
  && cd /tmp && rm -rf src

# Get pytorch source and build so that it runs on different GPUs.
ENV TORCH_CUDA_ARCH_LIST="8.0"
RUN cd /tmp && \
	git clone --recursive https://github.com/pytorch/pytorch.git && cd pytorch && \
	git checkout -b r2.3 origin/release/2.3 && \
	git submodule sync && git submodule update --init --recursive --jobs 0 && \
	MAX_JOBS=20 USE_CUDA=1 BUILD_TEST=0 USE_FBGEMM=0 USE_QNNPACK=0 USE_DISTRIBUTED=1 BUILD_CAFFE2=0 DEBUG=0 \
	  CMAKE_PREFIX_PATH=${PREFIX} python setup.py install && \
	rm -rf /tmp/pytorch

# FRNN
RUN cd /tmp/ \
	&& git clone https://github.com/asnaylor/prefix_sum.git \
    && git clone https://github.com/xju2/FRNN.git \
	&& cd prefix_sum \
	&& NVCC_FLAGS="--std=c++17 -gencode=arch=compute_80,code=sm_80" \
		python setup.py install \
    && cd /tmp/FRNN \
    && NVCC_FLAGS="--std=c++17 -gencode=arch=compute_80,code=sm_80" \
		python setup.py install && \
	rm -rf /tmp/prefix_sum && rm -rf /tmp/FRNN

# torchscatter
RUN cd /tmp/ && mkdir src \
	&& ${GET} https://github.com/rusty1s/pytorch_scatter/archive/refs/tags/2.1.2.tar.gz | ${UNPACK_TO_SRC} \
	&& cd src && FORCE_CUDA=1 pip3 install torch-scatter && rm -rf /tmp/src

# torch sparse
RUN cd /tmp/ && mkdir src \
	&& ${GET} https://github.com/rusty1s/pytorch_sparse/archive/refs/tags/0.6.18.tar.gz | ${UNPACK_TO_SRC} \
	&& cd src && FORCE_CUDA=1 pip3 install torch-sparse && rm -rf /tmp/src

# torch cluster
RUN cd /tmp/ && mkdir src \
	&& ${GET} https://github.com/rusty1s/pytorch_cluster/archive/refs/tags/1.6.3.tar.gz | ${UNPACK_TO_SRC} \
	&& cd src && FORCE_CUDA=1 pip3 install torch-cluster && rm -rf /tmp/src

RUN pip3 install torch_geometric lightning>=2.2

# Rapids AI
# cudf-cu12 dask-cudf-cu12 cuml-cu12 cugraph-cu12 cuspatial-cu12 cuproj-cu12 cuxfilter-cu12 cucim
RUN pip3 install --extra-index-url=https://pypi.nvidia.com cudf-cu12 cugraph-cu12 nx-cugraph-cu12

# Onnx (download of tar.gz does not work out of the box, since the build.sh script requires a git repository)
RUN pip3 install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
