# syntax=docker/dockerfile:experimental

# ------------------------
# Build stage: boost-builder
# Compile Boost 1.87 and install to /usr/local
# ------------------------
FROM ubuntu:20.04 AS boost-builder
ARG BOOST_VERSION_DOTTED=1.87.0
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN set -eux; \
  echo "tzdata tzdata/Areas select Etc" | debconf-set-selections; \
  echo "tzdata tzdata/Zones/Etc select UTC" | debconf-set-selections; \
  apt-get update && apt-get install -y --no-install-recommends \
    git build-essential autotools-dev automake libtool python3-dev ca-certificates curl libboost-regex-dev && \
  git clone --depth 1 --branch boost-${BOOST_VERSION_DOTTED} https://github.com/boostorg/boost.git /tmp/boost; \
  cd /tmp/boost; \
  git submodule update --init --recursive; \
  # Include graph library so CMake find_package(boost_graph) succeeds for PyModuleMapGraph
  ./bootstrap.sh --with-libraries=program_options,serialization,regex,filesystem,test,graph; \
  ./b2 -j2 link=shared cxxflags="-fPIC" variant=release install; \
  rm -rf /tmp/boost; \
  apt-get remove -y --purge git build-essential autotools-dev automake libtool python3-dev && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*;

FROM nvcr.io/nvidia/tritonserver:24.05-py3
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
# nvcc version: 12.4 ## nvcc --version
# cudnn version: 9.1.0  ## find / -name "libcudnn*" 2>/dev/null

LABEL description="Triton Server backend with other dependencies for Tracking-as-a-Service"
LABEL version="1.0"

# Install dependencies
# Update the CUDA Linux GPG Repository Key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

# See also https://root.cern.ch/build-prerequisites
# https://root.cern/install/dependencies/#ubuntu-and-other-debian-based-distributions
RUN apt-get update -y && apt-get install -y \
    build-essential cmake curl git freeglut3-dev libfreetype6-dev libpcre3-dev \
    libtbb-dev ninja-build time tree lld \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    rsync zlib1g-dev ccache vim unzip libblas-dev liblapack-dev swig rapidjson-dev \
    libexpat-dev libeigen3-dev libftgl-dev libgl2ps-dev libglew-dev libgsl-dev \
    liblz4-dev liblzma-dev libx11-dev libxext-dev libxft-dev libxpm-dev libxerces-c-dev \
    libzstd-dev libb64-dev graphviz gfortran  libglu1-mesa-dev  \
    libfftw3-dev libcfitsio-dev libgraphviz-dev \
    libavahi-compat-libdnssd-dev libldap2-dev libxml2-dev libkrb5-dev \
     qtwebengine5-dev nlohmann-json3-dev libmysqlclient-dev libxxhash-dev \
  && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3
RUN python3.10 -m pip install --upgrade pip

# Build Boost 1.76.0 from source (required by pymodulemapgraph ≥1.75)
COPY boost_1_76_0.tar.gz /tmp/boost_1_76_0.tar.gz
RUN cd /tmp && \
    tar xzf boost_1_76_0.tar.gz && \
    cd boost_1_76_0 && \
    ./bootstrap.sh --with-python=/usr/bin/python3.10 --with-libraries=program_options,test,graph,regex && \
    ./b2 -j$(nproc) link=shared threading=multi runtime-link=shared cxxflags="-fPIC" variant=release install && \
    cd / && rm -rf /tmp/boost_1_76_0*

# Force CMake to prefer CVMFS Boost over system Boost
# ENV BOOST_ROOT=/cvmfs/sft.cern.ch/lcg/releases/LCG_107_cuda/x86_64-el8-gcc11-opt/Boost \
#     BOOST_INCLUDEDIR=/cvmfs/sft.cern.ch/lcg/releases/LCG_107_cuda/x86_64-el8-gcc11-opt/Boost/include \
#     BOOST_LIBRARYDIR=/cvmfs/sft.cern.ch/lcg/releases/LCG_107_cuda/x86_64-el8-gcc11-opt/Boost/lib \
#     CMAKE_PREFIX_PATH=/cvmfs/sft.cern.ch/lcg/releases/LCG_107_cuda/x86_64-el8-gcc11-opt/Boost

# install ROOT - use pre-built binary to save ~40 minutes
ARG ROOT_VERSION=6.32.02
ENV ROOTSYS=/usr/local/root
ENV PATH="${ROOTSYS}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${ROOTSYS}/lib:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="${ROOTSYS}/lib:${PYTHONPATH}"

COPY root.tar.gz /tmp/root.tar.gz
RUN tar -xzf /tmp/root.tar.gz -C /usr/local/ && rm -f /tmp/root.tar.gz

# Environment variables
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib:/usr/local/lib"
ENV GET="curl --location --silent --create-dirs"
ENV UNPACK_TO_SRC="tar -xz --strip-components=1 --directory src"
ENV PREFIX="/usr/local"
ENV TORCH_CUDA_ARCH_LIST="80"
ENV PYTHONNOUSERSITE=True

# Python dependencies - use default PyPI
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --root-user-action=ignore pyyaml astunparse expecttest!=0.2.0 hypothesis numpy psutil requests types-dataclasses \
    typing-extensions>=4.8.0 filelock jinja2 fsspec lintrunner ninja packaging optree>=0.11.0 setuptools \
    git+https://github.com/LAL/trackml-library.git \
    click pytest pytest-cov class-resolver scipy pandas matplotlib uproot tqdm \
    ipykernel atlasify networkx seaborn wandb pygraphviz tritonclient[all] numba

# PyTorch from specific index
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --root-user-action=ignore torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# PyTorch Geometric extensions
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --root-user-action=ignore pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html && \
    pip3 install torch_geometric lightning>=2.2

# Onnx runtime
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --root-user-action=ignore onnxruntime-gpu \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# Rapids AI
# RUN --mount=type=cache,target=/root/.cache/pip \
#     pip3 install --extra-index-url=https://pypi.nvidia.com \
#     cudf-cu12==24.8.* cugraph-cu12==24.8.* nx-cugraph-cu12==24.8.* cugraph-pyg-cu12==24.8.*

# Install pymodulemapgraph (provides GraphBuilder)
# Clone and modify to apply patches with fuzz tolerance to handle version mismatches
# Use C++20 standard and force CUDA arch 80 via explicit CMake args
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /tmp/ \
    && git clone -b v1.22.0 https://gitlab.cern.ch/gnn4itkteam/pymodulemapgraph.git \
    && cd pymodulemapgraph \
    && sed -i 's/patch -p1/patch --fuzz=2 -p1/g' CMakeLists.txt \
    && (grep -q "<source_location>" src/pybindings/pybindmmg.cu || sed -i '1i#include <source_location>' src/pybindings/pybindmmg.cu) \
    && pip install . \
        --config-settings=cmake.args=-DCMAKE_CUDA_STANDARD=20 \
        --config-settings=cmake.args=-DCMAKE_CXX_STANDARD=20 \
        --config-settings=cmake.args=-DCMAKE_CUDA_ARCHITECTURES=80 \
        --config-settings=cmake.args="-DCMAKE_CUDA_FLAGS=--generate-code=arch=compute_80,code=[compute_80,sm_80]"


# FRNN - build from source with optimized parallelization
RUN cd /tmp/ && \
    git clone --depth 1 https://github.com/asnaylor/prefix_sum.git && \
    git clone --depth 1 https://github.com/xju2/FRNN.git && \
    cd prefix_sum && \
    NVCC_FLAGS="--std=c++17 -gencode=arch=compute_80,code=sm_80" \
        python setup.py install && \
    cd /tmp/FRNN && \
    NVCC_FLAGS="--std=c++17 -gencode=arch=compute_80,code=sm_80" \
        python setup.py install && \
    rm -rf /tmp/prefix_sum /tmp/FRNN

RUN pip install graph_segment