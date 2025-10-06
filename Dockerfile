# syntax=docker/dockerfile:experimental

FROM nvcr.io/nvidia/tritonserver:24.05-py3
# nvcc version: 12.4 ## nvcc --version
# cudnn version: 9.1.0  ## find / -name "libcudnn*" 2>/dev/null
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver

LABEL description="Triton Server backend with other dependencies for Tracking-as-a-Service"
LABEL version="1.0"

# Install dependencies
# Update the CUDA Linux GPG Repository Key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

# See also https://root.cern.ch/build-prerequisites
# https://root.cern/install/dependencies/#ubuntu-and-other-debian-based-distributions
RUN apt-get update -y && apt-get install -y \
    build-essential curl git freeglut3-dev libfreetype6-dev libpcre3-dev \
    libtbb-dev ninja-build time tree \
    python3 python3-dev python3-pip python3-numpy \
    rsync zlib1g-dev ccache vim unzip libblas-dev liblapack-dev swig rapidjson-dev \
    libexpat-dev libeigen3-dev libftgl-dev libgl2ps-dev libglew-dev libgsl-dev \
    liblz4-dev liblzma-dev libx11-dev libxext-dev libxft-dev libxpm-dev libxerces-c-dev \
    libzstd-dev libb64-dev graphviz gfortran  libglu1-mesa-dev  \
    libfftw3-dev libcfitsio-dev libgraphviz-dev \
    libavahi-compat-libdnssd-dev libldap2-dev libxml2-dev libkrb5-dev \
     qtwebengine5-dev nlohmann-json3-dev libmysqlclient-dev libxxhash-dev \
  && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip3 install --upgrade pip

# install BOOST
# http://www.boost.org/build/
ARG BOOST_VERSION_DOTTED=1.87.0
RUN cd /tmp && git clone --depth 1 --branch boost-${BOOST_VERSION_DOTTED} https://github.com/boostorg/boost.git /tmp/boost && \
  cd /tmp/boost && \
  git submodule update --init --recursive && \
  ./bootstrap.sh --with-python=/usr/bin/python3 && \
  ./b2 -j16 link=shared threading=multi runtime-link=shared cxxflags="-fPIC" variant=release install && \
  rm -rf /tmp/boost

# Manual builds for specific packages
# Install CMake v3.29.4
ARG CMAKE_VERSION=3.29.4
RUN cd /tmp && mkdir -p src \
  && ${GET} https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz \
    | ${UNPACK_TO_SRC} \
  && rsync -ru src/ ${PREFIX} \
  && cd /tmp && rm -rf /tmp/src

# install ROOT
ARG ROOT_VERSION=6.32.02
ENV ROOTSYS=/usr/local/root
RUN cd /tmp && \
    wget https://root.cern/download/root_v${ROOT_VERSION}.source.tar.gz && \
    tar -xzvf root_v${ROOT_VERSION}.source.tar.gz && \
    mkdir /tmp/root_build && \
    cd /tmp/root_build && \
    cmake ../root-${ROOT_VERSION} -G Ninja\
      -DCMAKE_INSTALL_PREFIX=${ROOTSYS} \
      -Dbuiltin_llvm=ON \
      -DLLVM_TARGETS_TO_BUILD="X86;NVPTX" \
      -DLLVM_PARALLEL_COMPILE_JOBS=1 \
      -DLLVM_PARALLEL_LINK_JOBS=1 \
      -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
      -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld" \
      -Dpyroot=OFF -Droofit=OFF -Dtmva=OFF -Dgnuinstall=OFF -Dminuit=OFF \
      -Drpath=ON && \
    cmake --build . --target install -- -j16 && \
    cd / && rm -rf /tmp/root_v${ROOT_VERSION}.source.tar.gz /tmp/root-${ROOT_VERSION} /tmp/root_build

# Environment variables
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib:/usr/local/lib"
ENV GET="curl --location --silent --create-dirs"
ENV UNPACK_TO_SRC="tar -xz --strip-components=1 --directory src"
ENV PREFIX="/usr/local"
ENV TORCH_CUDA_ARCH_LIST="80"
ENV PYTHONNOUSERSITE=True

RUN pip3 install pyyaml astunparse expecttest!=0.2.0 hypothesis numpy psutil requests types-dataclasses \
    typing-extensions>=4.8.0 sympy filelock jinja2 fsspec lintrunner ninja packaging optree>=0.11.0 setuptools \
    git+https://github.com/LAL/trackml-library.git \
    click pytest pytest-cov class-resolver scipy pandas matplotlib uproot tqdm \
    ipykernel atlasify networkx seaborn wandb pygraphviz tritonclient[all]

# Get pytorch source and build so that it runs on different GPUs.
RUN pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

RUN pip3 install torch_geometric lightning>=2.2

# Onnx (download of tar.gz does not work out of the box, since the build.sh script requires a git repository)
RUN pip3 install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# Rapids AI
# cudf-cu12 dask-cudf-cu12 cuml-cu12 cugraph-cu12 cuspatial-cu12 cuproj-cu12 cuxfilter-cu12 cucim
RUN pip3 install --extra-index-url=https://pypi.nvidia.com cudf-cu12 cugraph-cu12 nx-cugraph-cu12 cugraph-pyg-cu12
# RUN pip install pymodulemapgraph --index-url https://gitlab.cern.ch/api/v4/projects/210408/packages/pypi/simple

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

RUN pip install graph_segment
