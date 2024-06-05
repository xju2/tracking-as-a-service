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
    libzstd-dev ccache libb64-dev \
  && apt-get clean -y

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN pip3 install --upgrade pip
RUN pip3 install -U pandas matplotlib seaborn 

# Environment variables
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/lib:/usr/local/lib"
ENV GET="curl --location --silent --create-dirs"
ENV UNPACK_TO_SRC="tar -xz --strip-components=1 --directory src"
ENV PREFIX="/usr/local"
ENV TORCH_CUDA_ARCH_LIST="80"

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
  && cmake --build build -- install -j 20\
  && cd /tmp && rm -rf src build  

# libtorch (unzip cannot be used in a pipe...)
# This matches the CUDA version of the tritonserver image
ENV LIBTORCH_URL_GPU="https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcu121.zip"
# https://download.pytorch.org/whl/torch_stable.html

RUN ${GET} --output libtorch.zip ${LIBTORCH_URL_GPU} \
  && unzip libtorch.zip \
  && rsync -ruv libtorch/ ${PREFIX} \
  && rm -rf libtorch*

# torchscatter
RUN cd /tmp/ && mkdir src \
	&& ${GET} https://github.com/rusty1s/pytorch_scatter/archive/refs/tags/2.1.2.tar.gz | ${UNPACK_TO_SRC} \
	&& cd src && FORCE_CUDA=1 pip3 install torch-scatter && rm -rf /tmp/src

# torch sparse
RUN cd /tmp/ && mkdir src \
	&& ${GET} https://github.com/rusty1s/pytorch_sparse/archive/refs/tags/0.6.18.tar.gz | ${UNPACK_TO_SRC} \
	&& cd src && FORCE_CUDA=1 pip3 install torch-sparse && rm -rf /tmp/src

# torch cluster
RUN source activate $CONDA_ENV_NAME && cd /tmp/ && mkdir src \
	&& ${GET https://github.com/rusty1s/pytorch_cluster/archive/refs/tags/1.6.3.tar.gz | ${UNPACK_TO_SRC}} \
	&& cd src && FORCE_CUDA=1 pip3 install torch-cluster && rm -rf /tmp/src

RUN pip3 install torch_geometric lightning>=2.2

# root v6.24.06
# RUN cd /tmp && rm -rf src build && mkdir -p src \
#   && ${GET} https://root.cern/download/root_v6.24.06.source.tar.gz \
#     | ${UNPACK_TO_SRC} \
#   && cmake -B build -S src -GNinja \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_CXX_STANDARD=17 \
#     -DCMAKE_INSTALL_PREFIX=${PREFIX} \
#     -Dfail-on-missing=ON \
#     -Dgminimal=ON \
#     -Dgdml=ON \
#     -Dopengl=ON \
#     -Dpyroot=ON \
#   && cmake --build build -- install -j 20\
#   && rm -rf build src

# Rapids AI
# cudf-cu12 dask-cudf-cu12 cuml-cu12 cugraph-cu12 cuspatial-cu12 cuproj-cu12 cuxfilter-cu12 cucim
RUN pip3 install --extra-index-url=https://pypi.nvidia.com cudf-cu12 cugraph-cu12 nx-cugraph-cu12


# Onnx (download of tar.gz does not work out of the box, since the build.sh script requires a git repository)
RUN pip3 install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# RUN git clone https://github.com/microsoft/onnxruntime src \
#   && (cd src && git checkout v1.13.1) \
#   && ./src/build.sh \
#     --config MinSizeRel \
#     --build_shared_lib \
#     --build_dir build \
#     --use_cuda \
#     --cuda_home /usr/local/cuda \
#     --cudnn_home /usr/local/cuda \
#     --parallel 8 \
#     --skip_tests \
#     --cmake_extra_defines \
#       CMAKE_INSTALL_PREFIX=${PREFIX} \
#       CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
#   && cmake --build build/MinSizeRel -- install -j 20 \
#   && rm -rf build src

# faiss v1.7.4
  RUN cd /tmp && rm -rf src && mkdir -p src \
  && ${GET} https://github.com/facebookresearch/faiss/archive/refs/tags/v1.7.4.tar.gz \
    | ${UNPACK_TO_SRC} \
  && cd src && mkdir build && cd build \
  && cmake .. -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=ON \
        -DFAISS_ENABLE_C_API=ON -DBUILD_SHARED_LIBS=ON \
        -DPython_EXECUTABLE=/usr/bin/python -DPython_LIBRARIES=/usr/lib/python3.8 \
        -DCMAKE_INSTALL_PREFIX=${PREFIX} \
  && make -j8 faiss && make -j8 swigfaiss \
  && cd faiss/python && pip3 install . \
  && cd ../.. && make install -j8 && cd .. \
  && rm -rf src

# Install grpc
RUN git clone --recurse-submodules -b v1.64.1 --depth 1 https://github.com/grpc/grpc src\
    && cmake -B build -S src -DgRPC_INSTALL=ON \
        -DgRPC_BUILD_TESTS=OFF \
        -DCMAKE_INSTALL_PREFIX=${PREFIX} \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -- install -j20 \
    && rm -rf src build

# Install triton
RUN git clone -b r24.05 https://github.com/triton-inference-server/client.git \
    && cd client && mkdir build && cd build \
    && cmake ../src/c++ -DTRITON_ENABLE_CC_HTTP=OFF \
        -DTRITON_ENABLE_CC_GRPC=ON \
        -DTRITON_ENABLE_PYTHON_GRPC=ON \
        -DCMAKE_PREFIX_PATH="${PREFIX}/lib64/cmake;${PREFIX}/lib/cmake" \
        -DCMAKE_INSTALL_PREFIX=${PREFIX} \
        -DTRITON_USE_THIRD_PARTY=OFF  \
        -DTRITON_ENABLE_GPU=ON \
        -DTRITON_ENABLE_METRICS_GPU=ON \
        -DTRITON_ENABLE_PERF_ANALYZER=ON \
        -DTRITON_ENABLE_PERF_ANALYZER_C_API=ON \
    && make -j20 && make install \
    && cd ../.. && rm -rf client \
    && cd /tmp && rm -rf src build

