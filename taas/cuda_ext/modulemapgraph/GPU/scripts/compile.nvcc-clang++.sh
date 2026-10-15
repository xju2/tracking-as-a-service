#cmake -H. -B build/nvcc -DCMAKE_CXX_FLAGS="-w -Ofast" -DCMAKE_CUDA_FLAGS=" -w -pthread -lstdc++fs -O3 -allow-unsupported-compiler" -DCMAKE_POSITION_INDEPENDENT_CODE=OFF -DCMAKE_CXX_COMPILER=g++ -DCMAKE_CUDA_COMPILER=nvcc -DMMG_USE_CUDA=ON -DMMG_WITH_PYTORCH_EXPORT=NO -DMMG_MMG_USE_CUDA_LAUNCH_BOUNDS=YES -DCMAKE_INSTALL_PREFIX=install/nvcc/ -DMMG_ENABLE_TIMING=ON
cmake -H. -B build/nvcc-clang++ -DCMAKE_CXX_FLAGS="-w -O3" -DCMAKE_CUDA_FLAGS=" -w -pthread -lstdc++fs -O3 -allow-unsupported-compiler" -DCMAKE_POSITION_INDEPENDENT_CODE=OFF -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CUDA_COMPILER=nvcc -DMMG_USE_CUDA=ON -DMMG_WITH_PYTORCH_EXPORT=NO -DMMG_MMG_USE_CUDA_LAUNCH_BOUNDS=YES -DCMAKE_INSTALL_PREFIX=install/nvcc-clang++/ -DMMG_ENABLE_TIMING=OFF
cmake --build build/nvcc-clang++ -- -j
cmake --install build/nvcc-clang++
[[ ! -d $MMG_path/build/nvcc-clang++ ]] && mkdir $MMG_path/build/nvcc-clang++
BIN=build/nvcc-clang++/bin
LIB=build/nvcc-clang++/lib
mkdir $BIN
mkdir $LIB
