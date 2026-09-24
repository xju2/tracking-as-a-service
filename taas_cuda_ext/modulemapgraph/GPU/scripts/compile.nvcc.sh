debug_or_prod=${1:-'prod'}

if [[ $debug_or_prod == 'prod' ]];then
    cmake_preset="nvcc"
elif [[ $debug_or_prod == 'debug' ]];then
    cmake_preset="nvcc-timing-debug"
else
    echo "Unsupported debug_or_prod $debug_or_prod, please use either 'prod'or 'debug'"
    exit
fi

cmake --preset $cmake_preset
cmake --build --preset $cmake_preset
cmake --install build/nvcc
[[ ! -d $MMG_path/build/nvcc ]] && mkdir $MMG_path/build/nvcc
BIN=build/nvcc/bin
LIB=build/nvcc/lib
mkdir $BIN
mkdir $LIB
