#include <cuda.h>
#include <cuda_runtime.h>

//---------------------------------------------------------------------------------------------------------------------
int CUDA_SM (){
//---------------------------------------------------------------------------------------------------------------------
  int deviceID;
  cudaDeviceProp props;

  cudaGetDevice(&deviceID);
  cudaGetDeviceProperties(&props, deviceID);

  return props.multiProcessorCount;
}
