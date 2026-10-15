message (STATUS "Compiling CUDA libraries")

# TODO workaround since passing this via target_compile_options did not work
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda --expt-relaxed-constexpr")

enable_language(CUDA)

# Architecture configuration
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Local: auto-detect current GPU
        set(CMAKE_CUDA_ARCHITECTURES native)
        message(STATUS "Using native GPU architecture detection")
endif()

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

add_library (
    ModuleMapGraphGPU SHARED
    include/MMG/graph_creator.cuh
    src/cuda_tools.cu
    src/graph_creator.cu
    include/MMG/CUDA_graph_creator
)

# Add an alias target so we can also link to ModuleMapGraph::GPU when using FetchContent
add_library(ModuleMapGraph::GPU ALIAS ModuleMapGraphGPU)

target_include_directories(ModuleMapGraphGPU PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(ModuleMapGraphGPU CPU_deps)

target_compile_options (ModuleMapGraphGPU INTERFACE -O3)
target_compile_features (ModuleMapGraphGPU INTERFACE cxx_std_20)
# To remove stack-size warnings when using CUDA 12.9
# like : "_Z28partial_recursive_quick_sortPiS_S_i' cannot be statically determined (target: sm_75)"
target_link_options(ModuleMapGraphGPU PRIVATE
  $<DEVICE_LINK:-Xnvlink=--suppress-stack-size-warning>
)
set_target_properties(ModuleMapGraphGPU PROPERTIES 
  VERSION ${ModuleMapGraph_VERSION} SOVERSION ${ModuleMapGraph_VERSION_MAJOR} 
  BUILD_WITH_INSTALL_RPATH ON 
  INSTALL_RPATH_USE_LINK_PATH ON
  CUDA_SEPARABLE_COMPILATION ON )

if(MMG_ENABLE_TIMING)
  target_compile_definitions (ModuleMapGraphGPU PUBLIC MMG_ENABLE_TIMING)
endif()

if(MMG_ENABLE_DEBUG)
  target_compile_definitions (ModuleMapGraphGPU PUBLIC MMG_ENABLE_DEBUG)
endif()

if(MMG_USE_CUDA_LAUNCH_BOUNDS)
  target_compile_definitions (ModuleMapGraphGPU PUBLIC MMG_USE_CUDA_LAUNCH_BOUNDS)
endif()

if(MMG_WITH_NUMPY_EXPORT)
  target_compile_definitions(ModuleMapGraphGraph PUBLIC MMG_WITH_NUMPY_EXPORT)
  target_link_libraries(ModuleMapGraphGraph PUBLIC Boost::python Boost::numpy Python::Python Python::NumPy)
endif()

if(MMG_WITH_PYTORCH_EXPORT)
  target_compile_definitions(ModuleMapGraphGraph PUBLIC MMG_WITH_PYTORCH_EXPORT)
  target_link_libraries(ModuleMapGraphGraph PUBLIC ${TORCH_LIBRARIES} Boost::program_options)
endif()

if(MMG_BUILD_EXECUTABLES)
  add_executable (GraphBuilder.cu.exe utils/GraphBuilder.cu)
  add_executable (GraphBuilder.threads.cu.exe utils/GraphBuilder.threads.cu)
  add_executable (EdgeLayerConnector.cu.exe utils/EdgeLayerConnector.cu)
  add_executable (TrackDiff.cu.exe utils/TrackDiff.cu)

  target_link_libraries (GraphBuilder.cu.exe ModuleMapGraphHits ModuleMapGraphGPU)
  target_link_libraries (GraphBuilder.threads.cu.exe ModuleMapGraphHits ModuleMapGraphGPU)
  target_link_libraries (EdgeLayerConnector.cu.exe ModuleMapGraphHits ModuleMapGraphGPU)
  target_link_libraries (TrackDiff.cu.exe ModuleMapGraphHits ModuleMapGraphGPU)
endif()

### Install
install(
  TARGETS ModuleMapGraphGPU
  EXPORT ModuleMapGraph
  COMPONENT GPU
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if(MMG_BUILD_EXECUTABLES)
  install(
    TARGETS GraphBuilder.cu.exe GraphBuilder.threads.cu.exe EdgeLayerConnector.cu.exe TrackDiff.cu.exe
    EXPORT ModuleMapGraph
    COMPONENT GPU
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()
install(
  DIRECTORY include/
  COMPONENT GPU
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(
  DIRECTORY src/
  COMPONENT GPU
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
