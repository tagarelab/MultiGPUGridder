# https://devblogs.nvidia.com/building-cuda-applications-cmake/
# /opt/cmake-3.15.2-Linux-x86_64/bin/cmake-gui 


cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

#include(CTest)

SET(CMAKE_CXX_STANDARD 11)

#option(WITH_NVPROF    "Enable NVidia profiler bindings (requires CUDA)" FALSE)

#if(WITH_NVPROF)
# find_library(NVTX_LIBRARY
#     NAME libnvToolsExt.so
#     PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)

#   message(STATUS "NVidia tools extension library found in " ${NVTX_LIBRARY})
#endif()

find_package(Cuda)
add_library(MultiGPUGridder SHARED
		gpuBackProjectKernel.cu
		gpuForwardProjectKernel.cu
		gpuBackProject.h
		gpuForwardProject.h
		MemoryManager.cpp
		MultiGPUGridder.cpp
		MemoryManager.h
		MultiGPUGridder.h
  )

CUDA_ADD_CUFFT_TO_TARGET(MultiGPUGridder )


#find_package(Matlab)
#matlab_add_mex(NAME mexFunctionWrapper SRC mexFunctionWrapper.cpp LINK_TO MultiGPUGridder)
