# https://devblogs.nvidia.com/building-cuda-applications-cmake/
# /opt/cmake-3.15.2-Linux-x86_64/bin/cmake-gui 
# Have to select the x64 option in CMAKE

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

#include(CTest)

SET(CMAKE_CXX_STANDARD 11)

option(WITH_NVPROF    "Enable NVidia profiler bindings (requires CUDA)" FALSE)
option(WITH_MATLAB    "Compile the Matlab mex wrapper  (requires MATLAB)" TRUE)
option(WITH_PYTHON    "Compile the Python wrapper  (requires PYTHON)" TRUE)
option(WITH_CUFFT     "Include the CUDA CUFFT library (required for Windows)" False)

if(WITH_NVPROF)
 find_library(NVTX_LIBRARY
     NAME libnvToolsExt.so
     PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)

   message(STATUS "NVidia tools extension library found in " ${NVTX_LIBRARY})
endif()

#find_package(Cuda)
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

if(WITH_CUFFT)
	CUDA_ADD_CUFFT_TO_TARGET(MultiGPUGridder )
endif()

if(WITH_MATLAB)
	find_package(Matlab)
	matlab_add_mex(NAME mexFunctionWrapper SRC mexFunctionWrapper.cpp LINK_TO MultiGPUGridder)
	message(STATUS "Matlab library located")
endif()