# https://devblogs.nvidia.com/building-cuda-applications-cmake/
# /opt/cmake-3.15.2-Linux-x86_64/bin/cmake-gui 


cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

#include(CTest)

SET(CMAKE_CXX_STANDARD 11)

option(WITH_NVPROF    "Enable NVidia profiler bindings (requires CUDA)" FALSE)
option(WITH_CUPTI     "Enable CUPTI service (CUDA performance analysis)" FALSE)

# Shared libs option
option(BUILD_SHARED_LIBS "Build shared libraries" TRUE)

if(WITH_NVPROF)
 find_library(NVTX_LIBRARY
     NAME libnvToolsExt.so
     PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)

   message(STATUS "NVidia tools extension library found in " ${NVTX_LIBRARY})
endif()

if(WITH_CUPTI)
  find_package(CUDA REQUIRED)
  include(FindCUPTI)
  if (CUPTI_FOUND)
    set(CALIPER_HAVE_CUPTI TRUE)
    set(CALIPER_CUpti_CMAKE_MSG "Yes, using ${CUPTI_LIBRARY}")
    list(APPEND CALIPER_EXTERNAL_LIBS ${CUPTI_LIBRARY})
  endif()
endif()

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


find_package(Matlab)
matlab_add_mex(NAME mexFunctionWrapper SRC mexFunctionWrapper.cpp LINK_TO MultiGPUGridder)
