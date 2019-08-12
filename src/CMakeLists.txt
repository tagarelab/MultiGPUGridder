# https://devblogs.nvidia.com/building-cuda-applications-cmake/
# /opt/cmake-3.15.2-Linux-x86_64/bin/cmake-gui 


cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

#include(CTest)

SET(CMAKE_CXX_STANDARD 11)

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
