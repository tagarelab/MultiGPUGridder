cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

include(CTest)

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
