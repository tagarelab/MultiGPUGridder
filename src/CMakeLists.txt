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
#find_package(Matlab)

add_library(gridderLib SHARED

		AbstractGridder.h
		AbstractGridder.cpp
	
		gpuGridder.h
		gpuGridder.cpp
		
		gpuFFT.h
		gpuFFT.cu

		
  )

# Compile with -fPIC flag on
# set_property(TARGET gridderLib PROPERTY POSITION_INDEPENDENT_CODE ON)

if(WITH_CUFFT)
	CUDA_ADD_CUFFT_TO_TARGET(MultiGPUGridder )
endif()

if(WITH_MATLAB)
	find_package(Matlab)
	matlab_add_mex(NAME mexCreateGridder SRC mexCreateGridder.cpp LINK_TO gridderLib)
	matlab_add_mex(NAME mexDeleteGridder SRC mexDeleteGridder.cpp LINK_TO gridderLib)
	matlab_add_mex(NAME mexSetVariables SRC mexSetVariables.cpp LINK_TO gridderLib)
	matlab_add_mex(NAME mexGetVariables SRC mexGetVariables.cpp LINK_TO gridderLib)	
	matlab_add_mex(NAME mexForwardProject SRC mexForwardProject.cpp LINK_TO gridderLib)	
	

	message(STATUS "Matlab library located")
endif()




#mexFunctionWrapper.h
#mexCreateClass.cpp

#gpuBackProject.h
#gpuBackProjectKernel.cu
#gpuForwardProject.h
#gpuForwardProjectKernel.cu

#MemoryManager.h
#MemoryManager.cpp

#MultiGPUGridder.h
#MultiGPUGridder.cpp
