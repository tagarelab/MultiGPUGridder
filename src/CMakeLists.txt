# https://devblogs.nvidia.com/building-cuda-applications-cmake/
# /opt/cmake-3.15.2-Linux-x86_64/bin/cmake-gui 
# Have to select the x64 option in CMAKE

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
SET(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++11 support has been enabled by default.")

project(MultiGPUGridder LANGUAGES CXX CUDA)

#include(CTest)
find_package(CUDA 10.0 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})


option(WITH_NVPROF    "Enable NVidia profiler bindings (requires CUDA)" FALSE)
option(WITH_MATLAB    "Compile the Matlab mex wrapper  (requires MATLAB)" TRUE)
option(WITH_PYTHON    "Compile the Python wrapper  (requires PYTHON)" FALSE)
option(WITH_CUFFT     "Include the CUDA CUFFT library (required for Windows)" False)
option(WITH_WINDOWS  "Using a Windows PC" False)

if(WITH_NVPROF)
 find_library(NVTX_LIBRARY
     NAME libnvToolsExt.so
     PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib)

   message(STATUS "NVidia tools extension library found in " ${NVTX_LIBRARY})
endif()

if(WITH_WINDOWS)
	find_package(CUDA REQUIRED)
	list(APPEND CUDA_DEV_LIBRARIES
				${CUDA_cufft_LIBRARY})
endif()

#find_package(Cuda)
#find_package(Matlab)

set(SOURCE_Files 
		AbstractGridder.h
		AbstractGridder.cpp


		gpuGridder.h
		gpuGridder.cpp

		gpuForwardProject.h
		gpuForwardProjectKernel.cu

		gpuBackProject.h
		gpuBackProject.cu

		gpuFFT.h
		gpuFFT.cu
		
		MultiGPUGridder.h
		MultiGPUGridder.cpp
  )

set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cupti_LIBRARY} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_npp_LIBRARY} ${CUDA_nppc_LIBRARY} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY})

add_library(gridderLib SHARED ${SOURCE_Files}  )


target_link_libraries( gridderLib ${CUDA_LIBRARIES})


# Compile with -fPIC flag on
# set_property(TARGET gridderLib PROPERTY POSITION_INDEPENDENT_CODE ON)

#target_link_libraries( gridderLib ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cupti_LIBRARY} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_npp_LIBRARY} ${CUDA_nppc_LIBRARY} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY})


if(WITH_CUFFT)
	CUDA_ADD_CUFFT_TO_TARGET(MultiGPUGridder )
endif()

if(WITH_MATLAB)
	find_package(Matlab)


	if(NOT WITH_WINDOWS)
		find_package(Matlab)
		matlab_add_mex(NAME mexCreateGridder SRC mexCreateGridder.cpp LINK_TO gridderLib)
		matlab_add_mex(NAME mexDeleteGridder SRC mexDeleteGridder.cpp LINK_TO gridderLib)
		matlab_add_mex(NAME mexSetVariables SRC mexSetVariables.cpp LINK_TO gridderLib)
		matlab_add_mex(NAME mexGetVariables SRC mexGetVariables.cpp LINK_TO gridderLib)	
		matlab_add_mex(NAME mexForwardProject SRC mexMultiGPUForwardProject.cpp LINK_TO gridderLib)	
		matlab_add_mex(NAME mexBackProject SRC mexMultiGPUBackProject.cpp LINK_TO gridderLib)	

		message(STATUS "Matlab library located")
	endif()


	if(WITH_WINDOWS)
		matlab_add_mex(NAME mexDeleteGridder SRC mexDeleteGridder.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})
		matlab_add_mex(NAME mexCreateGridder SRC mexCreateGridder.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})
		matlab_add_mex(NAME mexSetVariables SRC mexSetVariables.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})
		matlab_add_mex(NAME mexGetVariables SRC mexGetVariables.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})
		matlab_add_mex(NAME mexForwardProject SRC mexMultiGPUForwardProject.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})
		matlab_add_mex(NAME mexBackProject SRC mexMultiGPUBackProject.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	

		message(STATUS "Matlab library located")
	endif()
endif()




#mexFunctionWrapper.h
#mexCreateClass.cpp

#gpuBackProject.h
#gpuBackProjectKernel.cu


#MemoryManager.h
#MemoryManager.cpp

#MultiGPUGridder.h
#MultiGPUGridder.cpp
