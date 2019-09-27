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

    Structs/MemoryStruct.h

    Structs/MemoryStructGPU.h

	Gridders/AbstractGridder.h
	Gridders/AbstractGridder.cpp

	Gridders/gpuGridder.h
	Gridders/gpuGridder.cpp

	Gridders/MultiGPUGridder.h
	Gridders/MultiGPUGridder.cpp


	Projection/gpuForwardProject.h
	Projection/gpuForwardProject.cu

	Projection/gpuBackProject.h
	Projection/gpuBackProject.cu

	Filters/AbstractFilter.h
	Filters/AbstractFilter.cu

	Filters/PadVolumeFilter.h
	Filters/PadVolumeFilter.cu

	Filters/AddVolumeFilter.h
	Filters/AddVolumeFilter.cu

	Filters/CropVolumeFilter.h
	Filters/CropVolumeFilter.cu

	Filters/CASToComplexFilter.h
	Filters/CASToComplexFilter.cu

	Filters/FFTShift2DFilter.h
	Filters/FFTShift2DFilter.cu

	Filters/FFTShift3DFilter.h
	Filters/FFTShift3DFilter.cu

	Filters/ComplexToCASFilter.h
	Filters/ComplexToCASFilter.cu

	Filters/DivideVolumeFilter.h
	Filters/DivideVolumeFilter.cu

	Filters/RealToComplexFilter.h
	Filters/RealToComplexFilter.cu

	Filters/ComplexToRealFilter.h
	Filters/ComplexToRealFilter.cu

	Filters/DivideScalarFilter.h
	Filters/DivideScalarFilter.cu

	Filters/MultiplyVolumeFilter.h
	Filters/MultiplyVolumeFilter.cu
  )

add_library(gridderLib SHARED ${SOURCE_Files}  )
target_include_directories(gridderLib PUBLIC Filters Gridders Projection Structs)

set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cupti_LIBRARY} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_npp_LIBRARY} ${CUDA_nppc_LIBRARY} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY})

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
		matlab_add_mex(NAME mexCreateGridder SRC Mex/mexCreateGridder.cpp LINK_TO gridderLib)
		matlab_add_mex(NAME mexDeleteGridder SRC Mex/mexDeleteGridder.cpp LINK_TO gridderLib)
		matlab_add_mex(NAME mexSetVariables SRC Mex/mexSetVariables.cpp LINK_TO gridderLib)
		matlab_add_mex(NAME mexGetVariables SRC Mex/mexGetVariables.cpp LINK_TO gridderLib)	
		matlab_add_mex(NAME mexForwardProject SRC Mex/mexMultiGPUForwardProject.cpp LINK_TO gridderLib)	
		matlab_add_mex(NAME mexBackProject SRC Mex/mexMultiGPUBackProject.cpp LINK_TO gridderLib)	

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
