# /opt/cmake-3.15.2-Linux-x86_64/bin/cmake-gui 
# When using Windows need to select the x64 option in CMAKE

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
SET(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++11 support has been enabled by default.")

# Options within the CMake GUI to ask the user
option(WITH_MATLAB   "Compile the Matlab mex wrapper  (requires MATLAB)" TRUE)
option(WITH_WINDOWS  "Using a Windows PC" False)

# Create the project using C++ and CUDA
project(MultiGPUGridder LANGUAGES CXX CUDA)

find_package(CUDA 10.0 REQUIRED)

if(WITH_WINDOWS)
	find_package(CUDA REQUIRED)
	list(APPEND CUDA_DEV_LIBRARIES
				${CUDA_cufft_LIBRARY})
endif()

# Create a variable with all the source file names
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

add_library(MultiGPUGridder SHARED ${SOURCE_Files}  )
target_include_directories(MultiGPUGridder PUBLIC Filters Gridders Projection Structs)

# Link the CUDA libraries and header files to the project
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cupti_LIBRARY} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_npp_LIBRARY} ${CUDA_nppc_LIBRARY} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY})
target_link_libraries( MultiGPUGridder ${CUDA_LIBRARIES})


if(WITH_MATLAB)
	find_package(Matlab)

	if(NOT WITH_WINDOWS)
		find_package(Matlab)
		matlab_add_mex(NAME mexCreateGridder SRC Mex/mexCreateGridder.cpp LINK_TO MultiGPUGridder)
		matlab_add_mex(NAME mexDeleteGridder SRC Mex/mexDeleteGridder.cpp LINK_TO MultiGPUGridder)
		matlab_add_mex(NAME mexSetVariables SRC Mex/mexSetVariables.cpp LINK_TO MultiGPUGridder)
		matlab_add_mex(NAME mexGetVariables SRC Mex/mexGetVariables.cpp LINK_TO MultiGPUGridder)	
		matlab_add_mex(NAME mexMultiGPUForwardProject SRC Mex/mexMultiGPUForwardProject.cpp LINK_TO MultiGPUGridder)	
		matlab_add_mex(NAME mexMultiGPUBackProject SRC Mex/mexMultiGPUBackProject.cpp LINK_TO MultiGPUGridder)	

		message(STATUS "Matlab library located")
	endif()

	if(WITH_WINDOWS)
		matlab_add_mex(NAME mexDeleteGridder SRC mexDeleteGridder.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})
		matlab_add_mex(NAME mexCreateGridder SRC mexCreateGridder.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})
		matlab_add_mex(NAME mexSetVariables SRC mexSetVariables.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})
		matlab_add_mex(NAME mexGetVariables SRC mexGetVariables.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})
		matlab_add_mex(NAME mexMultiGPUForwardProject SRC mexMultiGPUForwardProject.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})
		matlab_add_mex(NAME mexMultiGPUBackProject SRC mexMultiGPUBackProject.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	

		message(STATUS "Matlab library located")
	endif()
endif()
