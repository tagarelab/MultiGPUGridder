# /opt/cmake-3.15.2-Linux-x86_64/bin/cmake-gui 
# When using Windows need to select the x64 option in CMAKE

cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
SET(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "C++11 support has been enabled by default.")

# Options within the CMake GUI to ask the user
option(WITH_MATLAB   "Compile the Matlab mex wrapper  (requires MATLAB)" TRUE)
option(BUILD_TESTS   "Build Matlab tests  (requires MATLAB)" TRUE)


# Create the project using C++ and CUDA
project(MultiGPUGridder LANGUAGES CXX CUDA)

find_package(CUDA 10.0 REQUIRED)

if(WIN32)
	message(STATUS "Windows OS detected.")

	find_package(CUDA REQUIRED)
	list(APPEND CUDA_DEV_LIBRARIES
				${CUDA_cufft_LIBRARY})
endif()

include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cupti_LIBRARY} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDA_npp_LIBRARY} ${CUDA_nppc_LIBRARY} ${CUDA_nppi_LIBRARY} ${CUDA_npps_LIBRARY})



# Create a variable with all the source file names
set(SOURCE_Files 

    ${CMAKE_CURRENT_SOURCE_DIR}/Structs/HostMemory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Structs/DeviceMemory.h

	${CMAKE_CURRENT_SOURCE_DIR}/Gridders/AbstractGridder.h
	${CMAKE_CURRENT_SOURCE_DIR}/Gridders/AbstractGridder.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Gridders/gpuGridder.h
	${CMAKE_CURRENT_SOURCE_DIR}/Gridders/gpuGridder.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Gridders/MultiGPUGridder.h
	${CMAKE_CURRENT_SOURCE_DIR}/Gridders/MultiGPUGridder.cpp

	${CMAKE_CURRENT_SOURCE_DIR}/Projection/gpuForwardProject.h
	${CMAKE_CURRENT_SOURCE_DIR}/Projection/gpuForwardProject.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Projection/gpuBackProject.h
	${CMAKE_CURRENT_SOURCE_DIR}/Projection/gpuBackProject.cu

	${CMAKE_CURRENT_SOURCE_DIR}/Filters/AbstractFilter.h
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/AbstractFilter.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/PadVolumeFilter.h
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/PadVolumeFilter.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/AddVolumeFilter.h
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/AddVolumeFilter.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/CropVolumeFilter.h
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/CropVolumeFilter.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/CASToComplexFilter.h
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/CASToComplexFilter.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/FFTShift2DFilter.h
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/FFTShift2DFilter.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/FFTShift3DFilter.h
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/FFTShift3DFilter.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/ComplexToCASFilter.h
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/ComplexToCASFilter.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/DivideVolumeFilter.h
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/DivideVolumeFilter.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/RealToComplexFilter.h
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/RealToComplexFilter.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/ComplexToRealFilter.h
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/ComplexToRealFilter.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/DivideScalarFilter.h
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/DivideScalarFilter.cu
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/MultiplyVolumeFilter.h
	${CMAKE_CURRENT_SOURCE_DIR}/Filters/MultiplyVolumeFilter.cu
  )

# Variable for all the directories of source code
set(DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/Structs ${CMAKE_CURRENT_SOURCE_DIR}/Filters ${CMAKE_CURRENT_SOURCE_DIR}/Gridders ${CMAKE_CURRENT_SOURCE_DIR}/Projection ${CMAKE_CURRENT_SOURCE_DIR}/Tests ${CMAKE_CURRENT_SOURCE_DIR}/Mex)

if(NOT WIN32)
	add_library(MultiGPUGridder SHARED ${SOURCE_Files}  )
	# Link the CUDA libraries and header files to the project
	target_link_libraries( MultiGPUGridder ${CUDA_LIBRARIES})

	# Include all the header files
	target_include_directories(MultiGPUGridder PUBLIC DIRECTORIES)
endif()

if(WITH_MATLAB)
	find_package(Matlab)

	if(NOT WIN32)
		find_package(Matlab)
		matlab_add_mex(NAME mexCreateGridder SRC ${CMAKE_CURRENT_SOURCE_DIR}/Mex/mexCreateGridder.cpp LINK_TO MultiGPUGridder)
		matlab_add_mex(NAME mexDeleteGridder SRC ${CMAKE_CURRENT_SOURCE_DIR}/Mex/mexDeleteGridder.cpp LINK_TO MultiGPUGridder)
		matlab_add_mex(NAME mexSetVariables SRC ${CMAKE_CURRENT_SOURCE_DIR}/Mex/mexSetVariables.cpp LINK_TO MultiGPUGridder)
		matlab_add_mex(NAME mexMultiGPUForwardProject SRC ${CMAKE_CURRENT_SOURCE_DIR}/Mex/mexMultiGPUForwardProject.cpp LINK_TO MultiGPUGridder)	
		matlab_add_mex(NAME mexMultiGPUBackProject SRC ${CMAKE_CURRENT_SOURCE_DIR}/Mex/mexMultiGPUBackProject.cpp LINK_TO MultiGPUGridder)	
		matlab_add_mex(NAME mexMultiGPUGetVolume SRC ${CMAKE_CURRENT_SOURCE_DIR}/Mex/mexMultiGPUGetVolume.cpp LINK_TO MultiGPUGridder)	
		matlab_add_mex(NAME mexMultiGPUReconstructVolume SRC ${CMAKE_CURRENT_SOURCE_DIR}/Mex/mexMultiGPUReconstructVolume.cpp LINK_TO MultiGPUGridder)			
		message(STATUS "Matlab library located")

		if (BUILD_TESTS)
			matlab_add_mex(NAME mexPadVolume SRC ${CMAKE_CURRENT_SOURCE_DIR}/Tests/mexPadVolume.cpp ${SOURCE_Files} LINK_TO MultiGPUGridder)	

			message(STATUS "Matlab tests added")
		endif()

	endif()

	if(WIN32)
		matlab_add_mex(NAME mexDeleteGridder SRC ${CMAKE_CURRENT_SOURCE_DIR}/Mex/mexDeleteGridder.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})
		matlab_add_mex(NAME mexCreateGridder SRC ${CMAKE_CURRENT_SOURCE_DIR}/Mex/mexCreateGridder.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})
		matlab_add_mex(NAME mexSetVariables SRC ${CMAKE_CURRENT_SOURCE_DIR}/Mex/mexSetVariables.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})
		matlab_add_mex(NAME mexMultiGPUForwardProject SRC ${CMAKE_CURRENT_SOURCE_DIR}/Mex/mexMultiGPUForwardProject.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})
		matlab_add_mex(NAME mexMultiGPUBackProject SRC ${CMAKE_CURRENT_SOURCE_DIR}/Mex/mexMultiGPUBackProject.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
		matlab_add_mex(NAME mexMultiGPUGetVolume SRC ${CMAKE_CURRENT_SOURCE_DIR}/Mex/mexMultiGPUGetVolume.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
		matlab_add_mex(NAME mexMultiGPUReconstructVolume SRC ${CMAKE_CURRENT_SOURCE_DIR}/Mex/mexMultiGPUReconstructVolume.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	

		message(STATUS "Matlab library located")

		if (BUILD_TESTS)
			matlab_add_mex(NAME mexPadVolume SRC ${CMAKE_CURRENT_SOURCE_DIR}/Tests/mexPadVolume.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
			matlab_add_mex(NAME mexAddVolume SRC ${CMAKE_CURRENT_SOURCE_DIR}/Tests/mexAddVolume.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
			matlab_add_mex(NAME mexCropVolume SRC ${CMAKE_CURRENT_SOURCE_DIR}/Tests/mexCropVolume.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
			matlab_add_mex(NAME mexDivideScalar SRC ${CMAKE_CURRENT_SOURCE_DIR}/Tests/mexDivideScalar.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
			matlab_add_mex(NAME mexFFTShift2D SRC ${CMAKE_CURRENT_SOURCE_DIR}/Tests/mexFFTShift2D.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
			matlab_add_mex(NAME mexFFTShift3D SRC ${CMAKE_CURRENT_SOURCE_DIR}/Tests/mexFFTShift3D.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
			matlab_add_mex(NAME mexDivideVolume SRC ${CMAKE_CURRENT_SOURCE_DIR}/Tests/mexDivideVolume.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
			matlab_add_mex(NAME mexMultiplyVolume SRC ${CMAKE_CURRENT_SOURCE_DIR}/Tests/mexMultiplyVolume.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
			matlab_add_mex(NAME mexRealToComplex SRC ${CMAKE_CURRENT_SOURCE_DIR}/Tests/mexRealToComplex.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
			matlab_add_mex(NAME mexComplexToReal SRC ${CMAKE_CURRENT_SOURCE_DIR}/Tests/mexComplexToReal.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
			matlab_add_mex(NAME mexComplexToCAS SRC ${CMAKE_CURRENT_SOURCE_DIR}/Tests/mexComplexToCAS.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
			matlab_add_mex(NAME mexCASToComplex SRC ${CMAKE_CURRENT_SOURCE_DIR}/Tests/mexCASToComplex.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	

			message(STATUS "Matlab tests added")
		endif()

	endif()

	target_include_directories(mexDeleteGridder PUBLIC ${DIRECTORIES})
	target_include_directories(mexCreateGridder PUBLIC ${DIRECTORIES})
	target_include_directories(mexSetVariables PUBLIC ${DIRECTORIES})
	target_include_directories(mexMultiGPUForwardProject PUBLIC ${DIRECTORIES})
	target_include_directories(mexMultiGPUBackProject PUBLIC ${DIRECTORIES})
	target_include_directories(mexMultiGPUGetVolume PUBLIC ${DIRECTORIES})
	target_include_directories(mexMultiGPUReconstructVolume PUBLIC ${DIRECTORIES})

	if (BUILD_TESTS)
		target_include_directories(mexPadVolume PUBLIC ${DIRECTORIES})
		target_include_directories(mexAddVolume PUBLIC ${DIRECTORIES})
		target_include_directories(mexCropVolume PUBLIC ${DIRECTORIES})
		target_include_directories(mexDivideScalar PUBLIC ${DIRECTORIES})
		target_include_directories(mexFFTShift2D PUBLIC ${DIRECTORIES})
		target_include_directories(mexFFTShift3D PUBLIC ${DIRECTORIES})
		target_include_directories(mexDivideVolume PUBLIC ${DIRECTORIES})
		target_include_directories(mexMultiplyVolume PUBLIC ${DIRECTORIES})
		target_include_directories(mexRealToComplex PUBLIC ${DIRECTORIES})
		target_include_directories(mexComplexToReal PUBLIC ${DIRECTORIES})
		target_include_directories(mexComplexToCAS PUBLIC ${DIRECTORIES})
		target_include_directories(mexCASToComplex PUBLIC ${DIRECTORIES})

	endif()

endif()
