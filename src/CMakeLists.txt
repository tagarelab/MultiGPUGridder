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

    ${CMAKE_SOURCE_DIR}/Structs/gpuErrorCheck.h
    ${CMAKE_SOURCE_DIR}/Structs/HostMemory.h
    ${CMAKE_SOURCE_DIR}/Structs/DeviceMemory.h

	${CMAKE_SOURCE_DIR}/Gridders/AbstractGridder.h
	${CMAKE_SOURCE_DIR}/Gridders/AbstractGridder.cpp
	${CMAKE_SOURCE_DIR}/Gridders/gpuGridder.h
	${CMAKE_SOURCE_DIR}/Gridders/gpuGridder.cpp
	${CMAKE_SOURCE_DIR}/Gridders/MultiGPUGridder.h
	${CMAKE_SOURCE_DIR}/Gridders/MultiGPUGridder.cpp

	${CMAKE_SOURCE_DIR}/Projection/gpuForwardProject.h
	${CMAKE_SOURCE_DIR}/Projection/gpuForwardProject.cu
	${CMAKE_SOURCE_DIR}/Projection/gpuBackProject.h
	${CMAKE_SOURCE_DIR}/Projection/gpuBackProject.cu

	${CMAKE_SOURCE_DIR}/Filters/AbstractFilter.h
	${CMAKE_SOURCE_DIR}/Filters/AbstractFilter.cu
	${CMAKE_SOURCE_DIR}/Filters/PadVolumeFilter.h
	${CMAKE_SOURCE_DIR}/Filters/PadVolumeFilter.cu
	${CMAKE_SOURCE_DIR}/Filters/AddVolumeFilter.h
	${CMAKE_SOURCE_DIR}/Filters/AddVolumeFilter.cu
	${CMAKE_SOURCE_DIR}/Filters/CropVolumeFilter.h
	${CMAKE_SOURCE_DIR}/Filters/CropVolumeFilter.cu
	${CMAKE_SOURCE_DIR}/Filters/CASToComplexFilter.h
	${CMAKE_SOURCE_DIR}/Filters/CASToComplexFilter.cu
	${CMAKE_SOURCE_DIR}/Filters/FFTShift2DFilter.h
	${CMAKE_SOURCE_DIR}/Filters/FFTShift2DFilter.cu
	${CMAKE_SOURCE_DIR}/Filters/FFTShift3DFilter.h
	${CMAKE_SOURCE_DIR}/Filters/FFTShift3DFilter.cu
	${CMAKE_SOURCE_DIR}/Filters/ComplexToCASFilter.h
	${CMAKE_SOURCE_DIR}/Filters/ComplexToCASFilter.cu
	${CMAKE_SOURCE_DIR}/Filters/DivideVolumeFilter.h
	${CMAKE_SOURCE_DIR}/Filters/DivideVolumeFilter.cu
	${CMAKE_SOURCE_DIR}/Filters/RealToComplexFilter.h
	${CMAKE_SOURCE_DIR}/Filters/RealToComplexFilter.cu
	${CMAKE_SOURCE_DIR}/Filters/ComplexToRealFilter.h
	${CMAKE_SOURCE_DIR}/Filters/ComplexToRealFilter.cu
	${CMAKE_SOURCE_DIR}/Filters/DivideScalarFilter.h
	${CMAKE_SOURCE_DIR}/Filters/DivideScalarFilter.cu
	${CMAKE_SOURCE_DIR}/Filters/MultiplyVolumeFilter.h
	${CMAKE_SOURCE_DIR}/Filters/MultiplyVolumeFilter.cu
  )

# Variable for all the directories of source code
set(DIRECTORIES ${CMAKE_SOURCE_DIR}/Structs ${CMAKE_SOURCE_DIR}/Filters ${CMAKE_SOURCE_DIR}/Gridders ${CMAKE_SOURCE_DIR}/Projection ${CMAKE_SOURCE_DIR}/Tests ${CMAKE_SOURCE_DIR}/Mex)

if(NOT WIN32)
	add_library(MultiGPUGridder SHARED ${SOURCE_Files})

	# Link the CUDA libraries and header files to the project
	target_link_libraries( MultiGPUGridder ${CUDA_LIBRARIES})

	# Include all the header files from each sub folder
	target_include_directories(MultiGPUGridder PUBLIC ${CMAKE_SOURCE_DIR}/Filters ${CMAKE_SOURCE_DIR}/Gridders ${CMAKE_SOURCE_DIR}/Projection ${CMAKE_SOURCE_DIR}/Structs ${CMAKE_SOURCE_DIR}/Mex)
endif()

if(WITH_MATLAB)
	find_package(Matlab)
	message(STATUS "Matlab library located")

	if(NOT WIN32)
		matlab_add_mex(NAME mexCreateGridder SRC ${CMAKE_SOURCE_DIR}/Mex/mexCreateGridder.cpp LINK_TO MultiGPUGridder)
		matlab_add_mex(NAME mexDeleteGridder SRC ${CMAKE_SOURCE_DIR}/Mex/mexDeleteGridder.cpp LINK_TO MultiGPUGridder)
		matlab_add_mex(NAME mexSetVariables SRC ${CMAKE_SOURCE_DIR}/Mex/mexSetVariables.cpp LINK_TO MultiGPUGridder)
		matlab_add_mex(NAME mexMultiGPUForwardProject SRC ${CMAKE_SOURCE_DIR}/Mex/mexMultiGPUForwardProject.cpp LINK_TO MultiGPUGridder)	
		matlab_add_mex(NAME mexMultiGPUBackProject SRC ${CMAKE_SOURCE_DIR}/Mex/mexMultiGPUBackProject.cpp LINK_TO MultiGPUGridder)	
		matlab_add_mex(NAME mexMultiGPUGetVolume SRC ${CMAKE_SOURCE_DIR}/Mex/mexMultiGPUGetVolume.cpp LINK_TO MultiGPUGridder)	
		matlab_add_mex(NAME mexMultiGPUReconstructVolume SRC ${CMAKE_SOURCE_DIR}/Mex/mexMultiGPUReconstructVolume.cpp LINK_TO MultiGPUGridder)			
		

		if (BUILD_TESTS)
			matlab_add_mex(NAME mexPadVolume SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexPadVolume.cpp LINK_TO MultiGPUGridder)	
			matlab_add_mex(NAME mexAddVolume SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexAddVolume.cpp LINK_TO MultiGPUGridder)	
			matlab_add_mex(NAME mexCropVolume SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexCropVolume.cpp LINK_TO MultiGPUGridder)	
			matlab_add_mex(NAME mexDivideScalar SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexDivideScalar.cpp LINK_TO MultiGPUGridder)	
			matlab_add_mex(NAME mexFFTShift2D SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexFFTShift2D.cpp LINK_TO MultiGPUGridder)	
			matlab_add_mex(NAME mexFFTShift3D SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexFFTShift3D.cpp LINK_TO MultiGPUGridder)	
			matlab_add_mex(NAME mexDivideVolume SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexDivideVolume.cpp LINK_TO MultiGPUGridder)	
			matlab_add_mex(NAME mexMultiplyVolume SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexMultiplyVolume.cpp LINK_TO MultiGPUGridder)	
			matlab_add_mex(NAME mexRealToComplex SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexRealToComplex.cpp LINK_TO MultiGPUGridder)	
			matlab_add_mex(NAME mexComplexToReal SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexComplexToReal.cpp LINK_TO MultiGPUGridder)	
			matlab_add_mex(NAME mexComplexToCAS SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexComplexToCAS.cpp LINK_TO MultiGPUGridder)	
			matlab_add_mex(NAME mexCASToComplex SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexCASToComplex.cpp LINK_TO MultiGPUGridder)	
			message(STATUS "Matlab tests added")
		endif()

	endif()

	if(WIN32)
		matlab_add_mex(NAME mexDeleteGridder SRC ${CMAKE_SOURCE_DIR}/Mex/mexDeleteGridder.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})
		matlab_add_mex(NAME mexCreateGridder SRC ${CMAKE_SOURCE_DIR}/Mex/mexCreateGridder.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})
		matlab_add_mex(NAME mexSetVariables SRC ${CMAKE_SOURCE_DIR}/Mex/mexSetVariables.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})
		matlab_add_mex(NAME mexMultiGPUForwardProject SRC ${CMAKE_SOURCE_DIR}/Mex/mexMultiGPUForwardProject.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})
		matlab_add_mex(NAME mexMultiGPUBackProject SRC ${CMAKE_SOURCE_DIR}/Mex/mexMultiGPUBackProject.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
		matlab_add_mex(NAME mexMultiGPUGetVolume SRC ${CMAKE_SOURCE_DIR}/Mex/mexMultiGPUGetVolume.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
		matlab_add_mex(NAME mexMultiGPUReconstructVolume SRC ${CMAKE_SOURCE_DIR}/Mex/mexMultiGPUReconstructVolume.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
		message(STATUS "Matlab library located")

		if (BUILD_TESTS)
			matlab_add_mex(NAME mexPadVolume SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexPadVolume.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
			matlab_add_mex(NAME mexAddVolume SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexAddVolume.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
			matlab_add_mex(NAME mexCropVolume SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexCropVolume.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
			matlab_add_mex(NAME mexDivideScalar SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexDivideScalar.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
			matlab_add_mex(NAME mexFFTShift2D SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexFFTShift2D.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
			matlab_add_mex(NAME mexFFTShift3D SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexFFTShift3D.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
			matlab_add_mex(NAME mexDivideVolume SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexDivideVolume.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
			matlab_add_mex(NAME mexMultiplyVolume SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexMultiplyVolume.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
			matlab_add_mex(NAME mexRealToComplex SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexRealToComplex.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
			matlab_add_mex(NAME mexComplexToReal SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexComplexToReal.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
			matlab_add_mex(NAME mexComplexToCAS SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexComplexToCAS.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
			matlab_add_mex(NAME mexCASToComplex SRC ${CMAKE_SOURCE_DIR}/Tests/MexWrappers/mexCASToComplex.cpp ${SOURCE_Files} LINK_TO ${CUDA_LIBRARIES})	
			message(STATUS "Matlab tests added")
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
endif()
