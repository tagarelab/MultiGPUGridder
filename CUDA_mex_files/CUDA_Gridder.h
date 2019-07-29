#ifndef CUDA_GRIDDER_H // Only define the header once
#define CUDA_GRIDDER_H

#include "CPU_CUDA_Memory.h"
#include "gpuForwardProject.h"


#include <iostream>
#include <vector>
#include <limits>

// Include the CUDA Runtime
#include <cuda_runtime.h>
#include <cuda.h>

// NVTX labeling tools (for the nvidia profiling)
#include <nvToolsExt.h>
#include <cuda_profiler_api.h>


class CUDA_Gridder {

public:

	// Output image size parameter
	int* imgSize;

	// Mask radius parameter
	float* maskRadius;

	// Size of the coordinate axes
	int* axesSize;

	// Size of the volume
	int* volSize;
	
	// Size of the Kaiser bessel vector
	int kerSize = 501;    

	// Width of the Kaiser bessel function
	float kerHWidth = 2;

	// Create a CPU_CUDA_Memory class object for allocating, copying, and transferring array to  CPU, GPU, or Matlab memory
	CPU_CUDA_Memory * Mem_obj;
	
	// Constructor/Destructor
	CUDA_Gridder();
	~CUDA_Gridder(){};

	// Set GPU volume 
	void SetVolume( float* gpuVol, int* gpuVolSize);

	// Set coordinate axes
	void SetAxes(float* coordAxes, int* axesSize);

	// Set the output image size parameter
	void SetImgSize(int* imgSize);

	// Set the maskRadius parameter
	void SetMaskRadius(float* maskRadius);

	// Set the Kaiser Bessel Function vector
	void SetKaiserBesselFunction();

	// Run the forward projection CUDA kernel
	void Forward_Project();

};

#endif