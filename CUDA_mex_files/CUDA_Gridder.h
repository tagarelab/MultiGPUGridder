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

	// Create a CPU_CUDA_Memory class object for allocating, copying, and transferring array to  CPU, GPU, or Matlab memory
	CPU_CUDA_Memory * Mem_obj;
	
	// Constructor/Destructor
	CUDA_Gridder();
	~CUDA_Gridder(){};

	// Run the forward projection CUDA kernel
	void Forward_Project(std::vector<std::string> Input_Strings);

};

#endif