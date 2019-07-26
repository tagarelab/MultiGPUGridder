#ifndef CUDA_GRIDDER_H // Only define the header once
#define CUDA_GRIDDER_H

#include "CPU_CUDA_Memory.h"
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



	// Create a CPU_CUDA_Memory class object for allocating, copying, and transferring the CPU, GPU, and Matlab memory
	CPU_CUDA_Memory * Mem_obj;


public:

	// Constructor/Destructor
	//CUDA_Gridder_Initilize();
	//~CUDA_Gridder_Destruct();






};

#endif