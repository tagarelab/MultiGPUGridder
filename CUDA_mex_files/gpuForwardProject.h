//#include "mex.h"
#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <vector>

// includes CUDA Runtime
#include <cuda_runtime.h>

#include <cuda.h>

// NVTX labeling tools (for the nvidia profiling)
#include <nvToolsExt.h>
#include <cuda_profiler_api.h>

#ifndef __GPUFORWARDPROJECT_H__
#define __GPUFORWARDPROJECT_H__

extern void gpuForwardProject(
    std::vector<float*> gpuVol_Vector, std::vector<float*> gpuCASImgs_Vector,       // Vector of GPU array pointers
    std::vector<float*> gpuCoordAxes_Vector, std::vector<float*> ker_bessel_Vector, // Vector of GPU array pointers
    float * CASImgs_CPU_Pinned, float * coordAxes_CPU_Pinned, // Pointers to pinned CPU arrays for input / output
    int volSize, int imgSize, int nAxes, float maskRadius, int kerSize, float kerHWidth, // kernel Parameters and constants
    int numGPUs, int nStreams, int gridSize, int blockSize // Streaming parameters
);


#endif //__GPUFORWARDPROJECT_H__