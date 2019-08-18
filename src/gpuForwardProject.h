#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <vector>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda.h>

#include <cufft.h> // Library for CUDA FFT and inverse FFT functions see https://www.bu.edu/pasi/files/2011/07/Lecture83.pdf

// NVTX labeling tools (for the nvidia profiling)
//#include <nvToolsExt.h>
//#include <cuda_profiler_api.h>

#ifndef __GPUFORWARDPROJECT_H__
#define __GPUFORWARDPROJECT_H__

// Function for creating the CUDA streams and launchinh the forward projection kernel
extern void gpuForwardProject(
    std::vector<float *> gpuVol_Vector, std::vector<float *> gpuCASImgs_Vector,          // Vector of GPU array pointers
    std::vector<float *> gpuCoordAxes_Vector, std::vector<float *> ker_bessel_Vector,    // Vector of GPU array pointers
    // std::vector<cufftComplex *> gpuComplexImgs_Vector,                                   // Vector of GPU array pointers
    // std::vector<cufftComplex *> gpuComplexImgs_Shifted_Vector,                           // Vector of GPU array pointers
    float *CASImgs_CPU_Pinned, float *coordAxes_CPU_Pinned,                              // Pointers to pinned CPU arrays for input / output
    int volSize, int imgSize, int nAxes, float maskRadius, int kerSize, float kerHWidth, // kernel Parameters and constants
    int numGPUs, int nStreams, int gridSize, int blockSize, int nBatches,                 // Streaming parameters
	int MaxAxesAllocated
	);

#endif //__GPUFORWARDPROJECT_H__
