#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <vector>

#include "gpuGridder.h"

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda.h>

#include <cufft.h> // Library for CUDA FFT and inverse FFT functions see https://www.bu.edu/pasi/files/2011/07/Lecture83.pdf

// NVTX labeling tools (for the nvidia profiling)
//#include <nvToolsExt.h>
//#include <cuda_profiler_api.h>

#ifndef __GPUFORWARDPROJECT_H__
#define __GPUFORWARDPROJECT_H__

// Forward declaration of the gpu gridder class
class gpuGridder;

// Function for creating the CUDA streams and launchinh the forward projection kernel
static void gpuForwardProject(gpuGridder * gridder);

#endif //__GPUFORWARDPROJECT_H__
