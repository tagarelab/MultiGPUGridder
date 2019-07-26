#include "mex.h"
#include <cstdlib>
#include <stdio.h>
#include <iostream>


// includes CUDA Runtime
#include <cuda_runtime.h>

#include <cuda.h>

// NVTX labeling tools (for the nvidia profiling)
#include <nvToolsExt.h>
#include <cuda_profiler_api.h>

#ifndef __GPUFORWARDPROJECT_H__
#define __GPUFORWARDPROJECT_H__

extern void gpuForwardProject();

#endif //__GPUFORWARDPROJECT_H__