#pragma once

/**
 * @class   AbstractFilter
 * @brief   An interface class for GPU filters.
 *
 * AbstractFilter is the parent class for all the GPU filters which apply some calculation to a GPU array.
 * 
 * Host arrays can also be passed to AbstractFilter for filtering using the GPU. AbstractFilter will allocated GPU memory
 * and then copy the host array to the GPU as needed.
 * 
 * In order to set which GPU the filter should be run on, please use the cudaDeviceSet() function before calling the Update function.
 * 
 * */

#include "HostMemory.h"
#include "HostMemoryGPU.h"

#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <vector>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda.h>

#include <cufft.h> // Library for CUDA FFT and inverse FFT functions see https://www.bu.edu/pasi/files/2011/07/Lecture83.pdf

// If on a Linux machine
#if defined(__GNUC__)
#include <nvToolsExt.h> // NVTX labeling tools (for the nvidia profiling)
#endif

#include <cuda_profiler_api.h>

class AbstractFilter
{
protected:

    int dims;

    HostMemoryGPU<float> *d_input_struct;
    HostMemoryGPU<float> *d_output_struct;

    HostMemory<float> *h_input_struct;
    HostMemory<float> *h_output_struct;

    float *d_input_ptr;
    float *d_output_ptr;

public:
    // Constructor
    AbstractFilter()
    {

        this->dims = 3;

        this->d_input_struct = NULL;
        this->d_output_struct = NULL;
        this->h_input_struct = NULL;
        this->h_output_struct = NULL;
        this->d_input_ptr = NULL;
        this->d_output_ptr = NULL;
    };

    // Deconstructor
    ~AbstractFilter(){

    };

    void SetCPUInput(float *input, int *InputSize, int GPU_Device);

    void SetCPUOutput(float *output, int *OutputSize, int GPU_Device);

    void GetCPUOutput(float *output);
};
