#pragma once

#include "MemoryStruct.h"
#include "MemoryStructGPU.h"

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

    MemoryStructGPU<float> *d_input_struct;
    MemoryStructGPU<float> *d_output_struct;

    MemoryStruct<float> *h_input_struct;
    MemoryStruct<float> *h_output_struct;

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
