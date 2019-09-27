#pragma once

#include <cstdlib>
#include <stdio.h>
#include <iostream>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h> // Library for CUDA FFT and inverse FFT functions see https://www.bu.edu/pasi/files/2011/07/Lecture83.pdf

class gpuBackProject
{

public:
    // Constructor
    gpuBackProject() {}

    // Deconstructor
    ~gpuBackProject() {}

    // Run the forward projection
    static void RunKernel(
        float *d_CASVolume, float *d_CASImgs, float *d_KB_Table, float *d_CoordAxes,
        float *d_PlaneDensity, float kerHWidth, int nAxes, int GridSize, int BlockSize,
        int CASVolSize, int CASImgSize, int maskRadius, int KB_Table_Size, cudaStream_t *stream = NULL);
};
