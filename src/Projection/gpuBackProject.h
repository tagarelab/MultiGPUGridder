#pragma once

/**
 * @class   gpuBackProject
 * @brief   A class for back projection on a GPU
 *
 *
 * This class is used for back projection of a volume on a single NVIDIA GPU. It takes the required GPU memory pointers 
 * and associated parameters and runs the back projection kernel on the GPU. 
 * 
 * */

#include "gpuErrorCheck.h"

#include <cstdlib>
#include <stdio.h>
#include <iostream>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h> 

class gpuBackProject
{

public:
    // Constructor
    gpuBackProject() {}

    // Deconstructor
    ~gpuBackProject() {}

    /**
    * Run the back projection kernel on the current NVIDIA GPU. Prior to calling this function, use the cudaSetDevice() function
    * to specify which GPU to run the kernel on. The first four inputs are the pointers to the GPU arrays for the
    * CAS volume, CAS images, the Kaiser Bessel look up table, and the coordinate axes vector.
    * 
    * kerHWidth is the size of the Kaiser Bessel range (default of 2).
    * 
    * nAxes is the number of coordinate axes to process in this kernel call.
    * 
    * CASVolSize is the size of the CAS volume
    * 
    * CASImgSize is the size of the CAS images
    * 
    * maskRadius is the size of the CAS volume mask (default is volume size times interpolation factor divided by 2 and minus 1).
    * 
    * KB_Table_Size is the length of the Kaiser Bessel look up table (default is 501)
    * 
    * Optionally: Pass the pointer to a CUDA stream to assign the kernel call to that stream.
    */
    static void RunKernel(
        float *d_CASVolume, float *d_CASImgs, float *d_KB_Table, float *d_CoordAxes,
        float kerHWidth, int nAxes, int CASVolSize, int CASImgSize,
        int maskRadius, int KB_Table_Size, int extraPadding, cudaStream_t *stream = NULL);
};
