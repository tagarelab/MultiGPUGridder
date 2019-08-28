#include <cstdlib>
#include <stdio.h>
#include <cmath>

#include <iostream>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda.h>

#include <cufft.h> // Library for CUDA FFT and inverse FFT functions see https://www.bu.edu/pasi/files/2011/07/Lecture83.pdf



class gpuFFT
{
private:
    
public:
    gpuFFT();
    ~gpuFFT();

    static void PadVolume(float *inputVol, float* outputVol, int inputImgSize, int outputImgSize);

    static void VolumeToCAS(float* inputVol, int inputVolSize, float* outputVol, int interpFactor, int extraPadding);

    static void CASImgsToImgs(cudaStream_t& stream, int gridSize, int blockSize, int CASImgSize, int ImgSize, float* d_CASImgs, float* d_imgs, cufftComplex* d_CASImgsComplex, int numImgs);

};

