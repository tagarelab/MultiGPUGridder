#include <cstdlib>
#include <stdio.h>
#include <cmath>

#include <iostream>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda.h>

#include <cufft.h> // Library for CUDA FFT and inverse FFT functions see https://www.bu.edu/pasi/files/2011/07/Lecture83.pdf
//#include <cutil_inline.h>


class gpuFFT
{
private:
    
public:
    gpuFFT();
    ~gpuFFT();



    float *PadVolume(float *inputVol, int inputImgSize, int outputImgSize);


    float* VolumeToCAS(float* inputVol, int inputVolSize, int interpFactor, int extraPadding);

};

