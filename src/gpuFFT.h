#include <cstdlib>
#include <stdio.h>
#include <cmath>

#include <iostream>

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda.h>


class gpuFFT
{
private:
    
public:
    gpuFFT();
    ~gpuFFT();



    float *PadVolume(float *inputVol, int inputImgSize, int outputImgSize);



};

