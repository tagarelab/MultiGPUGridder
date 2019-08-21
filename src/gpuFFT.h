#include <cstdlib>
#include <stdio.h>
#include <cmath>

#include <iostream>


class gpuFFT
{
private:
    
public:
    gpuFFT();
    ~gpuFFT();



    float *PadVolume(float *inputVol, int inputImgSize, int outputImgSize);



};

