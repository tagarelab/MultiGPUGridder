#pragma once

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
    cufftHandle inverseFFTPlan;

    bool inverseFFTPlannedFlag;

public:
    // Constructor
    gpuFFT()
    {
        this->inverseFFTPlannedFlag = false;
    };

    // Deconstructor
    ~gpuFFT()
    {
        cufftDestroy(this->inverseFFTPlan);
    };

    ////////// Functions using host memory (i.e. CPU) //////////
    // Convert a volume to CAS volume
    static void VolumeToCAS(float *inputVol, int inputVolSize, float *outputVol, int interpFactor, int extraPadding);

    // Convert a CASImgs array to images
    void CASImgsToImgs(cudaStream_t &stream, int CASImgSize, int ImgSize, float *d_CASImgs, float *d_imgs, cufftComplex *d_CASImgsComplex, int numImgs);

    ////////// Wrappers for calling the CUDA kernels outside the .cu file //////////
    // Pad a GPU array with zeros
    template <typename T>
    static void PadVolume(T *d_inputVol, T *d_outputVol, int inputImgSize, int outputImgSize);

    // Convert a real array to complex type
    static void RealToComplex(float *d_Real, cufftComplex *d_Complex, int imgSize, int nSlices);

    // Run an inplace 2D cufftshift on each slice of a 3D array
    template <typename T>
    static void cufftShift_2D(T *d_Array, int imgSize, int nSlices, cudaStream_t &stream);

    // Run an inplace 3D cufftshift
    template <typename T>
    static void cufftShift_3D(T *d_Complex, int imgSize, int nSlices);

    // Convert a complex array type to real
    template <typename R>
    static void ComplexImgsToCAS(cufftComplex *d_Complex, R *d_Real, int imgSize);

    // Run a forward FFT
    static void FowardFFT(cufftComplex *d_Complex, int imgSize);

    // Convert a CAS array to complex
    static void CASImgsToComplex(float *d_CASImgs, cufftComplex *d_CASImgsComplex, int imgSize, int nSlices, cudaStream_t &stream);

    // Crop a complex array and normalize the array to remove the FFT forward scaling factor
    static void ComplexToCroppedNormalized(cufftComplex *d_Complex, float *d_imgs, int ComplexImgSize, int imgSize, int nSlices, cudaStream_t &stream);

};
