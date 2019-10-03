#pragma once

/**
 * @class   RealToComplexFilter
 * @brief   A filter for converting an array to cufftComplex type.
 *
 * RealToComplexFilter inherits from the AbstractFilter class.
 * 
 * This class copies the elements from an array and set as the real components 
 * of the output cufftComplex array on the GPU. The cufftComplex type is needed for
 * using the cufft CUDA library for the forward and inverse Fourier transform operations.
 * 
 * In order to set which GPU the filter should be run on, please use the cudaDeviceSet() function before calling the Update function.
 * 
 * */

#include "AbstractFilter.h"

class RealToComplexFilter : public AbstractFilter
{
private:
    float *Input;
    cufftComplex *Output;

    int VolumeSize;
    int nSlices;

public:
    // Constructor
    RealToComplexFilter() : AbstractFilter()
    {
        this->Input = NULL;
        this->Output = NULL;

        this->VolumeSize = 0;
        this->nSlices = 0;
    }
    /// Set the real input array
    void SetRealInput(float *Input) { this->Input = Input; }

    /// Set the output cufftComplex array
    void SetComplexOutput(cufftComplex *Output) { this->Output = Output; }

    /// Set the volume size
    void SetVolumeSize(int VolumeSize) { this->VolumeSize = VolumeSize; }

    /// If using a stack of 2D images, this is the number of images.
    void SetNumberOfSlices(int nSlices) { this->nSlices = nSlices; }

    /// Update the filter
    void Update(cudaStream_t *stream = NULL)
    {
        // Are we using the device pointers?
        if (this->Input != NULL && this->Output != NULL)
        {
            UpdateFilter(this->Input, this->Output, stream);
        }
        else
        {
            std::cerr << "RealToComplexFilter(): No valid inputs and/or output found." << '\n';
            return;
        }
    }

private:
    /// Run the CUDA kernel
    void UpdateFilter(float *Input, cufftComplex *Output, cudaStream_t *stream);
};
