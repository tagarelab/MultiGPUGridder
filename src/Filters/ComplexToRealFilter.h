#pragma once

/**
 * @class   ComplexToRealFilter
 * @brief   A filter for extracting the real component of a cufftComplex array.
 *
 * ComplexToRealFilter inherits from the AbstractFilter class.
 * 
 * This class extracts the real component of a cufftComplex array on the GPU. This is needed for 
 * obtaining the output of the forward and inverse Fourier transform operations.
 * 
 * In order to set which GPU the filter should be run on, please use the cudaDeviceSet() function before calling the Update function.
 * 
 * */

#include "AbstractFilter.h"

class ComplexToRealFilter : public AbstractFilter
{
private:
    cufftComplex *Input;
    float *Output;

    int VolumeSize;
    int nSlices;

public:
    // Constructor
    ComplexToRealFilter() : AbstractFilter()
    {
        this->Input = NULL;
        this->Output = NULL;

        this->VolumeSize = 0;
        this->nSlices = 0;
    }

    /// Set the complex cufftComplex input
    void SetComplexInput(cufftComplex *Input) { this->Input = Input; }

    /// Set the output array which will be the real component of the complex input array
    void SetRealOutput(float *Output) { this->Output = Output; }

    /// Set the length of the array. If using stacks of 2D images, this is the length along the X or Y dimension (which must be equal).
    void SetVolumeSize(int VolumeSize) { this->VolumeSize = VolumeSize; }

    /// If using a stack of 2D images, this is the number of images.
    void SetNumberOfSlices(int nSlices) { this->nSlices = nSlices; }

    /// Run the CUDA kernel
    void UpdateFilter(cufftComplex *Input, float *Output, cudaStream_t *stream);

    /// Update the filter
    void Update(cudaStream_t *stream = NULL)
    {
        // Are we using the device pointers?
        if (this->Input != NULL)
        {
            UpdateFilter(this->Input, this->Output, stream);
        }
        else
        {
            std::cerr << "ComplexToRealFilter(): No valid inputs found." << '\n';
            return;
        }
    }
};
