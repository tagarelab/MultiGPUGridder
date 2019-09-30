#pragma once

/**
 * @class   CASToComplexFilter
 * @brief   A filter for converting a GPU array to CAS type.
 *
 * CASToComplexFilter inherits from the AbstractFilter class.
 * 
 * This class converts a CAS type array to a complex cufftComplex array. The complex arrays are need for
 * running the forward and inverse Fourier transform with the cufft CUDA library.
 * 
 * In order to set which GPU the filter should be run on, please use the cudaDeviceSet() function before calling the Update function.
 * 
 * */

#include "AbstractFilter.h"

class CASToComplexFilter : public AbstractFilter
{
private:
    float *CASVolume;
    cufftComplex *ComplexVolume;

    int VolumeSize;
    int nSlices;

public:
    // Constructor
    CASToComplexFilter() : AbstractFilter()
    {
        this->CASVolume = NULL;
        this->ComplexVolume = NULL;

        this->VolumeSize = 0;
        this->nSlices = 0;
    }

    /// Set the CAS volume (the input array)
    void SetCASVolume(float *CASVolume) { this->CASVolume = CASVolume; }

    /// Set the output complex array
    void SetComplexOutput(cufftComplex *ComplexVolume) { this->ComplexVolume = ComplexVolume; }

    /// Set the length of the array. If using stacks of 2D images, this is the length along the X or Y dimension (which must be equal).
    void SetVolumeSize(int VolumeSize) { this->VolumeSize = VolumeSize; }

    /// If using a stack of 2D images, this is the number of images.
    void SetNumberOfSlices(int nSlices) { this->nSlices = nSlices; }

    /// Update the filter
    void Update(cudaStream_t *stream = NULL)
    {
        // Are we using the device pointers?
        if (this->CASVolume != NULL)
        {
            UpdateFilter(this->CASVolume, this->ComplexVolume, stream);
        }
        else
        {
            std::cerr << "CASToComplexFilter(): No valid inputs found." << '\n';
            return;
        }
    }

private:
    /// Run the CUDA kernel
    void UpdateFilter(float *Input, cufftComplex *Output, cudaStream_t *stream);
};
