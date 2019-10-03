#pragma once

/**
 * @class   ComplexToCASFilter
 * @brief   A filter for converting a cufftComplex GPU array to CAS type.
 *
 * ComplexToCASFilter inherits from the AbstractFilter class.
 * 
 * This class converts a complex GPU array to CAS type. The gpuForwardProject and gpuBackProject
 * classes need the arrays to be CAS type. The main advantage of using CAS for the projection is reduced
 * memory size and faster computation speed.
 * 
 * In order to set which GPU the filter should be run on, please use the cudaDeviceSet() function before calling the Update function.
 * 
 * */

#include "AbstractFilter.h"

class ComplexToCASFilter : public AbstractFilter
{
private:
    float *CASVolume;
    cufftComplex *ComplexVolume;

    int VolumeSize;
    int nSlices;

public:
    // Constructor
    ComplexToCASFilter() : AbstractFilter()
    {
        this->CASVolume = NULL;
        this->ComplexVolume = NULL;

        this->VolumeSize = 0;
        this->nSlices = 0;
    }

    /// Set the complex array input
    void SetComplexInput(cufftComplex *ComplexVolume) { this->ComplexVolume = ComplexVolume; }

    /// Set the output array
    void SetCASVolumeOutput(float *CASVolume) { this->CASVolume = CASVolume; }

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
        if (this->ComplexVolume != NULL && this->CASVolume != NULL)
        {
            UpdateFilter(this->ComplexVolume, this->CASVolume, stream);
        }
        else
        {
            std::cerr << "ComplexToCASFilter(): No valid inputs and/or outputs found." << '\n';
            return;
        }
    }
};
