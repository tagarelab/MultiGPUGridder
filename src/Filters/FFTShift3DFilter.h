#pragma once

/**
 * @class   FFTShift3DFilter
 * @brief   A filter for 3D FFT shift on a GPU array.
 *
 * FFTShift3DFilter inherits from the AbstractFilter class.
 * 
 * In order to set which GPU the filter should be run on, please use the cudaDeviceSet() function before calling the Update function.
 * 
 * */

#include "AbstractFilter.h"

class FFTShift3DFilter : public AbstractFilter
{
private:
    cufftComplex *Input;

    int VolumeSize;

public:
    // Constructor
    FFTShift3DFilter() : AbstractFilter()
    {
        this->Input = NULL;
        this->VolumeSize = 0;
    }

    /// Set the input GPU array
    void SetInput(cufftComplex *Input) { this->Input = Input; }

    /// Set the size of the GPU array
    void SetVolumeSize(int VolumeSize) { this->VolumeSize = VolumeSize; }

    /// Update the filter
    void Update(cudaStream_t *stream = NULL)
    {
        // Are we using the device pointers?
        if (this->Input != NULL)
        {
            UpdateFilter(this->Input, stream);
        }
        else
        {
            std::cerr << "FFTShift3DFilter(): No valid input found." << '\n';
            return;
        }
    }

private:
    /// Run the CUDA kernel
    void UpdateFilter(cufftComplex *Input, cudaStream_t *stream);
};
