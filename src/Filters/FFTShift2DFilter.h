#pragma once

/**
 * @class   FFTShift2DFilter
 * @brief   A filter for 2D FFT shift on a GPU array.
 *
 * FFTShift2DFilter inherits from the AbstractFilter class.
 * 
 * The FFTShift2DFilter runs a 2D FFT shift by iterating over each 2D slice of a 3D array.
 * 
 * In order to set which GPU the filter should be run on, please use the cudaDeviceSet() function before calling the Update function.
 * 
 * */

#include "AbstractFilter.h"

class FFTShift2DFilter : public AbstractFilter
{
private:
    cufftComplex *Input;

    int ImageSize;
    int nSlices;

public:
    // Constructor
    FFTShift2DFilter() : AbstractFilter()
    {
        this->Input = NULL;
        this->ImageSize = 0;
        this->nSlices = 0;
    }

    /// Set the input array
    void SetInput(cufftComplex *Input) { this->Input = Input; }

    /// Set the size of the image
    void SetImageSize(int ImageSize) { this->ImageSize = ImageSize; }

    /// Set the number of 2D images in the array
    void SetNumberOfSlices(int nSlices) { this->nSlices = nSlices; }

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
            std::cerr << "FFTShift2DFilter(): No valid inputs found." << '\n';
            return;
        }
    }

private:
    /// Run the CUDA kernel
    void UpdateFilter(cufftComplex *Input, cudaStream_t *stream);
};
