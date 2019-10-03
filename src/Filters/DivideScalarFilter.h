#pragma once

/**
 * @class   DivideScalarFilter
 * @brief   A filter for dividing a GPU array with a scalar.
 *
 * DivideScalarFilter inherits from the AbstractFilter class.
 * 
 * The DivideScalarFilter divides a GPU array with some scalar value
 * 
 * In order to set which GPU the filter should be run on, please use the cudaDeviceSet() function before calling the Update function.
 * 
 * */

#include "AbstractFilter.h"

class DivideScalarFilter : public AbstractFilter
{
private:
    float *Input;
    float Scalar;
    int VolumeSize;
    int nSlices;

public:
    // Constructor
    DivideScalarFilter() : AbstractFilter()
    {
        this->Input = NULL;
        this->Scalar = 0;
        this->VolumeSize = 0;
        this->nSlices = 0;
    }

    /// Set the input array
    void SetInput(float *Input) { this->Input = Input; }

    /// Set the float scalar to divide the array by
    void SetScalar(float Scalar) { this->Scalar = Scalar; }

    /// Set the volume size
    void SetVolumeSize(int VolumeSize) { this->VolumeSize = VolumeSize; }

    /// If using a stack of 2D images, this is the number of images.
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
            std::cerr << "DivideScalarFilter(): No valid inputs found." << '\n';
            return;
        }
    }

private:
    /// Run the CUDA kernel
    void UpdateFilter(float *Input, cudaStream_t *stream);
};
