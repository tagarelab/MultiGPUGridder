#pragma once

/**
 * @class   DivideVolumeFilter
 * @brief   A filter for dividing two GPU arrays.
 *
 * DivideVolumeFilter inherits from the AbstractFilter class.
 * 
 * The DivideVolumeFilter divides two GPU arrays together with the following formula:
 * VolumeOne = VolumeOne / VolumeTwo
 * 
 * In order to set which GPU the filter should be run on, please use the cudaDeviceSet() function before calling the Update function.
 * 
 * */

#include "AbstractFilter.h"

class DivideVolumeFilter : public AbstractFilter
{
private:
    float *d_VolumeOne;
    float *d_VolumeTwo;
    int VolumeSize;
    int nSlices;

public:
    // Constructor
    DivideVolumeFilter() : AbstractFilter()
    {
        this->d_VolumeOne = NULL;
        this->d_VolumeTwo = NULL;

        this->VolumeSize = 0;
        this->nSlices = 0;
    }

    /// Set the first volume
    void SetVolumeOne(float *d_VolumeOne) { this->d_VolumeOne = d_VolumeOne; }

    /// Set the second volume
    void SetVolumeTwo(float *d_VolumeTwo) { this->d_VolumeTwo = d_VolumeTwo; }

    /// Set the volume size
    void SetVolumeSize(int VolumeSize) { this->VolumeSize = VolumeSize; }

    /// If using a stack of 2D images, this is the number of images.
    void SetNumberOfSlices(int nSlices) { this->nSlices = nSlices; }

    /// Update the filter
    void Update(cudaStream_t *stream = NULL)
    {
        // Are we using the device pointers?
        if (this->d_VolumeOne != NULL)
        {
            UpdateFilter(this->d_VolumeOne, this->d_VolumeTwo, stream);
        }
        else if (this->d_input_struct != NULL) // Are we using structs for the pointers?
        {
            UpdateFilter(this->d_input_struct->GetPointer(), this->d_output_struct->GetPointer(), stream);
        }
        else
        {
            std::cerr << "DivideVolumeFilter(): No valid inputs found." << '\n';
            return;
        }
    }

private:
    /// Run the CUDA kernel
    void UpdateFilter(float *Input, float *Output, cudaStream_t *stream = NULL);
};
