#pragma once

/**
 * @class   AddVolumeFilter
 * @brief   A filter for adding two GPU arrays.
 *
 * AddVolumeFilter inherits from the AbstractFilter class.
 * 
 * This class simply adds two GPU arrays together with the following formula: VolumeOne = VolumeOne + VolumeTwo
 * 
 * In order to set which GPU the filter should be run on, please use the cudaDeviceSet() function before calling the Update function.
 * 
 * */

#include "AbstractFilter.h"

class AddVolumeFilter : public AbstractFilter
{
private:
    float *d_VolumeOne;
    float *d_VolumeTwo;
    int VolumeSize;
    int nSlices;

public:
    // Constructor
    AddVolumeFilter() : AbstractFilter()
    {
        this->d_VolumeOne = NULL;
        this->d_VolumeTwo = NULL;

        this->VolumeSize = 0;
        this->nSlices = 0;
    }

    /// Set the first volume by passing the GPU memory pointer
    void SetVolumeOne(float *d_VolumeOne) { this->d_VolumeOne = d_VolumeOne; }

    /// Set the second volume by passing the GPU memory pointer    
    void SetVolumeTwo(float *d_VolumeTwo) { this->d_VolumeTwo = d_VolumeTwo; }

    /// Set the length of the array. If added stacks of 2D images, this is the length along the X or Y dimension (which must be equal). 
    void SetVolumeSize(int VolumeSize) { this->VolumeSize = VolumeSize; }

    /// If adding stacks of 2D imgaes, this is the number of 2D slices.
    void SetNumberOfSlices(int nSlices) { this->nSlices = nSlices; }

    /// Run the AddVolume CUDA kernel
    void UpdateFilter(float *Input, float *Output, cudaStream_t *stream = NULL);

    /// Update the AddVolume filter
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
            std::cerr << "AddVolumeFilter(): No valid inputs found." << '\n';
            return;
        }
    }
};
