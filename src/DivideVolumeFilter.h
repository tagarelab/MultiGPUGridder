#pragma once

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

    // Deconstructor
    // ~DivideVolumeFilter();

    void SetVolumeOne(float *d_VolumeOne) { this->d_VolumeOne = d_VolumeOne; }
    void SetVolumeTwo(float *d_VolumeTwo) { this->d_VolumeTwo = d_VolumeTwo; }

    void SetVolumeSize(int VolumeSize) { this->VolumeSize = VolumeSize; }

    void SetNumberOfSlices(int nSlices) { this->nSlices = nSlices; }

    void UpdateFilter(float *Input, float *Output, cudaStream_t *stream = NULL);

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
};
