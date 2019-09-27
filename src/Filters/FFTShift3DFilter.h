#pragma once

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

    // Deconstructor
    // ~FFTShift3DFilter();

    void SetInput(cufftComplex *Input) { this->Input = Input; }

    void SetVolumeSize(int VolumeSize) { this->VolumeSize = VolumeSize; }

    void UpdateFilter(cufftComplex *Input, cudaStream_t *stream);

    void Update(cudaStream_t *stream = NULL)
    {
        // Are we using the device pointers?
        if (this->Input != NULL )
        {
            UpdateFilter(this->Input, stream);
        }
        // else if (this->d_input_struct != NULL) // Are we using structs for the pointers?
        // {
        //     UpdateFilter(this->d_input_struct->GetPointer(), this->d_output_struct->GetPointer());
        // }
        else
        {
            std::cerr << "FFTShift3DFilter(): No valid input found." << '\n';
            return;
        }
    }
};
