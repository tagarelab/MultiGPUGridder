#pragma once

#include "AbstractFilter.h"

class RealToComplexFilter : public AbstractFilter
{
private:
    float *Input;
    cufftComplex *Output;

    int VolumeSize;
    int nSlices;

public:
    // Constructor
    RealToComplexFilter() : AbstractFilter()
    {
        this->Input = NULL;
        this->Output = NULL;

        this->VolumeSize = 0;
        this->nSlices = 0;
    }

    // Deconstructor
    // ~RealToComplexFilter();

    void SetRealInput(float *Input) { this->Input = Input; }

    void SetComplexOutput(cufftComplex *Output) { this->Output = Output; }

    void SetVolumeSize(int VolumeSize) { this->VolumeSize = VolumeSize; }

    void SetNumberOfSlices(int nSlices) { this->nSlices = nSlices; }

    void UpdateFilter(float *Input, cufftComplex *Output, cudaStream_t *stream);

    void Update(cudaStream_t *stream = NULL)
    {
        // Are we using the device pointers?
        if (this->Input != NULL && this->Output != NULL)
        {
            UpdateFilter(this->Input, this->Output, stream);
        }
        // else if (this->d_input_struct != NULL) // Are we using structs for the pointers?
        // {
        //     UpdateFilter(this->d_input_struct->GetPointer(), this->d_output_struct->GetPointer());
        // }
        else
        {
            std::cerr << "RealToComplexFilter(): No valid inputs and/or output found." << '\n';
            return;
        }
    }
};
