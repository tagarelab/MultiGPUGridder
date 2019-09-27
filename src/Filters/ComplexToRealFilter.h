#pragma once

#include "AbstractFilter.h"

class ComplexToRealFilter : public AbstractFilter
{
private:
    cufftComplex *Input;
    float *Output;

    int VolumeSize;
    int nSlices;

public:
    // Constructor
    ComplexToRealFilter() : AbstractFilter()
    {
        this->Input = NULL;
        this->Output = NULL;

        this->VolumeSize = 0;
        this->nSlices = 0;
    }

    // Deconstructor
    // ~ComplexToRealFilter();

    void SetComplexInput(cufftComplex *Input) { this->Input = Input; }

    void SetRealOutput(float *Output) { this->Output = Output; }

    void SetVolumeSize(int VolumeSize) { this->VolumeSize = VolumeSize; }

    void SetNumberOfSlices(int nSlices) { this->nSlices = nSlices; }

    void UpdateFilter(cufftComplex *Input, float *Output, cudaStream_t *stream);

    void Update(cudaStream_t *stream = NULL)
    {
        // Are we using the device pointers?
        if (this->Input != NULL)
        {
            UpdateFilter(this->Input, this->Output, stream);
        }
        // else if (this->d_input_struct != NULL) // Are we using structs for the pointers?
        // {
        //     UpdateFilter(this->d_input_struct->GetPointer(), this->d_output_struct->GetPointer());
        // }
        else
        {
            std::cerr << "ComplexToRealFilter(): No valid inputs found." << '\n';
            return;
        }
    }
};
