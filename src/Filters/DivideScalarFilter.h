#pragma once

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

    // Deconstructor
    // ~DivideScalarFilter();

    void SetInput(float *Input) { this->Input = Input; }

    void SetScalar(float Scalar) { this->Scalar = Scalar; }

    void SetVolumeSize(int VolumeSize) { this->VolumeSize = VolumeSize; }

    void SetNumberOfSlices(int nSlices) { this->nSlices = nSlices; }

    void UpdateFilter(float *Input, cudaStream_t *stream);

    void Update(cudaStream_t *stream = NULL)
    {
        // Are we using the device pointers?
        if (this->Input != NULL)
        {
            UpdateFilter(this->Input, stream);
        }
        // else if (this->d_input_struct != NULL) // Are we using structs for the pointers?
        // {
        //     UpdateFilter(this->d_input_struct->GetPointer(), this->d_output_struct->GetPointer());
        // }
        else
        {
            std::cerr << "DivideScalarFilter(): No valid inputs found." << '\n';
            return;
        }
    }
};
