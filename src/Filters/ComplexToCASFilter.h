#pragma once

#include "AbstractFilter.h"

class ComplexToCASFilter : public AbstractFilter
{
private:
    float *CASVolume;
    cufftComplex *ComplexVolume;

    int VolumeSize;
    int nSlices;

public:
    // Constructor
    ComplexToCASFilter() : AbstractFilter()
    {
        this->CASVolume = NULL;
        this->ComplexVolume = NULL;

        this->VolumeSize = 0;
        this->nSlices = 0;
    }

    // Deconstructor
    // ~ComplexToCASFilter();

    void SetComplexInput(cufftComplex *ComplexVolume) { this->ComplexVolume = ComplexVolume; }

    void SetCASVolumeOutput(float *CASVolume) { this->CASVolume = CASVolume; }

    void SetVolumeSize(int VolumeSize) { this->VolumeSize = VolumeSize; }

    void SetNumberOfSlices(int nSlices) { this->nSlices = nSlices; }

    void UpdateFilter(cufftComplex *Input, float *Output, cudaStream_t *stream);

    void Update(cudaStream_t *stream = NULL)
    {
        // Are we using the device pointers?
        if (this->ComplexVolume != NULL && this->CASVolume != NULL)
        {
            UpdateFilter(this->ComplexVolume, this->CASVolume, stream);
        }
        // else if (this->d_input_struct != NULL) // Are we using structs for the pointers?
        // {
        //     UpdateFilter(this->d_input_struct->GetPointer(), this->d_output_struct->GetPointer());
        // }
        else
        {
            std::cerr << "ComplexToCASFilter(): No valid inputs and/or outputs found." << '\n';
            return;
        }
    }
};
