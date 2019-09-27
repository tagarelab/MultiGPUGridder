#pragma once

#include "AbstractFilter.h"

class CASToComplexFilter : public AbstractFilter
{
private:
    float *CASVolume;
    cufftComplex *ComplexVolume;

    int VolumeSize;
    int nSlices;

public:

    // Constructor
    CASToComplexFilter() : AbstractFilter()
    {
        this->CASVolume = NULL;
        this->ComplexVolume = NULL;

        this->VolumeSize = 0;
        this->nSlices = 0;
    }

    // Deconstructor
    // ~CASToComplexFilter();

    void SetCASVolume(float *CASVolume) { this->CASVolume = CASVolume; }

    void SetComplexOutput(cufftComplex *ComplexVolume) { this->ComplexVolume = ComplexVolume; }

    void SetVolumeSize(int VolumeSize) { this->VolumeSize = VolumeSize; }

    void SetNumberOfSlices(int nSlices) { this->nSlices = nSlices; }

    void UpdateFilter(float *Input, cufftComplex *Output, cudaStream_t *stream);

    void Update(cudaStream_t *stream = NULL)
    {
        // Are we using the device pointers?
        if (this->CASVolume != NULL)
        {
            UpdateFilter(this->CASVolume, this->ComplexVolume, stream);
        }
        // else if (this->d_input_struct != NULL) // Are we using structs for the pointers?
        // {
        //     UpdateFilter(this->d_input_struct->GetPointer(), this->d_output_struct->GetPointer());
        // }
        else
        {
            std::cerr << "CASToComplexFilter(): No valid inputs found." << '\n';
            return;
        }
    }
};
