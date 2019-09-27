#pragma once

#include "AbstractFilter.h"

class FFTShift2DFilter : public AbstractFilter
{
private:
    cufftComplex *Input;

    int ImageSize;
    int nSlices;

public:

    // Constructor
    FFTShift2DFilter() : AbstractFilter()
    {
        this->Input = NULL;
        this->ImageSize = 0;
        this->nSlices = 0;
    }

    // Deconstructor
    // ~FFTShift2DFilter();

    void SetInput(cufftComplex *Input) { this->Input = Input; }

    void SetImageSize(int ImageSize) { this->ImageSize = ImageSize; }

    void SetNumberOfSlices(int nSlices) { this->nSlices = nSlices; }

    void UpdateFilter(cufftComplex *Input, cudaStream_t *stream);

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
            std::cerr << "FFTShift2DFilter(): No valid inputs found." << '\n';
            return;
        }
    }
};
