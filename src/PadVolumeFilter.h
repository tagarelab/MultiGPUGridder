#pragma once

#include "AbstractFilter.h"

class PadVolumeFilter : public AbstractFilter
{
private:
    // How much to pad in each dimension?
    int paddingX;
    int paddingY;
    int paddingZ;

    int InputSizeX;
    int InputSizeY;
    int InputSizeZ;

    float *Input;
    float *Output;

public:
    // Constructor
    PadVolumeFilter() : AbstractFilter()
    {
        this->paddingX = 0;
        this->paddingY = 0;
        this->paddingZ = 0;
        this->InputSizeX = 0;
        this->InputSizeY = 0;
        this->InputSizeZ = 0;
        this->Input = NULL;
        this->Output = NULL;
    }

    // ~PadVolumeFilter();

    void SetInput(float *Input) { this->Input = Input; }

    void SetOutput(float *Output) { this->Output = Output; }

    void SetInputSize(int InputSize)
    {
        this->InputSizeX = InputSize;
        this->InputSizeY = InputSize;
        this->InputSizeZ = InputSize;
    }

    void SetNumberOfSlices(int nSlices)
    {
        this->InputSizeZ = nSlices;
    }

    // Set padding along the each dimension of the array
    void SetPaddingX(int paddingX) { this->paddingX = paddingX; }
    void SetPaddingY(int paddingY) { this->paddingY = paddingY; }
    void SetPaddingZ(int paddingZ) { this->paddingZ = paddingZ; }

    // Set padding in all three dimensions of the array
    void SetPadding(int padding)
    {
        this->paddingX = padding;
        this->paddingY = padding;
        this->paddingZ = padding;
    }

    // template <typename T>
    void UpdateFilter(float *Input, float *Output, cudaStream_t *stream = NULL);

    void Update(cudaStream_t *stream = NULL)
    {
        // Has the input and output been set?
        if (this->Input != NULL && this->Output != NULL)
        {
            this->UpdateFilter(this->Input, this->Output, stream);
        }
        else if (this->d_input_struct != NULL) // Are we using structs for the pointers?
        {
            this->UpdateFilter(this->d_input_struct->GetPointer(), this->d_output_struct->GetPointer(), stream);
        }
        else
        {
            std::cerr << "PadVolumeFilter(): No valid inputs and/or outputs found." << '\n';
            return;
        }
    }
};
