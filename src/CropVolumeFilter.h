#pragma once

#include "AbstractFilter.h"

class CropVolumeFilter : public AbstractFilter
{
private:
    // How much to pad in each dimension?
    int CropX;
    int CropY;
    int CropZ;

    int InputSizeX;
    int InputSizeY;
    int InputSizeZ;

    float *Input;
    float *Output;

public:
    // Constructor
    CropVolumeFilter() : AbstractFilter()
    {
        this->CropX = 0;
        this->CropY = 0;
        this->CropZ = 0;
        this->InputSizeX = 0;
        this->InputSizeY = 0;
        this->InputSizeZ = 0;
        this->Input = NULL;
        this->Output = NULL;
    }

    // ~CropVolumeFilter();

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
    void SetCropX(int CropX) { this->CropX = CropX; }
    void SetCropY(int CropY) { this->CropY = CropY; }
    void SetCropZ(int CropZ) { this->CropZ = CropZ; }

    // Set padding in all three dimensions of the array
    void SetPadding(int padding)
    {
        this->CropX = padding;
        this->CropY = padding;
        this->CropZ = padding;
    }

    void UpdateFilter(float *Input, float *Output, cudaStream_t *stream = NULL);

    void Update(cudaStream_t *stream = NULL)
    {
        // Are we using the device pointers?
        if (this->Input != NULL)
        {
            this->UpdateFilter(this->Input, this->Output, stream);
        }
        else if (this->d_input_struct != NULL) // Are we using structs for the pointers?
        {
            this->UpdateFilter(this->d_input_struct->GetPointer(), this->d_output_struct->GetPointer(), stream);
        }
        else
        {
            std::cerr << "CropVolumeFilter(): No valid inputs found." << '\n';
            return;
        }
    }
};
