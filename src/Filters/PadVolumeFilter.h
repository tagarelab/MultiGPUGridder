#pragma once

/**
 * @class   PadVolumeFilter
 * @brief   A filter for zero padding a GPU array.
 *
 * PadVolumeFilter inherits from the AbstractFilter class.
 * 
 * The PadVolumeFilter addes zero padding to a GPU array. This is needed for 
 * running the forward or inverse Fourier transform operations.
 * 
 * In order to set which GPU the filter should be run on, please use the cudaDeviceSet() function before calling the Update function.
 * 
 * */

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

    /// Set the input array
    void SetInput(float *Input) { this->Input = Input; }

    /// Set the output array
    void SetOutput(float *Output) { this->Output = Output; }

    /// Set the size of the input array
    void SetInputSize(int InputSize)
    {
        this->InputSizeX = InputSize;
        this->InputSizeY = InputSize;
        this->InputSizeZ = InputSize;
    }

    /// If using a stack of 2D images, this is the number of images.
    void SetNumberOfSlices(int nSlices)
    {
        this->InputSizeZ = nSlices;
    }

    /// Set padding along the X dimension of the array
    void SetPaddingX(int paddingX) { this->paddingX = paddingX; }

    /// Set padding along the Y dimension of the array
    void SetPaddingY(int paddingY) { this->paddingY = paddingY; }

    /// Set padding along the Z dimension of the array
    void SetPaddingZ(int paddingZ) { this->paddingZ = paddingZ; }

    /// Set padding in all three dimensions of the array
    void SetPadding(int padding)
    {
        this->paddingX = padding;
        this->paddingY = padding;
        this->paddingZ = padding;
    }

    /// Update the filter
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

private:
    /// Run the CUDA kernel
    void UpdateFilter(float *Input, float *Output, cudaStream_t *stream = NULL);
};
