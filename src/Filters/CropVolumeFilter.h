#pragma once

/**
 * @class   CropVolumeFilter
 * @brief   A filter for cropping a GPU array.
 *
 * CropVolumeFilter inherits from the AbstractFilter class.
 * 
 * The CropVolumeFilter crops a GPU array. This is needed for removing the zero padding
 * after running the forward or inverse Fourier transform operations.
 * 
 * In order to set which GPU the filter should be run on, please use the cudaDeviceSet() function before calling the Update function.
 * 
 * */

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

    /// Set the input (i.e. the larger) array
    void SetInput(float *Input) { this->Input = Input; }

    /// Set the output (i.e. the smaller) array
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

    /// Set cropping along the X dimension of the array
    void SetCropX(int CropX) { this->CropX = CropX; }

    /// Set cropping along the Y dimension of the array
    void SetCropY(int CropY) { this->CropY = CropY; }

    /// Set cropping along the Y dimension of the array
    void SetCropZ(int CropZ) { this->CropZ = CropZ; }

    // Set cropping in all three dimensions of the array
    void SetCropping(int padding)
    {
        this->CropX = padding;
        this->CropY = padding;
        this->CropZ = padding;
    }

    /// Run the CUDA kernel
    void UpdateFilter(float *Input, float *Output, cudaStream_t *stream = NULL);

    /// Update the filter
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
