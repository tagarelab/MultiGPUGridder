#include "CropVolumeFilter.h"

template <typename T>

__global__ void CropVolumeKernel(T *input, T *output,
                                 int InputSizeX, int InputSizeY, int InputSizeZ,
                                 int OutputSizeX, int OutputSizeY, int OutputSizeZ,
                                 int CropX, int CropY, int CropZ)
{

    // Crop a GPU volume (i.e. remove padding in all three dimensions)
    // inputImgSize is the size of the CASImgs (i.e. larger)
    // outputImgSize is the size of the images (i.e. smaller)

    // Index of the output (smaller) image
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row

    // Are we outside the bounds of the smaller output image?
    if (i >= OutputSizeX || i < 0 || j >= OutputSizeY || j < 0)
    {
        return;
    }

    for (int k = 0; k < OutputSizeZ; k++)
    {
        // Get the linear index of the output (smaller) image
        int ndx_1 = i + j * OutputSizeX + k * OutputSizeX * OutputSizeY;

        // Are we processing
        // Get the linear index of the input (larger) image
        int ndx_2 =
            (i + CropX) +
            (j + CropY) * InputSizeX +
            (k + CropZ) * InputSizeX * InputSizeY;

        output[ndx_1] = input[ndx_2];
    }
}

void CropVolumeFilter::UpdateFilter(float *Input, float *Output, cudaStream_t *stream)
{ // float *input, float *output
    // Pad a GPU allocated array (of dimensions 3) with zeros
    // Note: Output volume is larger than the input volume

    // std::cout << "CropVolumeFilter..." << '\n';
    // Check the input parameters
    if (this->InputSizeX <= 0 || this->InputSizeY <= 0 || this->InputSizeZ <= 0)
    {
        std::cerr << "Error CropVolumeFilter(): Input size must be a positive integer." << '\n';
        return;
    }
    else if (this->CropX < 0 || this->CropY < 0 || this->CropZ < 0)
    {
        std::cerr << "Error CropVolumeFilter(): The cropping values must be positive integers. Please use SetCropX(), SetCropY(), and SetCropZ()." << '\n';
        return;
    }

    // Calculate the size of the output volume
    int OutputSizeX = this->InputSizeX - this->CropX * 2;
    int OutputSizeY = this->InputSizeY - this->CropY * 2;
    int OutputSizeZ = this->InputSizeZ - this->CropZ * 2;

    // Define CUDA kernel launch dimensions
    // Iterate over the X,Y positions in the smaller image (i.e. the output image)
    int Grid = ceil(double(this->InputSizeX) / double(32));
    Grid = max(Grid, 1); // At least one

    dim3 dimGrid(Grid, Grid, 1);
    dim3 dimBlock(32, 32, 1); // i.e. 32*32 threads

    // std::cout << "CropVolumeFilter()..." << '\n';
    // std::cout << "InputSizeX: " << InputSizeX << '\n';
    // std::cout << "InputSizeY: " << InputSizeY << '\n';
    // std::cout << "InputSizeZ: " << InputSizeZ << '\n';
    // std::cout << "OutputSizeX: " << OutputSizeX << '\n';
    // std::cout << "OutputSizeY: " << OutputSizeY << '\n';
    // std::cout << "OutputSizeZ: " << OutputSizeZ << '\n';
    // std::cout << "Grid: " << Grid << '\n';

    // Use the CUDA stream if one was provided
    if (stream != NULL)
    {
        CropVolumeKernel<<<dimGrid, dimBlock, 0, *stream>>>(Input, Output,
                                                            this->InputSizeX, this->InputSizeY, this->InputSizeZ,
                                                            OutputSizeX, OutputSizeY, OutputSizeZ,
                                                            this->CropX, this->CropY, this->CropZ);
    }
    else
    {
        CropVolumeKernel<<<dimGrid, dimBlock>>>(Input, Output,
                                                this->InputSizeX, this->InputSizeY, this->InputSizeZ,
                                                OutputSizeX, OutputSizeY, OutputSizeZ,
                                                this->CropX, this->CropY, this->CropZ);
    }
}
