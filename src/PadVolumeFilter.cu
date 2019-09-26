#include "PadVolumeFilter.h"

// template <typename T>
__global__ void PadVolumeKernel(float *input, float *output,
                                int InputSizeX, int InputSizeY, int InputSizeZ,
                                int OutputSizeX, int OutputSizeY, int OutputSizeZ,
                                int paddingX, int paddingY, int paddingZ)
{

    // Zero pad a volume using the GPU

    // Index of the intput (smaller) image
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row

    // Are we outside the bounds of the image?
    if (i >= InputSizeX || i < 0 || j >= InputSizeY || j < 0)
    {
        return;
    }

    // Iterate over the input image (i.e. the smaller image)
    for (int k = 0; k < InputSizeZ; k++)
    {
        // Get the linear index of the output (smaller) image
        int ndx_1 = i + j * InputSizeX + k * InputSizeX * InputSizeY;

        // Get the linear index of the output (larger) image
        int ndx_2 =
            (i + paddingX) +
            (j + paddingY) * OutputSizeX +
            (k + paddingZ) * OutputSizeX * OutputSizeY;

        output[ndx_2] = input[ndx_1];
    }
}

// template <typename T>
void PadVolumeFilter::UpdateFilter(float *Input, float *Output, cudaStream_t *stream)
{ // float *input, float *output
    // Pad a GPU allocated array (of dimensions 3) with zeros
    // Note: Output volume is larger than the input volume

    // Check the input parameters
    if (this->InputSizeX <= 0 || this->InputSizeY <= 0 || this->InputSizeZ <= 0)
    {
        std::cerr << "Error PadVolumeFilter(): Input size must be a positive integer." << '\n';
        return;
    }
    else if (this->paddingX < 0 || this->paddingY < 0 || this->paddingZ < 0)
    {
        std::cerr << "Error PadVolumeFilter(): The padding values must be positive integers. Please use SetPaddingX(), SetPaddingY(), and SetPaddingZ()." << '\n';
        return;
    }

    // Calculate the size of the output volume
    int OutputSizeX = this->InputSizeX + this->paddingX * 2;
    int OutputSizeY = this->InputSizeY + this->paddingY * 2;
    int OutputSizeZ = this->InputSizeZ + this->paddingZ * 2;


    // Define CUDA kernel launch dimensions
    // Iterate over the X,Y positions in the smaller image (i.e. the input image)
    int Grid = ceil(double(max(OutputSizeX, OutputSizeY)) / double(32));
    Grid = max(Grid, 1); // At least one

    dim3 dimGrid(Grid, Grid, 1);
    dim3 dimBlock(32, 32, 1); // i.e. 32*32 threads

    // std::cout << "PadVolumeFilter()..." << '\n';
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
        PadVolumeKernel<<<dimGrid, dimBlock, 0, *stream>>>(Input, Output,
                                                           this->InputSizeX, this->InputSizeY, this->InputSizeZ,
                                                           OutputSizeX, OutputSizeY, OutputSizeZ,
                                                           this->paddingX, this->paddingY, this->paddingZ);
    }
    else
    {
        PadVolumeKernel<<<dimGrid, dimBlock>>>(Input, Output,
                                               this->InputSizeX, this->InputSizeY, this->InputSizeZ,
                                               OutputSizeX, OutputSizeY, OutputSizeZ,
                                               this->paddingX, this->paddingY, this->paddingZ);
    }
}
