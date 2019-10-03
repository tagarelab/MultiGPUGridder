#include "DivideVolumeFilter.h"

__global__ void DivideVolumesKernel(float *VolumeOne, float *VolumeTwo, int VolumeSize, int NumberSlices)
{
    // Divide two GPU arrays together (assume the arrays are the same size and square)
    // VolumeOne = VolumeOne / VolumeTwo

    // Index of volume
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row

    // Are we within the volume bounds?
    if (i < 0 || i >= VolumeSize || j < 0 || j >= VolumeSize)
    {
        return;
    }

    for (int k = 0; k < NumberSlices; k++)
    {
        // Get the linear index of the volume
        int ndx = i + j * VolumeSize + k * VolumeSize * VolumeSize;

        // Check that both values are not zero in order to prevent dividing by zero
        if (VolumeOne[ndx] != 0 && VolumeTwo[ndx] != 0)
        {
            VolumeOne[ndx] = VolumeOne[ndx] / VolumeTwo[ndx];
        }
        else
        {
            VolumeOne[ndx] = 0;
        }
    }
}

void DivideVolumeFilter::UpdateFilter(float* Input, float* Output, cudaStream_t *stream)
{
    // Add two GPU arrays (of dimensions 3) together
    // Equation: VolumeOne = VolumeOne + VolumeTwo

    // std::cout << "DivideVolumeFilter()..." << '\n';

    // Check the input parameters
    if (this->VolumeSize <= 0)
    {
        std::cerr << "Error DivideVolumeFilter(): Volume size parameter was not set. Please use SetVolumeSize() function first." << '\n';
        return;
    }

    int NumberSlices;

    // Running a volume if the number of slices is not specified
    if (this->nSlices <= 0)
    {
        NumberSlices = this->VolumeSize;
    }
    else
    {
        // Running a stack of 2D images
        NumberSlices = this->nSlices;
    }

    // Define CUDA kernel launch dimensions
    // Iterate over the X,Y positions for all slices
    int Grid = ceil(double(this->VolumeSize) / double(32));
    Grid = max(Grid, 1); // At least one

    dim3 dimGrid(Grid, Grid, 1);
    dim3 dimBlock(32, 32, 1); // i.e. 32*32 threads

    // Use the CUDA stream if one was provided
    if (stream != NULL)
    {
        DivideVolumesKernel<<<dimGrid, dimBlock, 0, *stream>>>(Input, Output, this->VolumeSize, NumberSlices);
    }
    else
    {
        DivideVolumesKernel<<<dimGrid, dimBlock>>>(Input, Output, this->VolumeSize, NumberSlices);
    }

    return;
};
