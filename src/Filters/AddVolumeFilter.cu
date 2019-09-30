#include "AddVolumeFilter.h"

template <typename T>
__global__ void AddVolumeKernel(T *VolumeOne, T *VolumeTwo, int VolumeSize, int nSlices)
{
    // Add two GPU arrays together (assume the arrays are the same size X, Y, and Z dimensions)
    // VolumeOne = VolumeOne + VolumeTwo

    // Index of volume in the XY plane
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row

    // Are we within the array bounds?
    if (i < 0 || i > VolumeSize || j < 0 || j > VolumeSize)
    {
        return;
    }

    // Iterate over all the slices at the current X and Y position
    for (int k = 0; k < nSlices; k++)
    {
        // Get the linear index of the current X, Y, and Z position
        int ndx = i + j * VolumeSize + k * VolumeSize * VolumeSize;

        VolumeOne[ndx] = VolumeOne[ndx] + VolumeTwo[ndx];
    }
}

void AddVolumeFilter::UpdateFilter(float* Input, float* Output, cudaStream_t *stream)
{
    // Add two GPU arrays (of dimensions 3) together
    // Equation: VolumeOne = VolumeOne + VolumeTwo

    // std::cout << "AddVolumeFilter()..." << '\n';

    // Check the input parameters
    if (this->VolumeSize <= 0)
    {
        std::cerr << "Error AddVolumeFilter(): Volume size parameter was not set. Please use SetVolumeSize() function first." << '\n';
        return;
    }
    else if (this->nSlices <= 0)
    {
        std::cerr << "Error AddVolumeFilter(): Number of slices parameter was not set. Please use SetNumberOfSlices() function first." << '\n';
        return;
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
        AddVolumeKernel<<<dimGrid, dimBlock, 0, *stream>>>(Input, Output, this->VolumeSize, this->nSlices);
    }
    else
    {
        AddVolumeKernel<<<dimGrid, dimBlock>>>(Input, Output, this->VolumeSize, this->nSlices);
    }

    return;
};
