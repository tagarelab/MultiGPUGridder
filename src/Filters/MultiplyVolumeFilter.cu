#include "MultiplyVolumeFilter.h"

__global__ void MultiplyVolumesKernel(cufftComplex *VolumeOne, float *VolumeTwo, int VolumeSize, int NumberSlices)
{
    // Multiply two GPU arrays together (assume the arrays are the same size and square)
    // VolumeOne = VolumeOne * VolumeTwo

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

        // Multiply together
        VolumeOne[ndx].x = VolumeOne[ndx].x * VolumeTwo[ndx];
        VolumeOne[ndx].y = VolumeOne[ndx].y * VolumeTwo[ndx];
    }
}

__global__ void MultiplyVolumesKernel(float *VolumeOne, float *VolumeTwo, int VolumeSize, int NumberSlices)
{
    // Multiply two GPU arrays together (assume the arrays are the same size and square)
    // VolumeOne = VolumeOne * VolumeTwo

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

        // Multiply together
        // Only multiply the real component of the cufftComplex array for now
        VolumeOne[ndx] = VolumeOne[ndx] * VolumeTwo[ndx];
    }
}

// Explicit template instantiation
template class MultiplyVolumeFilter<float>;
template class MultiplyVolumeFilter<cufftComplex>;

template <class T>
void MultiplyVolumeFilter<T>::UpdateFilter(T *Input, float *Output, cudaStream_t *stream)
{
    // Add two GPU arrays (of dimensions 3) together
    // Equation: VolumeOne = VolumeOne + VolumeTwo

    // Check the input parameters
    if (this->VolumeSize <= 0)
    {
        std::cerr << "Error MultiplyVolumeFilter(): Volume size parameter was not set. Please use SetVolumeSize() function first." << '\n';
        return;
    }

    // Running a volume if the number of slices is not specified
    int NumberSlices;
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
        MultiplyVolumesKernel<<<dimGrid, dimBlock, 0, *stream>>>(Input, Output, this->VolumeSize, NumberSlices);
    }
    else
    {
        MultiplyVolumesKernel<<<dimGrid, dimBlock>>>(Input, Output, this->VolumeSize, NumberSlices);
    }

    gpuErrorCheck(cudaPeekAtLastError());

    return;
};
