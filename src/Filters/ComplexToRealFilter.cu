#include "ComplexToRealFilter.h"

__global__ void ComplexToRealKernel(cufftComplex *ComplexImg, float *RealImg, int imgSize, int nSlices)
{
    // CUDA kernel for extracting the real component of a cufftComplex and then save the real component to a float array

    // CUDA kernel for converting the CASImgs to imgs
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row

    // Are we outside the bounds of the image?
    if (i >= imgSize || i < 0 || j >= imgSize || j < 0)
    {
        return;
    }

    // Each thread will do all the slices for some position X and Y
    for (int k = 0; k < nSlices; k++)
    {
        // Get the linear index of the current position
        int ndx = i + j * imgSize + k * imgSize * imgSize;

        RealImg[ndx] = ComplexImg[ndx].x;
    }
}

void ComplexToRealFilter::UpdateFilter(cufftComplex *Input, float *Output, cudaStream_t *stream)
{
    // Convert a cufftComplex array to CAS array

    // std::cout << "ComplexToRealFilter()" << '\n';

    // Check the input parameters
    if (this->VolumeSize <= 0)
    {
        std::cerr << "Error ComplexToRealFilter(): Volume size parameter was not set. Please use SetVolumeSize() function first." << '\n';
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
        ComplexToRealKernel<<<dimGrid, dimBlock, 0, *stream>>>(Input, Output, this->VolumeSize, NumberSlices);
    }
    else
    {
        ComplexToRealKernel<<<dimGrid, dimBlock>>>(Input, Output, this->VolumeSize, NumberSlices);
    }

    gpuErrorCheck(cudaPeekAtLastError());

    return;
};
