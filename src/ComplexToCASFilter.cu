#include "ComplexToCASFilter.h"

__global__ void ComplexToCASKernel(cufftComplex *ComplexInput, float *CASOutput, int imgSize, int nSlices)
{
    // CUDA kernel for converting CASImgs to imgs
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row

    // Are we outside the bounds of the image?
    if (i >= imgSize || i < 0 || j >= imgSize || j < 0)
    {
        return;
    }

    // Each thread will do all the slices for position X and Y
    for (int k = 0; k < nSlices; k++)
    {
        // CASimgs is the same dimensions as imgs
        int ndx = i + j * imgSize + k * imgSize * imgSize;

        // Summation of the real and imaginary components
        CASOutput[ndx] = ComplexInput[ndx].x + ComplexInput[ndx].y;
    }

    return;
}

void ComplexToCASFilter::UpdateFilter(cufftComplex *Input, float *Output, cudaStream_t *stream)
{
    // Convert a cufftComplex array to CAS array

    std::cout << "ComplexToCASFilter()" << '\n';

    // Check the input parameters
    if (this->VolumeSize <= 0)
    {
        std::cerr << "Error ComplexToCASFilter(): Volume size parameter was not set. Please use SetVolumeSize() function first." << '\n';
        return;
    }
    // Flag to specify that we are running a volume or a stack of 2D images
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

    std::cout << "ComplexToCASFilter()..." << '\n';
    std::cout << "VolumeSize: " << VolumeSize << '\n';
    std::cout << "NumberSlices: " << NumberSlices << '\n';
    std::cout << "this->nSlices: " << this->nSlices << '\n';
    std::cout << "Grid: " << Grid << '\n';

    // Use the CUDA stream if one was provided
    if (stream != NULL)
    {
        ComplexToCASKernel<<<dimGrid, dimBlock, 0, *stream>>>(Input, Output, this->VolumeSize, NumberSlices);
    }
    else
    {
        ComplexToCASKernel<<<dimGrid, dimBlock>>>(Input, Output, this->VolumeSize, NumberSlices);
    }

    return;
};
