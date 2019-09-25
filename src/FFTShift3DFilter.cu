#include "FFTShift3DFilter.h"

template <typename T>
__global__ void FFTShift3DKernel(T *data, int N)
{
    // In place FFT shift using GPU
    // Modified from https://raw.githubusercontent.com/marwan-abdellah/cufftShift/master/Src/CUDA/Kernels/in-place/cufftShift_3D_IP.cu
    // GNU Lesser General Public License

    // 3D Volume & 2D Slice & 1D Line
    int sLine = N;
    int sSlice = N * N;
    int sVolume = N * N * N;

    // Transformations Equations
    int sEq1 = (sVolume + sSlice + sLine) / 2;
    int sEq2 = (sVolume + sSlice - sLine) / 2;
    int sEq3 = (sVolume - sSlice + sLine) / 2;
    int sEq4 = (sVolume - sSlice - sLine) / 2;

    // Thread
    int xThreadIdx = threadIdx.x;
    int yThreadIdx = threadIdx.y;

    // Block Width & Height
    int blockWidth = blockDim.x;
    int blockHeight = blockDim.y;

    // Thread Index 2D
    int xIndex = blockIdx.x * blockWidth + xThreadIdx;
    int yIndex = blockIdx.y * blockHeight + yThreadIdx;

    // Are we within the image bounds?
    if (xIndex < 0 || xIndex >= N || yIndex < 0 || yIndex >= N)
    {
        return;
    }

    T regTemp;

    // Each thread will do all the slices for some X, Y position in the 3D matrix
    for (int zIndex = 0; zIndex < N; zIndex++)
    {

        // Thread Index Converted into 1D Index
        int index = (zIndex * sSlice) + (yIndex * sLine) + xIndex;

        if (zIndex < N / 2)
        {
            if (xIndex < N / 2)
            {
                if (yIndex < N / 2)
                {
                    regTemp = data[index];

                    // First Quad
                    data[index] = data[index + sEq1];

                    // Fourth Quad
                    data[index + sEq1] = regTemp;
                }
                else
                {
                    regTemp = data[index];

                    // Third Quad
                    data[index] = data[index + sEq3];

                    // Second Quad
                    data[index + sEq3] = regTemp;
                }
            }
            else
            {
                if (yIndex < N / 2)
                {
                    regTemp = data[index];

                    // Second Quad
                    data[index] = data[index + sEq2];

                    // Third Quad
                    data[index + sEq2] = regTemp;
                }
                else
                {
                    regTemp = data[index];

                    // Fourth Quad
                    data[index] = data[index + sEq4];

                    // First Quad
                    data[index + sEq4] = regTemp;
                }
            }
        }
    }
}

void FFTShift3DFilter::UpdateFilter(cufftComplex *Input, cudaStream_t *stream)
{
    // Apply a 3D FFT shift to an array

    std::cout << "FFTShift3DFilter()" << '\n';

    // Check the input parameters
    if (this->VolumeSize <= 0)
    {
        std::cerr << "Error FFTShift3DFilter(): Volume size parameter was not set. Please use SetVolumeSize() function first." << '\n';
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
        FFTShift3DKernel<<<dimGrid, dimBlock, 0, *stream>>>(Input, this->VolumeSize);
    }
    else
    {
        FFTShift3DKernel<<<dimGrid, dimBlock>>>(Input, this->VolumeSize);
    }

    return;
};
