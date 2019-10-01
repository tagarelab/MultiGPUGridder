#include "FFTShift2DFilter.h"

template <typename T>
__global__ void cufftShift_2D_kernel(T *data, int VolumeSize, int nSlices)
{
    // Modified from https://github.com/marwan-abdellah/cufftShift/blob/master/Src/CUDA/Kernels/in-place/cufftShift_2D_IP.cu
    // GNU Lesser General Public License

    // 2D Slice & 1D Line
    int sLine = VolumeSize;
    int sSlice = VolumeSize * VolumeSize;

    // Transformations Equations
    int sEq1 = (sSlice + sLine) / 2;
    int sEq2 = (sSlice - sLine) / 2;

    // Thread Index (2D)
    int X = blockIdx.x * blockDim.x + threadIdx.x;
    int Y = blockIdx.y * blockDim.y + threadIdx.y;

    // Are we outside the bounds of the image?
    if (X >= VolumeSize || X < 0 || Y >= VolumeSize || Y < 0)
    {
        return;
    }

    // Each thread will do all the slices for some X, Y position in the 3D matrix
    for (int Z = 0; Z < nSlices; Z++)
    {
        // Thread Index Converted into 1D Index
        int index = (Z * sSlice) + (Y * sLine) + X;

        T regTemp;

        if (X < VolumeSize / 2)
        {
            if (Y < VolumeSize / 2)
            {
                regTemp = data[index];

                // First Quad
                data[index] = data[index + sEq1];

                // Third Quad
                data[index + sEq1] = regTemp;
            }
        }
        else
        {
            if (Y < VolumeSize / 2)
            {
                regTemp = data[index];

                // Second Quad
                data[index] = data[index + sEq2];

                // Fourth Quad
                data[index + sEq2] = regTemp;
            }
        }
    }
}

// Explicit template instantiation
template class FFTShift2DFilter<float>;
template class FFTShift2DFilter<cufftComplex>;

template <class T>
void FFTShift2DFilter<T>::UpdateFilter(T *Input, cudaStream_t *stream)
{
    // Apply a 2D FFT shift to an array

    // std::cout << "FFTShift2DFilter()" << '\n';

    // Check the input parameters
    if (this->ImageSize <= 0)
    {
        std::cerr << "Error FFTShift2DFilter(): Image size parameter was not set. Please use SetImageSize() function first." << '\n';
        return;
    }
    else if (this->nSlices <= 0)
    {
        std::cerr << "Error FFTShift2DFilter(): Number of image slices parameter was not set. Please use SetNumberOfSlices() function first." << '\n';
        return;
    }

    // Define CUDA kernel launch dimensions
    // Iterate over the X,Y positions for all slices
    int Grid = ceil(double(this->ImageSize) / double(32));
    Grid = max(Grid, 1); // At least one

    dim3 dimGrid(Grid, Grid, 1);
    dim3 dimBlock(32, 32, 1); // i.e. 32*32 threads

    // Use the CUDA stream if one was provided
    if (stream != NULL)
    {
        cufftShift_2D_kernel<<<dimGrid, dimBlock, 0, *stream>>>(Input, this->ImageSize, this->nSlices);
    }
    else
    {
        cufftShift_2D_kernel<<<dimGrid, dimBlock>>>(Input, this->ImageSize, this->nSlices);
    }

    return;
};
