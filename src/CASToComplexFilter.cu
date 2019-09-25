#include "CASToComplexFilter.h"

__global__ void CASVolumeToComplexKernel(float *CASVolume, cufftComplex *ComplexVolume, int VolumeSize, int nSlices, bool VolumeFlag)
{
    // CUDA kernel for converting CASImgs to imgs
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row

    // Are we outside the bounds of the image?
    if (i > VolumeSize || i < 0 || j > VolumeSize || j < 0)
    {
        return;
    }

    // Each thread will do all the slices for position X and Y
    for (int k = 0; k < nSlices; k++)
    {
        int ndx_1 = i + j * VolumeSize + k * VolumeSize * VolumeSize;

        // Are we running a volume or a stack of 2D images
        if (VolumeFlag == true) // Running a volume
        {

            // Skip the first row, column, and slice
            if (i == 0 || j == 0 || k == 0)
            {
                // Real component
                ComplexVolume[ndx_1].x = 0;

                // Imaginary component
                ComplexVolume[ndx_1].y = 0;
            }
            else
            {
                // Find the second required index
                // Essentially, this is summing each pixel by the corresponding location counting from the end towards the beginning
                int i_end = VolumeSize - i;
                int j_end = VolumeSize - j;
                int k_end = VolumeSize - k;

                int ndx_2 = i_end + j_end * VolumeSize + k_end * VolumeSize * VolumeSize;

                // Real component
                ComplexVolume[ndx_1].x = 0.5 * (CASVolume[ndx_1] + CASVolume[ndx_2]);

                // Imaginary component
                ComplexVolume[ndx_1].y = 0.5 * (CASVolume[ndx_1] - CASVolume[ndx_2]);
            }
        }
        else // Running a stack of 2D images
        {
            // Skip the first row and column
            if (i == 0 || j == 0)
            {
                // Real component
                ComplexVolume[ndx_1].x = 0;

                // Imaginary component
                ComplexVolume[ndx_1].y = 0;
            }
            else
            {
                // Find the second required index
                // Essentially, this is summing each pixel by the corresponding location counting from the end towards the beginning
                int i_end = VolumeSize - i;
                int j_end = VolumeSize - j;
                int k_end = k;

                int ndx_2 = i_end + j_end * VolumeSize + k_end * VolumeSize * VolumeSize;

                // Real component
                ComplexVolume[ndx_1].x = 0.5 * (CASVolume[ndx_1] + CASVolume[ndx_2]);

                // Imaginary component
                ComplexVolume[ndx_1].y = 0.5 * (CASVolume[ndx_1] - CASVolume[ndx_2]);
            }
        }
    }

    return;
}

void CASToComplexFilter::UpdateFilter(float *Input, cufftComplex *Output, cudaStream_t *stream)
{
    // Convert a CAS array to cufftComplex type

    std::cout << "CASToComplexFilter()" << '\n';

    // Check the input parameters
    if (this->VolumeSize <= 0)
    {
        std::cerr << "Error CASToComplexFilter(): Volume size parameter was not set. Please use SetVolumeSize() function first." << '\n';
        return;
    }
    
    // Flag to specify that we are running a volume or a stack of 2D images
    bool VolumeFlag;
    int NumberSlices;

    // Running a volume if the number of slices is not specified
    if (this->nSlices <= 0)
    {
        VolumeFlag = true;
        NumberSlices = this->VolumeSize;
    }
    else
    {
        // Running a stack of 2D images
        VolumeFlag = false;
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
        CASVolumeToComplexKernel<<<dimGrid, dimBlock, 0, *stream>>>(Input, Output, this->VolumeSize, NumberSlices, VolumeFlag);
    }
    else
    {
        CASVolumeToComplexKernel<<<dimGrid, dimBlock>>>(Input, Output, this->VolumeSize, NumberSlices, VolumeFlag);
    }

    return;
};
