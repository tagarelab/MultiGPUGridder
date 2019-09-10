#include "gpuFFT.h"

#define Log(x)                  \
    {                           \
        std::cout << x << '\n'; \
    }

template <typename T>
__global__ void cufftShift_2D_kernel(T *data, int N, int nSlices)
{
    // 2D Slice & 1D Line
    int sLine = N;
    int sSlice = N * N;

    // Transformations Equations
    int sEq1 = (sSlice + sLine) / 2;
    int sEq2 = (sSlice - sLine) / 2;

    // Thread Index (1D)
    int xThreadIdx = threadIdx.x;
    int yThreadIdx = threadIdx.y;

    // Block Width & Height
    int blockWidth = blockDim.x;
    int blockHeight = blockDim.y;

    // Thread Index (2D)
    int xIndex = blockIdx.x * blockWidth + xThreadIdx;
    int yIndex = blockIdx.y * blockHeight + yThreadIdx;

    // Each thread will do all the slices for some X, Y position in the 3D matrix
    for (int zIndex = 0; zIndex < nSlices; zIndex++)
    {
        // Thread Index Converted into 1D Index
        int index = (zIndex * sSlice) + (yIndex * sLine) + xIndex;

        T regTemp;

        if (xIndex < N / 2)
        {
            if (yIndex < N / 2)
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
            if (yIndex < N / 2)
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

__global__ void cufftShift_3D_slice_kernel(cufftComplex *input, cufftComplex *output, int N, int nSlices)
{
    // 3D Volume, 2D Slice, 1D Line
    int sLine = N;
    int sSlice = N * N;
    int sVolume = N * N * N;

    // Transformations Equations
    int sEq1 = (sVolume + sSlice + sLine) / 2;
    int sEq2 = (sVolume + sSlice - sLine) / 2;
    int sEq3 = (sVolume - sSlice + sLine) / 2;
    int sEq4 = (sVolume - sSlice - sLine) / 2;

    // Thread Index 2D
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if (xIndex < 0 || xIndex > N || yIndex<0 | yIndex> N)
    {
        return;
    }

    // Each thread will do all the slices for some X, Y position in the 3D matrix
    for (int zIndex = 0; zIndex < nSlices; zIndex++)
    {
        // Thread Index Converted into 1D Index
        int index = (zIndex * sSlice) + (yIndex * sLine) + xIndex;

        if (zIndex < N / 2)
        {
            if (xIndex < N / 2)
            {
                if (yIndex < N / 2)
                {
                    // First Quad
                    output[index].x = input[index + sEq1].x;
                    output[index].y = input[index + sEq1].y;
                }
                else
                {
                    // Third Quad
                    output[index].x = input[index + sEq3].x;
                    output[index].y = input[index + sEq3].y;
                }
            }
            else
            {
                if (yIndex < N / 2)
                {
                    // Second Quad
                    output[index].x = input[index + sEq2].x;
                    output[index].y = input[index + sEq2].y;
                }
                else
                {
                    // Fourth Quad
                    output[index].x = input[index + sEq4].x;
                    output[index].y = input[index + sEq4].y;
                }
            }
        }
        else
        {
            if (xIndex < N / 2)
            {
                if (yIndex < N / 2)
                {
                    // First Quad
                    output[index].x = input[index - sEq4].x;
                    output[index].y = input[index - sEq4].y;
                }
                else
                {
                    // Third Quad
                    output[index].x = input[index - sEq2].x;
                    output[index].y = input[index - sEq2].y;
                }
            }
            else
            {
                if (yIndex < N / 2)
                {
                    // Second Quad
                    output[index].x = input[index - sEq3].x;
                    output[index].y = input[index - sEq3].y;
                }
                else
                {
                    // Fourth Quad
                    output[index].x = input[index - sEq1].x;
                    output[index].y = input[index - sEq1].y;
                }
            }
        }
    }
}

template <typename T>
__global__ void cufftShift_3D_slice_kernel(T *data, int N, int nSlices)
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
    for (int zIndex = 0; zIndex < nSlices; zIndex++)
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

__global__ void ComplexImgsToCASImgs(float *CASimgs, cufftComplex *imgs, int imgSize)
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
    for (int k = 0; k < imgSize; k++)
    {
        // CASimgs is the same dimensions as imgs
        int ndx = i + j * imgSize + k * imgSize * imgSize;

        // Summation of the real and imaginary components
        CASimgs[ndx] = imgs[ndx].x + imgs[ndx].y;
    }

    return;
}

__global__ void CASImgsToComplexImgs(float *CASimgs, cufftComplex *imgs, int imgSize, int nSlices)
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

        int ndx_1 = i + j * imgSize + k * imgSize * imgSize;

        // Skip the first row and first column
        if (i == 0 || j == 0)
        {
            // Real component
            imgs[ndx_1].x = 0;

            // Imaginary component
            imgs[ndx_1].y = 0;
        }
        else
        {
            // Offset to skip the first row then subtract from the end of the matrix and add the offset where the particular image starts in CASimgs
            int ndx_2 = imgSize + imgSize * imgSize - (i + j * imgSize) + k * imgSize * imgSize;

            // Real component
            imgs[ndx_1].x = 0.5 * (CASimgs[ndx_1] + CASimgs[ndx_2]);

            // Imaginary component
            imgs[ndx_1].y = 0.5 * (CASimgs[ndx_1] - CASimgs[ndx_2]);
        }
    }

    return;
}

__global__ void ComplexToReal(cufftComplex *ComplexImg, float *RealImg, int imgSize, int nSlices)
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

__global__ void PadVolumeKernel(float *input, float *output, int inputImgSize, int outputImgSize, int padding)
{
    // Zero pad a volume using the GPU

    // Index of the output (smaller) image
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row

    // Are we outside the bounds of the image?
    if (i >= inputImgSize || i < 0 || j >= inputImgSize || j < 0)
    {
        return;
    }

    // // Iterate over the input image (i.e. the smaller image)
    for (int k = 0; k < inputImgSize; k++)
    {

        // Get the linear index of the output (smaller) image
        int ndx_1 = i + j * inputImgSize + k * inputImgSize * inputImgSize;

        // Get the linear index of the output (larger) image
        int ndx_2 =
            (i + padding) +
            (j + padding) * outputImgSize +
            (k + padding) * outputImgSize * outputImgSize;

        output[ndx_2] = input[ndx_1];
    }
}

void gpuFFT::PadVolume(float *inputVol, float *outputVol, int inputImgSize, int outputImgSize)
{
    // Pad a volume (of dimensions 3) with zeros
    // Note: Output volume is larger than the input volume

    // Check the input parameters
    if (inputImgSize <= 0)
    {
        std::cerr << "CropVolume(): Invalid image size." << '\n';
    }

    // Create the output volume
    //float *outputVol = new float[outputImgSize * outputImgSize * outputImgSize];
    memset(outputVol, 0, outputImgSize * outputImgSize * outputImgSize * sizeof(float));

    // for (int i = 0; i < outputImgSize * outputImgSize * outputImgSize; i++)
    // {
    //     outputVol[i] = 0;
    // }

    // How much to add to each side?
    int padding = (outputImgSize - inputImgSize) / 2;

    // For very small matrix sizes it might be faster to use the CPU instead of the GPU
    bool use_gpu = true;

    if (use_gpu == true)
    {
        // Allocate GPU memory to hold the input and output arrays
        float *d_input;
        cudaMalloc(&d_input, sizeof(float) * inputImgSize * inputImgSize * inputImgSize);
        float *d_output;
        cudaMalloc(&d_output, sizeof(float) * outputImgSize * outputImgSize * outputImgSize);

        // Copy the input volume to the device
        cudaMemcpy(d_input, inputVol, sizeof(float) * inputImgSize * inputImgSize * inputImgSize, cudaMemcpyHostToDevice);

        // Run kernel to pad the intput array
        int gridSize = 32;
        int blockSize = ceil(inputImgSize / gridSize);

        dim3 dimGridCrop(gridSize, gridSize, 1);
        dim3 dimBlockCrop(blockSize, blockSize, 1);

        PadVolumeKernel<<<dimGridCrop, dimBlockCrop>>>(d_input, d_output, inputImgSize, outputImgSize, padding);

        // Copy the result back to the host
        cudaMemcpy(outputVol, d_output, sizeof(float) * outputImgSize * outputImgSize * outputImgSize, cudaMemcpyDeviceToHost);

        // Free the GPU memory
        cudaFree(d_input);
        cudaFree(d_output);
    }
    else
    {
        // Iterate over the input image (i.e. the smaller image)
        for (int i = 0; i < inputImgSize; i++)
        {
            for (int j = 0; j < inputImgSize; j++)
            {
                for (int k = 0; k < inputImgSize; k++)
                {

                    int input_ndx = i + j * inputImgSize + k * inputImgSize * inputImgSize;

                    int output_ndx = (i + padding) + (j + padding) * outputImgSize + (k + padding) * outputImgSize * outputImgSize;

                    outputVol[output_ndx] = inputVol[input_ndx];
                }
            }
        }
    }

    return;
}

__global__ void CropImgs(float *input, float *output, int inputImgSize, int outputImgSize, int nSlices)
{
    // Given the final projection images, crop out the zero padding to reduce memory size and transfer speed back to the CPU
    // inputImgSize is the size of the CASImgs (i.e. larger)
    // outputImgSize is the size of the images (i.e. smaller)

    // Index of the output (smaller) image
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row

    // Are we outside the bounds of the smaller output image?
    if (i >= outputImgSize || i < 0 || j >= outputImgSize || j < 0)
    {
        return;
    }

    // How much zero padding to remove from each side?
    int padding = (inputImgSize - outputImgSize) / 2;

    if (padding <= 0)
    {
        return;
    }

    for (int k = 0; k < nSlices; k++)
    {

        // Get the linear index of the output (smaller) image
        int ndx_1 = i + j * outputImgSize + k * outputImgSize * outputImgSize;

        // Get the linear index of the input (larger) image
        // NOTE: No padding in the Z direction because we are cropping each 2D images individually
        int ndx_2 =
            (i + padding) +
            (j + padding) * inputImgSize +
            k * inputImgSize * inputImgSize;

        output[ndx_1] = input[ndx_2];
    }
}

__global__ void ComplexToNormalizedImgs(cufftComplex *ComplexImg, float *output, int inputImgSize, int outputImgSize, int nSlices, int NormalizeFactor)
{
    // CUDA kernel for cropped a complex cufftComplex array and extracting the real component

    // Index of the output (smaller) image
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row

    // Are we outside the bounds of the smaller output image?
    if (i >= outputImgSize || i < 0 || j >= outputImgSize || j < 0)
    {
        return;
    }

    // How much zero padding to remove from each side?
    int padding = (inputImgSize - outputImgSize) / 2;

    if (padding <= 0)
    {
        return;
    }

    for (int k = 0; k < nSlices; k++)
    {

        // Get the linear index of the output (smaller) image
        int ndx_1 = i + j * outputImgSize + k * outputImgSize * outputImgSize;

        // Get the linear index of the input (larger) image
        // NOTE: No padding in the Z direction because we are cropping each 2D images individually
        int ndx_2 =
            (i + padding) +
            (j + padding) * inputImgSize +
            k * inputImgSize * inputImgSize;

        output[ndx_1] = ComplexImg[ndx_2].x / NormalizeFactor;
    }
}

__global__ void NormalizeImgs(float *input, int ImgSize, int numImgs, int NormalizeFactor)
{
    // Normalize images by dividing each voxel by some normalization factor

    // Index of the image
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row

    // Are we outside the bounds of the smaller output image?
    if (i >= ImgSize || i < 0 || j >= ImgSize || j < 0)
    {
        return;
    }

    for (int k = 0; k < numImgs; k++)
    {

        // Get the linear index of image
        int ndx = i + j * ImgSize + k * ImgSize * ImgSize;

        input[ndx] = input[ndx] / NormalizeFactor;
    }
}

void gpuFFT::VolumeToCAS(float *inputVol, int inputVolSize, float *outputVol, int interpFactor, int extraPadding)
{
    // Convert a CUDA array to CAS array
    // Note: The volume must be square (i.e. have the same dimensions for the X, Y, and Z)
    // Step 1: Pad with zeros
    // Step 2: fftshift
    // Step 3: Take discrete Fourier transform using cuFFT
    // Step 4: fftshift
    // Step 5: Convert to CAS volume using CUDA kernel

    // STEP 1
    // Example: input size = 128; interpFactor = 2; paddedVolSize = 256
    int paddedVolSize = inputVolSize * interpFactor;

    // Log("VolumeToCAS():");
    // Log(inputVolSize);
    // Log(interpFactor);
    // Log(extraPadding);
    // Log(paddedVolSize);

    int array_size = paddedVolSize * paddedVolSize * paddedVolSize;

    // Pad the input volume with zeros
    float *inputVol_Padded = new float[paddedVolSize * paddedVolSize * paddedVolSize];

    // Allocate memory for the resulting CAS volumes
    float *d_CAS_Vol, *h_CAS_Vol;
    cufftComplex *h_complex_array, *d_complex_array;
    h_CAS_Vol = (float *)malloc(sizeof(float) * array_size);
    h_complex_array = (cufftComplex *)malloc(sizeof(cufftComplex) * array_size);

    cudaMalloc(&d_CAS_Vol, sizeof(float) * array_size);
    cudaMalloc(&d_complex_array, sizeof(cufftComplex) * array_size);

    // Pad the input volume with zeros
    PadVolume(inputVol, inputVol_Padded, inputVolSize, paddedVolSize);

    // Plan the forward FFT
    cufftHandle forwardFFTPlan;
    cufftPlan3d(&forwardFFTPlan, paddedVolSize, paddedVolSize, paddedVolSize, CUFFT_C2C);

    // Convert the padded volume to a cufftComplex array
    // TO DO: Replace this with the CUDA kernel
    for (int k = 0; k < array_size; k++)
    {
        h_complex_array[k].x = inputVol_Padded[k]; // Real component
        h_complex_array[k].y = 0;                  // Imaginary component
    }

    // Copy the complex version of the GPU volume to the first GPU
    cudaMemcpy(d_complex_array, h_complex_array, array_size * sizeof(cufftComplex), cudaMemcpyHostToDevice);

    int gridSize = ceil(paddedVolSize / 32);
    int blockSize = 32; // i.e. 32*32 threads

    // Define CUDA kernel dimensions for converting the complex volume to a CAS volume
    dim3 dimGrid(gridSize, gridSize, 1);
    dim3 dimBlock(blockSize, blockSize, 1);

    // STEP 2
    // Apply an in place 3D FFT Shift
    cufftShift_3D_slice_kernel<<<dimGrid, dimBlock>>>(d_complex_array, paddedVolSize, paddedVolSize);

    // STEP 3
    // Execute the forward FFT on the 3D array
    cufftExecC2C(forwardFFTPlan, (cufftComplex *)d_complex_array, (cufftComplex *)d_complex_array, CUFFT_FORWARD);

    // STEP 4
    // Apply a second in place 3D FFT Shift
    cufftShift_3D_slice_kernel<<<dimGrid, dimBlock>>>(d_complex_array, paddedVolSize, paddedVolSize);

    // STEP 5
    // Convert the complex result of the forward FFT to a CAS img type
    ComplexImgsToCASImgs<<<dimGrid, dimBlock>>>(
        d_CAS_Vol, d_complex_array, paddedVolSize);

    // Copy the resulting CAS volume back to the host
    cudaMemcpy(h_CAS_Vol, d_CAS_Vol, array_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Wait for the stream to finish copying the result back to the host
    // cudaStreamSynchronize(Stream);
    cudaDeviceSynchronize(); // TO DO: replace with stream sync?

    // STEP 6
    // Pad the result with the additional padding
    // Example: input size = 128; interpFactor = 2; extra padding = 3; -> paddedVolSize_Extra= 262
    int paddedVolSize_Extra = paddedVolSize + extraPadding * 2;

    // Pad the padded volume with the extra zero padding
    // TO DO: make PadVolume completely on the GPU to remove the stream sync step above
    PadVolume(h_CAS_Vol, outputVol, paddedVolSize, paddedVolSize_Extra);

    // Free the temporary memory
    cudaFree(d_complex_array);
    cudaFree(d_CAS_Vol);
    std::free(inputVol_Padded);
    std::free(h_CAS_Vol);
    std::free(h_complex_array);
    cufftDestroy(forwardFFTPlan);

    // Return the resulting CAS volume
    return;
}

void gpuFFT::CASImgsToImgs(
    cudaStream_t &stream, int CASImgSize,
    int ImgSize, float *d_CASImgs, float *d_imgs,
    cufftComplex *d_CASImgsComplex,
    int numImgs)
{

    // Has the inverse FFT been planned? If not create one now
    if (this->inverseFFTPlannedFlag == false)
    {
        cufftPlan2d(&this->inverseFFTPlan, CASImgSize, CASImgSize, CUFFT_C2C);

        this->inverseFFTPlannedFlag = true;
    }
    
    int gridSize = 32;
    int blockSize = ceil(CASImgSize / gridSize);

    // Convert a CAS images array to images
    dim3 dimGrid(gridSize, gridSize, 1);
    dim3 dimBlock(blockSize, blockSize, 1);

    // Convert the CASImgs to complex cufft type
    CASImgsToComplexImgs<<<dimGrid, dimBlock, 0, stream>>>(d_CASImgs, d_CASImgsComplex, CASImgSize, numImgs);

    // Run FFTShift
    cufftShift_2D_kernel<<<dimGrid, dimBlock, 0, stream>>>(d_CASImgsComplex, CASImgSize, numImgs);

    // Execute the forward FFT on each 2D array
    // cufftPlanMany is not feasible since the number of images changes
    // cufftDestroy is then blocking the CPU and causes memory leaks if not called
    // This has similar computation speed as cufftPlanMany
    for (int i = 0; i < numImgs; i++)
    {
        // Set the FFT plan to the current stream to process
        cufftSetStream(inverseFFTPlan, stream);

        cufftResult_t result = cufftExecC2C(inverseFFTPlan,
                                            (cufftComplex *)&d_CASImgsComplex[i * CASImgSize * CASImgSize],
                                            (cufftComplex *)&d_CASImgsComplex[i * CASImgSize * CASImgSize],
                                            CUFFT_INVERSE);
    }

    // Run FFTShift
    cufftShift_2D_kernel<<<dimGrid, dimBlock, 0, stream>>>(d_CASImgsComplex, CASImgSize, numImgs);

    // Run kernel to crop the projection images (to remove the zero padding), extract the real value,
    // and normalize the scaling introduced during the FFT
    ComplexToNormalizedImgs<<<dimGrid, dimBlock, 0, stream>>>(d_CASImgsComplex, d_imgs, CASImgSize, ImgSize, numImgs, CASImgSize * CASImgSize);

    return;
}