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

template <typename T>
__global__ void cufftShift_3D_kernel(T *data, int N, int nSlices)
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

__global__ void RealToComplexKernel2(float *RealImg, cufftComplex *ComplexImg, int imgSize, int nSlices)
{
    // CUDA kernel for converting a real image to a complex type

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

        ComplexImg[ndx].x = RealImg[ndx];
    }
}

template <typename T>
__global__ void PadVolumeKernel(T *input, T *output, int inputImgSize, int outputImgSize, int padding)
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

template <typename T>
__global__ void CropImgs(T *input, T *output, int inputImgSize, int outputImgSize, int nSlices)
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

__global__ void ComplexToCroppedNormalizedImgs(cufftComplex *ComplexImg, float *output, int inputImgSize, int outputImgSize, int nSlices, int NormalizeFactor)
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
    // Step 1: Pad the input volume with zeros and convert to cufftComplex type
    // Step 2: fftshift
    // Step 3: Take discrete Fourier transform using cuFFT
    // Step 4: fftshift
    // Step 5: Convert to CAS volume using CUDA kernel
    // Step 6: Apply extra zero padding

    // Example: input size = 128; interpFactor = 2 -> paddedVolSize = 256
    int paddedVolSize = inputVolSize * interpFactor;

    // Example: input size = 128; interpFactor = 2; extra padding = 3; -> paddedVolSize_Extra = 262
    int paddedVolSize_Extra = paddedVolSize + extraPadding * 2;

    // Allocate memory for the intermediate steps
    float *d_Vol, *d_CAS_Vol, *d_CAS_Vol_Padded;
    cufftComplex *d_complex_array;
    cudaMalloc(&d_Vol, sizeof(float) * inputVolSize * inputVolSize * inputVolSize);
    cudaMalloc(&d_CAS_Vol, sizeof(float) * paddedVolSize * paddedVolSize * paddedVolSize);
    cudaMalloc(&d_CAS_Vol_Padded, sizeof(float) * pow(inputVolSize * interpFactor + extraPadding * 2, 3));
    cudaMalloc(&d_complex_array, sizeof(cufftComplex) * paddedVolSize * paddedVolSize * paddedVolSize);

    // Copy the volume to the corresponding GPU array
    cudaMemcpy(d_Vol, inputVol, sizeof(float) * inputVolSize * inputVolSize * inputVolSize, cudaMemcpyHostToDevice);

    // STEP 1: Pad the input volume with zeros and convert to cufftComplex type
    gpuFFT::PadVolume(d_Vol, d_CAS_Vol, inputVolSize, paddedVolSize);

    // Convert the d_CAS_Vol to complex type (need cufftComplex type for the forward FFT)
    gpuFFT::RealToComplex(d_CAS_Vol, d_complex_array, paddedVolSize, paddedVolSize);

    // STEP 2: Apply an in place 3D FFT Shift
    gpuFFT::cufftShift_3D(d_complex_array, paddedVolSize, paddedVolSize);

    // STEP 3: Execute the forward FFT on the 3D array
    gpuFFT::FowardFFT(d_complex_array, paddedVolSize);

    // STEP 4: Apply a second in place 3D FFT Shift
    gpuFFT::cufftShift_3D(d_complex_array, paddedVolSize, paddedVolSize);

    // STEP 5: Convert the complex result of the forward FFT to a CAS img type
    gpuFFT::ComplexImgsToCAS(d_complex_array, d_CAS_Vol, paddedVolSize);

    // STEP 6: Pad the result with the additional padding
    gpuFFT::PadVolume(d_CAS_Vol, d_CAS_Vol_Padded, paddedVolSize, paddedVolSize_Extra);

    // Copy the resulting CAS volume back to the host
    cudaMemcpy(outputVol, d_CAS_Vol_Padded, sizeof(float) * paddedVolSize * paddedVolSize * paddedVolSize, cudaMemcpyDeviceToHost);

    // Free the temporary memory allocated
    cudaFree(d_complex_array);
    cudaFree(d_CAS_Vol);
    cudaFree(d_CAS_Vol_Padded);
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

    // Convert the CASImgs to complex cufft type
    gpuFFT::CASImgsToComplex(d_CASImgs, d_CASImgsComplex, CASImgSize, numImgs, stream);

    // Run FFTShift on each 2D slice
    gpuFFT::cufftShift_2D(d_CASImgsComplex, CASImgSize, numImgs, stream);

    // Execute the forward FFT on each 2D array
    // cufftPlanMany is not feasible since the number of images changes and
    // cufftDestroy is blocks the CPU and causes memory leaks if not called
    // FFT on each 2D slice has similar computation speed as cufftPlanMany
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
    gpuFFT::cufftShift_2D(d_CASImgsComplex, CASImgSize, numImgs, stream);

    // Run kernel to crop the projection images (to remove the zero padding), extract the real value,
    // and normalize the scaling introduced during the FFT
    gpuFFT::ComplexToCroppedNormalized(d_CASImgsComplex, d_imgs, CASImgSize, ImgSize, numImgs, stream);

    return;
}

template <typename T>
void gpuFFT::PadVolume(T *d_inputVol, T *d_outputVol, int inputImgSize, int outputImgSize)
{
    // Pad a GPU allocated array (of dimensions 3) with zeros
    // Note: Output volume is larger than the input volume

    int padding = (outputImgSize - inputImgSize) / 2;

    // Check the input parameters
    if (inputImgSize <= 0)
    {
        std::cerr << "Error PadVolume(): inputImgSize must be a positive integer." << '\n';
        return;
    }
    else if (outputImgSize <= 0 || outputImgSize < inputImgSize)
    {
        std::cerr << "Error PadVolume(): The output image size must be larger than the input image size." << '\n';
        return;
    }

    // Define CUDA kernel launch dimensions
    int gridSize = ceil(outputImgSize / 32);
    int blockSize = 32; // i.e. 32*32 threads

    dim3 dimGrid(gridSize, gridSize, 1);
    dim3 dimBlock(blockSize, blockSize, 1);

    PadVolumeKernel<<<dimGrid, dimBlock>>>(d_inputVol, d_outputVol, inputImgSize, outputImgSize, padding);

    return;
}

void gpuFFT::RealToComplex(float *d_Real, cufftComplex *d_Complex, int imgSize, int nSlices)
{
    // Convert a real GPU allocated array (of dimensions 3) to a cufftComplex type

    // Check the input parameters
    if (imgSize <= 0)
    {
        std::cerr << "Error RealToComplex(): imgSize must be a positive integer." << '\n';
        return;
    }
    else if (nSlices <= 0)
    {
        std::cerr << "Error RealToComplex(): nSlices must be a positive integer." << '\n';
        return;
    }

    // Define CUDA kernel launch dimensions
    int gridSize = ceil(imgSize / 32);
    int blockSize = 32; // i.e. 32*32 threads

    dim3 dimGrid(gridSize, gridSize, 1);
    dim3 dimBlock(blockSize, blockSize, 1);

    RealToComplexKernel2<<<dimGrid, dimBlock>>>(d_Real, d_Complex, imgSize, nSlices);

    return;
}

template <typename T>
void gpuFFT::cufftShift_2D(T *d_Array, int imgSize, int nSlices, cudaStream_t &stream)
{
    // Run an inplace 2D FFT shift on a GPU array

    // Check the input parameters
    if (imgSize <= 0)
    {
        std::cerr << "Error cufftShift_2D(): imgSize must be a positive integer." << '\n';
        return;
    }
    else if (nSlices <= 0)
    {
        std::cerr << "Error cufftShift_2D(): nSlices must be a positive integer." << '\n';
        return;
    }

    // Define CUDA kernel launch dimensions
    int gridSize = ceil(imgSize / 32);
    int blockSize = 32; // i.e. 32*32 threads

    dim3 dimGrid(gridSize, gridSize, 1);
    dim3 dimBlock(blockSize, blockSize, 1);

    // Use the CUDA stream if one was provided
    if (stream != NULL)
    {
        cufftShift_2D_kernel<<<dimGrid, dimBlock, 0, stream>>>(d_Array, imgSize, nSlices);
    }
    else
    {
        cufftShift_2D_kernel<<<dimGrid, dimBlock>>>(d_Array, imgSize, nSlices);
    }

    return;
}

template <typename T>
void gpuFFT::cufftShift_3D(T *d_Complex, int imgSize, int nSlices)
{
    // Run an inplace 3D FFT shift on a cufftComplex type

    // Check the input parameters
    if (imgSize <= 0)
    {
        std::cerr << "Error cufftShift_3D(): imgSize must be a positive integer." << '\n';
        return;
    }
    else if (nSlices <= 0)
    {
        std::cerr << "Error cufftShift_3D(): nSlices must be a positive integer." << '\n';
        return;
    }

    // Define CUDA kernel launch dimensions
    int gridSize = ceil(imgSize / 32);
    int blockSize = 32; // i.e. 32*32 threads

    dim3 dimGrid(gridSize, gridSize, 1);
    dim3 dimBlock(blockSize, blockSize, 1);

    cufftShift_3D_kernel<<<dimGrid, dimBlock>>>(d_Complex, imgSize, nSlices);

    return;
}

template <typename R>
void gpuFFT::ComplexImgsToCAS(cufftComplex *d_Complex, R *d_Real, int imgSize)
{
    // Run an inplace 3D FFT shift on a cufftComplex type

    // Check the input parameter
    if (imgSize <= 0)
    {
        std::cerr << "Error ComplexImgsToCAS(): imgSize must be a positive integer." << '\n';
        return;
    }

    // Define CUDA kernel launch dimensions
    int gridSize = ceil(imgSize / 32);
    int blockSize = 32; // i.e. 32*32 threads

    dim3 dimGrid(gridSize, gridSize, 1);
    dim3 dimBlock(blockSize, blockSize, 1);

    ComplexImgsToCASImgs<<<dimGrid, dimBlock>>>(d_Real, d_Complex, imgSize);

    return;
}

void gpuFFT::FowardFFT(cufftComplex *d_Complex, int imgSize)
{
    // Run a forward FFT on a cufftComplex type array

    // Check the input parameter
    if (imgSize <= 0)
    {
        std::cerr << "Error FowardFFT(): imgSize must be a positive integer." << '\n';
        return;
    }

    // Plan the forward FFT
    cufftHandle forwardFFTPlan;
    cufftPlan3d(&forwardFFTPlan, imgSize, imgSize, imgSize, CUFFT_C2C);
    cufftExecC2C(forwardFFTPlan, (cufftComplex *)d_Complex, (cufftComplex *)d_Complex, CUFFT_FORWARD);

    // Free the plan
    cufftDestroy(forwardFFTPlan);
}

void gpuFFT::CASImgsToComplex(float *d_CASImgs, cufftComplex *d_CASImgsComplex, int imgSize, int nSlices, cudaStream_t &stream)
{
    // Convert a CAS array to complex

    // Check the input parameters
    if (imgSize <= 0)
    {
        std::cerr << "Error RealToComplex(): imgSize must be a positive integer." << '\n';
        return;
    }
    else if (nSlices <= 0)
    {
        std::cerr << "Error RealToComplex(): nSlices must be a positive integer." << '\n';
        return;
    }

    // Define CUDA kernel launch dimensions
    int gridSize = ceil(imgSize / 32);
    int blockSize = 32; // i.e. 32*32 threads

    dim3 dimGrid(gridSize, gridSize, 1);
    dim3 dimBlock(blockSize, blockSize, 1);

    // Use the CUDA stream if one was provided
    if (stream != NULL)
    {
        CASImgsToComplexImgs<<<dimGrid, dimBlock, 0, stream>>>(d_CASImgs, d_CASImgsComplex, imgSize, nSlices);
    }
    else
    {
        CASImgsToComplexImgs<<<dimGrid, dimBlock>>>(d_CASImgs, d_CASImgsComplex, imgSize, nSlices);
    }

    return;
}

void gpuFFT::ComplexToCroppedNormalized(cufftComplex *d_Complex, float *d_imgs, int ComplexImgSize, int imgSize, int nSlices, cudaStream_t &stream)
{
    // Convert a CAS array to complex

    // Check the input parameters
    if (imgSize <= 0)
    {
        std::cerr << "Error ComplexToCroppedNormalized(): imgSize must be a positive integer." << '\n';
        return;
    }
    else if (ComplexImgSize <= 0)
    {
        std::cerr << "Error ComplexToCroppedNormalized(): ComplexImgSize must be a positive integer." << '\n';
        return;
    }
    else if (nSlices <= 0)
    {
        std::cerr << "Error ComplexToCroppedNormalized(): nSlices must be a positive integer." << '\n';
        return;
    }

    // Define CUDA kernel launch dimensions
    int gridSize = ceil(imgSize / 32);
    int blockSize = 32; // i.e. 32*32 threads

    dim3 dimGrid(gridSize, gridSize, 1);
    dim3 dimBlock(blockSize, blockSize, 1);

    // Use the CUDA stream if one was provided
    if (stream != NULL)
    {
        ComplexToCroppedNormalizedImgs<<<dimGrid, dimBlock, 0, stream>>>(d_Complex, d_imgs, ComplexImgSize, imgSize, nSlices, ComplexImgSize * ComplexImgSize);
    }
    else
    {
        ComplexToCroppedNormalizedImgs<<<dimGrid, dimBlock>>>(d_Complex, d_imgs, ComplexImgSize, imgSize, nSlices, ComplexImgSize * ComplexImgSize);
    }

    return;
}
