#include "gpuFFT.h"

gpuFFT::gpuFFT(/* args */)
{
}

gpuFFT::~gpuFFT()
{
}


__global__ void PadVolumeKernel(float* input, float* output, int intputImgSize, int outputImgSize, int padding)
{
    // Zero pad a volume using the GPU

    // Index of the output (smaller) image
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row
    

    // Are we outside the bounds of the image?
    if (i >= intputImgSize || i < 0 || j >= intputImgSize || j < 0){
        return;
    }


    // // Iterate over the input image (i.e. the smaller image)
    for (int k = 0; k < intputImgSize; k++){

        // Get the linear index of the output (smaller) image
        int ndx_1 = i + j * intputImgSize + k * intputImgSize * intputImgSize;   

        // Get the linear index of the output (larger) image    
        int ndx_2 = 
        (i + padding) + 
        (j + padding) * outputImgSize +
        (k + padding) * outputImgSize *  outputImgSize;  

        output[ndx_1] = input[ndx_2];
    }
}


float *gpuFFT::PadVolume(float *inputVol, int inputImgSize, int outputImgSize)
{
    // Pad a volume (of dimensions 3) with zeros
    // Note: Output volume is larger than the input volume

    // Check the input parameters
    if(inputImgSize <=0)
    {
        std::cerr << "CropVolume(): Invalid image size." << '\n';
    }

    // Create the output volume
    //int outputImgSize = interpFactor * inputImgSize;

    float *outputVol = new float[outputImgSize * outputImgSize * outputImgSize];

    for (int i = 0; i < outputImgSize * outputImgSize * outputImgSize; i++)
    {
        outputVol[i] = 0; // Initilize the output volume to zeros first
    }

    std::cout << "Output volume size: " << outputImgSize << '\n';

    // How much to crop on each side?
    int padding = (outputImgSize - inputImgSize) / 2;

    bool use_gpu = true;

    if (use_gpu == true)
    {
        std::cout << "Padding: " << padding << '\n';
        std::cout << "inputImgSize: " << inputImgSize << '\n';
        std::cout << "outputImgSize: " << outputImgSize << '\n';

        // Allocate GPU memory to hold the input and output arrays
        float *d_input; 
        cudaMalloc(&d_input, sizeof(float) * inputImgSize * inputImgSize * inputImgSize);
        float *d_output; 
        cudaMalloc(&d_output, sizeof(float) * outputImgSize * outputImgSize * outputImgSize);


        // Run kernel to pad the intput array
        int gridSize = 32;
        int blockSize = inputImgSize / gridSize;

        dim3 dimGridCrop(gridSize, gridSize, 1);
        dim3 dimBlockCrop(blockSize, blockSize, 1);

        PadVolumeKernel<<< dimGridCrop, dimBlockCrop>>>(d_input, d_output, inputImgSize, outputImgSize, padding);

        // Copy the result back to the host
        cudaMemcpy(outputVol, d_output, sizeof(float) * outputImgSize * outputImgSize * outputImgSize, cudaMemcpyDeviceToHost);

        // Free the GPU memory
        cudaFree(d_input);
        cudaFree(d_output);

    } else 
    {
        // Iterate over the input image (i.e. the smaller image)
        for (int i = 0; i < inputImgSize; i++)
        {
            for (int j = 0; j < inputImgSize; j++)
            {
                for (int k = 0; k < inputImgSize; k++)
                {

                    int input_ndx = i + j*inputImgSize + k*inputImgSize*inputImgSize;

                    int output_ndx = (i + padding) + (j+padding)*outputImgSize + (k+padding)*outputImgSize*outputImgSize;

                    outputVol[output_ndx] = inputVol[input_ndx];
                }
            }
        }
    }
    
    return outputVol;

}