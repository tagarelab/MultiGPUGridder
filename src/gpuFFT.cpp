#include "gpuFFT.h"

gpuFFT::gpuFFT(/* args */)
{
}

gpuFFT::~gpuFFT()
{
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

    // padding = 0;

    std::cout << "Padding: " << padding << '\n';
    std::cout << "inputImgSize: " << inputImgSize << '\n';
    std::cout << "outputImgSize: " << outputImgSize << '\n';

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

    return outputVol;



}