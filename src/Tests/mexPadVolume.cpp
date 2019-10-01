#include "mexFunctionWrapper.h"
#include "PadVolumeFilter.h"
#include "DeviceMemory.h"

#define Log(x)                  \
    {                           \
        std::cout << x << '\n'; \
    }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if (nrhs != 6)
    {
        mexErrMsgTxt("mexCreateClass: There should be 6 inputs: Volume size, number of coordinate axes, the interpolation factor, number of GPUs, a vector of GPU device numbers, and a flag to run the FFTs on the GPU or not.");
    }

    int PaddingX = 10;
    int PaddingY = 10;
    int PaddingZ = 0;

    int GPU_Device = 0;

    float *h_InputVolume = (float *)mxGetData(prhs[0]);
    int *InputVolumeSize = (int *)mxGetData(prhs[1]);

    d_InputVolume = new DeviceMemory<float>(3, InputVolumeSize[0], InputVolumeSize[1], InputVolumeSize[2], GPU_Device);
    d_InputVolume->AllocateGPUArray();
    d_InputVolume->CopyToGPU(h_InputVolume);

    mwSize OutputSize[3];
    OutputSize[0] = InputVolumeSize[0] + PaddingX * 2;
    OutputSize[1] = InputVolumeSize[1] + PaddingY * 2;
    OutputSize[2] = InputVolumeSize[2] + PaddingZ * 2;

    d_OutputVolume = new DeviceMemory<float>(3, OutputSize[0], OutputSize[1], OutputSize[2], GPU_Device);
    d_OutputVolume->AllocateGPUArray();

    // Pad the input volume with zeros
    PadVolumeFilter *PadFilter = new PadVolumeFilter();
    PadFilter->SetInput(d_InputVolume->GetPointer());
    PadFilter->SetOutput(d_OutputVolume->GetPointer());
    PadFilter->SetInputSize(InputVolumeSize[0]);
    PadFilter->SetNumberOfSlices(InputVolumeSize[2]);
    PadFilter->SetPaddingX(PaddingX);
    PadFilter->SetPaddingY(PaddingY);
    PadFilter->SetPaddingZ(PaddingZ);
    PadFilter->Update();

    // Create the output matlab array as type float
    mxArray *Matlab_Pointer = mxCreateNumericArray(3, OutputSize, mxSINGLE_CLASS, mxREAL);

    h_OutputVolume = new float[OutputSize[0] * OutputSize[1] * OutputSize[2]];
    d_OutputVolume->CopyFromGPU(h_OutputVolume);

    // Copy the data to the Matlab array
    std::memcpy((float *)mxGetData(Matlab_Pointer), h_OutputVolume, sizeof(float) * OutputSize[0] * OutputSize[1] * OutputSize[2]);
}