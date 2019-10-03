#include "mexFunctionWrapper.h"
#include "CropVolumeFilter.h"
#include "DeviceMemory.h"

#define Log(x)                  \
    {                           \
        std::cout << x << '\n'; \
    }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if (nrhs != 6)
    {
        mexErrMsgTxt("mexPadVolume: There should be 6 inputs, (1) Volume as a single array, (2) Volume size as int32, (3) X crop, (4) Y crop, (5) Z crop, and (6) GPU device number as int32.");
    }

    float *h_InputVolume = (float *)mxGetData(prhs[0]);
    int *InputVolumeSize = (int *)mxGetData(prhs[1]);
    int CropX = (int)mxGetScalar(prhs[2]);
    int CropY = (int)mxGetScalar(prhs[3]);
    int CropZ = (int)mxGetScalar(prhs[4]);
    int GPU_Device = (int)mxGetScalar(prhs[5]);
    cudaSetDevice(GPU_Device);

    DeviceMemory<float> *d_InputVolume = new DeviceMemory<float>(3, InputVolumeSize[0], InputVolumeSize[1], InputVolumeSize[2], GPU_Device);
    d_InputVolume->AllocateGPUArray();
    d_InputVolume->CopyToGPU(h_InputVolume);

    mwSize OutputSize[3];
    OutputSize[0] = InputVolumeSize[0] - CropX * 2;
    OutputSize[1] = InputVolumeSize[1] - CropY * 2;
    OutputSize[2] = InputVolumeSize[2] - CropZ * 2;

    DeviceMemory<float> *d_OutputVolume = new DeviceMemory<float>(3, OutputSize[0], OutputSize[1], OutputSize[2], GPU_Device);
    d_OutputVolume->AllocateGPUArray();

    // Crop the input volume
    CropVolumeFilter *CropFilter = new CropVolumeFilter();
    CropFilter->SetInput(d_InputVolume->GetPointer());
    CropFilter->SetInputSize(InputVolumeSize[0]);
    CropFilter->SetNumberOfSlices(InputVolumeSize[2]);
    CropFilter->SetOutput(d_OutputVolume->GetPointer());
    CropFilter->SetCropX(CropX);
    CropFilter->SetCropY(CropY);
    CropFilter->SetCropZ(CropZ);
    CropFilter->Update();

    // Create the output matlab array as type float
    mxArray *Matlab_Pointer = mxCreateNumericArray(3, OutputSize, mxSINGLE_CLASS, mxREAL);

    float *h_OutputVolume = new float[OutputSize[0] * OutputSize[1] * OutputSize[2]];
    d_OutputVolume->CopyFromGPU(h_OutputVolume);

    // Copy the data to the Matlab array
    std::memcpy((float *)mxGetData(Matlab_Pointer), h_OutputVolume, sizeof(float) * OutputSize[0] * OutputSize[1] * OutputSize[2]);

    plhs[0] = Matlab_Pointer;
}