#include "mexFunctionWrapper.h"
#include "MultiplyVolumeFilter.h"
#include "DeviceMemory.h"

#define Log(x)                  \
    {                           \
        std::cout << x << '\n'; \
    }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if (nrhs != 4)
    {
        mexErrMsgTxt("mexMultiplyVolume: There should be 4 inputs, (1) VolumeOne as a single array, (2) VolumeTwo as a single array, (3) Volume size as int32, and (4) GPU device number as int32.");
    }

    float *h_VolumeOne = (float *)mxGetData(prhs[0]);
    float *h_VolumeTwo = (float *)mxGetData(prhs[1]);
    int *VolumeSize = (int *)mxGetData(prhs[2]);
    int GPU_Device = (int)mxGetScalar(prhs[3]);
    cudaSetDevice(GPU_Device);
    cudaDeviceReset();

    DeviceMemory<float> *d_VolumeOne = new DeviceMemory<float>(3, VolumeSize[0], VolumeSize[1], VolumeSize[2], GPU_Device);
    d_VolumeOne->AllocateGPUArray();
    d_VolumeOne->CopyToGPU(h_VolumeOne);

    DeviceMemory<float> *d_VolumeTwo = new DeviceMemory<float>(3, VolumeSize[0], VolumeSize[1], VolumeSize[2], GPU_Device);
    d_VolumeTwo->AllocateGPUArray();
    d_VolumeTwo->CopyToGPU(h_VolumeTwo);

    // Add the two volumes together
    MultiplyVolumeFilter<float>* MultiplyFilter = new MultiplyVolumeFilter<float>();
    MultiplyFilter->SetVolumeSize(VolumeSize[0]);
    MultiplyFilter->SetNumberOfSlices(VolumeSize[2]);
    MultiplyFilter->SetVolumeOne(d_VolumeOne->GetPointer());
    MultiplyFilter->SetVolumeTwo(d_VolumeTwo->GetPointer());
    MultiplyFilter->Update();

    // Create the output matlab array as type float
    mwSize OutputSize[3];
    OutputSize[0] = VolumeSize[0];
    OutputSize[1] = VolumeSize[1];
    OutputSize[2] = VolumeSize[2];

    mxArray *Matlab_Pointer = mxCreateNumericArray(3, OutputSize, mxSINGLE_CLASS, mxREAL);

    float *h_OutputVolume = new float[OutputSize[0] * OutputSize[1] * OutputSize[2]];
    d_VolumeOne->CopyFromGPU(h_OutputVolume);

    // Copy the data to the Matlab array
    std::memcpy((float *)mxGetData(Matlab_Pointer), h_OutputVolume, sizeof(float) * OutputSize[0] * OutputSize[1] * OutputSize[2]);

    plhs[0] = Matlab_Pointer;
}