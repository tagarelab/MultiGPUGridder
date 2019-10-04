#include "mexFunctionWrapper.h"
#include "FFTShift2DFilter.h"
#include "DeviceMemory.h"

#define Log(x)                  \
    {                           \
        std::cout << x << '\n'; \
    }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if (nrhs != 3)
    {
        mexErrMsgTxt("mexFFTShift2D: There should be 3 inputs, (1) Volume as a single array, (2) Volume size as int32, and (3) GPU device number as int32.");
    }

    float *h_InputVolume = (float *)mxGetData(prhs[0]);
    int *InputVolumeSize = (int *)mxGetData(prhs[1]);
    int GPU_Device = (int)mxGetScalar(prhs[2]);
    cudaSetDevice(GPU_Device);

    DeviceMemory<float> *d_InputVolume = new DeviceMemory<float>(3, InputVolumeSize[0], InputVolumeSize[1], InputVolumeSize[2], GPU_Device);
    d_InputVolume->AllocateGPUArray();
    d_InputVolume->CopyToGPU(h_InputVolume);

    // Run FFTShift on each 2D slice
    FFTShift2DFilter<float> *FFTShiftFilter = new FFTShift2DFilter<float>();
    FFTShiftFilter->SetInput(d_InputVolume->GetPointer());
    FFTShiftFilter->SetImageSize(InputVolumeSize[0]);
    FFTShiftFilter->SetNumberOfSlices(InputVolumeSize[2]);
    FFTShiftFilter->Update();

    mwSize OutputSize[3];
    OutputSize[0] = InputVolumeSize[0];
    OutputSize[1] = InputVolumeSize[1];
    OutputSize[2] = InputVolumeSize[2];

    // Create the output matlab array as type float
    mxArray *Matlab_Pointer = mxCreateNumericArray(3, OutputSize, mxSINGLE_CLASS, mxREAL);

    float *h_OutputVolume = new float[OutputSize[0] * OutputSize[1] * OutputSize[2]];
    d_InputVolume->CopyFromGPU(h_OutputVolume);

    // Copy the data to the Matlab array
    std::memcpy((float *)mxGetData(Matlab_Pointer), h_OutputVolume, sizeof(float) * OutputSize[0] * OutputSize[1] * OutputSize[2]);

    plhs[0] = Matlab_Pointer;
}