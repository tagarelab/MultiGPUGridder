#include "mexFunctionWrapper.h"
#include "DivideScalarFilter.h"
#include "DeviceMemory.h"

#define Log(x)                  \
    {                           \
        std::cout << x << '\n'; \
    }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if (nrhs != 4)
    {
        mexErrMsgTxt("mexPadVolume: There should be 4 inputs, (1) Volume as a single array, (2) Volume size as int32, (3) scalar (as int32), and (4) GPU device number as int32.");
    }

    float *h_InputVolume = (float *)mxGetData(prhs[0]);
    int *InputVolumeSize = (int *)mxGetData(prhs[1]);
    float *Scalar = (float *)mxGetData(prhs[2]);
    int GPU_Device = (int)mxGetScalar(prhs[3]);
    cudaSetDevice(GPU_Device);

    DeviceMemory<float> *d_InputVolume = new DeviceMemory<float>(3, InputVolumeSize[0], InputVolumeSize[1], InputVolumeSize[2], GPU_Device);
    d_InputVolume->AllocateGPUArray();
    d_InputVolume->CopyToGPU(h_InputVolume);

    // Divide the volume by the scalar
    DivideScalarFilter *DivideScalar = new DivideScalarFilter();
    DivideScalar->SetInput(d_InputVolume->GetPointer());
    DivideScalar->SetScalar(float(Scalar[0]));
    DivideScalar->SetVolumeSize(InputVolumeSize[0]);
    DivideScalar->SetNumberOfSlices(InputVolumeSize[2]);
    DivideScalar->Update();

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