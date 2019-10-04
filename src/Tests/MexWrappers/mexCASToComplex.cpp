#include "mexFunctionWrapper.h"
#include "CASToComplexFilter.h"
#include "DeviceMemory.h"

#define Log(x)                  \
    {                           \
        std::cout << x << '\n'; \
    }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if (nrhs != 3)
    {
        mexErrMsgTxt("mexCASToComplex: There should be 3 inputs, (1) Volume as a single array, (2) Volume size as int32, and (3) GPU device number as int32.");
    }

    float *h_InputVolume = (float *)mxGetData(prhs[0]);
    int *InputVolumeSize = (int *)mxGetData(prhs[1]);
    int GPU_Device = (int)mxGetScalar(prhs[2]);
    cudaSetDevice(GPU_Device);

    DeviceMemory<float> *d_CASVolume = new DeviceMemory<float>(3, InputVolumeSize[0], InputVolumeSize[1], InputVolumeSize[2], GPU_Device);
    d_CASVolume->AllocateGPUArray();
    d_CASVolume->CopyToGPU(h_InputVolume);

    DeviceMemory<cufftComplex> *d_ComplexOutput = new DeviceMemory<cufftComplex>(3, InputVolumeSize[0], InputVolumeSize[1], InputVolumeSize[2], GPU_Device);
    d_ComplexOutput->AllocateGPUArray();

    // Convert the CAS volume to cufftComplex type
    CASToComplexFilter *CASFilter = new CASToComplexFilter();
    CASFilter->SetCASVolume(d_CASVolume->GetPointer());
    CASFilter->SetComplexOutput(d_ComplexOutput->GetPointer());
    CASFilter->SetVolumeSize(InputVolumeSize[0]);
    CASFilter->SetNumberOfSlices(InputVolumeSize[2]);
    CASFilter->Update();

    mwSize OutputSize[3];
    OutputSize[0] = InputVolumeSize[0];
    OutputSize[1] = InputVolumeSize[1];
    OutputSize[2] = InputVolumeSize[2];

    // Create the output matlab array as type float
    mxArray *Matlab_Pointer = mxCreateNumericArray(3, OutputSize, mxSINGLE_CLASS, mxCOMPLEX);

    float *Output_Real, *Output_Complex;
    Output_Real = (float *)mxGetPr(Matlab_Pointer);
    Output_Complex = (float *)mxGetPi(Matlab_Pointer);

    cufftComplex *h_OutputVolume = new cufftComplex[OutputSize[0] * OutputSize[1] * OutputSize[2]];
    d_ComplexOutput->CopyFromGPU(h_OutputVolume);

    // Copy the data to the Matlab array
    for (int i = 0; i < d_ComplexOutput->length(); i++)
    {
        Output_Real[i] = h_OutputVolume[i].x;
        Output_Complex[i] = h_OutputVolume[i].y;
    }

    // std::memcpy((float *)mxGetData(Matlab_Pointer), h_OutputVolume, sizeof(float) * OutputSize[0] * OutputSize[1] * OutputSize[2]);

    plhs[0] = Matlab_Pointer;
}