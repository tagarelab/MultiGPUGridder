#include "mexFunctionWrapper.h"
#include "RealToComplexFilter.h"
#include "DeviceMemory.h"

#define Log(x)                  \
    {                           \
        std::cout << x << '\n'; \
    }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if (nrhs != 3)
    {
        mexErrMsgTxt("mexRealToComplex: There should be 4 inputs, (1) Volume as a single array, (2) Volume size as int32, and (3) GPU device number as int32.");
    }

    float *h_InputVolume = (float *)mxGetData(prhs[0]);
    int *InputVolumeSize = (int *)mxGetData(prhs[1]);
    int GPU_Device = (int)mxGetScalar(prhs[2]);
    cudaSetDevice(GPU_Device);

    DeviceMemory<float> *d_InputVolume = new DeviceMemory<float>(3, InputVolumeSize[0], InputVolumeSize[1], InputVolumeSize[2], GPU_Device);
    d_InputVolume->AllocateGPUArray();
    d_InputVolume->CopyToGPU(h_InputVolume);

    DeviceMemory<cufftComplex> *d_ComplexVolume = new DeviceMemory<cufftComplex>(3, InputVolumeSize[0], InputVolumeSize[1], InputVolumeSize[2], GPU_Device);
    d_ComplexVolume->AllocateGPUArray();

    // Convert the real volume to complex type (cufftComplex)
    RealToComplexFilter *RealToComplex = new RealToComplexFilter();
    RealToComplex->SetRealInput(d_InputVolume->GetPointer());
    RealToComplex->SetComplexOutput(d_ComplexVolume->GetPointer());
    RealToComplex->SetVolumeSize(InputVolumeSize[0]);
    RealToComplex->SetNumberOfSlices(InputVolumeSize[2]);
    RealToComplex->Update();

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
    d_ComplexVolume->CopyFromGPU(h_OutputVolume);

    // Copy the data to the Matlab array
    for (int i = 0; i < d_ComplexVolume->length(); i++)
    {
        Output_Real[i] = h_OutputVolume[i].x;
        Output_Complex[i] = h_OutputVolume[i].y;
    }

    // std::memcpy((float *)mxGetData(Matlab_Pointer), h_OutputVolume, sizeof(float) * OutputSize[0] * OutputSize[1] * OutputSize[2]);

    plhs[0] = Matlab_Pointer;
}