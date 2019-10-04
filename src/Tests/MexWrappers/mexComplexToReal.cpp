#include "mexFunctionWrapper.h"
#include "ComplexToRealFilter.h"
#include "DeviceMemory.h"

#define Log(x)                  \
    {                           \
        std::cout << x << '\n'; \
    }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if (nrhs != 3)
    {
        mexErrMsgTxt("mexComplexToReal: There should be 4 inputs, (1) Volume as a complex single array, (2) Volume size as int32, and (3) GPU device number as int32.");
    }

    float *Input_Real, *Input_Complex;
    Input_Real = (float *)mxGetPr(prhs[0]);
    Input_Complex = (float *)mxGetPi(prhs[0]);
    int *InputVolumeSize = (int *)mxGetData(prhs[1]);
    int GPU_Device = (int)mxGetScalar(prhs[2]);
    cudaSetDevice(GPU_Device);

    cufftComplex *h_ComplexVolume = new cufftComplex[InputVolumeSize[0] * InputVolumeSize[1] * InputVolumeSize[2]];
    
    for (int i = 0; i < InputVolumeSize[0] * InputVolumeSize[1] * InputVolumeSize[2]; i++)
    {
        h_ComplexVolume[i].x = Input_Real[i];
        h_ComplexVolume[i].y = Input_Complex[i];
    }

    DeviceMemory<cufftComplex> *d_ComplexVolume = new DeviceMemory<cufftComplex>(3, InputVolumeSize[0], InputVolumeSize[1], InputVolumeSize[2], GPU_Device);
    d_ComplexVolume->AllocateGPUArray();
    d_ComplexVolume->CopyToGPU(h_ComplexVolume);

    DeviceMemory<float> *d_RealVolume = new DeviceMemory<float>(3, InputVolumeSize[0], InputVolumeSize[1], InputVolumeSize[2], GPU_Device);
    d_RealVolume->AllocateGPUArray();

    // Convert the real volume to complex type (cufftComplex)
    ComplexToRealFilter *ComplexToReal = new ComplexToRealFilter();
    ComplexToReal->SetComplexInput(d_ComplexVolume->GetPointer());
    ComplexToReal->SetRealOutput(d_RealVolume->GetPointer());
    ComplexToReal->SetVolumeSize(InputVolumeSize[0]);
    ComplexToReal->SetNumberOfSlices(InputVolumeSize[2]);
    ComplexToReal->Update();

    mwSize OutputSize[3];
    OutputSize[0] = InputVolumeSize[0];
    OutputSize[1] = InputVolumeSize[1];
    OutputSize[2] = InputVolumeSize[2];

    // Create the output matlab array as type float
    mxArray *Matlab_Pointer = mxCreateNumericArray(3, OutputSize, mxSINGLE_CLASS, mxREAL);

    float *h_OutputVolume = new float[OutputSize[0] * OutputSize[1] * OutputSize[2]];
    d_RealVolume->CopyFromGPU(h_OutputVolume);

    // Copy the data to the Matlab array
    std::memcpy((float *)mxGetData(Matlab_Pointer), h_OutputVolume, sizeof(float) * OutputSize[0] * OutputSize[1] * OutputSize[2]);

    plhs[0] = Matlab_Pointer;
}