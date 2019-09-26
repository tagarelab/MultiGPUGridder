#include "mexFunctionWrapper.h"
#include "MultiGPUGridder.h"

#define Log(x)                  \
    {                           \
        std::cout << x << '\n'; \
    }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if (nrhs != 7)
    {
        mexErrMsgTxt("mexCreateClass: There should be 6 inputs: Volume size, number of coordinate axes, the interpolation factor, number of GPUs, a vector of GPU device numbers, a flag to run the FFTs on the GPU or not, and a flag to normalize the volume using the plane density.");
    }

    int *VolumeSize = (int *)mxGetData(prhs[0]);
    int *numCoordAxes = (int *)mxGetData(prhs[1]);
    float *interpFactor = (float *)mxGetData(prhs[2]);
    int Num_GPUs = (int)mxGetScalar(prhs[3]);
    int *GPU_Device = (int *)mxGetData(prhs[4]);
    int RunFFTOnDevice = (int)mxGetScalar(prhs[5]);
    int NormalizeByDensity = (int)mxGetScalar(prhs[6]);

    // Return a handle to a new C++ instance
    plhs[0] = convertPtr2Mat<MultiGPUGridder>(new MultiGPUGridder(*VolumeSize, *numCoordAxes, *interpFactor, Num_GPUs, GPU_Device, RunFFTOnDevice, NormalizeByDensity));

}