#include "mexFunctionWrapper.h"
#include "MultiGPUGridder.h"

#define Log(x)                  \
    {                           \
        std::cout << x << '\n'; \
    }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if (nrhs != 5)
    {
        mexErrMsgTxt("mexCreateClass: There should be 5 inputs: Volume size, number of coordinate axes, the interpolation factor, number of GPUs, and a vector of GPU device numbers.");
    }

    int *VolumeSize = (int *)mxGetData(prhs[0]);
    int *numCoordAxes = (int *)mxGetData(prhs[1]);
    float *interpFactor = (float *)mxGetData(prhs[2]);
    int Num_GPUs = (int)mxGetScalar(prhs[3]);
    int *GPU_Device = (int *)mxGetData(prhs[4]);

    // Return a handle to a new C++ instance
    plhs[0] = convertPtr2Mat<MultiGPUGridder>(new MultiGPUGridder(*VolumeSize, *numCoordAxes, *interpFactor, Num_GPUs, GPU_Device));

}