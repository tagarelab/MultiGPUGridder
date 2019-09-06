#include "mexFunctionWrapper.h"
#include "gpuGridder.h"

#define Log(x)                  \
    {                           \
        std::cout << x << '\n'; \
    }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

    if (nrhs != 3)
    {
        mexErrMsgTxt("mexCreateClass: There should be 3 inputs: Volume size, number of coordinate axes, and the interpolation factor.");
    }

    int *VolumeSize = (int *)mxGetData(prhs[0]);
    int *numCoordAxes = (int *)mxGetData(prhs[1]);
    float *interpFactor = (float *)mxGetData(prhs[2]);
    // int GPU_Device = 0;

    // Return a handle to a new C++ instance
    Log("Creating gpuGridder class");
    plhs[0] = convertPtr2Mat<gpuGridder>(new gpuGridder(*VolumeSize, *numCoordAxes, *interpFactor));
}