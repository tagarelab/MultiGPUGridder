#include "mexFunctionWrapper.h"
#include "MultiGPUGridder.h"

#define Log(x)                  \
    {                           \
        std::cout << x << '\n'; \
    }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 3 && nrhs != 4)
    {
        mexErrMsgTxt("mexCreateClass: There should be 3 inputs: Volume size, number of coordinate axes, and the interpolation factor.");
    }

    // Get the input command string
    char cmd[64];
    if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd)))
    {
        mexErrMsgTxt("First input should be a command string less than 64 characters long.");
    }

    // Get the class instance pointer from the second input
    MultiGPUGridder *MultiGPUGridderObj = convertMat2Ptr<MultiGPUGridder>(prhs[1]);

    // Set the pointer to the volume
    if (!strcmp("SetVolume", cmd))
    {
        // Pointer to the volume array and the dimensions of the array
        MultiGPUGridderObj->SetVolume((float *)mxGetData(prhs[2]), (int *)mxGetData(prhs[3]));
    }

    // Set the pointer to the coordinate axes
    if (!strcmp("SetCoordAxes", cmd))
    {
        // Pointer to the volume array and the dimensions of the array
        MultiGPUGridderObj->SetCoordAxes((float *)mxGetData(prhs[2]), (int *)mxGetData(prhs[3]));

    }

    // Set the mask radius parameter
    if (!strcmp("SetMaskRadius", cmd))
    {
        MultiGPUGridderObj->SetMaskRadius((float)mxGetScalar(prhs[2]));
    }

    // Set the pointer to the CAS volume
    if (!strcmp("SetCASVolume", cmd))
    {
        // Pointer to the volume array and the dimensions of the array
        MultiGPUGridderObj->SetCASVolume((float *)mxGetData(prhs[2]), (int *)mxGetData(prhs[3]));
    }

    // Set the maximum coordinate axes to allocate on the GPU
    if (!strcmp("SetMaxAxesToAllocate", cmd))
    {
        MultiGPUGridderObj->SetMaxAxesToAllocate((int)mxGetScalar(prhs[2]));
    }
    
    // Set the number of CUDA streams to use with each GPU
    if (!strcmp("SetNumberStreams", cmd))
    {
        MultiGPUGridderObj->SetNumStreams((int)mxGetScalar(prhs[2]));
    }

    // Set the pointer to the output images
    if (!strcmp("SetImages", cmd))
    {
        // Pointer to the images array and the dimensions of the array
        MultiGPUGridderObj->SetImages((float *)mxGetData(prhs[2]), (int *)mxGetData(prhs[3]));
    }

    // Set the pointer to the CAS images
    if (!strcmp("SetCASImages", cmd))
    {
        // Pointer to the CAS images array and the dimensions of the array
        MultiGPUGridderObj->SetCASImages((float *)mxGetData(prhs[2]), (int *)mxGetData(prhs[3]));
    }

    // Set the pointer to the Plane Density volume
    if (!strcmp("SetPlaneDensity", cmd))
    {
        // Pointer to the volume array and the dimensions of the array
        MultiGPUGridderObj->SetPlaneDensity((float *)mxGetData(prhs[2]), (int *)mxGetData(prhs[3]));
    }

    // Set the pointer to the Kaiser Bessel precompensation array (set using Matlab's getPreComp() functipn)
    if (!strcmp("SetKBPreCompArray", cmd))
    {
        // Pointer to the volume array and the dimensions of the array
        MultiGPUGridderObj->SetKBPreCompArray((float *)mxGetData(prhs[2]), (int *)mxGetData(prhs[3]));
    }
    
    // Set the GPUS to use
    if (!strcmp("SetGPUs", cmd))
    {
        // int *GPU_Array = (int *)mxGetData(prhs[2]);
        // int Number_GPUs = (int)mxGetScalar(prhs[3]);

        // Log("GPU_Array:");
        // Log(GPU_Array[0]);

        // Log("Number_GPUs:");
        // Log(Number_GPUs);

        // MultiGPUGridderObj->SetGPU(0);
    }

    // Set the Kaiser Bessel lookup table
    if (!strcmp("SetKBTable", cmd))
    {
        // KB Lookup table; Length of the table
        MultiGPUGridderObj->SetKerBesselVector((float *)mxGetData(prhs[2]), (int *)mxGetData(prhs[3]));
    }
}