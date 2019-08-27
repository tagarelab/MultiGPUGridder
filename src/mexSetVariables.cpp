#include "mexFunctionWrapper.h"
#include "gpuGridder.h"

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
    gpuGridder *gpuGridderObj = convertMat2Ptr<gpuGridder>(prhs[1]);

    // Set the volume size
    if (!strcmp("SetVolumeSize", cmd))
    {

        int *VolumeSize = (int *)mxGetData(prhs[2]);

        std::cout << "SetVolumeSize: " << VolumeSize[0] << '\n';

        gpuGridderObj->SetVolumeSize(VolumeSize[0]);
    }

    // Set the pointer to the volume
    if (!strcmp("SetVolume", cmd))
    {
        gpuGridderObj->SetVolume((float *)mxGetData(prhs[2]));
    }

    // Set the pointer to the CAS volume
    if (!strcmp("SetCASVolume", cmd))
    {
        gpuGridderObj->SetCASVolume((float *)mxGetData(prhs[2]));
    }

    // Set the output projection images size
    if (!strcmp("SetImageSize", cmd))
    {
        gpuGridderObj->SetImageSize((int *)mxGetData(prhs[2]));
    }

    // Set the pointer to the output images
    if (!strcmp("SetImages", cmd))
    {
        gpuGridderObj->SetImages((float *)mxGetData(prhs[2]));
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
    
        gpuGridderObj->SetGPU(0);
    }

    
}