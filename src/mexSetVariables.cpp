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

    // // Set the volume size
    // if (!strcmp("SetVolumeSize", cmd))
    // {
    //     int *VolumeSize = (int *)mxGetData(prhs[2]);

    //     std::cout << "SetVolumeSize: " << VolumeSize[0] << '\n';

    //     gpuGridderObj->SetVolumeSize(VolumeSize[0]);

    //     // DEBUG
    //     int* volSize = gpuGridderObj->GetVolumeSize();
    //     std::cout << "volSize: " << volSize[0] << " " << volSize[1] << " " << volSize[2] << '\n';

    // }

    // Set the pointer to the volume
    if (!strcmp("SetVolume", cmd))
    {
        // Pointer to the volume array and the dimensions of the array
        gpuGridderObj->SetVolume((float *)mxGetData(prhs[2]), (int*)mxGetData(prhs[3]));
    }

    // Set the pointer to the coordinate axes
    if (!strcmp("SetCoordAxes", cmd))
    {
        // Pointer to the volume array and the dimensions of the array
        gpuGridderObj->SetCoordAxes((float *)mxGetData(prhs[2]), (int*)mxGetData(prhs[3]));

        // Set the number of coordinate axes
        // gpuGridderObj->SetNumAxes((int)mxGetScalar(prhs[3]));
    }

    // Set the pointer to the CAS volume
    if (!strcmp("SetCASVolume", cmd))
    {
        // Pointer to the volume array and the dimensions of the array
        gpuGridderObj->SetCASVolume((float *)mxGetData(prhs[2]), (int*)mxGetData(prhs[3]));

    }

    // Set the output projection images size
    if (!strcmp("SetImageSize", cmd))
    {
        gpuGridderObj->SetImageSize((int *)mxGetData(prhs[2]));
    }

    // Set the pointer to the output images
    if (!strcmp("SetImages", cmd))
    {
        // Pointer to the images array and the dimensions of the array
        gpuGridderObj->SetImages((float *)mxGetData(prhs[2]), (int*)mxGetData(prhs[3]));
    }

    // Set the pointer to the CAS images
    if (!strcmp("SetCASImages", cmd))
    {
        // Pointer to the CAS images array and the dimensions of the array
        gpuGridderObj->SetCASImages((float *)mxGetData(prhs[2]), (int*)mxGetData(prhs[3]));

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

    // Set the Kaiser Bessel lookup table
    if (!strcmp("SetKBTable", cmd))
    {
        // KB Lookup table; Length of the table
        gpuGridderObj->SetKerBesselVector((float *)mxGetData(prhs[2]), (int*)mxGetData(prhs[3]));
    }
}