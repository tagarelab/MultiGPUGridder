#include "mexFunctionWrapper.h"
#include "gpuGridder.h"

#define Log(x) {std::cout << x << '\n';}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs != 2)
    {
        mexErrMsgTxt("mexGetVariables: There should be 2 inputs.");
    }

    // Get the input command string
    char cmd[64];
    if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd)))
    {
        mexErrMsgTxt("First input should be a command string less than 64 characters long.");
    }

    // Get the class instance pointer from the second input
    gpuGridder *gpuGridderObj = convertMat2Ptr<gpuGridder>(prhs[1]);

    // Return the summed volume from all of the GPUs (for getting the back projection kernel result)
    if (!strcmp("Volume", cmd))
    {

        // Get the matrix size of the GPU volume
        int* volSize = gpuGridderObj->GetVolumeSize();

        mwSize dims[3];
        dims[0] = volSize[0];
        dims[1] = volSize[1];
        dims[2] = volSize[2];

        std::cout << "dims: " << dims[0] << " " << dims[1] << " " << dims[2] << '\n';
      
        // Create the output matlab array as type float
        mxArray *Matlab_Pointer = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);

        // Call the method
        float *GPUVol = gpuGridderObj->GetVolume();

        // Copy the data to the Matlab array
        std::memcpy((float *)mxGetData(Matlab_Pointer), GPUVol, sizeof(float) * dims[0] * dims[1] * dims[2]);

        plhs[0] = Matlab_Pointer;

        return;
    }

    // Return the CAS volume from pinned memory (for debugging)
    if (!strcmp("CASVolume", cmd))
    {

        // Get the matrix size of the GPU volume
        int CASVolSize = gpuGridderObj->GetCASVolumeSize();

        mwSize dims[3];
        dims[0] = CASVolSize;
        dims[1] = CASVolSize;
        dims[2] = CASVolSize;

        std::cout << "dims: " << dims[0] << " " << dims[1] << " " << dims[2] << '\n';
      
        // Create the output matlab array as type float
        mxArray *Matlab_Pointer = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);

        // Call the method
        float *CASVolume = gpuGridderObj->GetCASVolume();

        // Copy the data to the Matlab array
        std::memcpy((float *)mxGetData(Matlab_Pointer), CASVolume, sizeof(float) * dims[0] * dims[1] * dims[2]);

        plhs[0] = Matlab_Pointer;

        return;
    }

    // Return the CAS images from pinned memory (for debugging)
    if (!strcmp("CASImages", cmd))
    {

        // Get the matrix size of the GPU volume
        int CASVolSize = gpuGridderObj->GetCASVolumeSize();
        int nAxes = gpuGridderObj->GetNumAxes();

        mwSize dims[3];
        dims[0] = CASVolSize;
        dims[1] = CASVolSize;
        dims[2] = nAxes;

        std::cout << "dims: " << dims[0] << " " << dims[1] << " " << dims[2] << '\n';
      
        // Create the output matlab array as type float
        mxArray *Matlab_Pointer = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);

        // Call the method
        float *CASImages = gpuGridderObj->GetCASImgsPtr_CPU();

        // Copy the data to the Matlab array
        std::memcpy((float *)mxGetData(Matlab_Pointer), CASImages, sizeof(float) * dims[0] * dims[1] * dims[2]);

        plhs[0] = Matlab_Pointer;

        return;
    }


    cudaDeviceReset(); // DEBUG

}