#include "mexFunctionWrapper.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // This is the wrapper for the corresponding Matlab class
    // Which allows for calling the C++ and CUDA functions while maintaining the memory pointers

    // Get the input command string
    char cmd[64];
    if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd)))
    {
        mexErrMsgTxt("First input should be a command string less than 64 characters long.");
    }

    // Create a new instance of the class
    if (!strcmp("new", cmd))
    {
        // Check parameters
        if (nlhs != 1)
        {
            mexErrMsgTxt("New: One output expected.");
        }

        // Return a handle to a new C++ instance
        plhs[0] = convertPtr2Mat<CUDA_Gridder>(new CUDA_Gridder);
        return;
    }

    // Get the class instance pointer from the second input
    CUDA_Gridder *CUDA_Gridder_instance = convertMat2Ptr<CUDA_Gridder>(prhs[1]);

    // Deleted the instance of the class
    if (!strcmp("delete", cmd))
    {
        // Free the memory of all the CPU arrays
        CUDA_Gridder_instance->Mem_obj->mem_Free("all");

        // Free the memory of all the GPU CUDA arrays
        CUDA_Gridder_instance->Mem_obj->CUDA_Free("all");

        // Destroy the C++ object
        destroyObject<CUDA_Gridder>(prhs[1]);

        // Warn if other commands were ignored
        if (nlhs != 0 || nrhs != 2)
        {
            mexWarnMsgTxt("Delete: Unexpected arguments ignored.");
        }

        return;
    }

    // Set the volume on all of the GPUs
    if (!strcmp("SetVolume", cmd))
    {
        // Check parameters
        if (nrhs != 3)
        {
            mexErrMsgTxt("SetVolume: Unexpected arguments. Please provide a Matlab array.");
        }

        // Get a pointer to the matlab array
        float *matlabArrayPtr = (float *)mxGetData(prhs[2]);

        // Get the matrix size of the input GPU volume
        const mwSize *dims_mwSize;
        dims_mwSize = mxGetDimensions(prhs[2]);

        int dims[3];
        dims[0] = (int)dims_mwSize[0];
        dims[1] = (int)dims_mwSize[1];
        dims[2] = (int)dims_mwSize[2];

        mwSize numDims;
        numDims = mxGetNumberOfDimensions(prhs[2]);

        if (numDims != 3)
        {
            mexErrMsgTxt("SetVolume: Unexpected arguments. Array should be a matrix with 3 dimensions.");
        }

        // Call the method
        CUDA_Gridder_instance->SetVolume(matlabArrayPtr, dims);

        return;
    }

    // Return the summed volume from all of the GPUs (for getting the back projection kernel result)
    if (!strcmp("GetVolume", cmd))
    {
        // Check parameters
        if (nrhs != 2)
        {
            mexErrMsgTxt("GetVolume: Unexpected arguments.");
        }

        // Get the matrix size of the GPU volume
        mwSize dims[3];
        dims[0] = CUDA_Gridder_instance->volSize[0];
        dims[1] = CUDA_Gridder_instance->volSize[1];
        dims[2] = CUDA_Gridder_instance->volSize[2];

        // Create the output matlab array as type float
        mxArray *Matlab_Pointer = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);

        // Get a pointer to the output matrix created above
        float *matlabArrayPtr = (float *)mxGetData(Matlab_Pointer);

        // Call the method
        float *GPUVol = CUDA_Gridder_instance->GetVolume();

        // Copy the data to the Matlab array
        std::memcpy(matlabArrayPtr, GPUVol, sizeof(float) * dims[0] * dims[1] * dims[2]);

        plhs[0] = Matlab_Pointer;

        return;
    }

    // Set the CASImgs array to pinned CPU memory
    if (!strcmp("SetImages", cmd))
    {
        // Check parameters
        if (nrhs != 3)
        {
            mexErrMsgTxt("SetImages: Unexpected arguments. Please provide a Matlab array.");
        }

        // Get a pointer to the matlab array
        float *matlabArrayPtr = (float *)mxGetData(prhs[2]);

        // Get the matrix size of the input GPU volume
        const mwSize *dims_mwSize;
        dims_mwSize = mxGetDimensions(prhs[2]);

        int dims[3];
        dims[0] = (int)dims_mwSize[0];
        dims[1] = (int)dims_mwSize[1];
        dims[2] = (int)dims_mwSize[2];

        mwSize numDims;
        numDims = mxGetNumberOfDimensions(prhs[2]);

        if (numDims != 3)
        {
            mexErrMsgTxt("SetVolume: Unexpected arguments. Array should be a matrix with 3 dimensions.");
        }

        // Call the method
        CUDA_Gridder_instance->SetImages(matlabArrayPtr);

        return;
    }

    // Reset the volume on all of the GPUs
    if (!strcmp("ResetVolume", cmd))
    {
        // Check parameters
        if (nrhs != 2)
        {
            mexErrMsgTxt("ResetVolume: Unexpected arguments.");
        }

        // Call the method
        CUDA_Gridder_instance->ResetVolume();

        return;
    }

    // Set the coordinate axes vector to pinned CPU memory
    if (!strcmp("SetAxes", cmd))
    {
        // Check parameters
        if (nrhs != 3)
        {
            mexErrMsgTxt("SetAxes: Unexpected arguments. Please provide a Matlab array.");
        }

        // Get a pointer to the matlab array
        float *matlabArrayPtr = (float *)mxGetData(prhs[2]);

        // Get the matrix size of the input GPU volume
        const mwSize *dims_mwSize;
        dims_mwSize = mxGetDimensions(prhs[2]);

        mwSize numDims;
        numDims = mxGetNumberOfDimensions(prhs[2]);

        int dims[3];
        dims[0] = (int)dims_mwSize[0];
        dims[1] = (int)dims_mwSize[1];
        dims[2] = 1;

        if (numDims != 2)
        {
            mexErrMsgTxt("SetAxes: Unexpected arguments. Array should be a row vector.");
        }

        // Call the method
        CUDA_Gridder_instance->SetAxes(matlabArrayPtr, dims);

        return;
    }

    // Set the size of the output images array
    if (!strcmp("SetImgSize", cmd))
    {
        // Check parameters
        if (nrhs != 3)
        {
            mexErrMsgTxt("SetImgSize: Unexpected arguments. Please provide a row vector.");
        }

        // Get a pointer to the matlab array
        int *matlabArrayPtr = (int *)mxGetData(prhs[2]);

        // Get the matrix size of the input GPU volume
        const mwSize *dims_mwSize;
        dims_mwSize = mxGetDimensions(prhs[2]);

        mwSize numDims;
        numDims = mxGetNumberOfDimensions(prhs[2]);

        int dims[3];
        dims[0] = (int)dims_mwSize[0];
        dims[1] = (int)dims_mwSize[1];
        dims[2] = 1;

        if (numDims != 2)
        {
            mexErrMsgTxt("SetImgSize: Unexpected arguments. Input should be a row vector.");
        }

        // Call the method
        CUDA_Gridder_instance->SetImgSize(matlabArrayPtr);

        return;
    }

    // Set the kernel mask radius
    if (!strcmp("SetMaskRadius", cmd))
    {
        // Check parameters
        if (nrhs != 3)
        {
            mexErrMsgTxt("SetMaskRadius: Unexpected arguments. Please provide a row vector.");
        }

        // Get a pointer to the matlab array
        float *MaskRadius = (float *)mxGetData(prhs[2]);

        // Call the method
        CUDA_Gridder_instance->SetMaskRadius(MaskRadius);

        return;
    }

    // Set the number of GPUs to use when calling the CUDA kernels
    if (!strcmp("SetNumberGPUs", cmd))
    {
        // Check parameters
        if (nrhs != 3)
        {
            mexErrMsgTxt("SetNumberGPUs: Unexpected arguments. Please provide a scalar value.");
        }

        int numGPUS = (int)mxGetScalar(prhs[2]);

        // Call the method
        CUDA_Gridder_instance->SetNumberGPUs(numGPUS);

        return;
    }

    // Set the number of CUDA streams to use when calling the CUDA kernels
    if (!strcmp("SetNumberStreams", cmd))
    {
        // Check parameters
        if (nrhs != 3)
        {
            mexErrMsgTxt("SetNumberStreams: Unexpected arguments. Please provide a scalar value.");
        }

        int nStreams = (int)mxGetScalar(prhs[2]);

        // Call the method
        CUDA_Gridder_instance->SetNumberStreams(nStreams);

        return;
    }

    // Set the number of batches to use when calling the CUDA kernels
    if (!strcmp("SetNumberBatches", cmd))
    {
        // Check parameters
        if (nrhs != 3)
        {
            mexErrMsgTxt("SetNumberBatches: Unexpected arguments. Please provide a scalar value.");
        }

        int nBatches = (int)mxGetScalar(prhs[2]);

        // Call the method
        CUDA_Gridder_instance->SetNumberBatches(nBatches);

        return;
    }

    // Allocate memory on the CPU
    if (!strcmp("mem_alloc", cmd))
    {
        // Check parameters
        if (nrhs != 5)
        {
            mexErrMsgTxt("mem_alloc: Unexpected arguments. Please provide (1) variable name as string, (2) data type as string, and (3) array size (i.e. [256, 256, 256]) as a Matlab vector.");
        }

        char varNameString[64]; // Unique name for the array as a string
        char varDataType[64];   // "int", "float", etc.
        int *varDataSize;       // e.g. [dim1, dim2, dim3]

        mxGetString(prhs[2], varNameString, sizeof(varNameString));
        mxGetString(prhs[3], varDataType, sizeof(varDataType));

        // Get a pointer to the matlab array
        varDataSize = (int *)mxGetData(prhs[4]);

        // Call the method
        CUDA_Gridder_instance->Mem_obj->mem_alloc(varNameString, varDataType, varDataSize);

        return;
    }

    // Pin a CPU array to pinned memory
    if (!strcmp("pin_mem", cmd))
    {
        // Check parameters
        if (nrhs != 3)
        {
            mexErrMsgTxt("pin_mem: Unexpected arguments. Please provide (1) variable name as string.");
        }

        // Corresponding unique name of the array as a string (needs to have been previously allocated using mem_alloc() function)
        char varNameString[64];
        mxGetString(prhs[2], varNameString, sizeof(varNameString));

        // Call the method
        CUDA_Gridder_instance->Mem_obj->pin_mem(varNameString);
        return;
    }

    // Display some information on the currently allocated CPU arrays to the Matlab console
    if (!strcmp("disp_mem", cmd))
    {
        // Check parameters
        if (nrhs != 3)
        {
            mexErrMsgTxt("disp_mem: Unexpected arguments. Please provide (1) variable name as string or 'all' to display information on all the arrays.");
        }

        // Get the unqiue name of the array as a string or an input of "all" will display information on all of the CPU arrays
        char varNameString[64];
        mxGetString(prhs[2], varNameString, sizeof(varNameString));

        // Call the method
        CUDA_Gridder_instance->Mem_obj->disp_mem(varNameString);
        return;
    }

    // Free the CPU memory of a specified array
    if (!strcmp("mem_Free", cmd))
    {
        // Check parameters
        char varNameString[64];

        if (nrhs != 3)
        {
            mexErrMsgTxt("mem_Free: Unexpected arguments. Please provide (1) variable name as string.");
        }

        // Unique name of the array given as a string
        mxGetString(prhs[2], varNameString, sizeof(varNameString));

        // Call the method
        CUDA_Gridder_instance->Mem_obj->mem_Free(varNameString);
        return;
    }

    // Copy an array from Matlab to a previously allocated CPU array
    if (!strcmp("mem_Copy", cmd))
    {
        // Check parameters
        if (nrhs != 4)
        {
            mexErrMsgTxt("mem_Copy: Unexpected arguments. Please provide (1) variable name as string, (2) a Matlab array of same dimensions and type as previously allocated.");
        }

        // Unique name of the array given as a string
        char varNameString[64];
        mxGetString(prhs[2], varNameString, sizeof(varNameString));

        // Get the pointer to the input Matlab array which should be float type (for now)
        float *matlabArrayPtr = (float *)mxGetData(prhs[3]);

        // Call the method
        CUDA_Gridder_instance->Mem_obj->mem_Copy(varNameString, matlabArrayPtr);
        return;
    }

    // Return a CPU array as a Matlab array
    if (!strcmp("mem_Return", cmd))
    {

        // Check parameters
        if (nrhs != 3)
        {
            mexErrMsgTxt("mem_Return: Unexpected arguments. Please provide (1) variable name as string.");
        }

        // Unique name of the array given as a string
        char varNameString[64];
        mxGetString(prhs[2], varNameString, sizeof(varNameString));

        // Create the output Matlab array (using the stored size of the corresponding C++ array)
        int *vol_dims;
        vol_dims = CUDA_Gridder_instance->Mem_obj->CPU_Get_Array_Size(varNameString);

        // Create the output Matlab array (using the stored size of the corresponding C++ array)
        mwSize dims[3]; 
        dims[0] = vol_dims[0];
        dims[1] = vol_dims[1];
        dims[2] = vol_dims[2];

        // Create the output matlab array as type float
        mxArray *Matlab_Pointer = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);

        // Get a pointer to the output matrix created above
        float *matlabArrayPtr = (float *)mxGetData(Matlab_Pointer);

        // Call the method
        float *OutputArray = CUDA_Gridder_instance->Mem_obj->ReturnCPUFloatPtr(varNameString);

        // Copy the data to the Matlab array
        std::memcpy(matlabArrayPtr, OutputArray, sizeof(float) * dims[0] * dims[1] * dims[2]);

        plhs[0] = Matlab_Pointer;

        return;
    }

    // Allocate memory on the GPU
    if (!strcmp("CUDA_alloc", cmd))
    {
        // Check parameters
        if (nrhs != 6)
        {
            mexErrMsgTxt("CUDA_alloc: Unexpected arguments. Please provide (1) variable name as string, (2) data type as string ('int', 'float'), (3) data size as Matlab vector ([256, 256, 256]), (4) GPU to allocate memory to as an integer (0 to number of GPUs minus 1).");
        }

        char varNameString[64]; // Unique name of the array given as a string
        char varDataType[64];   // "float", "int", etc.
        int *varDataSize;       // e.g. [dim1, dim2, dim3]
        int GPU_Device = 0;     // Default to the first GPU

        mxGetString(prhs[2], varNameString, sizeof(varNameString));
        mxGetString(prhs[3], varDataType, sizeof(varDataType));
        varDataSize = (int *)mxGetData(prhs[4]);
        GPU_Device = (int)mxGetScalar(prhs[5]);

        // Call the method
        CUDA_Gridder_instance->Mem_obj->CUDA_alloc(varNameString, varDataType, varDataSize, GPU_Device);
        return;
    }

    // Free memory on the GPU
    if (!strcmp("CUDA_Free", cmd))
    {
        // Check parameters
        if (nrhs != 3)
        {
            mexErrMsgTxt("CUDA_Free: Unexpected arguments. Please provide (1) variable name as string.");
        }

        // Unique name of the array given as a string
        char varNameString[64];
        mxGetString(prhs[2], varNameString, sizeof(varNameString));

        // Call the method
        CUDA_Gridder_instance->Mem_obj->CUDA_Free(varNameString);
        return;
    }

    // Display information of the previously allocated GPU arrays to the Matlab console
    if (!strcmp("CUDA_disp_mem", cmd))
    {
        // Check parameters
        if (nrhs != 3)
        {
            mexErrMsgTxt("CUDA_disp_mem: Unexpected arguments. Please provide (1) variable name as string or 'all' to display information on all the arrays.");
        }

        // Unique name of the array given as a string or use "all" to show information on all of the GPU arrays
        char varNameString[64];
        mxGetString(prhs[2], varNameString, sizeof(varNameString));

        // Call the method
        CUDA_Gridder_instance->Mem_obj->CUDA_disp_mem(varNameString);
        return;
    }

    // Copy a Matlab array to a specified GPU array
    if (!strcmp("CUDA_Copy", cmd))
    {
        // Check parameters
        if (nrhs != 4)
        {
            mexErrMsgTxt("CUDA_Copy: Unexpected arguments. Please provide (1) variable name as string, (2) a Matlab array of same dimensions and type as previously allocated.");
        }

        // Unique name of the array given as a string
        char varNameString[64];
        mxGetString(prhs[2], varNameString, sizeof(varNameString));

        // Get the pointer to the input Matlab array which should be float type (must be the same as previously allocated)
        float *matlabArrayPtr = (float *)mxGetData(prhs[3]);

        // Call the method
        CUDA_Gridder_instance->Mem_obj->CUDA_Copy(varNameString, matlabArrayPtr);
        return;
    }

    // Return a GPU array to Matlab as a Matlab array
    if (!strcmp("CUDA_Return", cmd))
    {
        // Check parameters
        if (nrhs != 3)
        {
            mexErrMsgTxt("CUDA_Return: Unexpected arguments. Please provide (1) variable name as string.");
        }

        // Unique name of the array given as a string
        char varNameString[64];
        mxGetString(prhs[2], varNameString, sizeof(varNameString));

        // Create the output Matlab array (using the stored size of the corresponding C++ array)
        int *vol_dims;
        vol_dims = CUDA_Gridder_instance->Mem_obj->CUDA_Get_Array_Size(varNameString);

        mwSize dims[3];
        dims[0] = vol_dims[0];
        dims[1] = vol_dims[1];
        dims[2] = vol_dims[2];

        // Create the output matlab array as type float
        mxArray *Matlab_Pointer = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);

        // Get a pointer to the output matrix created above
        float *matlabArrayPtr = (float *)mxGetData(Matlab_Pointer);

        // Call the method
        float *GPUVol = CUDA_Gridder_instance->Mem_obj->CUDA_Return(varNameString);

        std::cout << "GPUVol: " << GPUVol[0] << '\n';

        // Copy the data to the Matlab array
        std::memcpy(matlabArrayPtr, GPUVol, sizeof(float) * dims[0] * dims[1] * dims[2]);

        plhs[0] = Matlab_Pointer;

        return;
    }

    // Initilize the inputs of the forward and backwards kernels
    if (!strcmp("Projection_Initilize", cmd))
    {
        // Check parameters
        if (nrhs != 2)
        {
            mexErrMsgTxt("Projection_Initilize: Unexpected arguments. ");
        }

        // Call the method
        CUDA_Gridder_instance->Projection_Initilize();
        return;
    }

    // Run the forward projection CUDA kernel
    if (!strcmp("Forward_Project", cmd))
    {
        // Check parameters
        if (nrhs != 2)
        {
            mexErrMsgTxt("Forward_Project: Unexpected arguments. ");
        }

        // Call the method
        CUDA_Gridder_instance->Forward_Project();
        return;
    }

    // Run the back projection CUDA kernel
    if (!strcmp("Back_Project", cmd))
    {
        // Check parameters
        if (nrhs != 2)
        {
            mexErrMsgTxt("Back_Project: Unexpected arguments. ");
        }

        // Call the method
        CUDA_Gridder_instance->Back_Project();
        return;
    }

    // Check there is a second input, which should be the class instance handle
    if (nrhs < 2)
    {
        mexErrMsgTxt("Second input should be a class instance handle.");
    }

    // Got here, so command was not recognized
    mexErrMsgTxt("Command not recognized. Please check the associated .m file.");
}
