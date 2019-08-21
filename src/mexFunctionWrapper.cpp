#include "mexFunctionWrapper.h"

int *GetMatlabDimensions(const mxArray *MatlabInputPointer)
{

    // Get the matrix size of the input Matlab pointer
    const mwSize *dims_mwSize;
    dims_mwSize = mxGetDimensions(MatlabInputPointer);

    int *dims = new int[3];
    dims[0] = (int)dims_mwSize[0];
    dims[1] = (int)dims_mwSize[1];
    dims[2] = (int)dims_mwSize[2];

    return dims;
}

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
        plhs[0] = convertPtr2Mat<MultiGPUGridder>(new MultiGPUGridder);

        if (nrhs == 2) // No other inputs were provided
        {
            return;
        }
        else if (nrhs == 4)
        {
            // Parameters are: (1) volume size, (2) number of coordinate axes, and (3) interpolation factor

            // (1) Set the volume size
            // Get a pointer to the matlab array
            int *matlabArrayPtr = (int *)mxGetData(prhs[1]);

            // Check the dimensions of the given input
            mwSize numDims;
            numDims = mxGetNumberOfDimensions(prhs[1]);

            std::cout << "numDims: " << numDims << '\n';

            if (numDims != 2)
            {
                mexErrMsgTxt("SetVolumeSize: Input should be a single integer value.");
            }

            MultiGPUGridder *MultiGPUGridder_instance = convertMat2Ptr<MultiGPUGridder>(plhs[0]);

            // (2) Set the number of coordinate axes
            // Get a pointer to the matlab array
            int *matlabArrayPtr_Axes = (int *)mxGetData(prhs[2]);
            matlabArrayPtr_Axes[0] = matlabArrayPtr_Axes[0] * 9;

            // Check the dimensions of the given input
            mwSize numDims_Axes;
            numDims_Axes = mxGetNumberOfDimensions(prhs[2]);

            if (numDims_Axes != 2)
            {
                mexErrMsgTxt("SetAxes: Input should be a single integer value.");
            }

            // Allocate and set the coordinate axes
            MultiGPUGridder_instance->SetAxes(NULL, matlabArrayPtr_Axes);

            // (3) Set the interpolation factor
            // Get a pointer to the matlab array
            int *matlabArrayPtr_InterpFactor = (int *)mxGetData(prhs[3]);

            // Call the method to set the volume size
            // Additional 6 is for an extra padding of 3 in each dimension
            MultiGPUGridder_instance->SetVolumeSize(matlabArrayPtr[0] * matlabArrayPtr_InterpFactor[0] + 6);

            // Set the image size here
            int *imgSize = new int[3];
            imgSize[0] = matlabArrayPtr[0] * matlabArrayPtr_InterpFactor[0];
            imgSize[1] = matlabArrayPtr[0] * matlabArrayPtr_InterpFactor[0];
            imgSize[2] = matlabArrayPtr_Axes[0] / 9; // Number of coordinate axes since axes has 9 elements each

            MultiGPUGridder_instance->SetImgSize(imgSize);

            // Check the dimensions of the given input
            mwSize numDims_InterpFactor;
            numDims_InterpFactor = mxGetNumberOfDimensions(prhs[3]);

            if (numDims_InterpFactor != 2)
            {
                mexErrMsgTxt("SetInterpFactor: Input should be a single integer value.");
            }

            MultiGPUGridder_instance->SetInterpFactor(matlabArrayPtr_InterpFactor[0]);

            // Set the mask radius here
            float *maskRadius = new float;
            maskRadius[0] = (imgSize[0]) / 2 - 1;

            MultiGPUGridder_instance->SetMaskRadius(maskRadius);
        }

        return;
    }

    // Get the class instance pointer from the second input
    MultiGPUGridder *MultiGPUGridder_instance = convertMat2Ptr<MultiGPUGridder>(prhs[1]);

    // Deleted the instance of the class
    if (!strcmp("delete", cmd))
    {
        // Free the memory of all the CPU arrays
       // MultiGPUGridder_instance->Mem_obj->mem_Free("all");

        // Free the memory of all the GPU CUDA arrays
        //MultiGPUGridder_instance->Mem_obj->CUDA_Free("all");

        // Destroy the C++ object
        destroyObject<MultiGPUGridder>(prhs[1]);

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

        // Call the method
        MultiGPUGridder_instance->SetVolume((float *)mxGetData(prhs[2]), GetMatlabDimensions(prhs[2]));

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
        dims[0] = MultiGPUGridder_instance->volSize[0];
        dims[1] = MultiGPUGridder_instance->volSize[1];
        dims[2] = MultiGPUGridder_instance->volSize[2];

        // Create the output matlab array as type float
        mxArray *Matlab_Pointer = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);

        // Call the method
        float *GPUVol = MultiGPUGridder_instance->GetVolume();

        // Copy the data to the Matlab array
        std::memcpy((float *)mxGetData(Matlab_Pointer), GPUVol, sizeof(float) * dims[0] * dims[1] * dims[2]);

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

        // Call the method
        MultiGPUGridder_instance->SetImages((float *)mxGetData(prhs[2]));

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
        MultiGPUGridder_instance->ResetVolume();

        return;
    }

    // Set the coordinate axes vector to pinned CPU memory
    if (!strcmp("setCoordAxes", cmd))
    {
        // Check parameters
        if (nrhs != 3)
        {
            mexErrMsgTxt("setCoordAxes: Unexpected arguments. Please provide a Matlab array.");
        }

        // Call the method
        MultiGPUGridder_instance->SetAxes((float *)mxGetData(prhs[2]), GetMatlabDimensions(prhs[2]));

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

        // Call the method
        MultiGPUGridder_instance->SetImgSize((int *)mxGetData(prhs[2]));

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

        // Call the method
        MultiGPUGridder_instance->SetMaskRadius((float *)mxGetData(prhs[2]));

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

        // Call the method
        MultiGPUGridder_instance->SetNumberGPUs((int)mxGetScalar(prhs[2]));

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

        // Call the method
        MultiGPUGridder_instance->SetNumberStreams((int)mxGetScalar(prhs[2]));

        return;
    }

    // Set the kaiser bessel vector
    if (!strcmp("SetKerBesselVector", cmd))
    {
        // Check parameters
        if (nrhs != 3)
        {
            mexErrMsgTxt("SetKerBesselVector: Unexpected arguments. Please provide a Matlab array.");
        }

        // Call the method
        MultiGPUGridder_instance->SetKerBesselVector((float *)mxGetData(prhs[2]), GetMatlabDimensions(prhs[2])[0]);

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
        MultiGPUGridder_instance->Mem_obj->mem_alloc(varNameString, varDataType, varDataSize);

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
        MultiGPUGridder_instance->Mem_obj->pin_mem(varNameString);

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
        MultiGPUGridder_instance->Mem_obj->disp_mem(varNameString);
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
        MultiGPUGridder_instance->Mem_obj->mem_Free(varNameString);
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

        // Call the method
        MultiGPUGridder_instance->Mem_obj->mem_Copy(varNameString, (float *)mxGetData(prhs[3]));
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
        vol_dims = MultiGPUGridder_instance->Mem_obj->CPU_Get_Array_Size(varNameString);

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
        float *OutputArray = MultiGPUGridder_instance->Mem_obj->ReturnCPUFloatPtr(varNameString);

        // Copy the data to the Matlab array
        std::memcpy(matlabArrayPtr, OutputArray, sizeof(float) * dims[0] * dims[1] * dims[2]);

        plhs[0] = Matlab_Pointer;

        return;
    }

    // Return the CASImgs as a Matlab array
    if (!strcmp("GetImgs", cmd))
    {

        // Check parameters
        if (nrhs != 2)
        {
            mexErrMsgTxt("GetImgs: Unexpected arguments.");
        }

        // Create the output Matlab array (using the stored size of the corresponding C++ array)
        int *vol_dims;

        // The images are store in the following pinned CPU array: "CASImgs_CPU_Pinned"
        vol_dims = MultiGPUGridder_instance->Mem_obj->CPU_Get_Array_Size("CASImgs_CPU_Pinned");

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
        float *OutputArray = MultiGPUGridder_instance->Mem_obj->ReturnCPUFloatPtr("CASImgs_CPU_Pinned");

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
        MultiGPUGridder_instance->Mem_obj->CUDA_alloc(varNameString, varDataType, varDataSize, GPU_Device);

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
        MultiGPUGridder_instance->Mem_obj->CUDA_Free(varNameString);
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
        MultiGPUGridder_instance->Mem_obj->CUDA_disp_mem(varNameString);
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

        // Call the method
        MultiGPUGridder_instance->Mem_obj->CUDA_Copy(varNameString, (float *)mxGetData(prhs[3]));
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
        vol_dims = MultiGPUGridder_instance->Mem_obj->CUDA_Get_Array_Size(varNameString);

        mwSize dims[3];
        dims[0] = vol_dims[0];
        dims[1] = vol_dims[1];
        dims[2] = vol_dims[2];

        // Create the output matlab array as type float
        mxArray *Matlab_Pointer = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);

        // Get a pointer to the output matrix created above
        float *matlabArrayPtr = (float *)mxGetData(Matlab_Pointer);

        // Call the method
        float *GPUVol = MultiGPUGridder_instance->Mem_obj->CUDA_Return(varNameString);

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
        MultiGPUGridder_instance->Projection_Initilize();

        return;
    }

    // Run the forward projection CUDA kernel
    if (!strcmp("forwardProject", cmd))
    {
        // Accept either 2 or 3 parameters
        if (nrhs != 2 && nrhs != 3)
        {
            mexErrMsgTxt("forwardProject: Unexpected arguments.");
        }

        // If only 2 parameters were given run the forward projection
        if (nrhs == 1)
        {
            // Call the method
            MultiGPUGridder_instance->Forward_Project();
        }
        else
        {
            // Otherwise the user also provided us with a coordinate axes vector
            // Lets first set the coordinate axes array
            MultiGPUGridder_instance->SetAxes((float *)mxGetData(prhs[2]), GetMatlabDimensions(prhs[2]));

            // Now that the coordinate axes vector has been set run forward projection
            MultiGPUGridder_instance->Forward_Project();

            // Return the projection images
            // Create the output Matlab array (using the stored size of the corresponding C++ array)
            int *vol_dims;

            // The images are store in the following pinned CPU array: "CASImgs_CPU_Pinned"
            vol_dims = MultiGPUGridder_instance->Mem_obj->CPU_Get_Array_Size("CASImgs_CPU_Pinned");

            // Create the output Matlab array (using the stored size of the corresponding C++ array)
            mwSize dims_imgs[3];
            dims_imgs[0] = vol_dims[0];
            dims_imgs[1] = vol_dims[1];
            dims_imgs[2] = vol_dims[2];

            // Create the output matlab array as type float
            mxArray *Matlab_Pointer = mxCreateNumericArray(3, dims_imgs, mxSINGLE_CLASS, mxREAL);

            // Get a pointer to the output matrix created above
            float *matlabArrayPtr_imgs = (float *)mxGetData(Matlab_Pointer);

            // Call the method
            float *OutputArray = MultiGPUGridder_instance->Mem_obj->ReturnCPUFloatPtr("CASImgs_CPU_Pinned");

            // Copy the data to the Matlab array
            std::memcpy(matlabArrayPtr_imgs, OutputArray, sizeof(float) * dims_imgs[0] * dims_imgs[1] * dims_imgs[2]);

            plhs[0] = Matlab_Pointer;
        }
        return;
    }

    // Run the back projection CUDA kernel
    if (!strcmp("Back_Project", cmd))
    {
        // Accept 2 input parameters
        if (nrhs != 2 )
        {
            mexErrMsgTxt("Back_Project: Unexpected arguments.");
        }

        // If only 2 parameters were given run the back projection
        if (nrhs == 2)
        {
            // Call the method
            MultiGPUGridder_instance->Back_Project();
        }
        else if (nrhs == 4)
        {
            // Assume the third parameter is the CASImgs to back project
            // Set the CASImgs now
            // MultiGPUGridder_instance->SetImages((float *)mxGetData(prhs[2]));

            // Assume the fourth parameter is the coordinate axes array
            // Set the coordinate axes array now
            // Get a pointer to the matlab array
            // MultiGPUGridder_instance->SetAxes((float *)mxGetData(prhs[3]), GetMatlabDimensions(prhs[3]));

            // Lastly, run the back projection method
            // MultiGPUGridder_instance->Back_Project();
        }

        // Is the user provided an output array, lets output the
        // Summation of the volume on all of the GPUs
        if (nlhs == 1)
        {
            // Get the matrix size of the GPU volume
            mwSize dims[3];
            dims[0] = MultiGPUGridder_instance->volSize[0];
            dims[1] = MultiGPUGridder_instance->volSize[1];
            dims[2] = MultiGPUGridder_instance->volSize[2];

            // Create the output matlab array as type float
            mxArray *Matlab_Pointer = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);

            // Get a pointer to the output matrix created above
            float *matlabArrayPtr = (float *)mxGetData(Matlab_Pointer);

            // Call the method
            float *GPUVol = MultiGPUGridder_instance->GetVolume();

            // Copy the data to the Matlab array
            std::memcpy(matlabArrayPtr, GPUVol, sizeof(float) * dims[0] * dims[1] * dims[2]);

            plhs[0] = Matlab_Pointer;
        }

        return;
    }

    // Print the parameters of the multi GPU  gridder to the console for debugging
    if (!strcmp("Print", cmd))
    {

        MultiGPUGridder_instance->Print();
        return;
    }




    // Crop some image volume
    if (!strcmp("CropVolume", cmd))
    {
        // Check parameters
        if (nrhs != 3)
        {
            mexErrMsgTxt("CropVolume: Unexpected arguments. Please provide a Matlab array.");
        }


        // Get a pointer to the input matrix
        float *matlabArrayPtr = (float *)mxGetData(prhs[2]);

        // Get the matrix size of the input Matlab pointer
        const mwSize *dims_mwSize;
        dims_mwSize = mxGetDimensions(prhs[2]);

        int *dims = new int[3];
        dims[0] = (int)dims_mwSize[0];
        dims[1] = (int)dims_mwSize[1];
        dims[2] = (int)dims_mwSize[2];

        mwSize output_dims[3];
        output_dims[0] = dims[0] / 2;
        output_dims[1] = dims[1] / 2;
        output_dims[2] = dims[2] / 2;

        std::cout << "output_dims: " << output_dims[0] << " " <<  output_dims[1] << " " <<  output_dims[2] << '\n';
        // Call the method
        float * output_volume;// = new float[dims[0]/2 * dims[1]/2 * dims[2]/2];
        output_volume = MultiGPUGridder_instance->CropVolume(matlabArrayPtr, dims[0], dims[0] / 2);

        // Create the output matlab array as type float
        mxArray *Output_Matlab_Pointer = mxCreateNumericArray(3, output_dims, mxSINGLE_CLASS, mxREAL);

        // Get a pointer to the output matrix created above
        float *Output_matlabArrayPtr = (float *)mxGetData(Output_Matlab_Pointer);

        // Copy the data to the Matlab array
        std::memcpy(Output_matlabArrayPtr, output_volume, sizeof(float) * output_dims[0] * output_dims[1] * output_dims[2]);

        plhs[0] = Output_Matlab_Pointer;



        return;
    }
        
    // Pad some image volume
    if (!strcmp("PadVolume", cmd))
    {
        // Check parameters
        if (nrhs != 3)
        {
            mexErrMsgTxt("PadVolume: Unexpected arguments. Please provide a Matlab array.");
        }


        // Get a pointer to the input matrix
        float *matlabArrayPtr = (float *)mxGetData(prhs[2]);

        // Get the matrix size of the input Matlab pointer
        const mwSize *dims_mwSize;
        dims_mwSize = mxGetDimensions(prhs[2]);

        int *dims = new int[3];
        dims[0] = (int)dims_mwSize[0];
        dims[1] = (int)dims_mwSize[1];
        dims[2] = (int)dims_mwSize[2];

        mwSize output_dims[3];
        output_dims[0] = dims[0] * 2 + 6;
        output_dims[1] = dims[1] * 2 + 6;
        output_dims[2] = dims[2] * 2 + 6;

        // Call the method
        float * output_volume;
        output_volume = MultiGPUGridder_instance->PadVolume(matlabArrayPtr, dims[0], dims[0]*2 + 6);

        // Create the output matlab array as type float
        mxArray *Output_Matlab_Pointer = mxCreateNumericArray(3, output_dims, mxSINGLE_CLASS, mxREAL);

        // Get a pointer to the output matrix created above
        float *Output_matlabArrayPtr = (float *)mxGetData(Output_Matlab_Pointer);

        // Copy the data to the Matlab array
        std::memcpy(Output_matlabArrayPtr, output_volume, sizeof(float) * output_dims[0] * output_dims[1] * output_dims[2]);

        plhs[0] = Output_Matlab_Pointer;

        std::cout << "Done with PadVolume()" << '\n';


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
