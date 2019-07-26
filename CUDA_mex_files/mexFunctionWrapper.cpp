#include "mexFunctionWrapper.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{	

    // Get the command string
    char cmd[64];
	if (nrhs < 1 || mxGetString(prhs[0], cmd, sizeof(cmd)))
    {
		mexErrMsgTxt("First input should be a command string less than 64 characters long.");
    } 

    std::cout << "cmd: " << cmd << '\n';

    // New
    if (!strcmp("new", cmd)) {
        // Check parameters
        if (nlhs != 1)
        {
            mexErrMsgTxt("New: One output expected.");
        }

        // Return a handle to a new C++ instance
        plhs[0] = convertPtr2Mat<CUDA_Gridder>(new CUDA_Gridder);
        return;
    }    

    // Delete
    if (!strcmp("delete", cmd)) {

        // TODO: Delete all of the CPU arrays

        // TODO: Delete all of the CUDA arrays

        // Destroy the C++ object
        destroyObject<CUDA_Gridder>(prhs[1]);
        // Warn if other commands were ignored
        if (nlhs != 0 || nrhs != 2)
            mexWarnMsgTxt("Delete: Unexpected arguments ignored.");
        return;
    }    
    
    // Get the class instance pointer from the second input
    CUDA_Gridder* CUDA_Gridder_instance = convertMat2Ptr<CUDA_Gridder>(prhs[1]);
      
    // Call the various class methods    
    // mem_alloc    
    if (!strcmp("mem_alloc", cmd)) {
        // Check parameters

        if (nrhs !=  5)
        {
            mexErrMsgTxt("mem_alloc: Unexpected arguments. Please provide (1) variable name as string, (2) data type as string, and (3) array size (i.e. [256, 256, 256]) as a Matlab vector.");
        }

        char varNameString[64];      
        char varDataType[64]; 
        int * varDataSize; // e.g. [dim1, dim2, dim3]

        mxGetString(prhs[2], varNameString, sizeof(varNameString));
        mxGetString(prhs[3], varDataType, sizeof(varDataType));
        varDataSize = (int*)mxGetData(prhs[4]);

        // Call the method
        CUDA_Gridder_instance->Mem_obj->mem_alloc(varNameString, varDataType, varDataSize);

        return;
    }

    // pin_mem    
    if (!strcmp("pin_mem", cmd)) {
        // Check parameters

        if (nrhs !=  3)
        {
            mexErrMsgTxt("pin_mem: Unexpected arguments. Please provide (1) variable name as string.");
        }

        char varNameString[64]; 
        mxGetString(prhs[2], varNameString, sizeof(varNameString));

        // Call the method
        CUDA_Gridder_instance->Mem_obj->pin_mem(varNameString);
        return;
    }

    // disp_mem    
    if (!strcmp("disp_mem", cmd)) {
        // Check parameters

        if (nrhs !=  3)
        {
            mexErrMsgTxt("disp_mem: Unexpected arguments. Please provide (1) variable name as string or 'all' to display information on all the arrays.");
        }

        char varNameString[64]; 
        mxGetString(prhs[2], varNameString, sizeof(varNameString));

        // Call the method
        CUDA_Gridder_instance->Mem_obj->disp_mem(varNameString);
        return;
    }

    // mem_Free    
    if (!strcmp("mem_Free", cmd)) {
        // Check parameters
        char varNameString[64];   

        if (nrhs !=  3)
        {
            mexErrMsgTxt("mem_Free: Unexpected arguments. Please provide (1) variable name as string.");
        }

        mxGetString(prhs[2], varNameString, sizeof(varNameString));

        // Call the method
        CUDA_Gridder_instance->Mem_obj->mem_Free(varNameString);
        return;
    }

    // mem_Copy    
    if (!strcmp("mem_Copy", cmd)) {
        // Check parameters

        if (nrhs !=  4)
        {
            mexErrMsgTxt("mem_Copy: Unexpected arguments. Please provide (1) variable name as string, (2) a Matlab array of same dimensions and type as previously allocated.");
        }

        // Get the input variable name as a string
        char varNameString[64];  
        mxGetString(prhs[2], varNameString, sizeof(varNameString));

        // TO DO: Check to see if the matrix size is the same as the previously allocated array size


        // Call the method
        CUDA_Gridder_instance->Mem_obj->mem_Copy(varNameString, prhs[3]);

        return;
    }

    // mem_Return    
    if (!strcmp("mem_Return", cmd)) {
        // Check parameters

        if (nrhs !=  3)
        {
            mexErrMsgTxt("mem_Return: Unexpected arguments. Please provide (1) variable name as string.");
        }

        char varNameString[64];              

        // Get the input variable name as a string
        mxGetString(prhs[2], varNameString, sizeof(varNameString));
        
        // Call the method
        plhs[0] = CUDA_Gridder_instance->Mem_obj->mem_Return(varNameString, plhs[0]);     

        return;
    }

    // CUDA_alloc 
    if (!strcmp("CUDA_alloc", cmd)) {
        // Check parameters

        if (nrhs !=  6)
        {
            mexErrMsgTxt("CUDA_alloc: Unexpected arguments. Please provide (1) variable name as string, (2) data type as string ('int', 'float'), (3) data size as Matlab vector ([256, 256, 256]), (4) GPU to allocate memory to as an integer (0 to number of GPUs minus 1).");
        }

        char varNameString[64];      
        char varDataType[64]; 
        int * varDataSize; // e.g. [dim1, dim2, dim3]
        int GPU_Device = 0; // Default to the first GPU

        mxGetString(prhs[2], varNameString, sizeof(varNameString));
        mxGetString(prhs[3], varDataType, sizeof(varDataType));
        varDataSize = (int*)mxGetData(prhs[4]);
        GPU_Device = (int)mxGetScalar(prhs[5]);         

        std::cout << "GPU_Device: " << GPU_Device << '\n';

        // Call the method
        CUDA_Gridder_instance->Mem_obj->CUDA_alloc(varNameString, varDataType, varDataSize, GPU_Device);
        return;
    }

    // CUDA_Free    
    if (!strcmp("CUDA_Free", cmd)) {
        // Check parameters

        if (nrhs !=  3)
        {
            mexErrMsgTxt("CUDA_Free: Unexpected arguments. Please provide (1) variable name as string.");
        }

        char varNameString[64];             

        // Get the input variable name as a string
        mxGetString(prhs[2], varNameString, sizeof(varNameString));

        // Call the method
        CUDA_Gridder_instance->Mem_obj->CUDA_Free(varNameString);
        return;
    }

    // CUDA_disp_mem    
    if (!strcmp("CUDA_disp_mem", cmd)) {
        // Check parameters

        if (nrhs !=  3)
        {
            mexErrMsgTxt("CUDA_disp_mem: Unexpected arguments. Please provide (1) variable name as string or 'all' to display information on all the arrays.");
        }

        char varNameString[64];              
        if (nrhs >  2)
        {
            mxGetString(prhs[2], varNameString, sizeof(varNameString));
        }

        // Call the method
        CUDA_Gridder_instance->Mem_obj->CUDA_disp_mem(varNameString);
        return;
    }

    // CUDA_Copy    
    if (!strcmp("CUDA_Copy", cmd)) {
        // Check parameters

        if (nrhs !=  4)
        {
            mexErrMsgTxt("CUDA_Copy: Unexpected arguments. Please provide (1) variable name as string, (2) a Matlab array of same dimensions and type as previously allocated.");
        }

        // Get the input variable name as a string
        char varNameString[64];  
        mxGetString(prhs[2], varNameString, sizeof(varNameString));

        // TO DO: Check to see if the matrix size is the same as the previously allocated array size


        // Call the method
        CUDA_Gridder_instance->Mem_obj->CUDA_Copy(varNameString, prhs[3]);

        return;
    }

    // CUDA_Return    
    if (!strcmp("CUDA_Return", cmd)) {
        // Check parameters

        if (nrhs !=  3)
        {
            mexErrMsgTxt("CUDA_Return: Unexpected arguments. Please provide (1) variable name as string.");
        }

        char varNameString[64];              

        // Get the input variable name as a string
        mxGetString(prhs[2], varNameString, sizeof(varNameString));
        
        // Call the method
        plhs[0] = CUDA_Gridder_instance->Mem_obj->CUDA_Return(varNameString, plhs[0]);     

        return;
    }

    // Forward_Project    
    if (!strcmp("Forward_Project", cmd)) {
        // Check parameters

        if (nrhs <  3)
        {
            mexErrMsgTxt("Forward_Project: Unexpected arguments. Please provide (1) variable names as strings.");
        }                 

        // Get all the variable names as a vector   
        std::vector<std::string> Input_Strings;

        for (int i = 2; i<nrhs; i++) // Start the index at 2 (because c++ class handle is i=0; cmd is i=1)
        {
            char varNameString[64];  
            mxGetString(prhs[i], varNameString, sizeof(varNameString));
            Input_Strings.push_back(varNameString);
        }         

        // Call the method
        CUDA_Gridder_instance->Forward_Project(Input_Strings);
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
