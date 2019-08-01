#include "CPU_CUDA_Memory.h"

// The CPU_CUDA_Memory class functions

bool CPU_CUDA_Memory::GPUArrayAllocated(std::string varNameString, int GPU_Device)
{
    // Given the name of the array, is it already allocated on the given GPU?

    int arr_idx = FindArrayIndex(varNameString, this->CUDA_arr_names);    

    if (arr_idx != -1) // The array was found on one of the GPUs 
    {
        // Which GPU is the array assigned to?
        int GPU_assigned = this->CUDA_arr_GPU_Assignment[arr_idx];

        // Is the array on the requested GPU device already?
        if (GPU_assigned == GPU_Device)
        {
            return true;
        } else {
            // The array is NOT on the requested GPU device
            return false;
        }


    } else { // The array is NOT already allocated on any GPU        
        return false;
    }   

}

bool CPU_CUDA_Memory::CPUArrayAllocated(std::string varNameString)
{
    // Given the name of the array, is it already allocated on the CPU?

    int arr_idx = FindArrayIndex(varNameString, this->cpu_arr_names);    

    if (arr_idx != -1) // The array was found
    {
        return true;
    } else { // The array is NOT already allocated     
        return false;
    }   
}

int CPU_CUDA_Memory::FindArrayIndex(std::string varNameString, std::vector<std::string> NameVector){
    // Find the index in the NameVector vector which is equal to a given string

    // Initilize the array index variable
    int arr_idx = -1;

    // Loop through all of the array names and find the correct index
    for (int i = 0; i < NameVector.size(); i++)
    {            
        if ( NameVector[i] == varNameString ) // Are the strings equal?
        {
            arr_idx = i;
            break; // Stop once we reach the first index 
        }
    }

    // Check to make sure we found one        
    if (arr_idx >= NameVector.size() || arr_idx == -1) // String name wasn't found in the vector
    {
        //mexErrMsgTxt("Array name is not found. Please check spelling.");
        return -1;
    }

    return arr_idx;

}

int* CPU_CUDA_Memory::ReturnCPUIntPtr(std::string varNameString){
    // Given the name of a variable, return the memory pointer (supports only CPU int pointers)
    
    // Locate the index of the cpu_arr_names vector which correspondes to the given variable name 
    int arr_idx = FindArrayIndex(varNameString, cpu_arr_names);

    if (arr_idx < 0)
    {
        //mexErrMsgTxt("Failed to locate variable. Please check spelling.");
    }  

    if ( cpu_arr_types[arr_idx] == "int")
    {
        // Return the CPU memory pointer
        return cpu_arr_ptrs[arr_idx].i;

    } else {
        mexErrMsgTxt("This array is not of integer type. Please try ReturnCPUFloatPtr instead.");
    }
    
    return NULL; // Return null pointer since no pointer was found
}

float* CPU_CUDA_Memory::ReturnCPUFloatPtr(std::string varNameString){
    // Given the name of a variable, return the memory pointer (supports only CPU float pointers)
    
    // Locate the index of the cpu_arr_names vector which correspondes to the given variable name 
    int arr_idx = FindArrayIndex(varNameString, cpu_arr_names);

    if (arr_idx < 0)
    {
        //mexErrMsgTxt("Failed to locate variable. Please check spelling.");
    }  

    if ( cpu_arr_types[arr_idx] == "float")
    {
        // Return the CPU memory pointer
        return cpu_arr_ptrs[arr_idx].f;

    } else {
        mexErrMsgTxt("This array is not of integer type. Please try ReturnCPUIntPtr instead.");
    }
    
    return NULL; // Return null pointer since no pointer was found
}

int* CPU_CUDA_Memory::ReturnCUDAIntPtr(std::string varNameString){
    // Given the name of a variable, return the memory pointer (supports only CUDA int pointers)
    
    // Locate the index of the cpu_arr_names vector which correspondes to the given variable name 
    int arr_idx = FindArrayIndex(varNameString, CUDA_arr_names);

    if (arr_idx < 0)
    {
        //mexErrMsgTxt("Failed to locate variable. Please check spelling.");
    }  

    if ( CUDA_arr_types[arr_idx] == "int")
    {
        // Return the CUDA memory pointer
        return CUDA_arr_ptrs[arr_idx].i;

    } else {
        mexErrMsgTxt("This array is not of integer type. Please try ReturnCUDAFloatPtr instead.");
    }
    
    return NULL; // Return null pointer since no pointer was found
}

float* CPU_CUDA_Memory::ReturnCUDAFloatPtr(std::string varNameString){
    // Given the name of a variable, return the memory pointer (supports only CUDA float pointers)
    
    // Locate the index of the cpu_arr_names vector which correspondes to the given variable name 
    int arr_idx = FindArrayIndex(varNameString, CUDA_arr_names);

    if (arr_idx < 0)
    {
        //mexErrMsgTxt("Failed to locate variable. Please check spelling.");
    }  

    if ( CUDA_arr_types[arr_idx] == "float")
    {
        // Return the CUDA memory pointer
        return CUDA_arr_ptrs[arr_idx].f;

    } else {
        mexErrMsgTxt("This array is not of float type. Please try ReturnCUDAIntPtr instead.");
    }
    
    return NULL; // Return null pointer since no pointer was found
}

void CPU_CUDA_Memory::mem_alloc(std::string varNameString, std::string dataType, int * dataSize)
{        
    // Allocate memory based on the dataType (i.e. int, float, etc.)
    mexPrintf("Allocating memory: Name %s Type %s Size %i \n", varNameString.c_str(), dataType.c_str(),  dataSize[0] * dataSize[1] * dataSize[2]); 

    // Save the name of the variable and the array size to the corresponding vectors
    cpu_arr_names.push_back(varNameString);

    // Save the dataType of this array
    cpu_arr_types.push_back(dataType);

    // Deep copy the data size pointer
    int *new_dataSize_ptr = new int[3];
    for (int i=0; i<3; i++){
        new_dataSize_ptr[i] = dataSize[i];
    }
    
    // Save to the vector of array sizes
    cpu_arr_sizes.push_back(new_dataSize_ptr);

    // Allocate the memory and save the pointer to the corresponding vector
    Ptr_Types n;

    if ( dataType == "int")
    {
        // Need to convert the dataSize to long long int type to allow for array length larger than maximum int32 value
        unsigned long long *dataSizeLong = new  unsigned long long[3];
        dataSizeLong[0] = (unsigned long long)dataSize[0];
        dataSizeLong[1] = (unsigned long long)dataSize[1];
        dataSizeLong[2] = (unsigned long long)dataSize[2];

        std::cout << "CPU array size: " << dataSizeLong[0] * dataSizeLong[1] * dataSizeLong[2] * sizeof(int) << '\n';

        int *new_ptr = new int[ dataSizeLong[0] * dataSizeLong[1] * dataSizeLong[2] ]; // Multiply the X,Y,Z dimensions of the array
        n.i = new_ptr;
    } else if ( dataType == "unint")            
    {
        // Need to convert the dataSize to long long int type to allow for array length larger than maximum int32 value
        unsigned long long *dataSizeLong = new  unsigned long long[3];
        dataSizeLong[0] = (unsigned long long)dataSize[0];
        dataSizeLong[1] = (unsigned long long)dataSize[1];
        dataSizeLong[2] = (unsigned long long)dataSize[2];

        std::cout << "CPU array size: " << dataSizeLong[0] * dataSizeLong[1] * dataSizeLong[2] * sizeof(unsigned long long) << '\n';

        unsigned long long *new_ptr = new unsigned long long[ dataSizeLong[0] * dataSizeLong[1] * dataSizeLong[2] ]; // Multiply the X,Y,Z dimensions of the array
        n.un_int = new_ptr;

    } else if ( dataType == "float")
    {
        // Need to convert the dataSize to long long int type to allow for array length larger than maximum int32 value
        unsigned long long *dataSizeLong = new  unsigned long long[3];
        dataSizeLong[0] = (unsigned long long)dataSize[0];
        dataSizeLong[1] = (unsigned long long)dataSize[1];
        dataSizeLong[2] = (unsigned long long)dataSize[2];

        std::cout << "CPU array size: " << dataSizeLong[0] * dataSizeLong[1] * dataSizeLong[2] * sizeof(float) << '\n';

        float *new_ptr = new float[ dataSizeLong[0] * dataSizeLong[1] * dataSizeLong[2] ]; // Multiply the X,Y,Z dimensions of the array
        n.f = new_ptr;
    } else 
    {
        mexErrMsgTxt("Unrecognized data type. Please choose either int, float, or double.");
    }      

    // Save the new pointer to the allocated CPU array        
    cpu_arr_ptrs.push_back(n);

}

void CPU_CUDA_Memory::mem_Copy(std::string varNameString, float *New_Array)
{
    // Given a float array, copy the data to the corresponding C++ pointer
    mexPrintf("\n");
    mexPrintf("Copying memory: Name %s \n", varNameString.c_str()); 

    // Locate the index of the cpu_arr_names vector which correspondes to the given variable name 
    int arr_idx = FindArrayIndex(varNameString, cpu_arr_names);

    if (arr_idx < 0)
    {
        mexErrMsgTxt("Failed to locate variable. Please check spelling.");
        return;
    }     

    // TO DO: Check if the allocated memory is the same size as the input array



    // Copy the Matlab array to the class pointer for deep copy
    // Otherwise, Matlab seems to delete it's pointer once returning from the Mex file
    int dim_size = cpu_arr_sizes[arr_idx][0] * cpu_arr_sizes[arr_idx][1] * cpu_arr_sizes[arr_idx][2];

    if ( cpu_arr_types[arr_idx] == "int")
    {

        // Get the pointer to the input Matlab array which should be int type (same as previously allocated)
        //int* matlabArray = (int*)mxGetData(Matlab_Pointer);

        //std::memcpy(cpu_arr_ptrs[arr_idx].i, matlabArray, sizeof(int)*(dim_size));
    } else if ( cpu_arr_types[arr_idx] == "float")
    {
        std::memcpy(cpu_arr_ptrs[arr_idx].f, New_Array, sizeof(float)*(dim_size));
    } else 
    {
        mexErrMsgTxt("Unrecognized data type. Only float is supported currently for mem_Copy().");
    }      
    

}

mxArray* CPU_CUDA_Memory::mem_Return(std::string varNameString, mxArray *Matlab_Pointer)
{
    // Copy the data from the corresponding C++ array back to the Matlab array
    mexPrintf("\n");
    mexPrintf("Returning memory: Name %s \n", varNameString.c_str()); 

    // Locate the index of the cpu_arr_names vector which correspondes to the given variable name 
    int arr_idx = FindArrayIndex(varNameString, cpu_arr_names);

    if (arr_idx < 0)
    {
        mexErrMsgTxt("Failed to locate variable. Please check spelling.");
    }     

    // TO DO: Check if the allocated memory is the same size as the input array       


    // Create the output Matlab array (using the stored size of the corresponding C++ array)
    mwSize dims[3]; // matlab is row major (not column major)
    dims[0] = cpu_arr_sizes[arr_idx][0]; 
    dims[1] = cpu_arr_sizes[arr_idx][1];
    dims[2] = cpu_arr_sizes[arr_idx][2];

    // Need to convert the dims to long long int type to allow for array length larger than maximum int32 value
    unsigned long long *dim_size = new  unsigned long long[3];
    dim_size[0] = (unsigned long long)cpu_arr_sizes[arr_idx][0];
    dim_size[1] = (unsigned long long)cpu_arr_sizes[arr_idx][1];
    dim_size[2] = (unsigned long long)cpu_arr_sizes[arr_idx][2];

    // int dim_size = cpu_arr_sizes[arr_idx][0] * cpu_arr_sizes[arr_idx][1] * cpu_arr_sizes[arr_idx][2];

    if ( cpu_arr_types[arr_idx] == "int")
    {

        // Create the output matlab array as type int
        Matlab_Pointer = mxCreateNumericArray(3, dims, mxINT32_CLASS, mxREAL);   

        // Get a pointer to the output matrix created above
        int* matlabArrayPtr = (int*)mxGetData(Matlab_Pointer);

        std::memcpy(matlabArrayPtr, cpu_arr_ptrs[arr_idx].i, sizeof(int)*dim_size[0]*dim_size[1]*dim_size[2]);

        return Matlab_Pointer;

    }else if ( cpu_arr_types[arr_idx] == "un_int")
    {

        // Create the output matlab array as type float
        Matlab_Pointer = mxCreateNumericArray(3, dims, mxUINT64_CLASS, mxREAL);   

        // Get a pointer to the output matrix created above
        long long int* matlabArrayPtr = (long long int*)mxGetData(Matlab_Pointer);

        std::memcpy(matlabArrayPtr, cpu_arr_ptrs[arr_idx].un_int, sizeof(long long int)*dim_size[0]*dim_size[1]*dim_size[2]);

        return Matlab_Pointer;
    } else if ( cpu_arr_types[arr_idx] == "float")
    {

        // Create the output matlab array as type float
        Matlab_Pointer = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);   

        // Get a pointer to the output matrix created above
        float* matlabArrayPtr = (float*)mxGetData(Matlab_Pointer);

        std::memcpy(matlabArrayPtr, cpu_arr_ptrs[arr_idx].f, sizeof(float)*dim_size[0]*dim_size[1]*dim_size[2]);

        return Matlab_Pointer;
    } else 
    {
        mexErrMsgTxt("Unrecognized data type. Please choose either int, float, or double.");
    }        
}

void CPU_CUDA_Memory::pin_mem(std::string varNameString)
{
    // Given a variable name, pin the associated CPU array (i.e. make it non-pageable on the RAM so the GPUs can directly access it)
    
    std::cout << "pin_mem()..." << '\n';

    // Locate the index of the cpu_arr_names vector which correspondes to the given variable name         
    int arr_idx = FindArrayIndex(varNameString, cpu_arr_names);

    if (arr_idx < 0)
    {
        mexErrMsgTxt("Failed to locate variable. Please check spelling.");
        return;
    }     

    // Need to convert the cpu_arr_sizes[arr_idx] to long long int type to allow for array length larger than maximum int32 value
    unsigned long long *dataSizeLong = new  unsigned long long[3];
    dataSizeLong[0] = (unsigned long long)cpu_arr_sizes[arr_idx][0];
    dataSizeLong[1] = (unsigned long long)cpu_arr_sizes[arr_idx][1];
    dataSizeLong[2] = (unsigned long long)cpu_arr_sizes[arr_idx][2];
    
    // Get the size of the array    
    unsigned long long dim_size = dataSizeLong[0] * dataSizeLong[1] * dataSizeLong[2];

    // Register the CPU array as pinned memory using the CUDA function
    if ( cpu_arr_types[arr_idx] == "int")
    {            
        cudaHostRegister(cpu_arr_ptrs[arr_idx].i, sizeof(int)*dim_size, 0);
        return;
    } else if ( cpu_arr_types[arr_idx] == "un_int")
    {
        cudaHostRegister(cpu_arr_ptrs[arr_idx].un_int, sizeof(long long int)*dim_size, 0);
        return;
    } else if ( cpu_arr_types[arr_idx] == "float")
    {
        std::cout << "Pinning array " << varNameString << " with bytes " << sizeof(float)*dim_size << '\n';
        cudaHostRegister(cpu_arr_ptrs[arr_idx].f, sizeof(float)*dim_size, 0);
        return;
    } else 
    {
        mexErrMsgTxt("Unrecognized data type. Please choose either int, float, or double.");
    }       
}

void CPU_CUDA_Memory::disp_mem(std::string varNameString)
{    
    // Display in the Matlab console the size of a given matrix

    if ( varNameString == "all" ){
        // Display information on all of the arrays
        mexPrintf("\n"); 
        for (int arr_idx=0; arr_idx<cpu_arr_names.size(); arr_idx++)
        {
            mexPrintf("CPU Matrix size for %s: %i x %i x %i of type %s \n", cpu_arr_names[arr_idx].c_str(), cpu_arr_sizes[arr_idx][0], cpu_arr_sizes[arr_idx][1], cpu_arr_sizes[arr_idx][2], cpu_arr_types[arr_idx].c_str()); 
        }
        mexPrintf("\n"); 
        return;
    }

    // Locate the index of the cpu_arr_names vector which correspondes to the given variable name         
    int arr_idx = FindArrayIndex(varNameString, cpu_arr_names);

    if (arr_idx < 0)
    {
        mexErrMsgTxt("Failed to locate variable. Please check spelling.");
        return;
    }     

    mexPrintf("\n"); 
    mexPrintf("CPU Matrix size for %s: %i x %i x %i of type %s \n", cpu_arr_names[arr_idx].c_str(), cpu_arr_sizes[arr_idx][0], cpu_arr_sizes[arr_idx][1], cpu_arr_sizes[arr_idx][2], cpu_arr_types[arr_idx].c_str()); 

}

void CPU_CUDA_Memory::mem_Free(std::string varNameString)
{    
    // Free the memory of the corresponding array

    // If varNameString == "all" then free all the allocated CPU arrays
    if ( varNameString == "all" )
    {
        // Temporarily copy the cpu_arr_names to a new vector
        std::vector<std::string> cpu_arr_names_temp;
        for (int i=0; i<cpu_arr_names.size(); i++) 
        {
            cpu_arr_names_temp.push_back(cpu_arr_names[i]); 
        }     

        // Iterate over each name to free the memory
        for (int i = 0; i<cpu_arr_names_temp.size(); i++)
        {  
            // Start at the end of the cpu_arr_names vector and free each CPU array
            mem_Free(cpu_arr_names_temp[i]); 
        }

        return;
    }


    // Locate the index of the cpu_arr_names vector which correspondes to the given variable name         
    int arr_idx = FindArrayIndex(varNameString, cpu_arr_names);

    if (arr_idx < 0)
    {
        mexErrMsgTxt("Failed to locate variable. Please check spelling.");
        return;
    }  

    // Delete the arrays first from memory
    if ( cpu_arr_types[arr_idx] == "int")
    {

        std::free(cpu_arr_ptrs[arr_idx].i);

    } else if ( cpu_arr_types[arr_idx] == "un_int")
    {

        std::free(cpu_arr_ptrs[arr_idx].un_int);

    }else if ( cpu_arr_types[arr_idx] == "float")
    {

        std::free(cpu_arr_ptrs[arr_idx].f);

    }

    // Delete the corresponding information from all the cpu vectors
    cpu_arr_names.erase(cpu_arr_names.begin() + arr_idx);
    cpu_arr_types.erase(cpu_arr_types.begin() + arr_idx);
    cpu_arr_sizes.erase(cpu_arr_sizes.begin() + arr_idx);
    cpu_arr_ptrs.erase(cpu_arr_ptrs.begin() + arr_idx);

}

void CPU_CUDA_Memory::CUDA_alloc(std::string varNameString, std::string dataType, int * dataSize, int GPU_Device)
{
    // Allocate memory on the GPU based on the dataType (i.e. int, float, etc.)
    mexPrintf("Allocating GPU memory: Name %s Type %s Size %i on GPU %i \n", varNameString.c_str(), dataType.c_str(),  dataSize[0] * dataSize[1] * dataSize[2], GPU_Device); 

    // Save the name of the variable and the array size to the corresponding vectors
    CUDA_arr_names.push_back(varNameString);

    // Save the dataType of this array
    CUDA_arr_types.push_back(dataType);

    // Deep copy the data size pointer
    int *new_dataSize_ptr = new int[3];
    for (int i=0; i<3; i++){
        new_dataSize_ptr[i] = dataSize[i];
    }
    
    // Which GPU to allocate the memory to?
    int  numGPUs;
    cudaGetDeviceCount(&numGPUs);

    // Provide error message if no GPUs are found (i.e. all cards are busy)
    if ( numGPUs == 0 )
    {
        mexErrMsgTxt("No NVIDIA graphic cards located on your computer. All cards may be busy and unavailable.");  
    }

    if (GPU_Device < 0 || GPU_Device >= numGPUs){
        mexErrMsgTxt("Error in GPU selection. Please provide an integer between 0 and the number of NVIDIA graphic cards on your computer.");
    }

    CUDA_arr_GPU_Assignment.push_back(GPU_Device);
    cudaSetDevice(GPU_Device);

    // Save to the vector of array sizes
    CUDA_arr_sizes.push_back(new_dataSize_ptr);

    // Check to make sure the GPU has enough available memory left
    size_t mem_tot_0 = 0;
    size_t mem_free_0 = 0;
    cudaMemGetInfo(&mem_free_0, & mem_tot_0);
    std::cout<< "Free memory before copy: " << mem_free_0 << " Device: "<< GPU_Device << std::endl;

    // Allocate the memory and save the pointer to the corresponding vector
    Ptr_Types n;

    if ( dataType == "int")
    {
        int *devPtr = new int[ dataSize[0] * dataSize[1] * dataSize[2] ]; // Multiply the X,Y,Z dimensions of the array
        n.i = devPtr;

        std::cout << "CUDA Memory requested: " << sizeof(int)*(dataSize[0] * dataSize[1] * dataSize[2]) << " bytes" << '\n';

        // Is there enough available memory on the device to allocate this array?
        if ( mem_free_0 < sizeof(int)*(dataSize[0] * dataSize[1] * dataSize[2]))
        {
            std::cerr << "Not enough memory on the device to allocate the requested memory. Try fewer number of projections or a smaller volume. Or increase SetNumberBatches()." << '\n';
            mexErrMsgTxt("Not enough memory on the device to allocate the requested memory. Try fewer number of projections or a smaller volume. Or increase SetNumberBatches().");              
            return;
        }

        cudaMalloc(&n.i, sizeof(int)*(dataSize[0] * dataSize[1] * dataSize[2])); // Multiply the X,Y,Z dimensions of the array     
        cudaDeviceSynchronize();

    } else if ( dataType == "float")
    {
        std::cout << "CUDA Memory requested: " << sizeof(float)*(dataSize[0] * dataSize[1] * dataSize[2]) << " bytes" << '\n';
        // Is there enough available memory on the device to allocate this array?
        if ( mem_free_0 < sizeof(float)*(dataSize[0] * dataSize[1] * dataSize[2]))
        {
            mexErrMsgTxt("Not enough memory on the device to allocate the requested memory. Try fewer number of projections or a smaller volume.");  
            std::cerr << "Not enough memory on the device to allocate the requested memory. Try fewer number of projections or a smaller volume. Or increase SetNumberBatches()" << '\n';
            return;
        }

        float *devPtr = new float[ dataSize[0] * dataSize[1] * dataSize[2] ]; // Multiply the X,Y,Z dimensions of the array 
        n.f = devPtr;

        cudaMalloc(&n.f, sizeof(float)*(dataSize[0] * dataSize[1] * dataSize[2])); // Multiply the X,Y,Z dimensions of the array 
        cudaDeviceSynchronize();

    } else 
    {
        mexErrMsgTxt("Unrecognized data type. Please choose either int, float, or double.");
    }      

    // Save the new pointer to the allocated CUDA GPU array        
    CUDA_arr_ptrs.push_back(n);

}

void CPU_CUDA_Memory::CUDA_Free(std::string varNameString)
{
    // Free the memory of the corresponding CUDA GPU array

    std::cout << "varNameString: " << varNameString << '\n';

    // If varNameString == "all" then free all the allocated GPU arrays
    if ( varNameString == "all" )
    {
        // Temporarily copy the CUDA_arr_names to a new vector 
        std::vector<std::string> CUDA_arr_names_temp;
        for (int i=0; i<CUDA_arr_names.size(); i++) 
        {
            CUDA_arr_names_temp.push_back(CUDA_arr_names[i]); 
        }     

        // Iterate over each name to free the memory
        for (int i = 0; i<CUDA_arr_names_temp.size(); i++)
        {  
            // Start at the end of the CUDA_arr_names vector and free each GPU array
            CUDA_Free(CUDA_arr_names_temp[i]); 
        }

        return;
    }

    std::cout << "finding varNameString: " << varNameString << '\n';

    // Locate the index of the cpu_arr_names vector which correspondes to the given variable name         
    int arr_idx = FindArrayIndex(varNameString, CUDA_arr_names);

    if (arr_idx < 0)
    {
        mexErrMsgTxt("Failed to locate variable. Please check spelling.");
        return;
    }  

    // Set the GPU device to the device which contains the CUDA array
    cudaSetDevice(CUDA_arr_GPU_Assignment[arr_idx]);

    // Delete the array from the GPU memory  
    if ( CUDA_arr_types[arr_idx] == "int" )
    {
        cudaFree( CUDA_arr_ptrs[arr_idx].i );

    } else if ( CUDA_arr_types[arr_idx] == "float" )
    {
        cudaFree( CUDA_arr_ptrs[arr_idx].f );
    } else 
    {
        mexErrMsgTxt("Unrecognized data type. Please choose either int, float, or double.");
    }     

    // Delete the corresponding information from all the cpu vectors
    CUDA_arr_names.erase(CUDA_arr_names.begin() + arr_idx);
    CUDA_arr_types.erase(CUDA_arr_types.begin() + arr_idx);
    CUDA_arr_sizes.erase(CUDA_arr_sizes.begin() + arr_idx);
    CUDA_arr_ptrs.erase(CUDA_arr_ptrs.begin() + arr_idx);
    CUDA_arr_GPU_Assignment.erase(CUDA_arr_GPU_Assignment.begin() + arr_idx);

}

void CPU_CUDA_Memory::CUDA_disp_mem(std::string varNameString)
{    
    // Display in the Matlab console the size of a given GPU CUDA array

    // Display information on all of the arrays
    if ( varNameString == "all" ){
        
        mexPrintf("\n"); 
        for (int arr_idx=0; arr_idx<CUDA_arr_names.size(); arr_idx++)
        {
            mexPrintf("GPU Matrix size for %s: %i x %i x %i of type %s \n", CUDA_arr_names[arr_idx].c_str(), CUDA_arr_sizes[arr_idx][0], CUDA_arr_sizes[arr_idx][1], CUDA_arr_sizes[arr_idx][2], CUDA_arr_types[arr_idx].c_str()); 
        }
        mexPrintf("\n"); 
        return;
    }

    // Locate the index of the CUDA_arr_names vector which correspondes to the given variable name         
    int arr_idx = FindArrayIndex(varNameString, CUDA_arr_names);

    if (arr_idx < 0)
    {
        mexErrMsgTxt("Failed to locate variable. Please check spelling.");
        return;
    }     

    mexPrintf("\n"); 
    mexPrintf("GPU Matrix size for %s: %i x %i x %i of type %s \n", varNameString.c_str(), CUDA_arr_sizes[arr_idx][0], CUDA_arr_sizes[arr_idx][1], CUDA_arr_sizes[arr_idx][2], CUDA_arr_types[arr_idx].c_str()); 
    mexPrintf("\n"); 
}

void CPU_CUDA_Memory::CUDA_Copy(std::string varNameString, float *New_Array)
{
    // Given a Matlab array, copy the data to the corresponding CUDA pointer
    mexPrintf("\n");
    mexPrintf("Copying GPU memory: Name %s \n", varNameString.c_str()); 

    // Locate the index of the CUDA_arr_names vector which correspondes to the given variable name 
    int arr_idx = FindArrayIndex(varNameString, CUDA_arr_names);

    if (arr_idx < 0)
    {
        mexErrMsgTxt("Failed to locate variable. Please check spelling.");
        return;
    }     

    // TO DO: Check if the allocated memory is the same size as the input array

    // Set the GPU device to the device which contains the CUDA array
    cudaSetDevice(CUDA_arr_GPU_Assignment[arr_idx]);

    // Copy the Matlab array to the class pointer for deep copy
    // Otherwise, Matlab seems to delete it's pointer once returning from the Mex file
    int dim_size = CUDA_arr_sizes[arr_idx][0] * CUDA_arr_sizes[arr_idx][1] * CUDA_arr_sizes[arr_idx][2];

    if ( CUDA_arr_types[arr_idx] == "int")
    {
        // // Get the pointer to the input Matlab array which should be int type (same as previously allocated)
        // int* matlabArray = (int*)mxGetData(Matlab_Pointer);

        // // CUDA function to copy the data from device to host
        // cudaMemcpy(CUDA_arr_ptrs[arr_idx].i, matlabArray, dim_size*sizeof(int), cudaMemcpyHostToDevice);
        // cudaDeviceSynchronize();

    } else if ( CUDA_arr_types[arr_idx] == "float")
    {         
        // Sends data to device
        cudaMemcpy(CUDA_arr_ptrs[arr_idx].f, New_Array, dim_size*sizeof(float), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

    } else 
    {
        mexErrMsgTxt("Unrecognized data type. Please choose either int, float, or double.");
    }      
    

}

mxArray* CPU_CUDA_Memory::CUDA_Return(std::string varNameString, mxArray *Matlab_Pointer)
{
    // Copy the data from the corresponding CUDA array back to the Matlab array
    mexPrintf("\n");
    mexPrintf("Returning memory: Name %s \n", varNameString.c_str()); 

    // Locate the index of the CUDA_arr_names vector which correspondes to the given variable name 
    int arr_idx = FindArrayIndex(varNameString, CUDA_arr_names);

    if (arr_idx < 0)
    {
        mexErrMsgTxt("Failed to locate variable. Please check spelling.");
    }     

    // TO DO: Check if the allocated memory is the same size as the input array       

    // Set the GPU device to the device which contains the CUDA array
    cudaSetDevice(CUDA_arr_GPU_Assignment[arr_idx]);

    // Create the output Matlab array (using the stored size of the corresponding C++ array)
    mwSize dims[3]; // matlab is row major (not column major)
    dims[0] = CUDA_arr_sizes[arr_idx][0]; 
    dims[1] = CUDA_arr_sizes[arr_idx][1];
    dims[2] = CUDA_arr_sizes[arr_idx][2];

    // Copy the Matlab array to the class pointer for deep copy
    // Otherwise, Matlab seems to delete it's pointer once returning from the Mex file
    int dim_size = CUDA_arr_sizes[arr_idx][0] * CUDA_arr_sizes[arr_idx][1] * CUDA_arr_sizes[arr_idx][2];

    if ( CUDA_arr_types[arr_idx] == "int")
    {
        // Create the output matlab array as type int
        Matlab_Pointer = mxCreateNumericArray(3, dims, mxINT32_CLASS, mxREAL);   

        // Get a pointer to the output matrix created above
        int* matlabArrayPtr = (int*)mxGetData(Matlab_Pointer);

        // CUDA function to copy the data from device to host
        cudaMemcpy(matlabArrayPtr, CUDA_arr_ptrs[arr_idx].i, dim_size*sizeof(int), cudaMemcpyDeviceToHost);            
        cudaDeviceSynchronize();

        return Matlab_Pointer;

    } else if ( CUDA_arr_types[arr_idx] == "float")
    {
        
        // Create the output matlab array as type float
        Matlab_Pointer = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);   

        // Get a pointer to the output matrix created above
        float* matlabArrayPtr = (float*)mxGetData(Matlab_Pointer);

        // CUDA function to copy the data from device to host
        cudaMemcpy(matlabArrayPtr, CUDA_arr_ptrs[arr_idx].f, dim_size*sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        return Matlab_Pointer;

    } else 
    {
        mexErrMsgTxt("Unrecognized data type. Please choose either int, float, or double.");
    }   
}
