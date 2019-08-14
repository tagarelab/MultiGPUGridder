#include "MemoryManager.h"

// The MemoryManager class functions
bool MemoryManager::GPUArrayAllocated(std::string varNameString, int GPU_Device)
{
    // Given the name of some GPU array, check if it is already allocated on the given GPU

    int arr_idx = FindArrayIndex(varNameString, this->CUDA_arr_names);

    if (arr_idx != -1) // The array was found on one of the GPUs
    {
        // Which GPU is the array assigned to?
        int GPU_assigned = this->CUDA_arr_GPU_Assignment[arr_idx];

        // Is the array on the requested GPU device already?
        if (GPU_assigned == GPU_Device)
        {
            return true;
        }
        else
        {
            // The array is NOT on the requested GPU device
            return false;
        }
    }
    else
    { // The array is NOT already allocated on any GPU
        return false;
    }
}

bool MemoryManager::CPUArrayAllocated(std::string varNameString)
{
    // Given the name of some CPU array check to see if it is already allocated

    int arr_idx = FindArrayIndex(varNameString, this->cpu_arr_names);

    if (arr_idx != -1) // The array was found
    {
        return true;
    }
    else
    { // The array is NOT already allocated
        return false;
    }
}

int MemoryManager::FindArrayIndex(std::string varNameString, std::vector<std::string> NameVector)
{
    // Find the index in the NameVector vector which is equal to a given string

    // Initilize the array index variable
    int arr_idx = -1;

    // Loop through all of the array names and find the correct index
    for (int i = 0; i < NameVector.size(); i++)
    {
        if (NameVector[i] == varNameString) // Are the strings equal?
        {
            arr_idx = i;
            break; // Stop once we reach the first index
        }
    }

    // Check to make sure we found one
    if (arr_idx >= NameVector.size() || arr_idx == -1) // String name wasn't found in the vector
    {
        //std::cerr << "Failed to locate variable. " << varNameString << " Please check spelling." << '\n';
        return -1;
    }

    return arr_idx;
}

float *MemoryManager::ReturnCPUFloatPtr(std::string varNameString)
{
    // Given the name of a variable, return the memory pointer (supports only CPU float pointers)

    // Locate the index of the cpu_arr_names vector which correspondes to the given variable name
    int arr_idx = FindArrayIndex(varNameString, cpu_arr_names);

    if (arr_idx < 0)
    {
        std::cerr << "Failed to locate variable. Please check spelling." << '\n';
        return NULL;
    }

    // Return the CPU memory pointer
    return cpu_arr_ptrs[arr_idx];
}

float *MemoryManager::ReturnCUDAFloatPtr(std::string varNameString)
{
    // Given the name of a variable, return the memory pointer (supports only CUDA float pointers)

    // Locate the index of the cpu_arr_names vector which correspondes to the given variable name
    int arr_idx = FindArrayIndex(varNameString, CUDA_arr_names);

    if (arr_idx < 0)
    {
        std::cerr << "Failed to locate variable. Please check spelling." << '\n';
        return NULL; // Return null pointer since no pointer was found
    }

    // Return the CUDA memory pointer
    return CUDA_arr_ptrs[arr_idx].f;
}

cufftComplex *MemoryManager::ReturnCUDAComplexPtr(std::string varNameString)
{
    // Given the name of a variable, return the memory pointer (supports only CUDA cufftComplex pointers)

    // Locate the index of the cpu_arr_names vector which correspondes to the given variable name
    int arr_idx = FindArrayIndex(varNameString, CUDA_arr_names);

    if (arr_idx < 0)
    {
        std::cerr << "Failed to locate variable. Please check spelling." << '\n';
        return NULL; // Return null pointer since no pointer was found
    }

    // Return the CUDA memory pointer
    return CUDA_arr_ptrs[arr_idx].c;
}

void MemoryManager::mem_alloc(std::string varNameString, std::string dataType, int *dataSize)
{
    // Allocate memory based on the dataType (i.e. int, float, etc.)

    // Save the name of the variable and the array size to the corresponding vectors
    cpu_arr_names.push_back(varNameString);

    // Save the dataType of this array
    cpu_arr_types.push_back(dataType);

    // Deep copy the data size pointer
    int *new_dataSize_ptr = new int[3];
    for (int i = 0; i < 3; i++)
    {
        new_dataSize_ptr[i] = dataSize[i];
    }

    // Save to the vector of array sizes
    cpu_arr_sizes.push_back(new_dataSize_ptr);

    // Allocate the memory and save the pointer to the corresponding vector
    if (dataType == "float")
    {
        // Need to convert the dataSize to long long int type to allow for array length larger than maximum int32 value
        unsigned long long *dataSizeLong = new unsigned long long[3];
        dataSizeLong[0] = (unsigned long long)dataSize[0];
        dataSizeLong[1] = (unsigned long long)dataSize[1];
        dataSizeLong[2] = (unsigned long long)dataSize[2];

        float *new_ptr = new float[dataSizeLong[0] * dataSizeLong[1] * dataSizeLong[2]]; // Multiply the X,Y,Z dimensions of the array

        // Save the new pointer to the allocated CPU array
        cpu_arr_ptrs.push_back(new_ptr);
    }
    else
    {
        std::cerr << "Unrecognized data type. Please choose either int, float, or double." << '\n';
    }
}

void MemoryManager::mem_Copy(std::string varNameString, float *New_Array)
{
    // Given a float array, copy the data to the corresponding C++ pointer

    // Locate the index of the cpu_arr_names vector which correspondes to the given variable name
    int arr_idx = FindArrayIndex(varNameString, cpu_arr_names);

    if (arr_idx < 0)
    {
        std::cerr << "Failed to locate variable. Please check spelling." << '\n';
        return;
    }

    // Need to convert the dims to long long int type to allow for array length larger than maximum int32 value
    unsigned long long *dim_size = new unsigned long long[3];
    dim_size[0] = (unsigned long long)cpu_arr_sizes[arr_idx][0];
    dim_size[1] = (unsigned long long)cpu_arr_sizes[arr_idx][1];
    dim_size[2] = (unsigned long long)cpu_arr_sizes[arr_idx][2];

    if (cpu_arr_types[arr_idx] == "float")
    {
        std::memcpy(cpu_arr_ptrs[arr_idx], New_Array, sizeof(float) * (dim_size[0] * dim_size[1] * dim_size[2]));
    }
    else
    {
        std::cerr << "Unrecognized data type. Only float is supported currently for mem_Copy()." << '\n';
    }
}

void MemoryManager::pin_mem(std::string varNameString)
{
    // Given a variable name, pin the associated CPU array (i.e. make it non-pageable on the RAM so the GPUs can directly access it)

    // Locate the index of the cpu_arr_names vector which correspondes to the given variable name
    int arr_idx = FindArrayIndex(varNameString, cpu_arr_names);

    if (arr_idx < 0)
    {
        std::cerr << "Failed to locate variable. Please check spelling." << '\n';
        return;
    }

    // Need to convert the cpu_arr_sizes[arr_idx] to long long int type to allow for array length larger than maximum int32 value
    unsigned long long *dataSizeLong = new unsigned long long[3];
    dataSizeLong[0] = (unsigned long long)cpu_arr_sizes[arr_idx][0];
    dataSizeLong[1] = (unsigned long long)cpu_arr_sizes[arr_idx][1];
    dataSizeLong[2] = (unsigned long long)cpu_arr_sizes[arr_idx][2];

    // Get the size of the array
    unsigned long long dim_size = dataSizeLong[0] * dataSizeLong[1] * dataSizeLong[2];

    // Register the CPU array as pinned memory using the CUDA function
    if (cpu_arr_types[arr_idx] == "float")
    {
        cudaHostRegister(cpu_arr_ptrs[arr_idx], sizeof(float) * dim_size, 0);
        return;
    }
    else
    {
        std::cerr << "Unrecognized data type. Only float is currently supported." << '\n';
    }
}

void MemoryManager::disp_mem(std::string varNameString)
{
    // Display in the console the size of a given matrix

    if (varNameString == "all")
    {
        // Display information on all of the arrays
        std::cout << "\n";
        for (int arr_idx = 0; arr_idx < cpu_arr_names.size(); arr_idx++)
        {
            std::cout << "CPU Matrix size for " << cpu_arr_names[arr_idx] << ": "
                      << cpu_arr_sizes[arr_idx][0] << " x " << cpu_arr_sizes[arr_idx][1] << " x " << cpu_arr_sizes[arr_idx][2] << " of type " << cpu_arr_types[arr_idx] << '\n';
        }
        std::cout << "\n";
        return;
    }

    // Locate the index of the cpu_arr_names vector which correspondes to the given variable name
    int arr_idx = FindArrayIndex(varNameString, cpu_arr_names);

    if (arr_idx < 0)
    {
        std::cerr << "Failed to locate variable. Please check spelling." << '\n';
        return;
    }

    std::cout << "\n";
    std::cout << "CPU Matrix size for " << cpu_arr_names[arr_idx] << ": "
              << cpu_arr_sizes[arr_idx][0] << " x " << cpu_arr_sizes[arr_idx][1] << " x " << cpu_arr_sizes[arr_idx][2] << " of type " << cpu_arr_types[arr_idx] << '\n';
    std::cout << "\n";
}

void MemoryManager::mem_Free(std::string varNameString)
{
    // Free the memory of the corresponding array

    // If varNameString == "all" then free all the allocated CPU arrays
    if (varNameString == "all")
    {
        // Temporarily copy the cpu_arr_names to a new vector
        std::vector<std::string> cpu_arr_names_temp;
        for (int i = 0; i < cpu_arr_names.size(); i++)
        {
            cpu_arr_names_temp.push_back(cpu_arr_names[i]);
        }

        // Iterate over each name to free the memory
        for (int i = 0; i < cpu_arr_names_temp.size(); i++)
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
        std::cerr << "Failed to locate variable. Please check spelling." << '\n';
        return;
    }

    // Delete the arrays first from memory
    if (cpu_arr_types[arr_idx] == "float")
    {
        std::free(cpu_arr_ptrs[arr_idx]);
    }

    // Delete the corresponding information from all the cpu vectors
    cpu_arr_names.erase(cpu_arr_names.begin() + arr_idx);
    cpu_arr_types.erase(cpu_arr_types.begin() + arr_idx);
    cpu_arr_sizes.erase(cpu_arr_sizes.begin() + arr_idx);
    cpu_arr_ptrs.erase(cpu_arr_ptrs.begin() + arr_idx);
}

void MemoryManager::CUDA_alloc(std::string varNameString, std::string dataType, int *dataSize, int GPU_Device)
{
    // Allocate memory on the GPU based on the dataType (i.e. int, float, etc.)

    // Save the name of the variable and the array size to the corresponding vectors
    CUDA_arr_names.push_back(varNameString);

    // Save the dataType of this array
    CUDA_arr_types.push_back(dataType);

    // Deep copy the data size pointer
    int *new_dataSize_ptr = new int[3];
    for (int i = 0; i < 3; i++)
    {
        new_dataSize_ptr[i] = dataSize[i];
    }

    // Which GPU to allocate the memory to?
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);

    // Provide error message if no GPUs are found (i.e. all cards are busy)
    if (numGPUs == 0)
    {
        std::cerr << "No NVIDIA graphic cards located on your computer. All cards may be busy and unavailable." << '\n';
    }

    if (GPU_Device < 0 || GPU_Device >= numGPUs)
    {
        std::cerr << "Error in GPU selection. Please provide an integer between 0 and the number of NVIDIA graphic cards on your computer." << '\n';
    }

    CUDA_arr_GPU_Assignment.push_back(GPU_Device);
    cudaSetDevice(GPU_Device);

    // Save to the vector of array sizes
    CUDA_arr_sizes.push_back(new_dataSize_ptr);

    // Check to make sure the GPU has enough available memory left
    size_t mem_tot_0 = 0;
    size_t mem_free_0 = 0;
    cudaMemGetInfo(&mem_free_0, &mem_tot_0);
    // std::cout << "Free memory before copy: " << mem_free_0 << " Device: " << GPU_Device << std::endl;

    // Union of the pointer types
    Ptr_Types n;

    // Allocate the memory and save the pointer to the corresponding vector
    if (dataType == "float")
    {
        // std::cout << "CUDA Memory requested: " << sizeof(float) * (dataSize[0] * dataSize[1] * dataSize[2]) << " bytes" << '\n';
        // Is there enough available memory on the device to allocate this array?
        if (mem_free_0 < sizeof(float) * (dataSize[0] * dataSize[1] * dataSize[2]))
        {
            std::cerr << "Not enough memory on the device to allocate the requested memory. Try fewer number of projections or a smaller volume. Or increase SetNumberBatches()" << '\n';
            return;
        }

        float *devPtr = new float[dataSize[0] * dataSize[1] * dataSize[2]]; // Multiply the X,Y,Z dimensions of the array

        cudaMalloc(&devPtr, sizeof(float) * (dataSize[0] * dataSize[1] * dataSize[2])); // Multiply the X,Y,Z dimensions of the array

        // Save the new pointer to the allocated CUDA GPU array
        n.f = devPtr;       

        
    } else if (dataType == "cufftComplex")
    {
        // Is there enough available memory on the device to allocate this array?
        if (mem_free_0 < sizeof(cufftComplex) * (dataSize[0] * dataSize[1] * dataSize[2]))
        {
            std::cerr << "Not enough memory on the device to allocate the requested memory. Try fewer number of projections or a smaller volume. Or increase SetNumberBatches()" << '\n';
            return;
        }

        cufftComplex *devPtr = new cufftComplex[dataSize[0] * dataSize[1] * dataSize[2]]; // Multiply the X,Y,Z dimensions of the array

        cudaMalloc(&devPtr, sizeof(cufftComplex) * (dataSize[0] * dataSize[1] * dataSize[2])); // Multiply the X,Y,Z dimensions of the array

        // Save the new pointer to the allocated CUDA GPU array
        n.c = devPtr;;

    }
    else
    {
        std::cerr << "Unrecognized data type. Only float is currently supported." << '\n';
    }

    // Save the union of pointers to the vector of CUDA pointers
    CUDA_arr_ptrs.push_back(n);
}

void MemoryManager::CUDA_Free(std::string varNameString)
{
    // Free the memory of the corresponding CUDA GPU array

    // If varNameString == "all" then free all the allocated GPU arrays
    if (varNameString == "all")
    {
        // Temporarily copy the CUDA_arr_names to a new vector
        std::vector<std::string> CUDA_arr_names_temp;
        for (int i = 0; i < CUDA_arr_names.size(); i++)
        {
            CUDA_arr_names_temp.push_back(CUDA_arr_names[i]);
        }

        // Iterate over each name to free the memory
        for (int i = 0; i < CUDA_arr_names_temp.size(); i++)
        {
            // Start at the end of the CUDA_arr_names vector and free each GPU array
            CUDA_Free(CUDA_arr_names_temp[i]);
        }

        return;
    }

    // Locate the index of the cpu_arr_names vector which correspondes to the given variable name
    int arr_idx = FindArrayIndex(varNameString, CUDA_arr_names);

    if (arr_idx < 0)
    {
        std::cerr << "Failed to locate variable." << varNameString << " Please check spelling." << '\n';
        return;
    }

    // Set the GPU device to the device which contains the CUDA array
    cudaSetDevice(CUDA_arr_GPU_Assignment[arr_idx]);

    // Delete the array from the GPU memory
    if (CUDA_arr_types[arr_idx] == "float")
    {
        cudaFree(CUDA_arr_ptrs[arr_idx].f);
    
    } else if (CUDA_arr_types[arr_idx] == "cufftComplex")
    {
        cudaFree(CUDA_arr_ptrs[arr_idx].c);
    } else
    {
        std::cerr << "Unrecognized data type. Only float is currently supported." << '\n';
    }

    // Delete the corresponding information from all the cpu vectors
    CUDA_arr_names.erase(CUDA_arr_names.begin() + arr_idx);
    CUDA_arr_types.erase(CUDA_arr_types.begin() + arr_idx);
    CUDA_arr_sizes.erase(CUDA_arr_sizes.begin() + arr_idx);
    CUDA_arr_ptrs.erase(CUDA_arr_ptrs.begin() + arr_idx);
    CUDA_arr_GPU_Assignment.erase(CUDA_arr_GPU_Assignment.begin() + arr_idx);
}

void MemoryManager::CUDA_disp_mem(std::string varNameString)
{
    // Display in the Matlab console the size of a given GPU CUDA array

    // Display information on all of the arrays
    if (varNameString == "all")
    {

        std::cout << '\n';
        for (int arr_idx = 0; arr_idx < CUDA_arr_names.size(); arr_idx++)
        {
            std::cout << "GPU Matrix size for " << CUDA_arr_names[arr_idx] << ": " << CUDA_arr_sizes[arr_idx][0] << " x " << CUDA_arr_sizes[arr_idx][1] << " x " << CUDA_arr_sizes[arr_idx][2] << " of type " << CUDA_arr_types[arr_idx] << '\n';
        }
        std::cout << '\n';
        return;
    }

    // Locate the index of the CUDA_arr_names vector which correspondes to the given variable name
    int arr_idx = FindArrayIndex(varNameString, CUDA_arr_names);

    if (arr_idx < 0)
    {
        std::cerr << "Failed to locate variable. Please check spelling." << '\n';
        return;
    }

    std::cout << '\n';
    std::cout << "GPU Matrix size for " << varNameString << ": " << CUDA_arr_sizes[arr_idx][0] << " x " << CUDA_arr_sizes[arr_idx][1] << " x " << CUDA_arr_sizes[arr_idx][2] << " of type " << CUDA_arr_types[arr_idx] << '\n';
    std::cout << '\n';
}

void MemoryManager::CUDA_Copy(std::string varNameString, float *New_Array)
{
    // Given a float array, copy the data to the corresponding CUDA pointer

    // Locate the index of the CUDA_arr_names vector which correspondes to the given variable name
    int arr_idx = FindArrayIndex(varNameString, CUDA_arr_names);

    if (arr_idx < 0)
    {
        std::cerr << "Failed to locate variable. Please check spelling." << '\n';
        return;
    }

    // Set the GPU device to the device which contains the CUDA array
    cudaSetDevice(CUDA_arr_GPU_Assignment[arr_idx]);

    // Copy the Matlab array to the class pointer for deep copy
    // Otherwise, Matlab seems to delete it's pointer once returning from the Mex file
    int dim_size = CUDA_arr_sizes[arr_idx][0] * CUDA_arr_sizes[arr_idx][1] * CUDA_arr_sizes[arr_idx][2];

    if (CUDA_arr_types[arr_idx] == "float")
    {
        // Sends data to device asynchronously
        cudaMemcpy(CUDA_arr_ptrs[arr_idx].f, New_Array, dim_size * sizeof(float), cudaMemcpyHostToDevice);
    } else if (CUDA_arr_types[arr_idx] == "cufftComplex")
    {
        // Sends data to device asynchronously
        cudaMemcpy(CUDA_arr_ptrs[arr_idx].c, New_Array, dim_size * sizeof(float), cudaMemcpyHostToDevice);
    } else
    {
        std::cerr << "Only float type is currently supported." << '\n';
    }
}

void MemoryManager::CUDA_Copy_Asyc(std::string varNameString, float *New_Array, cudaStream_t stream)
{
    // Given a Matlab array, copy the data to the corresponding CUDA pointer using the given cudaStream

    // Locate the index of the CUDA_arr_names vector which correspondes to the given variable name
    int arr_idx = FindArrayIndex(varNameString, CUDA_arr_names);

    if (arr_idx < 0)
    {
        std::cerr << "Failed to locate variable. Please check spelling." << '\n';
        return;
    }

    // Set the GPU device to the device which contains the CUDA array
    cudaSetDevice(CUDA_arr_GPU_Assignment[arr_idx]);

    // Create the stream on the selected GPU
    cudaStreamCreate(&stream);

    // Copy the Matlab array to the class pointer for deep copy
    // Otherwise, Matlab seems to delete it's pointer once returning from the Mex file
    int dim_size = CUDA_arr_sizes[arr_idx][0] * CUDA_arr_sizes[arr_idx][1] * CUDA_arr_sizes[arr_idx][2];

    if (CUDA_arr_types[arr_idx] == "float")
    {
        // Sends data to device asynchronously
        cudaMemcpyAsync(CUDA_arr_ptrs[arr_idx].f, New_Array, dim_size * sizeof(float), cudaMemcpyHostToDevice, stream);
    }  else if (CUDA_arr_types[arr_idx] == "cufftComplex")
    {
        // Sends data to device asynchronously
        cudaMemcpyAsync(CUDA_arr_ptrs[arr_idx].c, New_Array, dim_size * sizeof(float), cudaMemcpyHostToDevice, stream);
    } else
    {
        std::cerr << "Only float type is currently supported." << '\n';
    }
}

int *MemoryManager::CPU_Get_Array_Size(std::string varNameString)
{
    // Get the size of the CPU array corresponding to the input array name (as a string)

    // Locate the index of the CUDA_arr_names vector which correspondes to the given variable name
    int arr_idx = FindArrayIndex(varNameString, cpu_arr_names);

    if (arr_idx < 0)
    {
        std::cerr << "Failed to locate variable. Please check spelling." << '\n';
    }

    int *dims = new int[3];
    dims[0] = cpu_arr_sizes[arr_idx][0];
    dims[1] = cpu_arr_sizes[arr_idx][1];
    dims[2] = cpu_arr_sizes[arr_idx][2];

    return dims;
}

int *MemoryManager::CUDA_Get_Array_Size(std::string varNameString)
{
    // Get the size of the CUDA GPU array corresponding to the input array name (as a string)

    // Locate the index of the CUDA_arr_names vector which correspondes to the given variable name
    int arr_idx = FindArrayIndex(varNameString, CUDA_arr_names);

    if (arr_idx < 0)
    {
        std::cerr << "Failed to locate variable. Please check spelling." << '\n';
    }

    int *dims = new int[3];
    dims[0] = CUDA_arr_sizes[arr_idx][0];
    dims[1] = CUDA_arr_sizes[arr_idx][1];
    dims[2] = CUDA_arr_sizes[arr_idx][2];

    return dims;
}

float *MemoryManager::CUDA_Return(std::string varNameString)
{
    // Copy the data from the corresponding CUDA array back to a float array

    // Locate the index of the CUDA_arr_names vector which correspondes to the given variable name
    int arr_idx = FindArrayIndex(varNameString, CUDA_arr_names);

    if (arr_idx < 0)
    {
        std::cerr << "Failed to locate variable. Please check spelling." << '\n';
    }

    // Set the GPU device to the device which contains the CUDA array
    cudaSetDevice(CUDA_arr_GPU_Assignment[arr_idx]);

    int dim_size = CUDA_arr_sizes[arr_idx][0] * CUDA_arr_sizes[arr_idx][1] * CUDA_arr_sizes[arr_idx][2];

    if (CUDA_arr_types[arr_idx] == "float")
    {
        // Copy from the GPU to the CPU
        float *CPU_Array = new float[dim_size];
        cudaMemcpy(CPU_Array, CUDA_arr_ptrs[arr_idx].f, dim_size * sizeof(float), cudaMemcpyDeviceToHost);

        return CPU_Array;
    } else if (CUDA_arr_types[arr_idx] == "cufftComplex")
    {
        // Copy from the GPU to the CPU
        cufftComplex *CPU_Array = new cufftComplex[dim_size];
        cudaMemcpy(CPU_Array, CUDA_arr_ptrs[arr_idx].c, dim_size * sizeof(float), cudaMemcpyDeviceToHost);

    } else
    {
        std::cerr << "Only float type is currently supported." << '\n';
    }

    return NULL;
}
