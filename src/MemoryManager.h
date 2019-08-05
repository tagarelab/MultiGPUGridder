#ifndef MemoryManager_H // Only define the header once
#define MemoryManager_H

#include "MemoryManager.h"
#include <iostream>
#include <vector>
#include <limits>
#include <cstring>
#include <string>

// Include the CUDA Runtime
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h> // Library for CUDA FFT and inverse FFT functions see https://www.bu.edu/pasi/files/2011/07/Lecture83.pdf


// NVTX labeling tools (for the nvidia profiling)
//#include <nvToolsExt.h>
#include <cuda_profiler_api.h>

// The class that we are interfacing to
class MemoryManager
{
public:
    // Class constructor
    MemoryManager()
    {
    }

    // Class deconstructor
    ~MemoryManager()
    {
        // Free all the GPU memory
        CUDA_Free("all");
        std::cout << "MemoryManager() deconstructor" << '\n';
    }

    bool GPUArrayAllocated(std::string varNameString, int GPU_Device);

    bool CPUArrayAllocated(std::string varNameString);

    int FindArrayIndex(std::string varNameString, std::vector<std::string> NameVector);

    float *ReturnCPUFloatPtr(std::string varNameString);

    float *ReturnCUDAFloatPtr(std::string varNameString);

<<<<<<< 1b4adfc5986cf104df88b8bf83462fd6e2ddbe70:src/MemoryManager.h
    void mem_alloc(std::string varNameString, std::string dataType, int *dataSize);
=======
    float* ReturnCUDAFloatPtr(std::string varNameString);

    cufftComplex* ReturnCUDAComplexPtr(std::string varNameString);

    void mem_alloc(std::string varNameString, std::string dataType, int * dataSize);
>>>>>>> Added support for cufftComplex allocation and returning pointer:CUDA_mex_files/CPU_CUDA_Memory.h

    void mem_Copy(std::string varNameString, float *New_Array);

    void pin_mem(std::string varNameString);

    void disp_mem(std::string varNameString);

    void mem_Free(std::string varNameString);

    void CUDA_alloc(std::string varNameString, std::string dataType, int *dataSize, int GPU_Device);

    void CUDA_Free(std::string varNameString);

    void CUDA_disp_mem(std::string varNameString);

    void CUDA_Copy(std::string varNameString, float *New_Array);

    void CUDA_Copy_Asyc(std::string varNameString, float *New_Array, cudaStream_t stream);

    int *CUDA_Get_Array_Size(std::string varNameString);

    int *CPU_Get_Array_Size(std::string varNameString);

<<<<<<< 1b4adfc5986cf104df88b8bf83462fd6e2ddbe70:src/MemoryManager.h
    float *CUDA_Return(std::string varNameString);
=======
    // Allow the vector of CPU pointers to be either int, float, or double type
    union Ptr_Types
    {
        int *i;        
        unsigned long long *un_int;
        float *f;
        cufftComplex * c;
    };
>>>>>>> Added support for cufftComplex allocation and returning pointer:CUDA_mex_files/CPU_CUDA_Memory.h

private:
    // Variables to hold the CPU arrays
    std::vector<std::string> cpu_arr_names; // Name of the variables, e.g. 'imgVol'
    std::vector<std::string> cpu_arr_types; // String of datatype, e.g. 'int', 'float', or 'double'
    std::vector<int *> cpu_arr_sizes;       // Size of the cpu array, e.g. [256, 256, 256]
    std::vector<float *> cpu_arr_ptrs;      // Memory pointer to the corresponding cpu array (supports int, float, or double)

    // Variables to hold the CUDA GPU arrays
    std::vector<std::string> CUDA_arr_names;  // Name of the variables, e.g. 'gpuVol'
    std::vector<std::string> CUDA_arr_types;  // String of datatype, e.g. 'int', 'float', or 'double'
    std::vector<int *> CUDA_arr_sizes;        // Size of the float array, e.g. [256, 256, 256]
    std::vector<float *> CUDA_arr_ptrs;       // Memory pointer to the corresponding int array
    std::vector<int> CUDA_arr_GPU_Assignment; // Which GPU is the array assigned to? (e.g. integer from 0 to 4 for 4 GPUs)
};

#endif