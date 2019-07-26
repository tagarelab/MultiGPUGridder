#ifndef CPU_CUDA_MEMORY_H // Only define the header once
#define CPU_CUDA_MEMORY_H

#include "mex.h"
#include "CPU_CUDA_Memory.h"
#include <iostream>
#include <vector>
#include <limits>

// Include the CUDA Runtime
#include <cuda_runtime.h>
#include <cuda.h>

// NVTX labeling tools (for the nvidia profiling)
#include <nvToolsExt.h>
#include <cuda_profiler_api.h>

// The class that we are interfacing to
class CPU_CUDA_Memory
{
public:
 
    CPU_CUDA_Memory() { mexPrintf("Calling constructor\n"); }
    ~CPU_CUDA_Memory() { mexPrintf("Calling destructor\n"); }

    int FindArrayIndex(std::string varNameString, std::vector<std::string> NameVector);

    void mem_alloc(std::string varNameString, std::string dataType, int * dataSize);

    void mem_Copy(std::string varNameString, const mxArray *Matlab_Pointer);

    mxArray* mem_Return(std::string varNameString, mxArray *Matlab_Pointer);

    void pin_mem(std::string varNameString);

    void disp_mem(std::string varNameString);

    void CUDA_alloc(std::string varNameString, std::string dataType, int * dataSize, int GPU_Device);

    void CUDA_Free(std::string varNameString);

    void CUDA_disp_mem(std::string varNameString);

    void CUDA_Copy(std::string varNameString, const mxArray *Matlab_Pointer);

    mxArray* CUDA_Return(std::string varNameString, mxArray *Matlab_Pointer);

private:

    // Allow the vector of CPU pointers to be either int, float, or double type
    union Ptr_Types
    {
        int *i;
        float *f;
    };

    // Variables to hold the CPU arrays
    std::vector<std::string> cpu_arr_names; // Name of the variables, e.g. 'imgVol'
    std::vector<std::string> cpu_arr_types; // String of datatype, e.g. 'int', 'float', or 'double'
    std::vector<int*> cpu_arr_sizes;        // Size of the cpu array, e.g. [256, 256, 256]
    std::vector<Ptr_Types> cpu_arr_ptrs;    // Memory pointer to the corresponding cpu array (supports int, float, or double)

    // Variables to hold the CUDA GPU arrays
    std::vector<std::string> CUDA_arr_names;   // Name of the variables, e.g. 'gpuVol'
    std::vector<std::string> CUDA_arr_types;   // String of datatype, e.g. 'int', 'float', or 'double'    
    std::vector<int*> CUDA_arr_sizes;          // Size of the float array, e.g. [256, 256, 256]
    std::vector<Ptr_Types> CUDA_arr_ptrs;      // Memory pointer to the corresponding int array
    std::vector<int> CUDA_arr_GPU_Assignment;  // Which GPU is the array assigned to? (e.g. integer from 0 to 4 for 4 GPUs)

};

#endif