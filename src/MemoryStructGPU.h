#pragma once

#include "MemoryStruct.h"

// Extend the memory struct from the abstract gridder class to include GPU related information
struct MemoryStructGPU : public MemoryStruct
{
    // Which GPU is the memory allocated on?
    int GPU_Device;

    // Error flag to rememeber if the allocation was succesful or not
    int ErrorFlag;

    // Extend the constructor from MemoryStruct
    MemoryStructGPU(int dims, int *ArraySize, int GPU_Device) : MemoryStruct(dims, ArraySize)
    {
        // Which GPU to use for allocating the array
        this->GPU_Device = GPU_Device;

        // Free the float array allocated in MemoryStruct. TO DO: make this not necessary
        std::free(this->ptr);

        // Allocate the array on the GPU
        AllocateGPUArray();
    }

    // Deconstructor to free the GPU memory
    ~MemoryStructGPU()
    {
        DeallocateGPUArray();
    }

    // Copy a float array from the CPU to the allocated array on the GPU
    void CopyToGPU(float *Array, int Bytes)
    {
        if (Bytes != this->bytes())
        {
            std::cerr << "Error in CopyToGPU(): supplied array has " << Bytes << " bytes while the allocated GPU array has " << this->bytes() << " bytes." << '\n';
        }

        // Given a float pointer (on host CPU) and number of bytes, copy the memory to this GPU array
        cudaMemcpy(this->ptr, Array, Bytes, cudaMemcpyHostToDevice);
    }

    // Copy the array from the GPU to a float array on the CPU
    void CopyFromGPU(float *Array, int Bytes)
    {
        if (Bytes != this->bytes())
        {
            std::cerr << "Error in CopyFromGPU(): supplied array has " << Bytes << " bytes while the allocated GPU array has " << this->bytes() << " bytes." << '\n';
        }

        // Given a float pointer (on host CPU) and number of bytes, copy the memory to this GPU array
        cudaMemcpy(Array, this->ptr, Bytes, cudaMemcpyDeviceToHost);
    }

    // Allocate the memory to a given GPU device
    void AllocateGPUArray()
    {
        std::cout << "GPU AllocateArray()" << '\n';

        // Set the current GPU
        cudaSetDevice(this->GPU_Device);

        // Check to make sure the GPU has enough available memory left
        size_t mem_tot_0 = 0;
        size_t mem_free_0 = 0;
        cudaMemGetInfo(&mem_free_0, &mem_tot_0);

        // Is there enough available memory on the device to allocate this array?
        if (mem_free_0 < this->bytes())
        {
            std::cerr << "Not enough memory on the device to allocate the requested memory. Try fewer number of projections or a smaller volume. Or increase SetNumberBatches()" << '\n';

            this->ptr = NULL; // Set the pointer to NULL

            this->ErrorFlag = 1; // Set the error flag to 1 to remember that this failed
        }
        else
        {
            // There is enough memory left on the current GPU
            cudaMalloc(&this->ptr, this->bytes());

            this->ErrorFlag = 0; // Set the error flag to 0 to remember that this was sucessful
        }
    }

    // Free the GPU memory
    void DeallocateGPUArray()
    {
        cudaFree(this->ptr);
    }
};
