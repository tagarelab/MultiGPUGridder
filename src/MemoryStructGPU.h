#pragma once

#include "MemoryStruct.h"

// Extend the memory struct from the abstract gridder class to include GPU related information
struct MemoryStructGPU : public MemoryStruct
{
    // Extend the constructor from MemoryStruct
    MemoryStructGPU(int dims, int *ArraySize, int GPU_Device) : MemoryStruct(dims, ArraySize)
    {
        // Which GPU to use for allocating the array
        this->GPU_Device = GPU_Device;

        // Allocate the memory on the GPU device
        this->AllocateGPUArray();
    }

    // Which GPU is the memory allocated on?
    int GPU_Device;

    // Error flag to rememeber if the allocation was succesful or not
    int ErrorFlag;

    void CopyToGPU(float *Array, int Bytes)
    {
        // Given a float pointer (on host CPU) and number of bytes, copy the memory to this GPU array
        cudaMemcpy(this->ptr, Array, Bytes, cudaMemcpyHostToDevice);
    }

    void CopyFromGPU(float *Array, int Bytes)
    {
        // Given a float pointer (on host CPU) and number of bytes, copy the memory to this GPU array
        cudaMemcpy(Array, this->ptr, Bytes, cudaMemcpyDeviceToHost);
    }

    void AllocateGPUArray()
    {
        // Allocate the memory to a given GPU device

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

    void DeallocateGPUArray()
    {
    }
};
