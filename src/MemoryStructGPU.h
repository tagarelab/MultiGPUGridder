#pragma once

#include "MemoryStruct.h"

// Extend the memory struct from the abstract gridder class to include GPU related information
template<class T = float>
struct MemoryStructGPU : public MemoryStruct< T >
{
    // Which GPU is the memory allocated on?
    int GPU_Device;

    // Error flag to rememeber if the allocation was succesful or not
    int ErrorFlag;

    // CUDA stream for asyc copying to / from the GPU
    cudaStream_t stream;

    // Extend the constructor from MemoryStruct
    MemoryStructGPU(int dims, int *ArraySize, int GPU_Device) : MemoryStruct<T>(dims, ArraySize)
    {
        // Which GPU to use for allocating the array
        this->GPU_Device = GPU_Device;

        // Set the GPU device to the device which contains the CUDA array
        cudaSetDevice(this->GPU_Device);

        // Create the stream on the selected GPU
        cudaStreamCreate(&this->stream);

        // Free the T type array allocated in MemoryStruct. TO DO: make this not necessary
        std::free(this->ptr);

        // Allocate the array on the GPU
        AllocateGPUArray();
    }

    // Deconstructor to free the GPU memory
    ~MemoryStructGPU()
    {
        DeallocateGPUArray();
    }

    // Copy a T type array from the CPU to the allocated array on the GPU
    void CopyToGPU(T *Array, int Bytes)
    {
        if (Bytes != this->bytes())
        {
            std::cerr << "Error in CopyToGPU(): supplied array has " << Bytes << " bytes while the allocated GPU array has " << this->bytes() << " bytes." << '\n';
        }

        // Given a T type pointer (on host CPU) and number of bytes, copy the memory to this GPU array
        cudaMemcpy(this->ptr, Array, Bytes, cudaMemcpyHostToDevice);
    }

    // Copy a T type array from the CPU to the allocated array on the GPU asynchronously
    void CopyToGPUAsyc(T *Array, int Bytes)
    {
        if (Bytes != this->bytes())
        {
            std::cerr << "Error in CopyToGPU(): supplied array has " << Bytes << " bytes while the allocated GPU array has " << this->bytes() << " bytes." << '\n';
        }

        // Given a T type pointer (on host CPU) and number of bytes, copy the memory to this GPU array
        cudaMemcpyAsync(this->ptr, Array, Bytes, cudaMemcpyHostToDevice, this->stream);
    }

    // Copy the array from the GPU to a T type array on the CPU
    void CopyFromGPU(T *Array, int Bytes)
    {
        if (Bytes != this->bytes())
        {
            std::cerr << "Error in CopyFromGPU(): supplied array has " << Bytes << " bytes while the allocated GPU array has " << this->bytes() << " bytes." << '\n';
        }

        // Given a T type pointer (on host CPU) and number of bytes, copy the memory to this GPU array
        cudaMemcpy(Array, this->ptr, Bytes, cudaMemcpyDeviceToHost);
    }

    // Allocate the memory to a given GPU device
    void AllocateGPUArray()
    {
        // Set the current GPU
        cudaSetDevice(this->GPU_Device);

        // Check to make sure the GPU has enough available memory left
        size_t mem_tot_0 = 0;
        size_t mem_free_0 = 0;
        cudaMemGetInfo(&mem_free_0, &mem_tot_0);

        // Is there enough available memory on the device to allocate this array?
        if (mem_free_0 < this->bytes())
        {
            std::cerr << "Error: Requested " << this->bytes() << " but only have " << mem_free_0 << " of memory remaining on GPU " << this->GPU_Device << '\n';

            std::cerr << "Size is: ";
            for (int i = 0; i < this->dims; i++)
            {
                std::cerr << this->GetSize(i) << " ";
            }
            std::cerr << '\n';
            std::cerr << "AllocateGPUArray(): Not enough memory on the device to allocate the requested memory. Try fewer number of projections or a smaller volume. Or increase SetNumberBatches()" << '\n';

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
