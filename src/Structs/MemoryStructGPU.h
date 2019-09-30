#pragma once

#include "MemoryStruct.h"

/**
 * @class   MemoryStructGPU
 * @brief   A class for allocating device (i.e GPU) memory
 *
 *
 * A class for allocating and deallocating GPU memory. MemoryStructGPU also remembers needed information for each allocated array
 * (e.g. CASImgs, images, coordinate axes, etc.) such as the array size, memory pointers, etc. This is the main GPU memory class.
 * 
 * MemoryStructGPU inherits from MemoryStruct and extends MemoryStructGPU to include GPU related information and functions. 
 * 
 * */

template <class T = float>
class MemoryStructGPU : public MemoryStruct<T>
{
public:
    /// Extend the MemoryStructGPU constructor from MemoryStruct
    MemoryStructGPU(int dims, int *ArraySize, int GPU_Device) : MemoryStruct<T>(dims, ArraySize)
    {
        // Which GPU to use for allocating the array
        this->GPU_Device = GPU_Device;

        // Set the GPU device to the device which contains the CUDA array
        cudaSetDevice(this->GPU_Device);

        // Create the stream on the selected GPU
        cudaStreamCreate(&this->stream);

        // Set the pointer to NULL
        this->ptr = NULL;
    }

    /// Extend the constructor from MemoryStruct: Array of 1 dimensions
    MemoryStructGPU(int dims, int ArraySizeX, int GPU_Device) : MemoryStruct<T>(dims, ArraySizeX)
    {
        // Which GPU to use for allocating the array
        this->GPU_Device = GPU_Device;

        // Set the GPU device to the device which contains the CUDA array
        cudaSetDevice(this->GPU_Device);

        // Create the stream on the selected GPU
        cudaStreamCreate(&this->stream);

        // Set the pointer to NULL
        this->ptr = NULL;
    }

    /// Extend the constructor from MemoryStruct: Array of 2 dimensions
    MemoryStructGPU(int dims, int ArraySizeX, int ArraySizeY, int GPU_Device) : MemoryStruct<T>(dims, ArraySizeX, ArraySizeY)
    {
        // Which GPU to use for allocating the array
        this->GPU_Device = GPU_Device;

        // Set the GPU device to the device which contains the CUDA array
        cudaSetDevice(this->GPU_Device);

        // Create the stream on the selected GPU
        cudaStreamCreate(&this->stream);

        // Set the pointer to NULL
        this->ptr = NULL;
    }

    /// Extend the constructor from MemoryStruct: Array of 3 dimensions
    MemoryStructGPU(int dims, int ArraySizeX, int ArraySizeY, int ArraySizeZ, int GPU_Device) : MemoryStruct<T>(dims, ArraySizeX, ArraySizeY, ArraySizeZ)
    {
        // Which GPU to use for allocating the array
        this->GPU_Device = GPU_Device;

        // Set the GPU device to the device which contains the CUDA array
        cudaSetDevice(this->GPU_Device);

        // Create the stream on the selected GPU
        cudaStreamCreate(&this->stream);

        // Set the pointer to NULL
        this->ptr = NULL;
    }

    /// Deconstructor to free any allocated GPU memory
    ~MemoryStructGPU()
    {
        if (this->Allocated == true)
        {
            DeallocateGPUArray();
        }
    }

    /// Copy an array array from the CPU to the previously allocated array on the GPU
    void CopyToGPU(T *Array, int Bytes)
    {
        if (Bytes != this->bytes())
        {
            std::cerr << "Error in CopyToGPU(): supplied array has " << Bytes << " bytes while the allocated GPU array has " << this->bytes() << " bytes." << '\n';
        }

        // Given a T type pointer (on host CPU) and number of bytes, copy the memory to this GPU array
        cudaMemcpy(this->ptr, Array, Bytes, cudaMemcpyHostToDevice);
    }

    /// Copy an array from the CPU to the allocated array on the GPU asynchronously.
    void CopyToGPUAsyc(T *Array, int Bytes)
    {
        if (Bytes != this->bytes())
        {
            std::cerr << "Error in CopyToGPUAsyc(): supplied array has " << Bytes << " bytes while the allocated GPU array has " << this->bytes() << " bytes." << '\n';
        }

        // Given a T type pointer (on host CPU) and number of bytes, copy the memory to this GPU array
        cudaError_t status = cudaMemcpyAsync(this->ptr, Array, Bytes, cudaMemcpyHostToDevice, this->stream);

        if (status != 0)
        {
            std::cerr << "cudaMalloc: " << cudaGetErrorString(status) << '\n';
            this->ErrorFlag = -1; // Set the error flag to -1 to remember that this failed

            int *curr_device = new int[1];
            cudaGetDevice(curr_device);

            std::cerr << "Current device is " << curr_device[0] << " while GPU_Device is " << this->GPU_Device << '\n';
        }
    }

    /// Copy the array from the GPU to a previously allocated array on the host (i.e. CPU).
    void CopyFromGPU(T *Array, int Bytes)
    {
        if (Bytes != this->bytes())
        {
            std::cerr << "Error in CopyFromGPU(): supplied array has " << Bytes << " bytes while the allocated GPU array has " << this->bytes() << " bytes." << '\n';
        }

        // Given a T type pointer (on host CPU) and number of bytes, copy the memory to this GPU array
        cudaMemcpy(Array, this->ptr, Bytes, cudaMemcpyDeviceToHost);
    }

    /// Allocate the memory on the GPU
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
            cudaSetDevice(this->GPU_Device);

            // There is enough memory left on the current GPU
            cudaError_t status = cudaMalloc((void **)&this->ptr, this->bytes());

            if (status != 0)
            {
                std::cerr << "cudaMalloc: " << cudaGetErrorString(status) << '\n';
                this->ErrorFlag = -1; // Set the error flag to -1 to remember that this failed

                int *curr_device = new int[1];
                cudaGetDevice(curr_device);

                std::cerr << "Current device is " << curr_device[0] << " while GPU_Device is " << this->GPU_Device << '\n';
            }

            this->ErrorFlag = 0; // Set the error flag to 0 to remember that this was sucessful
            this->Allocated = true;
        }
    }

    /// Reset the GPU array back to all zeros
    void Reset()
    {
        cudaMemset(this->ptr, 0, this->bytes());
    }

    /// Free the GPU memory
    void DeallocateGPUArray()
    {
        cudaFree(this->ptr);
    }

protected:
    // Which GPU is the memory allocated on?
    int GPU_Device;

    // CUDA stream for asyc copying to / from the GPU
    cudaStream_t stream;
};
