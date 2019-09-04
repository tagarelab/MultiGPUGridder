#pragma once

#include <cstdlib>
#include <stdio.h>
#include <cstring>

// Include the CUDA Runtime
#include <cuda_runtime.h>
#include <cuda.h>

// Struct to contain the needed information for each allocated array (e.g. CASImgs, images, coordinate axes, etc.)
struct MemoryStruct
{
    MemoryStruct(int dims, int *ArraySize)
    {
        // Constructor for the structure
        this->dims = dims;
        this->size = new int[dims];

        // Set size to values in ArraySize
        for (int i = 0; i < dims; i++)
        {
            this->size[i] = ArraySize[i];
        }

        this->ptr = new float[this->length()];
    }

    // Deminsions of the array
    int dims;

    // Size of the array
    int *size;

    // Pointer to the memory allocated
    float *ptr;

    // Function to return the number of bytes the array is
    int bytes()
    {
        // Return the number of bytes
        int bytes = 1;
        for (int i = 0; i < this->dims; i++)
        {
            bytes = bytes * size[i];
        }

        bytes = bytes * sizeof(float);

        return bytes;
    };

    // Function to return the array length
    int length()
    {
        std::cout << "size[0]: " << size[0] << '\n';
        // Return the length of the array
        int len = 1;
        for (int i = 0; i < this->dims; i++)
        {
            len = len * size[i];
            std::cout << "len: " << len << '\n';
        }

        return len;
    }

    // Copy a given array
    void CopyArray(float *Array)
    {
        std::memcpy(this->ptr, Array, this->bytes());
    }

    // Copy a given pointer
    void CopyPointer(float *&ptr)
    {   
        // Free the currently allocated memory
        std::free(this->ptr);

        // Copy the pointer to the structure
        this->ptr = ptr;
    }

    // Pin the memory to the CPU (in order to enable the async CUDA stream copying)
    void PinArray()
    {
        cudaHostRegister(this->ptr, this->bytes(), 0);
    }
};