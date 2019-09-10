#pragma once

#include <cstdlib>
#include <stdio.h>
#include <cstring>

// Include the CUDA Runtime
#include <cuda_runtime.h>
#include <cuda.h>

// Struct to contain the needed information for each allocated array (e.g. CASImgs, images, coordinate axes, etc.)
template<class T = float>
struct MemoryStruct
{
    // Deminsions of the array
    int dims;

    // Size of the array
    int *size;

    // Pointer to the memory allocated
    T *ptr;

    // Constructor for the structure
    MemoryStruct(int dims, int *ArraySize)
    {
        this->dims = dims;
        this->size = new int[dims];

        // Set size to values in ArraySize
        for (int i = 0; i < dims; i++)
        {
            this->size[i] = ArraySize[i];
        }
        
        // Allocate the memory for the T type array
        AllocateArray();
    }

    // Deconstructor to free the memory
    ~MemoryStruct()
    {
        DeallocateArray();
    }

    // Allocate the memory for the T type array
    void AllocateArray()
    {
        this->ptr = new T[this->length()];
    }

    // Function to return the number of bytes the array is
    unsigned int bytes()
    {
        // Return the number of bytes
        int bytes = 1;
        for (int i = 0; i < this->dims; i++)
        {
            bytes = bytes * size[i];
        }

        bytes = bytes * sizeof(T);

        return bytes;
    };

    // Function to return the array length
    int length()
    {
        // Return the length of the array
        int len = 1;
        for (int i = 0; i < this->dims; i++)
        {
            len = len * size[i];
        }

        return len;
    }

    // Function to return the given array dimension
    int GetSize(int dim)
    {
        if (dim > this->dims)
        {
            std::cerr << "Error in size(): requested dim " << dim << " but the array has dimsions of size " << this->dims << '\n';
            return 0;
        }

        return size[dim];
    }

    // Function to return the given array dimension as int array (i.e. no input given)
    int* GetSize()
    {
        return this->size;
    }

    // Function to return the dimesion of the array
    int GetDim()
    {
        return this->dims;
    }

    // Copy a given array
    void CopyArray(T *Array)
    {
        std::memcpy(this->ptr, Array, this->bytes());
    }

    // Copy a given pointer
    void CopyPointer(T *&ptr)
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

    // Free the memory
    void DeallocateArray()
    {
        std::free(this->ptr);
    }

    // Get the pointer
    T *GetPointer()
    {
        return this->ptr;
    }

    // Get the pointer using some offset from the beginning of the array
    T *GetPointer(int offset)
    {
        return &this->ptr[offset];
    } 

    void Reset()
    {
        // Reset the array back to all zeros
        std::memset(this->ptr, 0, this->bytes());
    }
};