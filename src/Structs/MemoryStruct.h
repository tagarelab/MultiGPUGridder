#pragma once

/**
 * @class   MemoryStruct
 * @brief   A class for allocating host (i.e CPU) memory
 *
 *
 * A class for allocating and deallocating host memory. MemoryStruct also remembers needed information for each allocated array
 * (e.g. CASImgs, images, coordinate axes, etc.) such as the array size, memory pointers, etc. This is the main CPU memory class.
 * 
 * */

#include <cstdlib>
#include <stdio.h>
#include <cstring>
#include <iostream>

// Include the CUDA Runtime
#include <cuda_runtime.h>
#include <cuda.h>

template <class T = float>
class MemoryStruct
{
public:
    /// Constructor for the class. dims is the dimensions of the array to allocate while ArraySize is an int vector of size dims containing the
    /// array size along each dimension.
    MemoryStruct(int dims, int *ArraySize)
    {
        this->dims = dims;
        this->size = new int[dims];
        this->ErrorFlag = 0;
        this->Allocated = false;
        this->Initialized = false;

        // Set size to values in ArraySize
        for (int i = 0; i < dims; i++)
        {
            this->size[i] = ArraySize[i];
        }
    }
    /// Additional constructor for the structure: Array of 1 dimension
    MemoryStruct(int dims, int ArraySizeX)
    {
        this->dims = dims;
        this->size = new int[dims];
        this->ErrorFlag = 0;
        this->Allocated = false;
        this->Initialized = false;

        // Set size to values in ArraySize
        this->size[0] = ArraySizeX;
    }

    /// Additional constructor for the class: Array of 2 dimensions
    MemoryStruct(int dims, int ArraySizeX, int ArraySizeY)
    {
        this->dims = dims;
        this->size = new int[dims];
        this->ErrorFlag = 0;
        this->Allocated = false;
        this->Initialized = false;

        // Set size to values in ArraySize
        this->size[0] = ArraySizeX;
        this->size[1] = ArraySizeY;
    }

    /// Additional constructor for the class: Array of 3 dimensions
    MemoryStruct(int dims, int ArraySizeX, int ArraySizeY, int ArraySizeZ)
    {
        this->dims = dims;
        this->size = new int[dims];
        this->ErrorFlag = 0;
        this->Allocated = false;
        this->Initialized = false;

        // Set size to values in ArraySize
        this->size[0] = ArraySizeX;
        this->size[1] = ArraySizeY;
        this->size[2] = ArraySizeZ;
    }

    /// Deconstructor to free the memory
    ~MemoryStruct()
    {
        DeallocateArray();
    }

    /// Allocate the memory for the host array
    void AllocateArray()
    {
        this->ptr = new T[this->length()];
        this->Allocated = true;
        this->Initialized = true;
    }

    /// Function to return the number of bytes the array is
    long long int bytes()
    {
        // Return the number of bytes
        long long bytes = 1;
        for (int i = 0; i < this->dims; i++)
        {
            bytes = bytes * size[i];
        }

        bytes = bytes * sizeof(T);

        if (bytes < 0)
        {
            std::cerr << "bytes(): Array size is too long. Bytes is negative." << '\n';
            this->ErrorFlag = 1;
        }

        return bytes;
    };

    /// Function to return the array length (i.e. the number of elements of the array).
    long long int length()
    {
        // Return the length of the array
        long long int len = 1;
        for (int i = 0; i < this->dims; i++)
        {
            len = len * size[i];
        }

        return len;
    }

    /// Function to return the length of a given dimension of the array.
    int GetSize(int dim)
    {
        if (dim > this->dims)
        {
            std::cerr << "Error in size(): requested dim " << dim << " but the array has dimsions of size " << this->dims << '\n';
            return 0;
        }

        return size[dim];
    }

    /// Function to return a vector containing the length along all the array dimensions.
    int *GetSize()
    {
        return this->size;
    }

    /// Function to return the dimesion of the array.
    int GetDim()
    {
        return this->dims;
    }

    /// Copy the array to a second array.
    void CopyArray(T *Array)
    {
        if (this->IsAllocated() == true || this->IsInitialized() == true)
        {
            std::memcpy(this->ptr, Array, this->bytes());
        }
        else
        {
            std::cerr << "CopyArray: Array must be Initialized or allocated before copying to it." << '\n';
        }
    }

    /// Copy a given pointer to the class
    void CopyPointer(T *&ptr)
    {
        // Free the currently allocated memory
        if (this->IsAllocated() == true)
        {
            std::free(this->ptr);
        }

        // Copy the pointer to the structure
        this->ptr = ptr;

        this->Initialized = true;
    }

    /// Pin the memory to the CPU in order to enable asynchronous memory transfers to and from the GPU
    void PinArray()
    {
        if (this->IsAllocated() == true || this->IsInitialized() == true)
        {
            cudaHostRegister(this->ptr, this->bytes(), 0);
        }
        else
        {
            std::cerr << "PinArray: Array must be Initialized or allocated before pinning to memory." << '\n';
        }
    }

    /// Free the memory if it was allocated by this class
    void DeallocateArray()
    {
        // Free the currently allocated memory
        if (this->IsAllocated() == true)
        {
            std::free(this->ptr);
        }
    }

    /// Get the pointer to the array
    T *GetPointer()
    {
        return this->ptr;
    }

    /// Get the pointer using some offset from the beginning of the array
    T *GetPointer(int offset)
    {
        return &this->ptr[offset];
    }

    /// Reset the array back to all zeros
    void Reset()
    {
        if (this->IsAllocated() == true || this->IsInitialized() == true)
        {
            std::memset(this->ptr, 0, this->bytes());
        }
        else
        {
            std::cerr << "Reset: Array must be Initialized or allocated before reseting to zeros." << '\n';
        }
    }

    /// Get the status of the error flag
    bool GetErrorFlag()
    {
        return this->ErrorFlag;
    }

    /// Flag to determine whether the array was allocated by this class (true) or by another class (false).
    bool IsAllocated()
    {
        return this->Allocated;
    }

    /// Flag to determine whether the array was initilized already. AllocateArray() and CopyPointer() will both set this flag to true.
    bool IsInitialized()
    {
        return this->Initialized;
    }

protected:
    // Deminsions of the array
    int dims;

    // Size of the array
    int *size;

    // Pointer to the memory allocated
    T *ptr;

    // Error flag to rememeber if the allocation was succesful or not
    int ErrorFlag;

    // Flag to see if the array has been allocated by this object or not
    // If the array was allocated elsewhere and only the pointer was copied
    // Then allocated should be set to false (to prevent attempting to free the array)
    bool Allocated;

    // This flag refers to when simplying copying a pointer (instead of this class allocating the memory)
    // Represents whether this class should free the associated memory or not
    bool Initialized;
};