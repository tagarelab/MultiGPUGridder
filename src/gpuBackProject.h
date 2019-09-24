#pragma once

#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <vector>

#include "gpuFFT.h"
#include "MemoryStruct.h"
#include "MemoryStructGPU.h"

// includes CUDA Runtime
#include <cuda_runtime.h>
#include <cuda.h>
#include <cufft.h> // Library for CUDA FFT and inverse FFT functions see https://www.bu.edu/pasi/files/2011/07/Lecture83.pdf

#define Log2(x, y)                          \
    {                                       \
        std::cout << x << " " << y << '\n'; \
    }

class gpuBackProject
{
private:
    // A structure for holding all of the pointer offset values when running the forward projection kernel
    struct Offsets
    {
        std::vector<int> numAxesPerStream;
        std::vector<int> CoordAxes_CPU_Offset;
        std::vector<int> coord_Axes_CPU_streamBytes;
        std::vector<int> gpuImgs_Offset;
        std::vector<int> gpuCASImgs_streamBytes;
        std::vector<int> gpuCASImgs_Offset;
        std::vector<int> gpuCoordAxes_Stream_Offset;
        std::vector<unsigned long long> CASImgs_CPU_Offset;
        std::vector<unsigned long long> Imgs_CPU_Offset;
        std::vector<int> gpuImgs_streamBytes;
        std::vector<int> stream_ID;
        int num_offsets;
    };

    // Pointers to the previously allocated GPU arrays
    MemoryStructGPU<float> *d_Imgs;
    MemoryStructGPU<float> *d_CASImgs;
    MemoryStructGPU<float> *d_KB_Table;
    MemoryStructGPU<float> *d_CoordAxes;
    MemoryStructGPU<float> *d_CASVolume;
    MemoryStructGPU<float> *d_PlaneDensity;
    MemoryStructGPU<cufftComplex> *d_CASImgsComplex;

    // Pointer to the previously allocated pinned CPU arrays
    MemoryStruct<float> *Imgs_CPU_Pinned;
    MemoryStruct<float> *CASImgs_CPU_Pinned;
    MemoryStruct<float> *coordAxes_CPU_Pinned;

    // Pointer to an array of CUDA streams
    cudaStream_t *streams;

    // Grid size for the CUDA kernel
    int gridSize;

    // Block size for the CUDA kernel
    int blockSize;

    int interpFactor;

    // Number of coordinate axes to process
    int nAxes;

    // Maximum number of coordinate axes which were allocated
    int MaxAxesAllocated;

    // Number of CUDA streams to use
    int nStreams;

    // Which GPU device to use
    int GPU_Device;

    // Offset for processing a subset of all the coordinate axes
    // This is in number of axes from the beginning of the pinned CPU coordinate axes array
    int coordAxesOffset;

    // Mask radius for the forward projection
    float maskRadius;

    // Width of the kaiser bessel lookup table
    float kerHWidth;

    // Plan the pointer offset values for running the CUDA kernels
    Offsets PlanOffsetValues();

    // gpuFFT object for running forward and inverse FFT
    gpuFFT *gpuFFT_obj;

public:
    // Constructor
    gpuBackProject()
    {
        // Create a new gpuFFT object for running the forward and inverse FFT
        this->gpuFFT_obj = new gpuFFT();

        // Default value for the optional CAS images array
        this->CASImgs_CPU_Pinned = NULL;

        interpFactor = 2;
    }

    // Deconstructor
    ~gpuBackProject()
    {
        delete this->gpuFFT_obj;
    }

    // Setter functions
    void SetCASVolume(MemoryStructGPU<float> *&CASVolume) { this->d_CASVolume = CASVolume; }
    void SetPlaneDensity(MemoryStructGPU<float> *&PlaneDensity) { this->d_PlaneDensity = PlaneDensity; }
    void SetCASImages(MemoryStructGPU<float> *&CASImgs) { this->d_CASImgs = CASImgs; }
    void SetComplexCASImages(MemoryStructGPU<cufftComplex> *&CASImgsComplex) { this->d_CASImgsComplex = CASImgsComplex; }
    void SetImages(MemoryStructGPU<float> *&Imgs) { this->d_Imgs = Imgs; }
    void SetCoordinateAxes(MemoryStructGPU<float> *&CoordAxes) { this->d_CoordAxes = CoordAxes; }
    void SetKBTable(MemoryStructGPU<float> *&KB_Table) { this->d_KB_Table = KB_Table; }
    void SetPinnedCoordinateAxes(MemoryStruct<float> *coordAxes_CPU_Pinned) { this->coordAxes_CPU_Pinned = coordAxes_CPU_Pinned; }
    void SetPinnedCASImages(MemoryStruct<float> *CASImgs_CPU_Pinned) { this->CASImgs_CPU_Pinned = CASImgs_CPU_Pinned; }
    void SetPinnedImages(MemoryStruct<float> *Imgs_CPU_Pinned) { this->Imgs_CPU_Pinned = Imgs_CPU_Pinned; }
    void SetCUDAStreams(cudaStream_t *streams) { this->streams = streams; }
    void SetCoordinateAxesOffset(int coordAxesOffset) { this->coordAxesOffset = coordAxesOffset; }
    void SetGridSize(int gridSize) { this->gridSize = gridSize; }
    void SetBlockSize(int blockSize) { this->blockSize = blockSize; }
    void SetNumberOfAxes(int nAxes) { this->nAxes = nAxes; }
    void SetMaxAxesAllocated(int MaxAxesAllocated) { this->MaxAxesAllocated = MaxAxesAllocated; }
    void SetNumberOfStreams(int nStreams) { this->nStreams = nStreams; }
    void SetGPUDevice(int GPU_Device) { this->GPU_Device = GPU_Device; }
    void SetMaskRadius(int maskRadius) { this->maskRadius = maskRadius; }
    void SetKerHWidth(float kerHWidth) { this->kerHWidth = kerHWidth; }

    // Run the forward projection
    void Execute();
};
