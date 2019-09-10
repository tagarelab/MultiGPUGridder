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

// NVTX labeling tools (for the nvidia profiling)
//#include <nvToolsExt.h>
//#include <cuda_profiler_api.h>

// #ifndef __GPUFORWARDPROJECT_H__
// #define __GPUFORWARDPROJECT_H__

class gpuForwardProject
{
private:

    // Create vectors to hold all of the pointer offset values when running the CUDA kernels
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

    // Pointer to the CASVolume array on the device (i.e. the GPU)
    MemoryStructGPU<float> *d_CASVolume;

    // Pointer to the CAS images array on the device (i.e. the GPU)
    MemoryStructGPU<float> *d_CASImgs;

    // Pointer to the images array on the device (i.e. the GPU)
    MemoryStructGPU<float> *d_Imgs;

    // Pointer to the complex CAS images array on the device (i.e. the GPU)
    MemoryStructGPU<cufftComplex> *d_CASImgsComplex;

    // Pointer to the coordinate axes vector on the device (i.e. the GPU)
    MemoryStructGPU<float> *d_CoordAxes;

    // Pointer to the Kaiser bessel vector on the device (i.e. the GPU)
    MemoryStructGPU<float> *d_KB_Table;

    // Pointer to the coordinate axes array which is pinned to the host (i.e. the CPU)
    float *coordAxes_CPU_Pinned;

    // Pointer to the CASImgs array which is pinned to the host (i.e. the CPU)
    float *CASImgs_CPU_Pinned;

    // Pointer to the images array which is pinned to the host (i.e. the CPU)
    float *Imgs_CPU_Pinned;

    // Pointer to an array of CUDA streams
    cudaStream_t * streams;

    // Grid size for the CUDA kernel
    int gridSize;

    // Block size for the CUDA kernel
    int blockSize;

    // Number of coordinate axes to process
    int nAxes;

    // Maximum number of coordinate axes which were allocated
    int MaxAxesAllocated;

    // Number of CUDA streams to use
    int nStreams;

    // Which GPU device to use
    int GPU_Device;

    // Mask radius for the forward projection
    float maskRadius;

    // Width of the kaiser bessel lookup table
    float kerHWidth;

    // Offset for processing a subset of all the coordinate axes
    // This is in number of axes from the beginning of the pinned CPU coordinate axes array
    int coordAxesOffset;

    // Plan the pointer offset values for running the CUDA kernels
    Offsets PlanOffsetValues();

    // gpuFFT object for running forward and inverse FFT
    gpuFFT * gpuFFT_obj;

public:

    gpuForwardProject()
    {
        // Create a new gpuFFT object for running the forward and inverse FFT
        this->gpuFFT_obj = new gpuFFT();
    }

    // Deconstructor
    ~gpuForwardProject()
    {
        delete this->gpuFFT_obj;
    }


    // Setter functions
    void SetCASVolume(MemoryStructGPU<float> *&CASVolume) { this->d_CASVolume = CASVolume; }
    void SetCASImages(MemoryStructGPU<float> *&CASImgs) { this->d_CASImgs = CASImgs; }
    void SetComplexCASImages(MemoryStructGPU<cufftComplex> *&CASImgsComplex) { this->d_CASImgsComplex = CASImgsComplex; }
    void SetImages(MemoryStructGPU<float> *&Imgs) { this->d_Imgs = Imgs; }
    void SetCoordinateAxes(MemoryStructGPU<float> *&CoordAxes) { this->d_CoordAxes = CoordAxes; }
    void SetKBTable(MemoryStructGPU<float> *&KB_Table) { this->d_KB_Table = KB_Table; }
    
    void SetPinnedCoordinateAxes(float *&coordAxes_CPU_Pinned) { this->coordAxes_CPU_Pinned = coordAxes_CPU_Pinned; }
    void SetPinnedCASImages(float *&CASImgs_CPU_Pinned) { this->CASImgs_CPU_Pinned = CASImgs_CPU_Pinned; }
    void SetPinnedImages(float *&Imgs_CPU_Pinned) { this->Imgs_CPU_Pinned = Imgs_CPU_Pinned; }
    void SetCUDAStreams(cudaStream_t * streams) { this->streams = streams; }
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



// gpuForwardProject::~gpuForwardProject()
// {
// }

// Function for creating the CUDA streams and launchinh the forward projection kernel
// void gpuForwardProjectLaunch(gpuGridder *gridder);

// #endif //__GPUFORWARDPROJECT_H__
