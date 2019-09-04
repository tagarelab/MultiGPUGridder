
#pragma once

#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include <vector>

// #include "gpuGridder.h"
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
    // Pointer to the CASVolume array on the device (i.e. the GPU)
    MemoryStructGPU *d_CASVolume;

    // Pointer to the CAS images array on the device (i.e. the GPU)
    MemoryStructGPU *d_CASImgs;

    // Pointer to the images array on the device (i.e. the GPU)
    MemoryStructGPU *d_Imgs;

    // Pointer to the coordinate axes vector on the device (i.e. the GPU)
    MemoryStructGPU *d_CoordAxes;

    // Pointer to the Kaiser bessel vector on the device (i.e. the GPU)
    MemoryStructGPU *d_KB_Table;

    // Pointer to the coordinate axes array which is pinned to the host (i.e. the CPU)
    float *coordAxes_CPU_Pinned;

    // Pointer to the CASImgs array which is pinned to the host (i.e. the CPU)
    float *CASImgs_CPU_Pinned;

    // Pointer to the images array which is pinned to the host (i.e. the CPU)
    float *Imgs_CPU_Pinned;

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

public:
    void SetPinnedCoordinateAxes(float *&coordAxes_CPU_Pinned) { this->coordAxes_CPU_Pinned = coordAxes_CPU_Pinned; }

    void SetPinnedCASImages(float *&CASImgs_CPU_Pinned) { this->CASImgs_CPU_Pinned = CASImgs_CPU_Pinned; }

    void SetPinnedImages(float *&Imgs_CPU_Pinned) { this->Imgs_CPU_Pinned = Imgs_CPU_Pinned; }

    void SetCASVolume(MemoryStructGPU *&CASVolume) { this->d_CASVolume = CASVolume; }

    void SetCASImages(MemoryStructGPU *&CASImgs) { this->d_CASImgs = CASImgs; }

    void SetImages(MemoryStructGPU *&Imgs) { this->d_Imgs = Imgs; }

    void SetCoordinateAxes(MemoryStructGPU *&CoordAxes) { this->d_CoordAxes = CoordAxes; }

    void SetKBTable(MemoryStructGPU *&KB_Table) { this->d_KB_Table = KB_Table; }

    void SetGridSize(int gridSize) { this->gridSize = gridSize; }

    void SetBlockSize(int blockSize) { this->blockSize = blockSize; }

    void SetNumberOfAxes(int nAxes) { this->nAxes = nAxes; }

    void SetMaxAxesAllocated(int MaxAxesAllocated) { this->MaxAxesAllocated = MaxAxesAllocated; }

    void SetNumberOfStreams(int nStreams) { this->nStreams = nStreams; }

    void SetGPUDevice(int GPU_Device) { this->GPU_Device = GPU_Device; }

    void SetMaskRadius(int maskRadius) { this->maskRadius = maskRadius; }

    void Execute();
};

// gpuForwardProject::gpuForwardProject(/* args */)
// {
// }

// gpuForwardProject::~gpuForwardProject()
// {
// }

// Function for creating the CUDA streams and launchinh the forward projection kernel
// void gpuForwardProjectLaunch(gpuGridder *gridder);

// #endif //__GPUFORWARDPROJECT_H__
