#pragma once

#include "AbstractGridder.h"
#include "gpuFFT.h"
#include "gpuForwardProject.h"
#include "gpuBackProject.h"
#include "MemoryStruct.h"
#include "MemoryStructGPU.h"

#include <cstdlib>
#include <stdio.h>
#include <vector>

// Include the CUDA Runtime
#include <cuda_runtime.h>
#include <cuda.h>

// Are we compiling on a windows or linux machine?
#if defined(_MSC_VER) //  Microsoft
#include <algorithm>
#elif defined(__GNUC__) // Linux

#else

#endif



// NVTX labeling tools (for the nvidia profiling)
// #include <nvToolsExt.h>
// #include <cuda_profiler_api.h>

class gpuGridder : public AbstractGridder
{

public:
    // Constructor
    gpuGridder(int VolumeSize, int numCoordAxes, float interpFactor, int GPU_Device) : AbstractGridder(VolumeSize, numCoordAxes, interpFactor)
    {
        // Set default values
        this->VolumeToCASVolumeFlag = false;
        this->GPUArraysAllocatedFlag = false;
        this->nStreams = 1;
        this->MaxAxesToAllocate = 0;

        // Set the GPU device
        SetGPU(GPU_Device);

        if (this->ErrorFlag == 1)
        {
            std::cerr << "Failed to initilize gpuGridder. Please check the GPU device number." << '\n';
        }
    };

    // Deconstructor
    ~gpuGridder() { FreeMemory(); };

    // Convert the volume to a CAS volume
    void VolumeToCASVolume();

    // Copy the CAS volume to the GPU asynchronously
    void CopyCASVolumeToGPUAsyc();

    // Allocate needed GPU memory
    void Allocate();

    // Run the forward projection kernel
    void ForwardProject();

    // Run forward projection on some subset of the coordinate axes
    void ForwardProject(int AxesOffset, int nAxesToProcess);

    // Run the back projection and return the volume
    void BackProject(int AxesOffset, int nAxesToProcess);
    void BackProject();

    // Setter functions
	void SetNumStreams(int nStreams) { this->nStreams = nStreams; }
    void SetGPU(int GPU_Device);

    // Getter functions
    cudaStream_t *GetStreamsPtr() { return this->streams; }
    float *GetVolumeFromDevice();
    float *GetCASVolumeFromDevice();
    float *GetPlaneDensityFromDevice();
    int GetGPUDevice() { return this->GPU_Device; }
    int GetNumStreams() { return this->nStreams; }
    int GetGridSize() { return this->gridSize; }
    int GetBlockSize() { return this->blockSize; }
    float *GetCASVolumePtr_Device() { return this->d_CASVolume->GetPointer(); }
    float *GetCASImgsPtr_Device() { return this->d_CASImgs->GetPointer(); }
    float *GetImgsPtr_Device() { return this->d_Imgs->GetPointer(); }
    float *GetCoordAxesPtr_Device() { return this->d_CoordAxes->GetPointer(); }
    float *GetKBTablePtr_Device() { return this->d_KB_Table->GetPointer(); }

private:
    // Initilize pointers for allocating memory on the GPU
    MemoryStructGPU<cufftComplex> *d_CASImgsComplex; // For forward / inverse FFT
    MemoryStructGPU<float> *d_CASVolume;
    MemoryStructGPU<float> *d_CASImgs;
    MemoryStructGPU<float> *d_Imgs;     // Output images
    MemoryStructGPU<float> *d_KB_Table; // Kaiser bessel lookup table
    MemoryStructGPU<float> *d_CoordAxes;
    MemoryStructGPU<float> *d_PlaneDensity;
    MemoryStructGPU<float> *d_Volume;

    // Kernel launching parameters
    int gridSize;
    int blockSize;
    int GPU_Device; // Which GPU to use?
    int nStreams;   // Streams to use on this GPU

    // CUDA streams to use for forward / back projection
    cudaStream_t *streams;

    // Forward projection object
    gpuForwardProject *ForwardProject_obj;

    // Back projection object
    gpuBackProject *BackProject_obj;

    // Array allocation flag
    bool GPUArraysAllocatedFlag;

    // Flag to run / not run the volume to CAS volume
    bool VolumeToCASVolumeFlag;

    // Initilization functions
    void InitilizeGPUArrays();
    void InitilizeCUDAStreams();
    void InitilizeForwardProjection(int AxesOffset, int nAxesToProcess);
    void InitilizeBackProjection(int AxesOffset, int nAxesToProcess);

    // Estimate the maximum number of coordinate axes to allocate on the GPUs
    int EstimateMaxAxesToAllocate(int VolumeSize, int interpFactor);

    // Print to the console how much GPU memory is remaining
    void PrintMemoryAvailable();

    // Free all of the allocated memory
    void FreeMemory();
};
