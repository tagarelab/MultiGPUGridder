#ifndef GPU_GRIDDER
#define GPU_GRIDDER

#include "AbstractGridder.h"
#include "gpuFFT.h"
#include "gpuForwardProject.h"
#include "MemoryStruct.h"
#include "MemoryStructGPU.h"

#include <cstdlib>
#include <stdio.h>
#include <vector>

// Include the CUDA Runtime
#include <cuda_runtime.h>
#include <cuda.h>

// NVTX labeling tools (for the nvidia profiling)
#include <nvToolsExt.h>
#include <cuda_profiler_api.h>
// #include <cufft.h> // Library for CUDA FFT and inverse FFT functions see https://www.bu.edu/pasi/files/2011/07/Lecture83.pdf

class gpuGridder : public AbstractGridder
{

public:
    // Constructor
    gpuGridder(int VolumeSize, int numCoordAxes, float interpFactor) : AbstractGridder(VolumeSize, numCoordAxes, interpFactor)
    {
        this->FP_initilized = false;
        this->newVolumeFlag = true;
        this->nStreams = 1;

        // Estimate the maximum number of coordinate axes to allocate

        // Set the GPU device
        // SetGPU(GPU_Device);

        if (this->ErrorFlag == 1)
        {
            std::cerr << "Failed to initilize gpuGridder. Please check the GPU device number." << '\n';
        }


        this->MaxAxesToAllocate = EstimateMaxAxesToAllocate(VolumeSize, interpFactor);

    };

    // ~gpuGridder() : ~AbstractGridder() { };

    // ~gpuGridder(){};

    // Run the forward projection and return the projection images
    void ForwardProject();

    // // Run the back projection and return the volume
    // float *BackProject();

    // Set the volume
    // void SetVolume(float *Volume, int *VolumeSize);

    // // Reset the volume to all zeros
    // void ResetVolume();

    // Set which GPUs to use
    void SetGPU(int GPU_Device);

    // // Set the GPU for this object to use for processiong
    // void SetGPU(int GPU){this->GPU = GPU;};

    // Get the GPU device number to use for processing
    int GetGPUDevice() { return this->GPU_Device; }

    // Get the number of CUDA streams
    int GetNumStreams() { return this->nStreams; }

    // Set the number of CUDA streams
    int SetNumStreams(int nStreams) { this->nStreams = nStreams; }

    // Get the pointer to the CUDA streams
    cudaStream_t *GetStreamsPtr() { return this->streams; }

    // Get the CUDA grid size
    int GetGridSize() { return this->gridSize; }

    // Get the CUDA block size
    int GetBlockSize() { return this->blockSize; }

    // Get the device CAS volume pointer
    float *GetCASVolumePtr_Device() { return this->d_CASVolume->ptr; }

    // Get the device CAS images pointer
    float *GetCASImgsPtr_Device() { return this->d_CASImgs->ptr; }

    // Get the device complex CAS images pointer
    //cufftComplex *GetComplexCASImgsPtr_Device() { return this->d_CASImgsComplex; }

    // Get the device images pointer
    float *GetImgsPtr_Device() { return this->d_Imgs->ptr; }

    // Get the device coordinate axes pointer
    float *GetCoordAxesPtr_Device() { return this->d_CoordAxes->ptr; }

    // Get the device kaiser bessel lookup table pointer
    float *GetKBTablePtr_Device() { return this->d_KB_Table->ptr; }

protected:
    // // Get a new images array and then convert them to CAS
    // void SetImages(float *imgs);

    // Convert the volume to a CAS volume
    void VolumeToCASVolume();

    // // Convert the CAS volume back to volume
    // void CASToVolume();

    // How many streams to use on this device?
    int nStreams;

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

    // Kernel launching parameters
    int gridSize;
    int blockSize;

    // Several CUDA streams
    cudaStream_t *streams;

private:
    // Which GPU to use for processing?
    int GPU_Device;

    // Initilize the GPU arrays
    void InitilizeGPUArrays();

    // Initilize the forward projection object
    void InitilizeForwardProjection();

    // Forward projection object
    gpuForwardProject *ForwardProject_obj;

    // Forward projection initilization flag
    bool FP_initilized;

    // Forward projection new volume flag
    bool newVolumeFlag;

    // Estimate the maximum number of coordinate axes to allocate on the GPUs
    int EstimateMaxAxesToAllocate(int VolumeSize, int interpFactor);

    // Allocate GPU arrays
    // float *AllocateGPUArray(int GPU_Device, int ArraySize);

    // // Free all of the allocated memory
    // void FreeMemory();
};

#endif