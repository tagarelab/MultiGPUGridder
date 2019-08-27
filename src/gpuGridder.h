#ifndef GPU_GRIDDER
#define GPU_GRIDDER

#include "AbstractGridder.h"
#include "gpuFFT.h"
#include "gpuForwardProject.h"

#include <cstdlib>
#include <stdio.h>
#include <vector>

// Include the CUDA Runtime
#include <cuda_runtime.h>
#include <cuda.h>

// NVTX labeling tools (for the nvidia profiling)
// #include <nvToolsExt.h>
// #include <cuda_profiler_api.h>
// #include <cufft.h> // Library for CUDA FFT and inverse FFT functions see https://www.bu.edu/pasi/files/2011/07/Lecture83.pdf

class gpuGridder : public AbstractGridder
{

public:
    // Constructor
    gpuGridder(int VolumeSize, int numCoordAxes, float interpFactor) : AbstractGridder(VolumeSize, numCoordAxes, interpFactor){};

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

    // Set the volume
    void SetVolume(float *Volume);

    // Return the volume
    float *GetVolume();

    // Set which GPUs to use
    void SetGPU(int GPU_Device);

    // // Set the GPU for this object to use for processiong
    // void SetGPU(int GPU){this->GPU = GPU;};

    // Get the GPU device number to use for processing
    int GetGPUDevice() { return this->GPU_Device; }

    // Get the number of CUDA streams
    int GetNumStreams() { return this->nStreams; }

    // Get the pointer to the CUDA streams
    cudaStream_t *GetStreamsPtr() { return this->streams; }

    // Get the CUDA grid size
    int GetGridSize() { return this->gridSize; }

    // Get the CUDA block size
    int GetBlockSize() { return this->blockSize; }

    // Get the device CAS volume pointer
    float *GetCASVolumePtr() { return this->d_CASVolume; }

    // Get the device CAS images pointer
    float *GetCASImgsPtr() { return this->d_CASImgs; }

    // Get the device coordinate axes pointer
    float *GetCoordAxesPtr() { return this->d_CoordAxes; }

    // Get the device kaiser bessel lookup table pointer
    float *GetKBTablePtr() { return this->d_KB_Table; }

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
    float *d_CASVolume;

    // Pointer to the CAS images array on the device (i.e. the GPU)
    float *d_CASImgs;

    // Pointer to the coordinate axes vector on the device (i.e. the GPU)
    float *d_CoordAxes;

    // Pointer to the Kaiser bessel vector on the device (i.e. the GPU)
    float *d_KB_Table;

    // Kernel launching parameters
    int gridSize;
    int blockSize;

    // Several CUDA streams
    cudaStream_t *streams;

private:
    // Which GPU to use for processing?
    int GPU_Device;

    // Create the CUDA streams
    void CreateCUDAStreams();

    // Delete the CUDA streams
    void DestroyCUDAStreams();

    // Initilize the GPU arrays
    void InitilizeGPUArrays();

    // Copy the volume to the GPU
    void CopyVolumeToGPU();

    // Allocate GPU arrays
    void AllocateGPUArray(int GPU_Device, float *d_Ptr, int ArraySize);


    // // Free all of the allocated memory
    // void FreeMemory();
};

#endif