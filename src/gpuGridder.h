#pragma once

#include "AbstractGridder.h"
#include "gpuFFT.h"
#include "gpuForwardProject.h"
#include "gpuBackProject.h"
#include "MemoryStruct.h"
#include "MemoryStructGPU.h"

#include "CropVolumeFilter.h"
#include "CASToComplexFilter.h"
#include "FFTShift2DFilter.h"
#include "FFTShift3DFilter.h"
#include "PadVolumeFilter.h"
#include "ComplexToCASFilter.h"
#include "DivideVolumeFilter.h"
#include "RealToComplexFilter.h"
#include "ComplexToRealFilter.h"
#include "MultiplyVolumeFilter.h"
#include "DivideScalarFilter.h"

#include <cstdlib>
#include <stdio.h>
#include <vector>
#include <iostream>
// #include <math.h>

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
    gpuGridder(int VolumeSize, int numCoordAxes, float interpFactor, int RunFFTOnDevice, int NormalizeByDensity, int GPU_Device) : AbstractGridder(VolumeSize, numCoordAxes, interpFactor)
    {
        // Set default values
        this->VolumeToCASVolumeFlag = false;
        this->GPUArraysAllocatedFlag = false;
        this->nStreams = 1;
        this->MaxAxesToAllocate = 0;
        this->VolumeSize = VolumeSize;

        this->inverseFFTImagesFlag = false;
        this->forwardFFTVolumePlannedFlag = false;
        this->forwardFFTImagesFlag = false;

        this->RunFFTOnDevice = RunFFTOnDevice;
        this->NormalizeByDensity = NormalizeByDensity;

        // Set the GPU device
        SetGPU(GPU_Device);

        if (this->ErrorFlag == 1)
        {
            std::cerr << "Failed to initilize gpuGridder. Please check the GPU device number." << '\n';
        }

        // Create a new gpuFFT object for running the forward and inverse FFT
        this->gpuFFT_obj = new gpuFFT();
    };

    // Deconstructor
    ~gpuGridder() { FreeMemory(); };

    // Convert the volume to a CAS volume
    void VolumeToCASVolume();

    // Copy the CAS volume to the GPU asynchronously
    void CopyCASVolumeToGPUAsyc();

    // Allocate needed GPU memory
    void Allocate();

    // Run forward projection on some subset of the coordinate axes
    void ForwardProject(int AxesOffset, int nAxesToProcess);

    // Run the back projection and return the volume
    void BackProject(int AxesOffset, int nAxesToProcess);

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
    float *GetCASVolumePtr();
    float *GetPlaneDensityPtr();
    void CopyCASVolumeToHost();
    void CopyVolumeToHost();
    void CopyPlaneDensityToHost();

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

    // Convert projection images to CAS images by running a forward FFT
    void ImgsToCASImgs(cudaStream_t &stream, float *CASImgs, float *Imgs, cufftComplex *CASImgsComplex, int numImgs);

    // Reconstruct the volume by converting the CASVolume to Volume
    void ReconstructVolume();

private:
    // Initilize pointers for allocating memory on the GPU
    MemoryStructGPU<cufftComplex> *d_CASImgsComplex;      // For forward / inverse FFT
    MemoryStructGPU<cufftComplex> *d_PaddedVolumeComplex; // For converting volume to CAS volume
    MemoryStructGPU<float> *d_CASVolume;
    MemoryStructGPU<float> *d_PaddedVolume;
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

    int VolumeSize;
    // CUDA streams to use for forward / back projection
    cudaStream_t *streams;

    // Array allocation flag
    bool GPUArraysAllocatedFlag;

    // Flag to run / not run the volume to CAS volume
    bool VolumeToCASVolumeFlag;

    // Flag to run the forward and inverse FFT on the GPU
    int RunFFTOnDevice;

    // Flag to normalize the back projected volume by the plane density
	int NormalizeByDensity;

    // Initilization functions
    void InitilizeGPUArrays();
    void InitilizeCUDAStreams();

    // Estimate the maximum number of coordinate axes to allocate on the GPUs
    int EstimateMaxAxesToAllocate(int VolumeSize, int interpFactor);

    // Print to the console how much GPU memory is remaining
    void PrintMemoryAvailable();

    // Free all of the allocated memory
    void FreeMemory();

    // gpuFFT object for running forward and inverse FFT
    gpuFFT *gpuFFT_obj;

    // For converting the volume to CAS volume
    bool forwardFFTVolumePlannedFlag;
    cufftHandle forwardFFTVolume;

    // For converting images to CAS images
    bool forwardFFTImagesFlag;
    cufftHandle forwardFFTImages;

    // For converting CAS images to images
    bool inverseFFTImagesFlag;
    cufftHandle inverseFFTImages;

protected:
    // Plan the pointer offset values for running the CUDA kernels
    Offsets PlanOffsetValues(int coordAxesOffset, int nAxes);

    // Convert CAS images to images using an inverse FFT
    void CASImgsToImgs(cudaStream_t &stream, float *CASImgs, float *Imgs, cufftComplex *CASImgsComplex, int numImgs);
};
