#pragma once

/**
 * @class   gpuGridder
 * @brief   A class for gridding on the GPU
 *
 *
 * This class is used for forward and back projection of a volume on a single NVIDIA GPU. The gpuGridder 
 * inherits from the AbstractClass which is used for setting and getting the host (i.e. CPU) memory pointers to the 
 * volume, coordinate axes vector, etc, and for setting various parameters such as the interpolation factor.
 * 
 * The gpuGridder estimates the number of coordinate axes which the GPU has available memory for. The GPU memory has
 * type DeviceMemory which remembers various required information about the GPU array.
 * 
 * The ForwardProject and BackProject functions both take the same inputs: AxesOffset and nAxesToProcess which represent
 * the offset (in number of coordinate axes) of the host (i.e. CPU) coordinate axes array to start the forward or back projection
 * on. The nAxesToProcess represents how many coordinate axes this gridder should process. For example, if the host memory has
 * 1,000 coordinate axes and we wanted the gpuGridder to process axes 500 to 700, then simply set the AxesOffset to 500 and the 
 * nAxesToProcess to 200.
 * 
 * The gpuGridder also provides functions for executing the forward and inverse Fourier transforms on the GPU (see the CASVolumeToVolume and
 * ImgsToCASImgs functions).
 * */

#include "AbstractGridder.h"
#include "gpuForwardProject.h"
#include "gpuBackProject.h"
#include "HostMemory.h"
#include "DeviceMemory.h"

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

#include "gpuErrorCheck.h"

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
    // / gpuGridder Constructor. Set RunFFTOnDevice to 1 for true and 0 for false. GPU_Device is the NVIDIA GPU device number and starts at 0.
    gpuGridder(int VolumeSize, int numCoordAxes, float interpFactor, int RunFFTOnDevice, int GPU_Device, bool verbose = false) : AbstractGridder(VolumeSize, numCoordAxes, interpFactor)
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

        this->verbose = verbose;

        // Set the GPU device
        this->SetGPU(GPU_Device);

        if (this->ErrorFlag == 1)
        {
            std::cerr << "Failed to initilize gpuGridder. Please check the GPU device number." << '\n';
        }
    };

    // Deconstructor
    ~gpuGridder() { FreeMemory(); };

    /// Allocate required GPU memory
    void Allocate();

    /// Run forward projection on some subset of the coordinate axes. AxesOffset is the number of coordinate axes from the beginning of the
    /// host array (starting at 0) while the nAxesToProcess is the number of coordinate axes to process. For example, if the host memory has
    /// 1,000 coordinate axes, and we wanted the to process axes 500 to 700 then simply set the AxesOffset to 500 and the nAxesToProcess to 200.
    void ForwardProject(int AxesOffset, int nAxesToProcess);

    /// Run back projection on some subset of the coordinate axes. AxesOffset is the number of coordinate axes from the beginning of the
    /// host array (starting at 0) while the nAxesToProcess is the number of coordinate axes to process. For example, if the host memory has
    /// 1,000 coordinate axes, and we wanted the to process axes 500 to 700 then simply set the AxesOffset to 500 and the nAxesToProcess to 200.
    void BackProject(int AxesOffset, int nAxesToProcess);

    /// Set the number of CUDA streams to use for the forward and back projection kernels.
    void SetNumStreams(int nStreams) { this->nStreams = nStreams; }

    /// Set the GPU device number starting at 0. For example, if a computer has 4 GPUs the GPU_Device number could be 0, 1, 2, or 3 and determines
    /// which GPU to use for processing.
    void SetGPU(int GPU_Device);

    /// Copy the GPU volume array back to the host (i.e. CPU) and return the pointer to the new host array.
    float *GetVolumeFromDevice();

    /// Copy the GPU volume array back to the host (i.e. CPU) and return the pointer to the new host array.
    float *GetCASVolumeFromDevice();

    /// Copy the GPU volume array back to the host (i.e. CPU) and return the pointer to the new host array.
    float *GetPlaneDensityFromDevice();
    
    /// Get the pointer to the volume array on the GPU.
    float *GetCASVolumePtr();
    
    /// Get the pointer to the plane density array on the GPU.
    float *GetPlaneDensityPtr();

    /// Copy the volume array from the GPU back to the pinned host (i.e. CPU) memory. This is used to return the output of the back projection.
    void CopyVolumeToHost();

    /// Copy the CAS volume array from the GPU back to the pinned host (i.e. CPU) memory. This is needed if running the Fourier transform on the CPU (such as within Matlab or Python).
    void CopyCASVolumeToHost();
    
    /// Copy the CAS volume array from the GPU back to the pinned host (i.e. CPU) memory. This is needed if running the Fourier transform on the CPU (such as within Matlab or Python).
    void CopyPlaneDensityToHost();

    /// A structure for holding all of the pointer offset values when running the forward and back projection kernels.
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
        std::vector<int> currBatch;
        int Batches;
        int num_offsets;
    };

    /// Convert projection images to CAS images by running a forward FFT
    void ImgsToCASImgs(cudaStream_t &stream, float *CASImgs, float *Imgs, cufftComplex *CASImgsComplex, int numImgs);

    /// Reconstruct the volume by converting the CASVolume to Volume and running an inverse FFT. This volume is normalized by the plane density array and the 
    /// Kaiser Bessel pre-compensation array.
    void ReconstructVolume();

    /// Convert the CASVolume to volume by running an inverse FFT. This function does not normalize using the plane density.
    void CASVolumeToVolume();

    /// Calculate the plane density by running the back projection kernel with the CASimages array equal to one. The plane density
    /// is needed for normalizing the volume during reconstruction.
    void CalculatePlaneDensity(int AxesOffset, int nAxesToProcess);

private:
    // Initilize pointers for allocating memory on the GPU
    DeviceMemory<cufftComplex> *d_CASImgsComplex;      // For forward / inverse FFT
    DeviceMemory<cufftComplex> *d_PaddedVolumeComplex; // For converting volume to CAS volume
    DeviceMemory<float> *d_CASVolume;
    DeviceMemory<float> *d_PaddedVolume;
    DeviceMemory<float> *d_CASImgs;
    DeviceMemory<float> *d_Imgs;     // Output images
    DeviceMemory<float> *d_KB_Table; // Kaiser bessel lookup table
    DeviceMemory<float> *d_CoordAxes;
    DeviceMemory<float> *d_PlaneDensity;
    DeviceMemory<float> *d_Volume;

    // Kernel launching parameters
    // int gridSize;
    // int blockSize;
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

    // Initilization functions
    void InitilizeGPUArrays();
    void InitilizeCUDAStreams();

    // Estimate the maximum number of coordinate axes to allocate on the GPUs
    int EstimateMaxAxesToAllocate(int VolumeSize, int interpFactor);

    // Print to the console how much GPU memory is remaining
    void PrintMemoryAvailable();

    // Free all of the allocated memory
    void FreeMemory();

    // For converting the volume to CAS volume
    bool forwardFFTVolumePlannedFlag;
    cufftHandle forwardFFTVolume;

    // For converting images to CAS images
    bool forwardFFTImagesFlag;
    cufftHandle forwardFFTImages;

    // For converting CAS images to images
    bool inverseFFTImagesFlag;
    cufftHandle inverseFFTImages;

    // Convert the volume to a CAS volume
    void VolumeToCASVolume();

protected:
    // Plan the pointer offset values for running the CUDA kernels
    Offsets PlanOffsetValues(int coordAxesOffset, int nAxes);

    // Convert CAS images to images using an inverse FFT
    void CASImgsToImgs(cudaStream_t &stream, float *CASImgs, float *Imgs, cufftComplex *CASImgsComplex, int numImgs);

    // Should we print status information to the console?
    bool verbose;
};
