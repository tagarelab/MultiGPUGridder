#pragma once

/**
 * @class   gpuProjection
 * @brief   Parent class for the forward and back projection classes
 *
 * */

#include "HostMemory.h"
#include "DeviceMemory.h"
#include "gpuErrorCheck.h"

#include "gpuForwardProject.h"
#include "gpuBackProject.h"

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

class gpuProjection
{

public:
    // Deconstructor
    ~gpuProjection()
    {
        FreeMemory();
    }

    // gpuProjection Constructor
    gpuProjection(int GPU_Device, int nStreamsFP, int nStreamsBP, int VolumeSize, float interpFactor, int numCoordAxes, int extraPadding, bool RunFFTOnDevice = true, bool verbose = false)
    {
        this->GPU_Device = GPU_Device;
        this->nStreamsFP = nStreamsFP;
        this->nStreamsBP = nStreamsBP;
        this->VolumeSize = VolumeSize;
        this->interpFactor = interpFactor;
        this->numCoordAxes = numCoordAxes;
        this->extraPadding = extraPadding;
        this->RunFFTOnDevice = RunFFTOnDevice;
        this->verbose = verbose;

        // Set default values
        this->GPUArraysAllocatedFlag = false;
        this->KB_Table_Size = 501;
        this->kerSize = 501;
        this->kerHWidth = 2;
        this->ErrorFlag = 0;
        this->ApplyCTFs = true;

        this->inverseFFTImagesFlag = false;
        this->forwardFFTVolumePlannedFlag = false;
        this->forwardFFTImagesFlag = false;

        // Initialize the CUDA streams and allocate the GPU memory
        Allocate();
    }

    // Setter functions for previously allocated host memory
    void SetHostVolume(HostMemory<float> *h_Volume) { this->h_Volume = h_Volume; }
    void SetHostCASVolume(HostMemory<float> *h_CASVolume) { this->h_CASVolume = h_CASVolume; }
    void SetHostImages(HostMemory<float> *h_Imgs) { this->h_Imgs = h_Imgs; }
    void SetHostCASImages(HostMemory<float> *h_CASImgs) { this->h_CASImgs = h_CASImgs; }
    // void SetHostCTF(HostMemory<float> *h_CTF) { this->h_CTF = h_CTF; }
    void SetHostCoordinateAxes(HostMemory<float> *h_CoordAxes) { this->h_CoordAxes = h_CoordAxes; }
    void SetHostKBTable(HostMemory<float> *h_KB_Table) { this->h_KB_Table = h_KB_Table; }
    void SetHostKBPreCompArray(HostMemory<float> *h_KBPreComp) { this->h_KBPreComp = h_KBPreComp; }

    // Setter functions for previously allocated GPU memory
    void SetDeviceVolume(DeviceMemory<float> *d_Volume) { this->d_Volume = d_Volume; }
    void SetDeviceCASVolume(DeviceMemory<float> *d_CASVolume) { this->d_CASVolume = d_CASVolume; }
    void SetDevicePlaneDensity(DeviceMemory<float> *d_PlaneDensity) { this->d_PlaneDensity = d_PlaneDensity; }

    /// Run forward projection on some subset of the coordinate axes. AxesOffset is the number of coordinate axes from the beginning of the
    /// host array (starting at 0) while the nAxesToProcess is the number of coordinate axes to process. For example, if the host memory has
    /// 1,000 coordinate axes, and we wanted the to process axes 500 to 700 then simply set the AxesOffset to 500 and the nAxesToProcess to 200.
    void ForwardProject(int AxesOffset, int nAxesToProcess);

    /// Run back projection on some subset of the coordinate axes. AxesOffset is the number of coordinate axes from the beginning of the
    /// host array (starting at 0) while the nAxesToProcess is the number of coordinate axes to process. For example, if the host memory has
    /// 1,000 coordinate axes, and we wanted the to process axes 500 to 700 then simply set the AxesOffset to 500 and the nAxesToProcess to 200.
    void BackProject(int AxesOffset, int nAxesToProcess);

    /// Calculate the plane density by running the back projection kernel with the CASimages array equal to one. The plane density
    /// is needed for normalizing the volume during reconstruction.
    void CalculatePlaneDensity(int AxesOffset, int nAxesToProcess);

    /// The mask radius for forward and back projection. The default value is the volume size times interpolation factor divided by two minus one.
    void SetMaskRadius(float maskRadius) { this->maskRadius = maskRadius; };

    /// Allocate required GPU memory and CUDA streams
    void Allocate();

    /// Reconstruct the volume by converting the CASVolume to Volume and running an inverse FFT. This volume is normalized by the plane density array and the
    /// Kaiser Bessel pre-compensation array.
    void ReconstructVolume();

    /// Convert the CASVolume to volume by running an inverse FFT. This function does not normalize using the plane density.
    void CASVolumeToVolume();

protected:
    // Initilization functions
    void InitializeGPUArrays();
    void InitializeCUDAStreams();

    // Print to the console how much GPU memory is remaining
    void PrintMemoryAvailable();

    /// Estimate how many coordinate axes we have memory to allocate
    int EstimateMaxAxesToAllocate(int VolumeSize, int interpFactor);

    /// Convert projection images to CAS images by running a forward FFT
    void ImgsToCASImgs(cudaStream_t &stream, float *CASImgs, cufftComplex *CASImgsComplex, float *Imgs, float *CTFs, float *CTFsPadded, int numImgs);

private:
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

    int GPU_Device; // Which GPU to use?
    int nStreamsFP; // Streams to use on this GPU for the forward projection operation
    int nStreamsBP; // Number of streams for the back projection operation

    int VolumeSize;
    float interpFactor;
    int extraPadding;
    int MaxAxesToAllocate;

    // Size of the Kaiser bessel vector
    int kerSize;

    // Width of the Kaiser bessel function
    float kerHWidth;

    // Mask radius for the forward / back projection kernel
    float maskRadius;

    // CUDA streams to use for forward / back projection
    cudaStream_t *FP_streams;

    // CUDA streams to use for back projection
    cudaStream_t *BP_streams;

    // Flag to test that all arrays were allocated successfully
    bool ErrorFlag;

    // Number of coordinate axes to allocate on this GPU
    int numCoordAxes;

    // Length of the Kaiser Bessel look up table
    int KB_Table_Size;

    // Initialize pointers for the allocated CPU memory
    HostMemory<float> *h_Imgs;
    HostMemory<float> *h_Volume;
    HostMemory<float> *h_CoordAxes;
    HostMemory<float> *h_KB_Table;
    HostMemory<float> *h_KBPreComp;    // Kaiser Bessel precompensation array (currently set using Matlab getPreComp())
    HostMemory<float> *h_CASVolume;    // Optional inputs
    HostMemory<float> *h_CASImgs;      // Optional inputs
    HostMemory<float> *h_PlaneDensity; // Optional inputs

    // Initialize pointers for the allocated GPU memory
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
    DeviceMemory<float> *d_CASVolume_Cropped;
    DeviceMemory<cufftComplex> *d_CASVolume_Cropped_Complex; // For converting volume to CAS volume
    DeviceMemory<float> *d_KBPreComp;
    DeviceMemory<float> *d_CTFs;               // For applying the CTFs
    DeviceMemory<float> *d_CTFsPadded;         // For applying the CTFs

    // For converting the volume to CAS volume
    bool forwardFFTVolumePlannedFlag;
    cufftHandle forwardFFTVolume;

    // For converting images to CAS images
    bool forwardFFTImagesFlag;
    cufftHandle forwardFFTImages;

    // For converting CAS images to images
    bool inverseFFTImagesFlag;
    cufftHandle inverseFFTImages;

    // Plan the pointer offset values for running the CUDA kernels
    Offsets PlanOffsetValues(int coordAxesOffset, int nAxes, int numStreams);

    // Convert CAS images to images using an inverse FFT
    void CASImgsToImgs(cudaStream_t &stream, float *CASImgs, float *Imgs, int numImgs, cufftComplex *CASImgsComplex);

    // Should we print status information to the console?
    bool verbose;

    // Convert the volume to a CAS volume
    void VolumeToCASVolume();

    // Array allocation flag
    bool GPUArraysAllocatedFlag;

    // Flag to run the forward and inverse FFT on the GPU
    int RunFFTOnDevice;

    // Should we apply the CTFs before backprojection?
    bool ApplyCTFs;

    // Free all of the allocated memory
    void FreeMemory();
};