#include "AbstractGridder.h"
#include "gpuFFT.h"

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

    // Return the volume
    float *GetVolume();

    // // Set the GPU for this object to use for processiong
    // void SetGPU(int GPU){this->GPU = GPU;};

protected:
    // // Get a new images array and then convert them to CAS
    // void SetImages(float *imgs);

    // Convert the volume to a CAS volume
    void VolumeToCASVolume();

    // // Convert the CAS volume back to volume
    // void CASToVolume();

private:
    // Which GPU(s) to use for processing?
    std::vector<int> GPUs;

    // Pointers to the CASVolume array on the device (i.e. the GPU)
    std::vector<float *> d_CASVolume;

    // Pointers to the CAS images array on the device (i.e. the GPU)
    std::vector<float *> d_CASImgs;

    // Pointers to the coordinate axes vector on the device (i.e. the GPU)
    std::vector<float *> d_CoordAxes;

    // Pointers to the Kaiser bessel vector on the device (i.e. the GPU)
    std::vector<float *> d_KB_Table;

    // Initilize the GPU arrays
    void InitilizeGPUArrays();

    // Copy the volume to each of the GPUs
    void CopyVolumeToGPUs();

    // Allocate GPU arrays
    void AllocateGPUArray(int GPU_Device, std::vector<float *> Ptr_Vector, int ArraySize);

    // Flag to test that all arrays were allocated successfully
    bool ErrorFlag = false;
    // // Free all of the allocated memory
    // void FreeMemory();


};
