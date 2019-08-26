#include "AbstractGridder.h"
#include "gpuFFT.h"

#include <cstdlib>
#include <stdio.h>

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

    // // Convert the volume to a CAS volume
    // void VolumeToCAS();

    // // Convert the CAS volume back to volume
    // void CASToVolume();

private:
    // // Free all of the allocated memory
    // void FreeMemory();

    // // Which GPU to use for processing?
    // int GPU;
};
