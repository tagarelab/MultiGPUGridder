#include "MultiGPUGridder.h"

// Are we compiling on a windows or linux machine?
#if defined(_MSC_VER)
//  Microsoft
#define EXPORT __declspec(dllexport)
#define IMPORT __declspec(dllimport)
#elif defined(__GNUC__)
//  GCC
#define EXPORT __attribute__((visibility("default")))
#define IMPORT
#else
//  Do nothing and provide a warning to the user
#define EXPORT
#define IMPORT
#pragma warning Unknown dynamic link import / export semantics.
#endif

void MultiGPUGridder::SetVolume(float *Volume, int *VolumeSize)
{
    // Set the volume on each gpuGridder object
    for (int i = 0; i < Num_GPUs; i++)
    {
        gpuGridder_vec[i]->SetVolume(Volume, VolumeSize);
    }
}

void MultiGPUGridder::SetCASVolume(float *Volume, int *VolumeSize)
{
    // Set the CASvolume on each gpuGridder object
    for (int i = 0; i < Num_GPUs; i++)
    {
        gpuGridder_vec[i]->SetCASVolume(Volume, VolumeSize);
    }
}

void MultiGPUGridder::SetCoordAxes(float *Volume, int *VolumeSize)
{
    std::cout << "MultiGPUGridder::SetCoordAxes()" << '\n';

    // Set the coordinate axes array on each gpuGridder object
    for (int i = 0; i < Num_GPUs; i++)
    {
        std::cout << "Setting coordaxes for GPU device " << i << '\n';

        gpuGridder_vec[i]->SetCoordAxes(Volume, VolumeSize);
    }
}

void MultiGPUGridder::SetImages(float *Volume, int *VolumeSize)
{
    // Set the output projection images array on each gpuGridder object
    for (int i = 0; i < Num_GPUs; i++)
    {
        gpuGridder_vec[i]->SetImages(Volume, VolumeSize);
    }
}

void MultiGPUGridder::SetCASImages(float *Volume, int *VolumeSize)
{
    // Set the output projection CAS images array on each gpuGridder object
    for (int i = 0; i < Num_GPUs; i++)
    {
        gpuGridder_vec[i]->SetCASImages(Volume, VolumeSize);
    }
}

void MultiGPUGridder::SetKerBesselVector(float *Volume, int *VolumeSize)
{
    // Set the Kaiser Bessel lookup table array on each gpuGridder object
    for (int i = 0; i < Num_GPUs; i++)
    {
        gpuGridder_vec[i]->SetKerBesselVector(Volume, VolumeSize);
    }
}

void MultiGPUGridder::SetNumStreams(int nStreams)
{
    // Set the number of CUDA streams to use with each GPU
    for (int i = 0; i < Num_GPUs; i++)
    {
        gpuGridder_vec[i]->SetNumStreams(nStreams);
    }
}

void MultiGPUGridder::ForwardProject()
{
    // Run the forward projection kernel on each gpuGridder object
    for (int i = 0; i < Num_GPUs; i++)
    {
        gpuGridder_vec[i]->ForwardProject();
    }

    // Sync the GPUs
    for (int i = 0; i < Num_GPUs; i++)
    {
        cudaSetDevice(this->GPU_Devices[i]);
        cudaDeviceSynchronize();
    }
}

// Define C functions for the C++ class since Python ctypes can only talk to C (not C++)
// #define USE_EXTERN_C true
// #if USE_EXTERN_C == true

// extern "C"
// {
//     EXPORT MultiGPUGridder *Gridder_new() { return new MultiGPUGridder(); }
//     EXPORT void SetNumberGPUs(MultiGPUGridder *gridder, int numGPUs) { gridder->SetNumberGPUs(numGPUs); }
//     EXPORT void SetNumberStreams(MultiGPUGridder *gridder, int nStreams) { gridder->SetNumberStreams(nStreams); }
//     EXPORT void SetVolume(MultiGPUGridder *gridder, float *gpuVol, int *gpuVolSize) { gridder->SetVolume(gpuVol, gpuVolSize); }
//     EXPORT float *GetVolume(MultiGPUGridder *gridder) { return gridder->GetVolume(); }
//     EXPORT void ResetVolume(MultiGPUGridder *gridder) { gridder->ResetVolume(); }
//     EXPORT void SetImages(MultiGPUGridder *gridder, float *newCASImgs) { gridder->SetImages(newCASImgs); }
//     EXPORT float *GetImages(MultiGPUGridder *gridder, float *CASImgs) { return gridder->GetImages(); }
//     EXPORT void SetAxes(MultiGPUGridder *gridder, float *coordAxes, int *axesSize) { gridder->SetAxes(coordAxes, axesSize); }
//     EXPORT void SetImgSize(MultiGPUGridder *gridder, int *imgSize) { gridder->SetImgSize(imgSize); }
//     EXPORT void SetMaskRadius(MultiGPUGridder *gridder, float *maskRadius) { gridder->SetMaskRadius(maskRadius); }
//     EXPORT void Forward_Project(MultiGPUGridder *gridder) { gridder->Forward_Project(); }
//     EXPORT void Back_Project(MultiGPUGridder *gridder) { gridder->Back_Project(); }
// }
// #endif