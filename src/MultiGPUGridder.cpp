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
    // Each GPU will process a subset of the coordinate axes
    // Since the axes array is static all objects share the same variable
    // So we just need to pass an offset (in number of coordinate axes) from the beginng
    // To select the subset of axes to process

    // Estimate number of coordinate axes per GPU
    int EstimatedNumAxesPerGPU = ceil((double)this->GetNumAxes() / (double)this->Num_GPUs);

    std::cout << "EstimatedNumAxesPerGPU: " << EstimatedNumAxesPerGPU << '\n';

    // Create a vector to hold the number of axes to assign for each GPU
    std::vector<int> NumAxesPerGPU;

    int NumAxesAssigned = 0;

    // TO DO: Consider different GPU specs within the same computer and assign more axes to more capable devices
    for (int i = 0; i < this->Num_GPUs; i++)
    {
        if ((NumAxesAssigned + EstimatedNumAxesPerGPU) < this->GetNumAxes())
        {
            // Assign the number of coordinate axes for this GPU
            NumAxesPerGPU.push_back(EstimatedNumAxesPerGPU);

            NumAxesAssigned = NumAxesAssigned + EstimatedNumAxesPerGPU;
        }
        else if (NumAxesAssigned < this->GetNumAxes()) // Are the axes left to assign?
        {
            // Otherwise we would assign more axes than we have so take the remainder
            NumAxesPerGPU.push_back(this->GetNumAxes() - NumAxesAssigned);

            NumAxesAssigned = NumAxesAssigned + this->GetNumAxes() - NumAxesAssigned;
        }

        std::cout << "NumAxesPerGPU[i]: " << NumAxesPerGPU[i] << '\n';
    }

    // Calculate the offset for each GPU
    int coordAxesOffset[this->Num_GPUs];

    // First GPU has no offset so assign a value of zero
    coordAxesOffset[0] = 0;

    std::cout << "coordAxesOffset[0]: " << coordAxesOffset[0] << '\n';

    // Calculate the offset for the rest of the GPUs
    if (this->Num_GPUs > 1)
    {
        for (int i = 1; i < this->Num_GPUs; i++)
        {
            coordAxesOffset[i] = coordAxesOffset[i - 1] + NumAxesPerGPU[i - 1];

            std::cout << "coordAxesOffset[i]: " << coordAxesOffset[i] << '\n';
        }
    }

    // Convert the volume to CAS volume using the first GPU
    // The CASVolume is shared amoung all the objects since CASVolume is a static member in the AbstractGridder class
    gpuGridder_vec[0]->VolumeToCASVolume();

    for (int i = 0; i < Num_GPUs; i++)
    {
        gpuGridder_vec[i]->newVolumeFlag = false;
        gpuGridder_vec[i]->ForwardProject(coordAxesOffset[i], NumAxesPerGPU[i]);
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