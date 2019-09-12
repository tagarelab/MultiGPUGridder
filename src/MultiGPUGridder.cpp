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

MultiGPUGridder::CoordinateAxesPlan MultiGPUGridder::PlanCoordinateAxes()
{
    // Plan which GPU will process which coordinate axes
    // NumAxesPerGPU is how many coordinate axes each GPU will process
    // coordAxesOffset is the index of the starting coordinate axes for each GPU

    CoordinateAxesPlan AxesPlan_obj;
    AxesPlan_obj.NumAxesPerGPU.clear();
    AxesPlan_obj.coordAxesOffset = new int[this->Num_GPUs];

    // Estimate equal number of coordinate axes to process on each GPU
    int EstimatedNumAxesPerGPU = ceil((double)this->GetNumAxes() / (double)this->Num_GPUs);

    int NumAxesAssigned = 0;

    // TO DO: Consider different GPU specs within the same computer and assign more axes to more capable devices
    // Currently we assume each GPU is equally capable
    for (int i = 0; i < this->Num_GPUs; i++)
    {
        if ((NumAxesAssigned + EstimatedNumAxesPerGPU) < this->GetNumAxes())
        {
            // Assign the number of coordinate axes for this GPU
            AxesPlan_obj.NumAxesPerGPU.push_back(EstimatedNumAxesPerGPU);

            NumAxesAssigned = NumAxesAssigned + EstimatedNumAxesPerGPU;
        }
        else if (NumAxesAssigned < this->GetNumAxes()) // Are the axes left to assign?
        {
            // Otherwise we would assign more axes than we have so take the remainder
            AxesPlan_obj.NumAxesPerGPU.push_back(this->GetNumAxes() - NumAxesAssigned);

            NumAxesAssigned = NumAxesAssigned + this->GetNumAxes() - NumAxesAssigned;
        }
    }

    // First GPU has no offset so assign a value of zero
    AxesPlan_obj.coordAxesOffset[0] = 0;

    // Calculate the offset for the rest of the GPUs
    if (this->Num_GPUs > 1)
    {
        for (int i = 1; i < this->Num_GPUs; i++)
        {
            AxesPlan_obj.coordAxesOffset[i] = AxesPlan_obj.coordAxesOffset[i - 1] + AxesPlan_obj.NumAxesPerGPU[i - 1];
        }
    }

    return AxesPlan_obj;
}

void MultiGPUGridder::ForwardProject()
{
    // Run the forward projection kernel on each gpuGridder object
    // Each GPU will process a subset of the coordinate axes
    // Since the axes array is static all objects share the same variable
    // So we just need to pass an offset (in number of coordinate axes) from the beginng
    // To select the subset of axes to process

    // Plan which GPU will process which coordinate axes
    CoordinateAxesPlan AxesPlan_obj = PlanCoordinateAxes();

    // If this is the first time running allocate the needed GPU memory
    if (this->ForwardProjectInitializedFlag == false)
    {
        for (int i = 0; i < Num_GPUs; i++)
        {
            gpuGridder_vec[i]->SetNumAxes(AxesPlan_obj.NumAxesPerGPU[i]);
            gpuGridder_vec[i]->Allocate();
        }
        this->ForwardProjectInitializedFlag = true;
    }

    // Convert the volume to CAS volume using the first GPU
    // The CASVolume is shared amoung all the objects since CASVolume is a static member in the AbstractGridder class
    cudaSetDevice(this->GPU_Devices[0]);
    gpuGridder_vec[0]->VolumeToCASVolume();
    cudaDeviceSynchronize(); // Wait for the first GPU to convert the volume to CAS volume

    // Copy the CAS Volume to each GPU at the same time
    for (int i = 0; i < Num_GPUs; i++)
    {
        gpuGridder_vec[i]->CopyCASVolumeToGPUAsyc();
    }

    // Synchronize all of the GPUs
    GPU_Sync(); // needed?

    for (int i = 0; i < Num_GPUs; i++)
    {
        gpuGridder_vec[i]->ForwardProject(AxesPlan_obj.coordAxesOffset[i], AxesPlan_obj.NumAxesPerGPU[i]);
    }

    // Synchronize all of the GPUs
    GPU_Sync();
}

void MultiGPUGridder::GPU_Sync()
{
    // Synchronize all of the GPUs
    for (int i = 0; i < this->Num_GPUs; i++)
    {
        cudaSetDevice(this->GPU_Devices[i]);
        cudaDeviceSynchronize();
    }
}

void MultiGPUGridder::FreeMemory()
{
    std::cout << "FreeMemory(): " << '\n';
    // Free all of the allocated GPU memory
    for (int i = 0; i < Num_GPUs; i++)
    {
        std::cout << "GPU " << i << '\n';
        delete gpuGridder_vec[i];
    }

    // Free all of the allocated CPU memory
    // Let Matlab delete this instead for now
    // delete this->Volume;
    // delete this->CASVolume;
    // delete this->imgs;
    // delete this->CASimgs;
    // delete this->coordAxes;
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