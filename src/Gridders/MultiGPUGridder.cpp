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

void MultiGPUGridder::SetNumStreamsFP(int nStreamsFP)
{
    // Set the number of CUDA streams to use with each GPU for the forward projection
    for (int i = 0; i < Num_GPUs; i++)
    {
        gpuGridder_vec[i]->SetNumStreamsFP(nStreamsFP);
    }
}

void MultiGPUGridder::SetNumStreamsBP(int nStreamsBP)
{
    // Set the number of CUDA streams to use with each GPU for the back projection
    for (int i = 0; i < Num_GPUs; i++)
    {
        gpuGridder_vec[i]->SetNumStreamsBP(nStreamsBP);
    }
}

MultiGPUGridder::CoordinateAxesPlan MultiGPUGridder::PlanCoordinateAxes()
{
    // Plan which GPU will process which coordinate axes
    // NumAxesPerGPU is how many coordinate axes each GPU will process
    // coordAxesOffset is the index of the starting coordinate axes for each GPU

    if (this->verbose == true)
    {
        std::cout << "MultiGPUGridder::CoordinateAxesPlan()" << '\n';
    }

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

            if (this->verbose == true)
            {
                std::cout << "GPU " << this->GPU_Devices[i] << " coordinate axes offset: " << AxesPlan_obj.coordAxesOffset[i] << '\n';
                std::cout << "GPU " << this->GPU_Devices[i] << " axes assigned: " << AxesPlan_obj.NumAxesPerGPU[i] << '\n';
            }
        }
    }

    return AxesPlan_obj;
}

void MultiGPUGridder::ForwardProject()
{
    // Run the forward projection kernel on each gpuGridder object
    // Each GPU will process a subset of the coordinate axes
    // So we just need to pass an offset (in number of coordinate axes) from the beginning
    // To select the subset of axes to process

    std::vector<std::thread> CPUThreads;

    if (this->UseMultiThread == true)
    {
        // Reserve space for CPU threads with one CPU thread for each GPU
        // Ensures the GPU process concurently if a CPU thread blocking CUDA API call is made
        // Such as cudaMalloc or cudaDeviceSynchronize
        CPUThreads.reserve(Num_GPUs);
    }

    if (this->verbose == true)
    {
        std::cout << "MultiGPUGridder::ForwardProject()" << '\n';
    }

    // Plan which GPU will process which coordinate axes
    CoordinateAxesPlan AxesPlan_obj = PlanCoordinateAxes();

    // Pass the host memory pointers to each of the gpu gridder objects
    for (int i = 0; i < Num_GPUs; i++)
    {
        gpuGridder_vec[i]->h_Imgs = this->h_Imgs;
        gpuGridder_vec[i]->h_Volume = this->h_Volume;
        gpuGridder_vec[i]->h_CoordAxes = this->h_CoordAxes;
        gpuGridder_vec[i]->h_KB_Table = this->h_KB_Table;
        gpuGridder_vec[i]->h_CASVolume = this->h_CASVolume;
        gpuGridder_vec[i]->h_CASImgs = this->h_CASImgs;
    }

    // If this is the first time running allocate the needed GPU memory
    if (this->ProjectInitializedFlag == false)
    {
        for (int i = 0; i < Num_GPUs; i++)
        {
            gpuGridder_vec[i]->h_Volume = this->h_Volume;
            gpuGridder_vec[i]->SetNumAxes(AxesPlan_obj.NumAxesPerGPU[i]);
            gpuGridder_vec[i]->Allocate();
        }

        this->ProjectInitializedFlag = true;
    }

    // Update the mask radius parameter
    for (int i = 0; i < Num_GPUs; i++)
    {
        gpuGridder_vec[i]->SetMaskRadius(this->maskRadius);
    }

    // Synchronize all of the GPUs
    GPU_Sync(); // needed?

    for (int i = 0; i < Num_GPUs; i++)
    {
        if (this->UseMultiThread == true)
        {
            // Multi thread version
            CPUThreads.push_back(std::thread(&gpuGridder::ForwardProject, gpuGridder_vec[i], AxesPlan_obj.coordAxesOffset[i], AxesPlan_obj.NumAxesPerGPU[i]));
        }
        else
        {
            // Single thread version: gpuGridder_vec[i]->ForwardProject(AxesPlan_obj.coordAxesOffset[i], AxesPlan_obj.NumAxesPerGPU[i]);
            gpuGridder_vec[i]->ForwardProject(AxesPlan_obj.coordAxesOffset[i], AxesPlan_obj.NumAxesPerGPU[i]);
        }
    }

    if (this->UseMultiThread == true)
    {
        // Join CPU threads together
        for (int i = 0; i < Num_GPUs; i++)
        {
            CPUThreads[i].join();
        }
    }

    CPUThreads.clear();
    CPUThreads.shrink_to_fit();

    // Synchronize all of the GPUs
    GPU_Sync();
}

void MultiGPUGridder::BackProject()
{
    // Run the back projection kernel on each gpuGridder object
    // Each GPU will process a subset of the coordinate axes
    // So we just need to pass an offset (in number of coordinate axes) from the beginning
    // To select the subset of axes to process

    std::vector<std::thread> CPUThreads;
    if (this->UseMultiThread == true)
    {
        // Reserve space for CPU threads with one CPU thread for each GPU
        // Ensures the GPU process concurently if a CPU thread blocking CUDA API call is made
        // Such as cudaMalloc or cudaDeviceSynchronize
        CPUThreads.reserve(Num_GPUs);
    }

    if (this->verbose == true)
    {
        std::cout << "MultiGPUGridder::BackProject()" << '\n';
    }

    // Plan which GPU will process which coordinate axes
    CoordinateAxesPlan AxesPlan_obj = PlanCoordinateAxes();

    // Pass the host memory pointers to each of the gpu gridder objects
    for (int i = 0; i < Num_GPUs; i++)
    {
        gpuGridder_vec[i]->h_Imgs = this->h_Imgs;
        gpuGridder_vec[i]->h_Volume = this->h_Volume;
        gpuGridder_vec[i]->h_CoordAxes = this->h_CoordAxes;
        gpuGridder_vec[i]->h_KB_Table = this->h_KB_Table;
        gpuGridder_vec[i]->h_CASVolume = this->h_CASVolume;
        gpuGridder_vec[i]->h_CASImgs = this->h_CASImgs;
    }

    // If this is the first time running allocate the needed GPU memory
    if (this->ProjectInitializedFlag == false)
    {
        for (int i = 0; i < Num_GPUs; i++)
        {
            gpuGridder_vec[i]->SetNumAxes(AxesPlan_obj.NumAxesPerGPU[i]);
            gpuGridder_vec[i]->Allocate();
        }
        this->ProjectInitializedFlag = true;
    }

    // Update the mask radius parameter
    for (int i = 0; i < Num_GPUs; i++)
    {
        gpuGridder_vec[i]->SetMaskRadius(this->maskRadius);
    }

    // Synchronize all of the GPUs
    GPU_Sync();

    for (int i = 0; i < Num_GPUs; i++)
    {
        if (this->UseMultiThread == true)
        {
            // Multi thread version
            CPUThreads.push_back(std::thread(&gpuGridder::BackProject, gpuGridder_vec[i], AxesPlan_obj.coordAxesOffset[i], AxesPlan_obj.NumAxesPerGPU[i]));
        }
        else
        {
            // Single thread version
            gpuGridder_vec[i]->BackProject(AxesPlan_obj.coordAxesOffset[i], AxesPlan_obj.NumAxesPerGPU[i]);
        }
    }

    if (this->UseMultiThread == true)
    {
        // Join CPU threads together
        for (int i = 0; i < Num_GPUs; i++)
        {
            CPUThreads[i].join();
        }
    }

    CPUThreads.clear();
    CPUThreads.shrink_to_fit();

    // Synchronize all of the GPUs
    GPU_Sync();
}

void MultiGPUGridder::CASVolumeToVolume()
{
    // Combine the CASVolume from each GPU and convert it to volume
    if (this->verbose == true)
    {
        std::cout << "MultiGPUGridder::CASVolumeToVolume()" << '\n';
    }

    // Pass the host memory pointers to each of the gpu gridder objects
    // TO DO: is this needed?
    for (int i = 0; i < Num_GPUs; i++)
    {
        gpuGridder_vec[i]->h_Imgs = this->h_Imgs;
        gpuGridder_vec[i]->h_Volume = this->h_Volume;
        gpuGridder_vec[i]->h_CoordAxes = this->h_CoordAxes;
        gpuGridder_vec[i]->h_KB_Table = this->h_KB_Table;
        gpuGridder_vec[i]->h_CASVolume = this->h_CASVolume;
        gpuGridder_vec[i]->h_CASImgs = this->h_CASImgs;
    }

    // Synchronize all of the GPUs
    GPU_Sync();

    // Combine the CAS volume arrays from each GPU and copy back to the host
    // SumCASVolumes(); // Slower function but seems more reliable

    // This function is faster, but seems to be less reliable if we clear the gridder and remake it
    int GPU_For_Reconstruction = this->GPU_Devices[0];
    EnablePeerAccess(GPU_For_Reconstruction);
    AddCASVolumes(GPU_For_Reconstruction);

    // Synchronize all of the GPUs
    GPU_Sync();
}

void MultiGPUGridder::EnablePeerAccess(int GPU_For_Reconstruction)
{
    // Allow the first GPU to access the memory of the other GPUs
    // This is needed for the reconstruct volume function

    if (this->PeerAccessEnabled == false)
    {
        gpuErrorCheck(cudaSetDevice(GPU_For_Reconstruction));
        for (int i = 0; i < this->Num_GPUs; i++)
        {
            if (i != GPU_For_Reconstruction)
            {
                // Is peer access already enabled?
                int canAccessPeer;
                cudaDeviceCanAccessPeer(&canAccessPeer, GPU_For_Reconstruction, i);

                if (canAccessPeer == 1)
                {
                    // The first GPU can now access GPU device number i
                    gpuErrorCheck(cudaDeviceEnablePeerAccess(i, 0));
                }
                else
                {
                    std::cerr << "The GPUs appear to not support peer access for sharing memory. \
                    ReconstructVolume() and CASVolumeToVolume() cannot run without this ability.Please try \
                    reconstructing on the CPU instead of the GPU."
                              << '\n';
                }
            }
        }

        this->PeerAccessEnabled = true;
    }
}

void MultiGPUGridder::DisablePeerAccess(int GPU_For_Reconstruction)
{
    // Allow the first GPU to access the memory of the other GPUs
    // This is needed for the reconstruct volume function

    if (this->PeerAccessEnabled == true)
    {
        gpuErrorCheck(cudaSetDevice(GPU_For_Reconstruction));
        for (int i = 0; i < this->Num_GPUs; i++)
        {
            if (i != GPU_For_Reconstruction)
            {
                // Is peer access already enabled?
                int canAccessPeer;
                cudaDeviceCanAccessPeer(&canAccessPeer, GPU_For_Reconstruction, i);

                if (canAccessPeer == 1)
                {
                    // The first GPU can no longer access GPU device number i
                    gpuErrorCheck(cudaDeviceDisablePeerAccess(i));
                }
                else
                {
                    std::cerr << "The GPUs appear to not support peer access for sharing memory. \
                    ReconstructVolume() and CASVolumeToVolume() cannot run without this ability.Please try \
                    reconstructing on the CPU instead of the GPU."
                              << '\n';
                }
            }
        }

        this->PeerAccessEnabled = false;
    }
}

void MultiGPUGridder::AddCASVolumes(int GPU_For_Reconstruction)
{
    // Add the CASVolume from all the GPUs to the given GPU device (without needing to copy to host memory first)
    // This is needed for reconstructing the volume after back projection

    if (this->verbose == true)
    {
        std::cout << "MultiGPUGridder::AddCASVolumes()" << '\n';
        std::cout << "Using GPU " << this->GPU_Devices[GPU_For_Reconstruction] << " for adding." << '\n';
    }

    gpuErrorCheck(cudaDeviceSynchronize());
    gpuErrorCheck(cudaSetDevice(this->GPU_Devices[GPU_For_Reconstruction]));
    std::unique_ptr<AddVolumeFilter> AddFilter(new AddVolumeFilter());

    int CASVolumeSize = this->h_Volume->GetSize(0) * this->interpFactor + this->extraPadding * 2;

    if (this->Num_GPUs > 1)
    {
        for (int i = 0; i < this->Num_GPUs; i++)
        {
            if (this->GPU_Devices[i] != this->GPU_Devices[GPU_For_Reconstruction])
            {
                AddFilter->SetVolumeSize(CASVolumeSize);
                AddFilter->SetNumberOfSlices(CASVolumeSize);
                AddFilter->SetVolumeOne(gpuGridder_vec[GPU_For_Reconstruction]->GetCASVolumePtr());
                AddFilter->SetVolumeTwo(gpuGridder_vec[i]->GetCASVolumePtr());
                AddFilter->Update();
                gpuErrorCheck(cudaDeviceSynchronize());
            }
        }
    }

    // Copy the resulting array to the pinned host memory
    gpuGridder_vec[GPU_For_Reconstruction]->CopyCASVolumeToHost();
}

void MultiGPUGridder::GPU_Sync()
{
    // Synchronize all of the GPUs
    for (int i = 0; i < this->Num_GPUs; i++)
    {
        gpuErrorCheck(cudaSetDevice(this->GPU_Devices[i]));
        gpuErrorCheck(cudaDeviceSynchronize());
    }
}

void MultiGPUGridder::SumCASVolumes()
{
    // Get the CAS volume off each GPU and sum the arrays together within host memory
    // This function is used to get the result after the back projection

    if (this->verbose == true)
    {
        std::cout << "MultiGPUGridder::SumCASVolumes()" << '\n';
    }

    // Synchronize all of the GPUs
    GPU_Sync();

    // Temporary volume array
    float *SummedVolume = new float[this->h_CASVolume->length()];

    for (int i = 0; i < this->h_CASVolume->length(); i++)
    {
        SummedVolume[i] = 0; // TO DO: replace with memset
    }

    for (int i = 0; i < Num_GPUs; i++)
    {
        // Get the volume from the current GPU
        float *tempVolume = gpuGridder_vec[i]->GetCASVolumeFromDevice();

        // Add the volumes together
        for (int i = 0; i < this->h_CASVolume->length(); i++)
        {
            SummedVolume[i] = SummedVolume[i] + tempVolume[i];
        }

        delete[] tempVolume;
    }

    // Copy the resulting summed volume to the pinned CPU array (if a pointer was previously provided)
    //if (this->h_CASVolume != NULL)
    //{
    this->h_CASVolume->CopyArray(SummedVolume);
    //}

    // Release the temporary memory
    delete[] SummedVolume;

    // Synchronize all of the GPUs
    GPU_Sync();
}

void MultiGPUGridder::FreeMemory()
{

    // Free all of the allocated GPU memory
    for (int i = 0; i < Num_GPUs; i++)
    {
        if (this->verbose == true)
        {
            std::cout << "MultiGPUGridder::FreeMemory() on GPU " << this->GPU_Devices[i] << '\n';
        }
    }

    // If the GPU peer access is enabled, disable it (i.e. the default setting)
    if (this->PeerAccessEnabled == true)
    {
        this->DisablePeerAccess(this->GPU_Devices[0]);
    }

    //delete [] gpuGridder_vec;
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