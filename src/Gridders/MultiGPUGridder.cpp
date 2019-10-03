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

        return AxesPlan_obj;
    }
}

void MultiGPUGridder::ForwardProject()
{
    // Run the forward projection kernel on each gpuGridder object
    // Each GPU will process a subset of the coordinate axes
    // Since the axes array is static all objects share the same variable
    // So we just need to pass an offset (in number of coordinate axes) from the beginng
    // To select the subset of axes to process

    if (this->verbose == true)
    {
        std::cout << "MultiGPUGridder::ForwardProject()" << '\n';
    }

    // Plan which GPU will process which coordinate axes
    CoordinateAxesPlan AxesPlan_obj = PlanCoordinateAxes();

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
    GPU_Sync(); // needed?

    for (int i = 0; i < Num_GPUs; i++)
    {
        // Single thread version: gpuGridder_vec[i]->ForwardProject(AxesPlan_obj.coordAxesOffset[i], AxesPlan_obj.NumAxesPerGPU[i]);
        gpuGridder_vec[i]->ForwardProject(AxesPlan_obj.coordAxesOffset[i], AxesPlan_obj.NumAxesPerGPU[i]);
    }

    // Synchronize all of the GPUs
    GPU_Sync();
}

void MultiGPUGridder::BackProject()
{
    // Run the back projection kernel on each gpuGridder object
    // Each GPU will process a subset of the coordinate axes
    // Since the axes array is static all objects share the same variable
    // So we just need to pass an offset (in number of coordinate axes) from the beginng
    // To select the subset of axes to process

    if (this->verbose == true)
    {
        std::cout << "MultiGPUGridder::BackProject()" << '\n';
    }

    // Plan which GPU will process which coordinate axes
    CoordinateAxesPlan AxesPlan_obj = PlanCoordinateAxes();

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
        gpuGridder_vec[i]->BackProject(AxesPlan_obj.coordAxesOffset[i], AxesPlan_obj.NumAxesPerGPU[i]);
    }

    // Synchronize all of the GPUs
    GPU_Sync();
}

void MultiGPUGridder::EnablePeerAccess(int GPU_For_Reconstruction)
{
    // Allow the first GPU to access the memory of the other GPUs
    // This is needed for the reconstruct volume function

    if (this->verbose == true)
    {
        std::cout << "MultiGPUGridder::EnablePeerAccess()" << '\n';
        std::cout << "GPU_For_Reconstruction: " << this->GPU_Devices[GPU_For_Reconstruction] << '\n';
    }

    if (this->PeerAccessEnabled == false)
    {
        gpuErrorCheck(cudaSetDevice(this->GPU_Devices[GPU_For_Reconstruction]));
        for (int i = 0; i < this->Num_GPUs; i++)
        {
            if (i != GPU_For_Reconstruction)
            {
                // Can peer access be enabled?
                int canAccessPeer;
                cudaDeviceCanAccessPeer(&canAccessPeer, this->GPU_Devices[GPU_For_Reconstruction], this->GPU_Devices[i]);
                
                if (canAccessPeer == 1)
                {
                    // The first GPU can now access GPU device number i
                    gpuErrorCheck(cudaDeviceEnablePeerAccess(this->GPU_Devices[i], 0));
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

void MultiGPUGridder::CASVolumeToVolume()
{
    // Combine the CASVolume from each GPU and convert it to volume

    if (this->verbose == true)
    {
        std::cout << "MultiGPUGridder::CASVolumeToVolume()" << '\n';
    }

    if (this->RunFFTOnDevice == 1)
    {
        // We have to combine the output from each GPU in the frequency domain and not spatial domain
        int GPU_For_Reconstruction = 0; // Use the first GPU for reconstructing the volume from CAS volume

        // Allow the first GPU to access the memory of the other GPUs
        // This is needed for the reconstruct volume function
        EnablePeerAccess(GPU_For_Reconstruction);

        // Add the CASVolume from all the GPUs to the first GPU (for reconstructing the volume)
        AddCASVolumes(GPU_For_Reconstruction);

        // Reconstruct the volume using the GPU_For_Reconstruction GPU
        gpuErrorCheck(cudaSetDevice(this->GPU_Devices[GPU_For_Reconstruction]));
        gpuGridder_vec[GPU_For_Reconstruction]->CASVolumeToVolume();
        gpuGridder_vec[GPU_For_Reconstruction]->CopyVolumeToHost();
    }
    else
    {
        // We're not running the FFT on the GPU so send the required arrays back to the CPU memory

        // Combine the CAS volume arrays from each GPU and copy back to the host
        SumCASVolumes();
    }
}

void MultiGPUGridder::ReconstructVolume()
{
    // First calculate the plane density on each GPU
    // Then combine the CASVolume and plane density arrays and convert to volume

    if (this->verbose == true)
    {
        std::cout << "MultiGPUGridder::ReconstructVolume()" << '\n';
    }

    // Plan which GPU will process which coordinate axes
    CoordinateAxesPlan AxesPlan_obj = PlanCoordinateAxes();

    for (int i = 0; i < this->Num_GPUs; i++)
    {
        // Calculate the plane densities on each GPU
        gpuGridder_vec[i]->CalculatePlaneDensity(AxesPlan_obj.coordAxesOffset[i], AxesPlan_obj.NumAxesPerGPU[i]);
    }

    // Synchronize all of the GPUs
    GPU_Sync();

    if (this->RunFFTOnDevice == 1)
    {
        // We have to combine the output from each GPU in the frequency domain and not spatial domain
        int GPU_For_Reconstruction = 0; // Use the first GPU for reconstructing the volume from CAS volume

        // Allow the first GPU to access the memory of the other GPUs
        // This is needed for the reconstruct volume function
        EnablePeerAccess(GPU_For_Reconstruction);

        // Add the CASVolume from all the GPUs to the first GPU (for reconstructing the volume)
        AddCASVolumes(GPU_For_Reconstruction);

        // Add the plane densities from all the GPUs to the first GPU (for reconstructing the volume)
        AddPlaneDensities(GPU_For_Reconstruction);

        // Reconstruct the volume using the GPU_For_Reconstruction GPU
        gpuGridder_vec[GPU_For_Reconstruction]->ReconstructVolume();
        gpuGridder_vec[GPU_For_Reconstruction]->CopyVolumeToHost();
    }
    else
    {
        // We're not running the FFT on the GPU so send the need arrays back to the CPU memory

        // Combine the CAS volume arrays from each GPU and copy back to the host
        SumCASVolumes();

        // Combine the plane density arrays from each GPU and copy back to the host
        SumPlaneDensity();
    }
}

void MultiGPUGridder::AddCASVolumes(int GPU_For_Reconstruction)
{
    // Add the CASVolume from all the GPUs to the given GPU device
    // This is needed for reconstructing the volume after back projection

    if (this->verbose == true)
    {
        std::cout << "MultiGPUGridder::AddCASVolumes()" << '\n';
        std::cout << "Using GPU " << this->GPU_Devices[GPU_For_Reconstruction] << " for adding." << '\n';
    }

    gpuErrorCheck(cudaDeviceSynchronize());
    gpuErrorCheck(cudaSetDevice(this->GPU_Devices[GPU_For_Reconstruction]));
    AddVolumeFilter *AddFilter = new AddVolumeFilter();

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

    delete AddFilter;

    // Copy the resulting array to the pinned host memory if the pointer exists
    // if (this->h_CASVolume != NULL)
    // {
    //     gpuGridder_vec[GPU_Device]->CopyCASVolumeToHost();
    // }
}

void MultiGPUGridder::AddPlaneDensities(int GPU_For_Reconstruction)
{
    // Add the plane density from all the GPUs to the given GPU device
    // This is needed for reconstructing the volume after back projection

    if (this->verbose == true)
    {
        std::cout << "MultiGPUGridder::AddPlaneDensities()" << '\n';
        std::cout << "Using GPU " << this->GPU_Devices[GPU_For_Reconstruction] << " for adding." << '\n';
    }

    gpuErrorCheck(cudaDeviceSynchronize());
    gpuErrorCheck(cudaSetDevice(this->GPU_Devices[GPU_For_Reconstruction]));
    AddVolumeFilter *AddFilter = new AddVolumeFilter();

    int PlaneDensityVolumeSize = this->h_Volume->GetSize(0) * this->interpFactor + this->extraPadding * 2;

    if (this->Num_GPUs > 1)
    {
        for (int i = 0; i < this->Num_GPUs; i++)
        {
            if (this->GPU_Devices[i] != this->GPU_Devices[GPU_For_Reconstruction])
            {
                AddFilter->SetVolumeSize(PlaneDensityVolumeSize);
                AddFilter->SetNumberOfSlices(PlaneDensityVolumeSize);
                AddFilter->SetVolumeOne(gpuGridder_vec[GPU_For_Reconstruction]->GetPlaneDensityPtr());
                AddFilter->SetVolumeTwo(gpuGridder_vec[i]->GetPlaneDensityPtr());
                AddFilter->Update();
                gpuErrorCheck(cudaDeviceSynchronize());
            }
        }
    }

    delete AddFilter;

    // Copy the resulting array to the pinned host memory if the pointer exists
    // if (this->h_PlaneDensity != NULL)
    // {
    //     gpuGridder_vec[GPU_Device]->CopyPlaneDensityToHost();
    // }
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
    // Get the CAS volume off each GPU and sum the arrays together
    // This function is used to get the result after the back projection

    if (this->verbose == true)
    {
        std::cout << "MultiGPUGridder::SumCASVolumes()" << '\n';
    }

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
    }

    // Copy the resulting summed volume to the pinned CPU array (if a pointer was previously provided)
    if (this->h_CASVolume != NULL)
    {
        this->h_CASVolume->CopyArray(SummedVolume);
    }

    // Release the temporary memory
    delete[] SummedVolume;
}

void MultiGPUGridder::SumVolumes()
{
    // Get the volume off each GPU and sum the arrays together

    if (this->verbose == true)
    {
        std::cout << "MultiGPUGridder::SumVolumes()" << '\n';
    }

    // Temporary volume array
    float *SummedVolume = new float[this->h_Volume->length()];

    for (int i = 0; i < this->h_Volume->length(); i++)
    {
        SummedVolume[i] = 0; // TO DO: replace with memset
    }

    for (int i = 0; i < Num_GPUs; i++)
    {
        // Get the volume from the current GPU
        float *tempVolume = gpuGridder_vec[i]->GetVolumeFromDevice();

        // Add the volumes together
        for (int i = 0; i < this->h_Volume->length(); i++)
        {
            SummedVolume[i] = SummedVolume[i] + tempVolume[i];
        }
    }

    // Copy the resulting summed volume to the pinned CPU array (if a pointer was previously provided)
    if (this->h_Volume != NULL)
    {
        this->h_Volume->CopyArray(SummedVolume);
    }

    // Release the temporary memory
    delete[] SummedVolume;
}

void MultiGPUGridder::SumPlaneDensity()
{
    // Get the plane density off each GPU and sum the arrays together
    // This function is used to get the result after the back projection

    if (this->verbose == true)
    {
        std::cout << "MultiGPUGridder::SumPlaneDensity()" << '\n';
    }

    // Temporary volume array
    float *SummedVolume = new float[this->h_PlaneDensity->length()];

    for (int i = 0; i < this->h_PlaneDensity->length(); i++)
    {
        SummedVolume[i] = 0; // TO DO: replace with memset
    }

    for (int i = 0; i < Num_GPUs; i++)
    {
        // Get the volume from the current GPU
        float *tempVolume = gpuGridder_vec[i]->GetPlaneDensityFromDevice();

        std::cout << "this->h_PlaneDensity->CopyArray(SummedVolume):" << '\n';

        // Add the volumes together
        for (int i = 0; i < this->h_PlaneDensity->length(); i++)
        {
            SummedVolume[i] = SummedVolume[i] + tempVolume[i];
        }
    }

    // Copy the resulting summed plane densities to the pinned CPU array (if a pointer was previously provided)
    if (this->h_PlaneDensity != NULL)
    {
        this->h_PlaneDensity->CopyArray(SummedVolume);
    }

    // Release the temporary memory
    delete[] SummedVolume;
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