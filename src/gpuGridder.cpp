#include "gpuGridder.h"

#define Log(x)                  \
    {                           \
        std::cout << x << '\n'; \
    }

int gpuGridder::EstimateMaxAxesToAllocate(int VolumeSize, int interpFactor)
{
    // Estimate the maximum number of coordinate axes to allocate on the GPU
    cudaSetDevice(this->GPU_Device);
    size_t mem_tot = 0;
    size_t mem_free = 0;
    cudaMemGetInfo(&mem_free, &mem_tot);

    // Throw error if mem_free is zero
    if (mem_free <= 0)
    {
        std::cerr << "No free memory on GPU " << this->GPU_Device << '\n';
        this->ErrorFlag = 1;
        return -1;
    }

    // Estimate how many bytes of memory is needed to process each coordinate axe
    int CASImg_Length = (VolumeSize * interpFactor + this->extraPadding * 2) * (VolumeSize * interpFactor + this->extraPadding * 2);
    int Img_Length = (VolumeSize * interpFactor) * (VolumeSize * interpFactor);
    int Bytes_per_Img = Img_Length * sizeof(float);
    int Bytes_per_CASImg = CASImg_Length * sizeof(float);
    int Bytes_per_ComplexCASImg = CASImg_Length * sizeof(cufftComplex);
    int Bytes_for_CASVolume = pow((VolumeSize * interpFactor + this->extraPadding * 2), 3) * sizeof(float);
    int Bytes_for_CoordAxes = 9 * sizeof(float); // 9 elements per axes

    // How many coordinate axes would fit in the remaining free GPU memory?
    int EstimatedMaxAxes = (mem_free - Bytes_for_CASVolume) / (Bytes_per_Img + Bytes_per_CASImg + Bytes_per_ComplexCASImg + Bytes_for_CoordAxes);

    // Leave room on the GPU to run the FFTs and CUDA kernels so only use 30% of the maximum possible
    EstimatedMaxAxes = floor(EstimatedMaxAxes * 0.3);

    return EstimatedMaxAxes;
}

float *gpuGridder::GetVolumeFromDevice()
{
    float *Volume = new float[this->d_Volume->length()];
    this->d_Volume->CopyFromGPU(Volume, this->d_Volume->bytes());

    return Volume;
}

float *gpuGridder::GetCASVolumeFromDevice()
{
    float *CASVolume = new float[this->d_CASVolume->length()];
    this->d_CASVolume->CopyFromGPU(CASVolume, this->d_CASVolume->bytes());

    return CASVolume;
}

float *gpuGridder::GetPlaneDensityFromDevice()
{
    float *PlaneDensity = new float[this->d_PlaneDensity->length()];
    this->d_PlaneDensity->CopyFromGPU(PlaneDensity, this->d_PlaneDensity->bytes());

    return PlaneDensity;
}

void gpuGridder::VolumeToCASVolume()
{
    cudaSetDevice(this->GPU_Device);

    // Convert the volume to CAS volume
    gpuFFT::VolumeToCAS(
        this->h_Volume->GetPointer(),
        this->h_Volume->GetSize(0),
        this->h_CASVolume->GetPointer(),
        this->interpFactor,
        this->extraPadding);
}

void gpuGridder::CopyCASVolumeToGPUAsyc()
{
    // Copy the CAS volume to the GPU asynchronously
    this->d_CASVolume->CopyToGPUAsyc(this->h_CASVolume->GetPointer(), this->h_CASVolume->bytes());
}

void gpuGridder::SetGPU(int GPU_Device)
{
    // Set which GPU to use

    // Check how many GPUs there are on the computer
    int numGPUDetected;
    cudaGetDeviceCount(&numGPUDetected);

    // Check wether the given GPU_Device value is valid
    if (GPU_Device < 0 || GPU_Device >= numGPUDetected) //  An invalid numGPUs selection was chosen
    {
        std::cerr << "GPU_Device number provided " << GPU_Device << '\n';
        std::cerr << "Error in GPU selection. Please provide an integer between 0 and the number of NVIDIA graphic cards on your computer. Use SetNumberGPUs() function." << '\n';
        this->ErrorFlag = 1;
        return;
    }

    this->GPU_Device = GPU_Device;
}

void gpuGridder::InitilizeGPUArrays()
{
    // Initilize the GPU arrays and allocate the needed memory on the GPU
    cudaSetDevice(this->GPU_Device);

    if (this->MaxAxesToAllocate == 0)
    {
        std::cerr << "MaxAxesToAllocate must be a positive integer. Please run EstimateMaxAxesToAllocate() first. " << '\n';
        this->ErrorFlag = 1;
        return;
    }

    // Allocate the volume
    this->d_Volume = new MemoryStructGPU<float>(this->h_Volume->GetDim(), this->h_Volume->GetSize(), this->GPU_Device);

    // Allocate the CAS volume
    this->d_CASVolume = new MemoryStructGPU<float>(this->h_CASVolume->GetDim(), this->h_CASVolume->GetSize(), this->GPU_Device);
    this->d_CASVolume->AllocateGPUArray();

    // Allocate the plane density array (for the back projection)
    this->d_PlaneDensity = new MemoryStructGPU<float>(this->h_CASVolume->GetDim(), this->h_CASVolume->GetSize(), this->GPU_Device);
    this->d_PlaneDensity->AllocateGPUArray();

    // Allocate the CAS images
    if (this->h_CASImgs != nullptr)
    {
        // The pinned CASImgs was previously created so use its deminsions (i.e. creating CASImgs is optional)
        this->d_CASImgs = new MemoryStructGPU<float>(this->h_CASImgs->GetDim(), this->h_CASImgs->GetSize(), this->GPU_Device);
        this->d_CASImgs->AllocateGPUArray();
        // this->d_CASImgs->CopyToGPUAsyc(this->h_CASImgs->GetPointer(), this->h_CASImgs->bytes());
    }
    else
    {
        // First, create a dims array of the correct size of d_CASImgs
        int *size = new int[3];
        size[0] = this->h_Imgs->GetSize(0) * this->interpFactor;
        size[1] = this->h_Imgs->GetSize(1) * this->interpFactor;
        size[2] = std::min(this->GetNumAxes(), this->MaxAxesToAllocate);

        this->d_CASImgs = new MemoryStructGPU<float>(3, size, this->GPU_Device);
        this->d_CASImgs->AllocateGPUArray();
        delete[] size;
    }

    // Allocate the complex CAS images array
    this->d_CASImgsComplex = new MemoryStructGPU<cufftComplex>(this->h_Imgs->GetDim(), this->d_CASImgs->GetSize(), this->GPU_Device);
    this->d_CASImgsComplex->AllocateGPUArray();

    // Limit the number of axes to allocate to be MaxAxesToAllocate
    int *imgs_size = new int[3];
    imgs_size[0] = this->h_Imgs->GetSize(0);
    imgs_size[1] = this->h_Imgs->GetSize(1);
    imgs_size[2] = std::min(this->GetNumAxes(), this->MaxAxesToAllocate);

    // Allocate the images
    this->d_Imgs = new MemoryStructGPU<float>(this->h_Imgs->GetDim(), imgs_size, this->GPU_Device);
    this->d_Imgs->AllocateGPUArray();
    delete[] imgs_size;

    // Allocate the coordinate axes array
    int *axes_size = new int[1];
    axes_size[0] = std::min(this->GetNumAxes(), this->MaxAxesToAllocate);
    axes_size[0] = axes_size[0] * 9; // 9 elements per coordinate axes

    this->d_CoordAxes = new MemoryStructGPU<float>(this->h_CoordAxes->GetDim(), this->h_CoordAxes->GetSize(), this->GPU_Device);
    this->d_CoordAxes->AllocateGPUArray();
    delete[] axes_size;

    // Allocate the Kaiser bessel lookup table
    this->d_KB_Table = new MemoryStructGPU<float>(this->h_KB_Table->GetDim(), this->h_KB_Table->GetSize(), this->GPU_Device);
    this->d_KB_Table->AllocateGPUArray();
    this->d_KB_Table->CopyToGPUAsyc(this->h_KB_Table->GetPointer(), this->h_KB_Table->bytes());
}

void gpuGridder::InitilizeCUDAStreams()
{
    cudaSetDevice(this->GPU_Device);

    // Create the CUDA streams
    this->streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * this->nStreams);

    for (int i = 0; i < this->nStreams; i++) // Loop through the streams
    {
        cudaStreamCreate(&this->streams[i]);
    }
}

void gpuGridder::Allocate()
{
    // Allocate the needed GPU memory

    // Have the GPU arrays already been allocated?
    if (this->GPUArraysAllocatedFlag == false)
    {
        // Estimate the maximum number of coordinate axes to allocate per stream
        this->MaxAxesToAllocate = EstimateMaxAxesToAllocate(this->h_Volume->GetSize(0), this->interpFactor);

        // Initilize the needed arrays on the GPU
        InitilizeGPUArrays();

        // Initilize the CUDA streams
        InitilizeCUDAStreams();

        // Create a forward projection object
        this->ForwardProject_obj = new gpuForwardProject();

        // Create a back projection object
        this->BackProject_obj = new gpuBackProject();

        this->GPUArraysAllocatedFlag = true;
    }
}

// Initilize the forward projection object
void gpuGridder::InitilizeForwardProjection(int AxesOffset, int nAxesToProcess)
{
    // Log("InitilizeForwardProjection()");
    cudaSetDevice(this->GPU_Device);

    // Have the GPU arrays been allocated?
    if (this->GPUArraysAllocatedFlag == false)
    {
        Allocate();
        this->GPUArraysAllocatedFlag = true;
    }

    // Pass the float pointers to the forward projection object
    this->ForwardProject_obj->SetPinnedCoordinateAxes(this->h_CoordAxes);
    this->ForwardProject_obj->SetPinnedImages(this->h_Imgs);

    // Set the CASImgs pointer if it was previously allocated (i.e. this is optional)
    // if (this->CASimgs != nullptr)
    // {
    this->ForwardProject_obj->SetPinnedCASImages(this->h_CASImgs);
    // }

    // Calculate the block size for running the CUDA kernels
    // NOTE: gridSize times blockSize needs to equal CASimgSize
    this->gridSize = 32;
    this->blockSize = ceil((this->h_Imgs->GetSize(0) * this->interpFactor) / this->gridSize);

    // Pass the pointers and parameters to the forward projection object
    this->ForwardProject_obj->SetCASVolume(this->d_CASVolume);
    this->ForwardProject_obj->SetImages(this->d_Imgs);
    this->ForwardProject_obj->SetCoordinateAxes(this->d_CoordAxes);
    this->ForwardProject_obj->SetKBTable(this->d_KB_Table);
    this->ForwardProject_obj->SetCUDAStreams(this->streams);
    this->ForwardProject_obj->SetGridSize(this->gridSize);
    this->ForwardProject_obj->SetBlockSize(this->blockSize);
    this->ForwardProject_obj->SetNumberOfAxes(this->GetNumAxes());
    this->ForwardProject_obj->SetMaxAxesAllocated(this->MaxAxesToAllocate);
    this->ForwardProject_obj->SetNumberOfStreams(this->nStreams);
    this->ForwardProject_obj->SetGPUDevice(this->GPU_Device);
    this->ForwardProject_obj->SetMaskRadius(this->maskRadius);
    this->ForwardProject_obj->SetKerHWidth(this->kerHWidth);
    this->ForwardProject_obj->SetCASImages(this->d_CASImgs);
    this->ForwardProject_obj->SetComplexCASImages(this->d_CASImgsComplex);
    this->ForwardProject_obj->SetCoordinateAxesOffset(AxesOffset); // Offset in number of coordinate axes from the beginning of the CPU array
    this->ForwardProject_obj->SetNumberOfAxes(nAxesToProcess);     // Number of axes to process
}

// Initilize the forward back object
void gpuGridder::InitilizeBackProjection(int AxesOffset, int nAxesToProcess)
{

    cudaSetDevice(this->GPU_Device);

    // Have the GPU arrays been allocated?
    if (this->GPUArraysAllocatedFlag == false)
    {
        Allocate();
        this->GPUArraysAllocatedFlag = true;
    }

    // Pass the float pointers to the forward projection object
    this->BackProject_obj->SetPinnedCoordinateAxes(this->h_CoordAxes);

    this->BackProject_obj->SetPinnedImages(this->h_Imgs);

    // Set the CASImgs pointer if it was previously allocated (i.e. this is optional)
    // if (this->h_CASImgs != nullptr)
    // {

    this->BackProject_obj->SetPinnedCASImages(this->h_CASImgs);
    // }

    // Calculate the block size for running the CUDA kernels
    // NOTE: gridSize times blockSize needs to equal CASimgSize
    this->gridSize = 32;
    this->blockSize = ceil((this->h_Imgs->GetSize(0) * this->interpFactor) / this->gridSize);

    // Pass the pointers and parameters to the forward projection object
    this->BackProject_obj->SetCASVolume(this->d_CASVolume);
    this->BackProject_obj->SetPlaneDensity(this->d_PlaneDensity);
    this->BackProject_obj->SetImages(this->d_Imgs);
    this->BackProject_obj->SetCoordinateAxes(this->d_CoordAxes);
    this->BackProject_obj->SetKBTable(this->d_KB_Table);
    this->BackProject_obj->SetCUDAStreams(this->streams);
    this->BackProject_obj->SetGridSize(this->gridSize);
    this->BackProject_obj->SetBlockSize(this->blockSize);
    this->BackProject_obj->SetNumberOfAxes(this->GetNumAxes());
    this->BackProject_obj->SetMaxAxesAllocated(this->MaxAxesToAllocate);
    this->BackProject_obj->SetNumberOfStreams(this->nStreams);
    this->BackProject_obj->SetGPUDevice(this->GPU_Device);
    this->BackProject_obj->SetMaskRadius(this->maskRadius);
    this->BackProject_obj->SetKerHWidth(this->kerHWidth);
    this->BackProject_obj->SetCASImages(this->d_CASImgs);
    this->BackProject_obj->SetComplexCASImages(this->d_CASImgsComplex);
    this->BackProject_obj->SetCoordinateAxesOffset(AxesOffset); // Offset in number of coordinate axes from the beginning of the CPU array
    this->BackProject_obj->SetNumberOfAxes(nAxesToProcess);     // Number of axes to process
}


void gpuGridder::PrintMemoryAvailable()
{
    // Check to make sure the GPU has enough available memory left
    size_t mem_tot_0 = 0;
    size_t mem_free_0 = 0;
    cudaMemGetInfo(&mem_free_0, &mem_tot_0);
    std::cout << "Memory remaining on GPU " << this->GPU_Device << " " << mem_free_0 << " out of " << mem_tot_0 << '\n';
}

void gpuGridder::ForwardProject(int AxesOffset, int nAxesToProcess)
{
    // Run the forward projection on some subset of the coordinate axes (needed when using multiple GPUs)
    cudaSetDevice(this->GPU_Device);

    // Have the GPU arrays been allocated?
    if (this->GPUArraysAllocatedFlag == false)
    {
        Allocate();

        this->GPUArraysAllocatedFlag = true;
    }

    // PrintMemoryAvailable();

    // Initilize the forward projection object
    InitilizeForwardProjection(AxesOffset, nAxesToProcess);

    // Do we need to run Volume to CASVolume? (Can skip if using multiple GPUs for example)
    if (this->VolumeToCASVolumeFlag == false)
    {
        // Run the volume to CAS volume function
        // VolumeToCASVolume();
    }

    // Copy the CAS volume to the corresponding GPU array
    this->d_CASVolume->CopyToGPU(this->h_CASVolume->GetPointer(), this->h_CASVolume->bytes());

    // Check the error flags to see if we had any issues during the initilization
    if (this->ErrorFlag == 1 ||
        this->d_CASVolume->GetErrorFlag() == 1 ||
        this->d_CASImgs->GetErrorFlag() == 1 ||
        this->d_Imgs->GetErrorFlag() == 1 ||
        this->d_CoordAxes->GetErrorFlag() == 1 ||
        this->d_KB_Table->GetErrorFlag() == 1)
    {
        std::cerr << "Error during intilization." << '\n';
        return; // Don't run the kernel and return
    }

    // Run the forward projection CUDA kernel
    cudaSetDevice(this->GPU_Device);
    this->ForwardProject_obj->Execute();
}

void gpuGridder::BackProject(int AxesOffset, int nAxesToProcess)
{
    // Run the forward projection on some subset of the coordinate axes (needed when using multiple GPUs)
    cudaSetDevice(this->GPU_Device);

    // Have the GPU arrays been allocated?
    if (this->GPUArraysAllocatedFlag == false)
    {
        Allocate();

        this->GPUArraysAllocatedFlag = true;
    }

    // PrintMemoryAvailable();

    // Initilize the back projection object
    InitilizeBackProjection(AxesOffset, nAxesToProcess);

    // Do we need to run Volume to CASVolume? (Can skip if using multiple GPUs for example)
    // if (this->VolumeToCASVolumeFlag == true)
    // {
    //     // Run the volume to CAS volume function
    //     VolumeToCASVolume();
    // }

    // Copy the CAS volume to the corresponding GPU array
    this->d_CASVolume->CopyToGPU(this->h_CASVolume->GetPointer(), this->h_CASVolume->bytes());
    // this->d_CASImgs->CopyToGPU(this->CASimgs->GetPointer(), this->CASimgs->bytes());

    // Check the error flags to see if we had any issues during the initilization
    if (this->ErrorFlag == 1 ||
        this->d_CASVolume->GetErrorFlag() == 1 ||
        this->d_CASImgs->GetErrorFlag() == 1 ||
        this->d_Imgs->GetErrorFlag() == 1 ||
        this->d_CoordAxes->GetErrorFlag() == 1 ||
        this->d_KB_Table->GetErrorFlag() == 1)
    {
        std::cerr << "Error during intilization." << '\n';
        return; // Don't run the kernel and return
    }

    // Run the back projection CUDA kernel
    this->BackProject_obj->Execute();
}

void gpuGridder::FreeMemory()
{
    // Free all of the allocated memory
    std::cout << "gpuGridder FreeMemory()" << '\n';

    // Free the GPU memory
    // this->d_Imgs->DeallocateGPUArray();
    // this->d_CASImgs->DeallocateGPUArray();
    // this->d_KB_Table->DeallocateGPUArray();
    // this->d_CASVolume->DeallocateGPUArray();
    // this->d_CoordAxes->DeallocateGPUArray();
    // this->d_CASImgsComplex->DeallocateGPUArray();

    // Reset the GPU
    cudaSetDevice(this->GPU_Device);
    cudaDeviceReset();
}