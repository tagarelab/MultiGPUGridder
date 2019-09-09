#include "gpuGridder.h"

#define Log(x)                  \
    {                           \
        std::cout << x << '\n'; \
    }

// gpuGridder::gpuGridder()
// {
//     // Constructor
// }

// gpuGridder::~gpuGridder()
// {
//     // Deconstructor
// }

int gpuGridder::EstimateMaxAxesToAllocate(int VolumeSize, int interpFactor)
{
    Log("EstimateMaxAxesToAllocate()...");

    // Estimate the maximum number of coordinate axes to allocate on the GPU
    cudaSetDevice(this->GPU_Device);
    size_t mem_tot = 0;
    size_t mem_free = 0;
    cudaMemGetInfo(&mem_free, &mem_tot);

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

    Log("mem_free:");
    Log(mem_free);
    Log("Bytes_for_CASVolume:");
    Log(Bytes_for_CASVolume);
    Log("Bytes_per_Img:");
    Log(Bytes_per_Img);
    Log("Bytes_per_CASImg:");
    Log(Bytes_per_CASImg);
    Log("Bytes_per_ComplexCASImg:");
    Log(Bytes_per_ComplexCASImg);
    Log("EstimatedMaxAxes:");
    Log(EstimatedMaxAxes);

    // Leave room on the GPU to run the FFTs and CUDA kernels so only use 30% of the maximum possible
    EstimatedMaxAxes = floor(EstimatedMaxAxes * 0.3);

    Log("EstimatedMaxAxes:");
    Log(EstimatedMaxAxes);

    return EstimatedMaxAxes;
}

void gpuGridder::VolumeToCASVolume()
{
    Log("VolumeToCASVolume()");

    cudaSetDevice(this->GPU_Device);

    // Convert the volume to CAS volume
    gpuFFT::VolumeToCAS(
        this->Volume->GetPointer(),
        this->Volume->GetSize(0),
        this->CASVolume->GetPointer(),
        this->interpFactor,
        this->extraPadding);
}

void gpuGridder::SetGPU(int GPU_Device)
{
    // Set which GPUs to use

    // Check how many GPUs there are on the computer
    int numGPUDetected;
    cudaGetDeviceCount(&numGPUDetected);

    Log("numGPUDetected:");
    Log(numGPUDetected);

    // Check wether the given GPU_Device value is valid
    if (GPU_Device < 0 || GPU_Device >= numGPUDetected) //  An invalid numGPUs selection was chosen
    {
        std::cerr << "GPU_Device number provided " << GPU_Device << '\n';
        std::cerr << "Error in GPU selection. Please provide an integer between 0 and the number of NVIDIA graphic cards on your computer. Use SetNumberGPUs() function." << '\n';
        this->ErrorFlag = 1;
        return;
    }

    this->GPU_Device = GPU_Device;
    Log("GPU Added:");
    Log(GPU_Device);
}

void gpuGridder::InitilizeGPUArrays()
{
    // Initilize the GPU arrays and allocate the needed memory on the GPU

    cudaSetDevice(this->GPU_Device);

    Log("InitilizeGPUArrays()");
    Log("this->MaxAxesToAllocate");
    Log(this->MaxAxesToAllocate);
    Log("this->GPU_Device");
    Log(this->GPU_Device);

    // Allocate the CAS volume
    this->d_CASVolume = new MemoryStructGPU(this->CASVolume->GetDim(), this->CASVolume->GetSize(), this->GPU_Device);
    this->d_CASVolume->CopyToGPU(this->CASVolume->GetPointer(), this->CASVolume->bytes());

    Log("this->CASVolume->GetSize(0)");
    Log(this->CASVolume->GetSize(0));
    Log("this->CASVolume->GetSize(1)");
    Log(this->CASVolume->GetSize(1));
    Log("this->CASVolume->GetSize(2)");
    Log(this->CASVolume->GetSize(2));

    // Allocate the CAS images
    Log("CASImgs");
    if (this->CASimgs != nullptr)
    {
        // The pinned CASImgs was previously created so use its deminsions (i.e. creating CASImgs is optional)
        this->d_CASImgs = new MemoryStructGPU(this->CASimgs->GetDim(), this->CASimgs->GetSize(), this->GPU_Device);
        this->d_CASImgs->CopyToGPU(this->CASimgs->GetPointer(), this->CASimgs->bytes());
    }
    else
    {
        // First, create a dims array of the correct size of d_CASImgs
        int *size = new int[3];
        size[0] = this->imgs->GetSize(0) * this->interpFactor;
        size[1] = this->imgs->GetSize(1) * this->interpFactor;
        size[2] = std::min(this->coordAxes->GetSize(0) / 9, this->MaxAxesToAllocate);

        Log("size");
        Log(size[0]);
        Log(size[1]);
        Log(size[2]);

        this->d_CASImgs = new MemoryStructGPU(3, size, this->GPU_Device);

        delete[] size;
    }

    // Allocate the complex CAS images array
    int CASImgLength = this->d_CASImgs->length();
    cudaMalloc(&this->d_CASImgsComplex, sizeof(cufftComplex) * CASImgLength);

    Log("d_Imgs");
    // Limit the number of axes to allocate to be MaxAxesToAllocate
    int *imgs_size = new int[3];
    imgs_size[0] = this->imgs->GetSize(0);
    imgs_size[1] = this->imgs->GetSize(1);
    imgs_size[2] = std::min(this->coordAxes->GetSize(0) / 9, this->MaxAxesToAllocate); // 9 elements per coordinate axes

    Log("imgs_size[0]");
    Log(imgs_size[0]);
    Log("imgs_size[1]");
    Log(imgs_size[1]);
    Log("imgs_size[2]");
    Log(imgs_size[2]);

    // Allocate the images
    this->d_Imgs = new MemoryStructGPU(this->imgs->GetDim(), imgs_size, this->GPU_Device);
    delete[] imgs_size;

    Log("d_CoordAxes");
    // Allocate the coordinate axes array
    this->d_CoordAxes = new MemoryStructGPU(this->coordAxes->GetDim(), this->coordAxes->GetSize(), this->GPU_Device);

    // Allocate the Kaiser bessel lookup table
    this->d_KB_Table = new MemoryStructGPU(this->ker_bessel_Vector->GetDim(), this->ker_bessel_Vector->GetSize(), this->GPU_Device);
    this->d_KB_Table->CopyToGPU(this->ker_bessel_Vector->GetPointer(), this->ker_bessel_Vector->bytes());
}

void gpuGridder::InitilizeCUDAStreams()
{
    cudaSetDevice(this->GPU_Device); // needed?

    // Create the CUDA streams
    this->streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * this->nStreams);

    for (int i = 0; i < this->nStreams; i++) // Loop through the streams
    {
        cudaStreamCreate(&this->streams[i]);
    }
}

// Initilize the forward projection object
void gpuGridder::InitilizeForwardProjection()
{
    Log("InitilizeForwardProjection()");
    cudaSetDevice(this->GPU_Device);

    this->ForwardProject_obj = new gpuForwardProject();

    // Pass the float pointers to the forward projection object
    float *coordAxesPtr = this->coordAxes->GetPointer();
    float *ImgsPtr = this->imgs->GetPointer();
    this->ForwardProject_obj->SetPinnedCoordinateAxes(coordAxesPtr);
    this->ForwardProject_obj->SetPinnedImages(ImgsPtr);

    // Set the CASImgs pointer if it was previously allocated (i.e. this is optional)
    if (this->CASimgs != nullptr)
    {
        float *CASImagesPtr = this->CASimgs->GetPointer();
        this->ForwardProject_obj->SetPinnedCASImages(CASImagesPtr);
    }
    else
    {
        // Set the pinned CASImgs pointer to be the null pointer
        float *tempPtr = nullptr;
        this->ForwardProject_obj->SetPinnedCASImages(tempPtr);
    }

    // Pass the pointer to the MemoryStructGPU to the forward projection object
    this->ForwardProject_obj->SetCASVolume(this->d_CASVolume);
    this->ForwardProject_obj->SetImages(this->d_Imgs);
    this->ForwardProject_obj->SetCoordinateAxes(this->d_CoordAxes);
    this->ForwardProject_obj->SetCoordinateAxesOffset(0); // Default is no offset
    this->ForwardProject_obj->SetKBTable(this->d_KB_Table);

    // Pass the pointer to the array of CUDA streams
    this->ForwardProject_obj->SetCUDAStreams(this->streams);

    // Set the various parameters for the forward projection object
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

    // Set the initilization flag to true
    this->FP_initilized = true;
}

void gpuGridder::ForwardProject()
{
    Log("ForwardProject()");

    // Run the forward projection on all the coordinate axes with no offset
    ForwardProject(0, this->GetNumAxes());

    return;
}

void gpuGridder::ForwardProject(int AxesOffset, int nAxesToProcess)
{
    // Run the forward projection on some subset of the coordinate axes (needed when using multiple GPUs)
    Log("ForwardProject(int AxesOffset, int nAxesToProcess)");
    cudaSetDevice(this->GPU_Device);

    // this->newVolumeFlag = true;
    this->FP_initilized = false;

    // NOTE: gridSize times blockSize needs to equal CASimgSize
    this->gridSize = 32;
    this->blockSize = ceil((this->imgs->GetSize(0) * this->interpFactor) / this->gridSize);

    // Has the forward projection been initilized?
    if (this->FP_initilized == false)
    {
        // Initilize the needed arrays on the GPU
        InitilizeGPUArrays();

        // Initilize the CUDA streams
        InitilizeCUDAStreams();

        // Initilize the forward projection object
        InitilizeForwardProjection();

        // Set the coordinate axes offset ( in number of coordinate axes from the beginning of the pinned CPU coordinate axes array)
        this->ForwardProject_obj->SetCoordinateAxesOffset(AxesOffset);

        // Set the number of axes to process
        this->ForwardProject_obj->SetNumberOfAxes(nAxesToProcess);

        // Reset the flag
        this->FP_initilized = true;
    }

    // Do we have a new volume? If so, run the volume to CAS volume function
    // Assume for now that we have a new volume for each call to ForwardProject()
    if (this->newVolumeFlag == true)
    {
        // Run the volume to CAS volume function
        VolumeToCASVolume();

        // Reset the flag
        // this->newVolumeFlag = false;
    }

    // Check the error flags to see if we had any issues during the initilization
    if (this->ErrorFlag == 1 ||
        this->d_CASVolume->ErrorFlag == 1 ||
        this->d_CASImgs->ErrorFlag == 1 ||
        this->d_Imgs->ErrorFlag == 1 ||
        this->d_CoordAxes->ErrorFlag == 1 ||
        this->d_KB_Table->ErrorFlag == 1)
    {
        std::cerr << "Error during intilization." << '\n';
        return; // Don't run the kernel and return
    }

    // Copy the CAS volume to the corresponding GPU array
    this->d_CASVolume->CopyToGPU(this->CASVolume->GetPointer(), this->CASVolume->bytes());

    // Run the forward projection CUDA kernel
    Log("gpuForwardProjectLaunch()");
    this->ForwardProject_obj->Execute();

    // cudaDeviceSynchronize();
    Log("gpuForwardProjectLaunch() Done");

    return;
}
