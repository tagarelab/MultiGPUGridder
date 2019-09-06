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

void gpuGridder::VolumeToCASVolume()
{
    Log("VolumeToCASVolume()");

    // Convert the volume to CAS volume
    gpuFFT::VolumeToCAS(
        this->Volume->GetPointer(),
        this->Volume->GetSize(0),
        this->CASVolume->GetPointer(),
        this->interpFactor,
        this->extraPadding);

    // Copy the resulting CAS volume to the corresponding GPU array
    this->d_CASVolume->CopyToGPU(this->CASVolume->GetPointer(), this->CASVolume->bytes());
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
    Log("InitilizeGPUArrays()");

    // Allocate the CAS volume
    this->d_CASVolume = new MemoryStructGPU(this->CASVolume->GetDim(), this->CASVolume->GetSize(), this->GPU_Device);
    this->d_CASVolume->CopyToGPU(this->CASVolume->GetPointer(), this->CASVolume->bytes());

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
        size[2] = std::min(this->coordAxes->GetSize(0) / 9, this->MaxGPUAxesToAllocate) * this->interpFactor;

        Log("size");
        Log(size[0]);
        Log(size[1]);
        Log(size[2]);

        this->d_CASImgs = new MemoryStructGPU(3, size, this->GPU_Device);

        delete[] size;
    }

    Log("d_Imgs");
    // Allocate the images
    // this->d_Imgs = new MemoryStructGPU(this->imgs->GetDim(), this->imgs->GetSize(), this->GPU_Device);
    // this->d_Imgs->CopyToGPU(this->imgs->GetPointer(), this->imgs->bytes());

    // Limit the number of axes to allocate to be MaxGPUAxesToAllocate
    int *imgs_size = new int[3];
    imgs_size[0] = this->imgs->GetSize(0);
    imgs_size[1] = this->imgs->GetSize(1);
    imgs_size[2] = std::min(this->coordAxes->GetSize(0) / 9, this->MaxGPUAxesToAllocate); // 9 elements per coordinate axes

    // Allocate the images
    this->d_Imgs = new MemoryStructGPU(this->imgs->GetDim(), imgs_size, this->GPU_Device);
    delete[] imgs_size;


    Log("d_CoordAxes");
    // Allocate the coordinate axes array
    this->d_CoordAxes = new MemoryStructGPU(this->coordAxes->GetDim(), this->coordAxes->GetSize(), this->GPU_Device);
    // this->d_CoordAxes->CopyToGPU(this->coordAxes->GetPointer(), this->coordAxes->bytes());

    // Allocate the Kaiser bessel lookup table
    this->d_KB_Table = new MemoryStructGPU(this->ker_bessel_Vector->GetDim(), this->ker_bessel_Vector->GetSize(), this->GPU_Device);
    this->d_KB_Table->CopyToGPU(this->ker_bessel_Vector->GetPointer(), this->ker_bessel_Vector->bytes());
}

void gpuGridder::SetVolume(float *Volume, int *ArraySize)
{
    Log("SetVolume()");

    // First save the given pointer
    this->Volume = new MemoryStruct(3, ArraySize);
    this->Volume->CopyPointer(Volume);

    // Next, pin the volume to host (i.e. CPU) memory in order to enable the async CUDA stream copying
    // This will let us copy the volume to all GPUs at the same time
    this->Volume->PinArray();
}

// Initilize the forward projection object
void gpuGridder::InitilizeForwardProjection()
{
    Log("InitilizeForwardProjection()");

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
    } else 
    {
        // Set the pinned CASImgs pointer to be the null pointer
        float * tempPtr = nullptr;
        this->ForwardProject_obj->SetPinnedCASImages(tempPtr);
    }

    // Pass the pointer to the MemoryStructGPU to the forward projection object
    this->ForwardProject_obj->SetCASVolume(this->d_CASVolume);
    this->ForwardProject_obj->SetImages(this->d_Imgs);
    this->ForwardProject_obj->SetCoordinateAxes(this->d_CoordAxes);
    this->ForwardProject_obj->SetKBTable(this->d_KB_Table);

    // Set the various parameters for the forward projection object
    this->ForwardProject_obj->SetGridSize(this->gridSize);
    this->ForwardProject_obj->SetBlockSize(this->blockSize);
    this->ForwardProject_obj->SetNumberOfAxes(this->GetNumAxes());
    this->ForwardProject_obj->SetMaxAxesAllocated(this->MaxGPUAxesToAllocate);
    this->ForwardProject_obj->SetNumberOfStreams(this->nStreams);
    this->ForwardProject_obj->SetGPUDevice(this->GPU_Device);
    this->ForwardProject_obj->SetMaskRadius(this->maskRadius);
    this->ForwardProject_obj->SetKerHWidth(this->kerHWidth);
    this->ForwardProject_obj->SetCASImages(this->d_CASImgs);
}

void gpuGridder::ForwardProject()
{
    Log("ForwardProject()");

    this->newVolumeFlag = true;
    this->FP_initilized = false;

    // NOTE: gridSize times blockSize needs to equal CASimgSize
    this->gridSize = 32;
    this->blockSize = ceil((this->imgs->GetSize(0) * this->interpFactor) / this->gridSize);

    // Has the forward projection been initilized?
    if (this->FP_initilized == false)
    {
        // Initilize the needed arrays on the GPU
        InitilizeGPUArrays();

        // Initilize the forward projection object
        InitilizeForwardProjection();

        // Reset the flag
        this->FP_initilized = true;
    }

    // Do we have a new volume? If so, run the volume to CAS volume function
    // Assume for now that we have a new volume for each call to ForwardProject()
    if (this->newVolumeFlag == true)
    {
        cudaDeviceSynchronize(); // needed?

        // Run the volume to CAS volume function
        VolumeToCASVolume();

        // Reset the flag
        this->newVolumeFlag = false;
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

    // Synchronize before running the kernel
    cudaDeviceSynchronize(); // needed?

    // Run the forward projection CUDA kernel
    Log("gpuForwardProjectLaunch()");
    this->ForwardProject_obj->Execute();

    cudaDeviceSynchronize();
    Log("gpuForwardProjectLaunch() Done");

    return;

    // Note: This modifies the Matlab array in-place
}
