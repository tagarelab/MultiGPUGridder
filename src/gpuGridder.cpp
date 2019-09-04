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
    gpuFFT::VolumeToCAS(this->Volume->ptr, this->Volume->size[0], this->CASVolume->ptr, this->interpFactor, this->extraPadding);
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
    this->d_CASVolume = new MemoryStructGPU(this->CASVolume->dims, this->CASVolume->size, this->GPU_Device);
    this->d_CASVolume->CopyToGPU(this->CASVolume->ptr, this->CASVolume->bytes());

    // Allocate the CAS images
    this->d_CASImgs = new MemoryStructGPU(this->CASimgs->dims, this->CASimgs->size, this->GPU_Device);
    this->d_CASImgs->CopyToGPU(this->CASimgs->ptr, this->CASimgs->bytes());

    // Allocate the images
    this->d_Imgs = new MemoryStructGPU(this->imgs->dims, this->imgs->size, this->GPU_Device);
    this->d_Imgs->CopyToGPU(this->imgs->ptr, this->imgs->bytes());

    // Allocate the coordinate axes array
    this->d_CoordAxes = new MemoryStructGPU(this->coordAxes->dims, this->coordAxes->size, this->GPU_Device);
    this->d_CoordAxes->CopyToGPU(this->coordAxes->ptr, this->coordAxes->bytes());

    // Allocate the Kaiser bessel lookup table
    this->d_KB_Table = new MemoryStructGPU(this->ker_bessel_Vector->dims, this->ker_bessel_Vector->size, this->GPU_Device);
    this->d_KB_Table->CopyToGPU(this->ker_bessel_Vector->ptr, this->ker_bessel_Vector->bytes());
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
    this->ForwardProject_obj = new gpuForwardProject();

    // Pass the pointer to the MemoryStructGPU to the forward projection object
    this->ForwardProject_obj->SetCASVolume(this->d_CASVolume);
    this->ForwardProject_obj->SetCASImages(this->d_CASImgs);
    this->ForwardProject_obj->SetImages(this->d_Imgs);
    this->ForwardProject_obj->SetCoordinateAxes(this->d_CoordAxes);
    this->ForwardProject_obj->SetKBTable(this->d_KB_Table);

    // Set the various parameters for the forward projection object
    this->ForwardProject_obj->SetGridSize(this->gridSize);
    this->ForwardProject_obj->SetBlockSize(this->blockSize);
    this->ForwardProject_obj->SetNumberOfAxes(this->GetNumAxes());
    this->ForwardProject_obj->SetMaxAxesAllocated(this->MaxAxesAllocated);
    this->ForwardProject_obj->SetNumberOfStreams(this->nStreams);
    this->ForwardProject_obj->SetGPUDevice(this->GPU_Device);
    this->ForwardProject_obj->SetMaskRadius(this->maskRadius);
    
}

void gpuGridder::ForwardProject()
{
    Log("ForwardProject()");

    std::cout << "this->Volume->ptr: " << this->Volume->ptr << '\n';
    std::cout << "this->CASVolume->ptr: " << this->CASVolume->ptr << '\n';
    std::cout << "this->imgs->ptr: " << this->imgs->ptr << '\n';
    std::cout << "this->CASimgs->ptr: " << this->CASimgs->ptr << '\n';
    std::cout << "this->coordAxes->ptr: " << this->coordAxes->ptr << '\n';
    std::cout << "this->ker_bessel_Vector->ptr: " << this->ker_bessel_Vector->ptr << '\n';

    // Do we need to convert the volume, copy to the GPUs, etc?
    // Assume for now that we have a new volume for each call to ForwardProject()
    bool newVolumeFlag = 1;

    if (newVolumeFlag == 1)
    {
        // Initilize the needed arrays on the GPU
        InitilizeGPUArrays();

        // Initilize the forward projection object
        InitilizeForwardProjection();

        // Run the volume to CAS volume function
        VolumeToCASVolume();

        return;
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

    // NOTE: gridSize times blockSize needs to equal CASimgSize
    this->gridSize = 32;
    this->blockSize = ceil(this->CASimgs->size[0] / gridSize);

    // Run the forward projection CUDA kernel
    Log("gpuForwardProjectLaunch()");
    // gpuForwardProjectLaunch(this);

    cudaDeviceSynchronize();
    Log("gpuForwardProjectLaunch() Done");

    return;

    // Note: This modifies the Matlab array in-place
}
