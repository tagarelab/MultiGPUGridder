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

        // (2): Run the volume to CAS volume function
        VolumeToCASVolume();

        cudaDeviceSynchronize(); // needed?

        // (3): Initilize the needed arrays on the GPU
        InitilizeGPUArrays();

        cudaDeviceSynchronize(); // needed?

        return;
    }

    // Check the error flag to see if we had any issues during the initilization
    if (this->ErrorFlag != 0)
    {
        std::cerr << "Error during intilization." << '\n';
        return; // Don't run the kernel and return
    }

    // Synchronize all of the CUDA streams before running the kernel
    // TO DO: cudaEventSyncronize() may be faster than cudaDeviceSynchronize()
    cudaDeviceSynchronize();

    // NOTE: gridSize times blockSize needs to equal CASimgSize
    this->gridSize = 32;
    this->blockSize = ceil(this->CASimgs->size[0] / gridSize);

    // Run the forward projection CUDA kernel
    Log("gpuForwardProjectLaunch()");
    gpuForwardProjectLaunch(this);

    cudaDeviceSynchronize();
    Log("gpuForwardProjectLaunch() Done");

    return;

    // Note: This modifies the Matlab array in-place
}
