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

    // Save the volume size of the CAS volume
    // Example: A volume size of 128, interp factor 2, extra padding of 3 would give -> 262 CAS volume size
    this->CASVolumeSize = this->VolumeSize[0] * this->interpFactor + this->extraPadding * 2;

    // Also save the CAS image size
    int CASimgSize[3];
    CASimgSize[0] = this->CASVolumeSize;
    CASimgSize[1] = this->CASVolumeSize;
    CASimgSize[2] = this->numCoordAxes;

    std::cout << "this->CASVolumeSize: " << this->CASVolumeSize << '\n';
    std::cout << "this->VolumeSize[0]: " << this->VolumeSize[0] << '\n';
    std::cout << "this->interpFactor: " << this->interpFactor << '\n';
    std::cout << "this->extraPadding: " << this->extraPadding << '\n';
    std::cout << "CASVolume: ";

    this->SetCASImageSize(CASimgSize);

    // Convert the volume to CAS volume 
    // this->CASVolume = new float[this->CASVolumeSize * this->CASVolumeSize * this->CASVolumeSize];
    gpuFFT::VolumeToCAS(this->Volume, this->VolumeSize[0], this->CASVolume, this->interpFactor, this->extraPadding);

    // DEBUG
    for (int i = 0; i < 10; i++)
    {
        std::cout << this->CASVolume[i] << " ";
    }
    std::cout << '\n';
    // END DEBUG

}

void gpuGridder::AllocateGPUArray(int GPU_Device, float *d_Ptr, int ArraySize)
{
    // Set the current GPU
    cudaSetDevice(GPU_Device);

    // Check to make sure the GPU has enough available memory left
    size_t mem_tot_0 = 0;
    size_t mem_free_0 = 0;
    cudaMemGetInfo(&mem_free_0, &mem_tot_0);

    // Is there enough available memory on the device to allocate this array?
    if (mem_free_0 < sizeof(float) * (ArraySize))
    {
        std::cerr << "Not enough memory on the device to allocate the requested memory. Try fewer number of projections or a smaller volume. Or increase SetNumberBatches()" << '\n';

        d_Ptr = NULL; // Set the pointer to NULL

        this->ErrorFlag = 1; // Set the error flag to 1 to remember that this failed
    }
    else
    {
        // There is enough memory left on the current GPU
        cudaMalloc(&d_Ptr, sizeof(float) * (ArraySize));
    }
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

    Log("GPU_Device");
    Log(this->GPU_Device);

    Log("this->CASVolumeSize");
    Log(this->CASVolumeSize);

    cudaSetDevice(this->GPU_Device);

    // Allocate the CAS volume
    AllocateGPUArray(this->GPU_Device, d_CASVolume, this->CASVolumeSize * this->CASVolumeSize * this->CASVolumeSize);

    // Allocate the CAS images
    AllocateGPUArray(this->GPU_Device, d_CASImgs, this->CASimgSize[0] * this->CASimgSize[1] * this->CASimgSize[2]);

    // Allocate the coordinate axes array
    AllocateGPUArray(this->GPU_Device, d_CoordAxes, this->numCoordAxes * 9); // 9 float elements per cordinate axes

    // Allocate the Kaiser bessel lookup table
    AllocateGPUArray(this->GPU_Device, d_KB_Table, this->kerSize);
}

void gpuGridder::SetVolume(float *Volume)
{
    Log("SetVolume()");

    // First save the given pointer
    this->Volume = Volume;

    // Next, pin the volume to host (i.e. CPU) memory in order to enable the async CUDA stream copying
    // This will let us copy the volume to all GPUs at the same time
    this->VolumeBytes = sizeof(float) * this->VolumeSize[0] * this->VolumeSize[1] * this->VolumeSize[2];
    cudaHostRegister(this->Volume, this->VolumeBytes, 0);
}

void gpuGridder::CopyVolumeToGPU()
{
    // Copy the volume to the GPUs (the volume is already pinned to CPU memory during SetVolume())
    Log("CopyVolumeToGPU()");

    // Set the current GPU device
    cudaSetDevice(this->GPU_Device);

    // Sends data to device asynchronously
    // TO DO: Input a stream to use instead of just the first one?
    cudaMemcpyAsync(this->d_CASVolume, this->CASVolume, VolumeBytes, cudaMemcpyHostToDevice, this->streams[0]);
}

void gpuGridder::CreateCUDAStreams()
{
    // Create the CUDA streams to usefor async memory transfers and for running the kernels
    if (this->nStreams >= 1)
    {
        this->streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * this->nStreams);
    }
    else
    {
        std::cerr << "Failed to create CUDA streams. The number of streams must be a positive integer." << '\n';
        this->ErrorFlag = 1;
    }
}

void gpuGridder::DestroyCUDAStreams()
{
    // Destroy the streams
    for (int i = 0; i < this->nStreams; i++)
    {
        cudaStreamDestroy(this->streams[i]);
    }
}


void gpuGridder::ForwardProject()
{
    Log("ForwardProject()");

    // Do we need to convert the volume, copy to the GPUs, etc?
    // Assume for now that we have a new volume for each call to ForwardProject()
    bool newVolumeFlag = 1;

    if (newVolumeFlag == 1)
    {
        // (1): Create the CUDA streams
        this->nStreams = 4;

        CreateCUDAStreams();

        // (2): Run the volume to CAS volume function
        VolumeToCASVolume();

        // (3): Initilize the needed arrays on each GPU
        // InitilizeGPUArrays();



        return; // debug

        // (4): Copy the CASVolume to each of the GPUs
        // TO DO: might need to run device sync here?
        CopyVolumeToGPU();
    }

    // Check the error flag to see if we had any issues during the initilization
    if (this->ErrorFlag != 0)
    {
        std::cerr << "Error during intilization." << '\n';
        return; // Don't run the kernel and return
    }

    // Synchronize all of the CUDA streams before running the kernel
    // TO DO: cudaEventSyncronize() may be faster than cudaDeviceSynchronize()
    // cudaDeviceSynchronize();

    // Run the forward projection CUDA kernel
    Log("gpuForwardProjectLaunch()");
    gpuForwardProjectLaunch(this);

    Log("gpuForwardProjectLaunch() Done");

    return;

    // Note: This modifies the Matlab array in-place
}

float *gpuGridder::GetVolume()
{
    std::cout << "Volume: ";
    this->Volume[0] = 12;
    for (int i = 0; i < 10; i++)
    {
        std::cout << this->Volume[i] << " ";
    }
    std::cout << '\n';

    return this->Volume;
}