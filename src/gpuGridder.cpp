#include "gpuGridder.h"

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
    // Convert the volume to CAS volume
    this->CASVolume = gpuFFT::VolumeToCAS(this->Volume, this->VolumeSize[0], this->interpFactor, this->extraPadding);

    // DEBUG
    for (int i = 0; i < 10; i++)
    {
        std::cout << this->CASVolume[i] << " ";
    }
    std::cout << '\n';
    // END DEBUG

    // Save the volume size of the CAS volume
    // Example: A volume size of 128, interp factor 2, extra padding of 3 would give -> 262 CAS volume size
    this->CASVolumeSize = this->VolumeSize[0] * this->interpFactor + this->extraPadding * 2;

    // Also save the CAS image size
    int CASimgSize[3];
    CASimgSize[0] = this->CASVolumeSize;
    CASimgSize[1] = this->CASVolumeSize;
    CASimgSize[2] = this->numCoordAxes;

    this->SetCASImageSize(CASimgSize);

    std::cout << "this->CASVolumeSize: " << this->CASVolumeSize << '\n';
    std::cout << "CASVolume: ";
}

void gpuGridder::AllocateGPUArray(int GPU_Device, std::vector<float *> Ptr_Vector, int ArraySize)
{
    // Allocate an array on a GPU
    cudaSetDevice(GPU_Device);

    // Check to make sure the GPU has enough available memory left
    size_t mem_tot_0 = 0;
    size_t mem_free_0 = 0;
    cudaMemGetInfo(&mem_free_0, &mem_tot_0);

    // Is there enough available memory on the device to allocate this array?
    if (mem_free_0 < sizeof(float) * (ArraySize))
    {
        std::cerr << "Not enough memory on the device to allocate the requested memory. Try fewer number of projections or a smaller volume. Or increase SetNumberBatches()" << '\n';

        this->ErrorFlag = 1; // Set the error flag to 1 to remember that this failed
    }
    else
    {
        // There is enough memory left on the current GPU
        float *devPtr;
        cudaMalloc(&devPtr, sizeof(float) * (ArraySize));

        // Add the pointer to the corresponding vector of device pointers
        Ptr_Vector.push_back(devPtr);
    }
}

void gpuGridder::InitilizeGPUArrays()
{
    // Initilize the GPU arrays

    // First make sure the vectors of pointers are empty
    d_CASVolume.clear();
    d_CASImgs.clear();
    d_CoordAxes.clear();
    d_KB_Table.clear();

    // Loop through each of the GPUs and allocate the needed memory
    for (int i = 0; i < GPUs.size(); i++)
    {
        int GPU_Device = GPUs[i];
        cudaSetDevice(GPU_Device);

        // Allocate the CAS volume
        AllocateGPUArray(GPU_Device, d_CASVolume, this->CASVolumeSize * this->CASVolumeSize * this->CASVolumeSize);

        // Allocate the CAS images
        AllocateGPUArray(GPU_Device, d_CASImgs, this->CASimgSize[0] * this->CASimgSize[1] * this->CASimgSize[2]);

        // Allocate the coordinate axes array
        AllocateGPUArray(GPU_Device, d_CoordAxes, this->numCoordAxes * 9); // 9 float elements per cordinate axes

        // Allocate the Kaiser bessel lookup table
        AllocateGPUArray(GPU_Device, d_KB_Table, this->kerSize);
    }
}


void gpuGridder::SetVolume(float *Volume)
{
    // First save the given pointer
    this->Volume = Volume;

    // Next, pin the volume to host (i.e. CPU) memory in order to enable the async CUDA stream copying
    // This will let us copy the volume to all GPUs at the same time
    this->VolumeBytes = sizeof(float) * this->VolumeSize[0] * this->VolumeSize[1] * this->VolumeSize[2];
    cudaHostRegister(this->Volume, this->VolumeBytes, 0);
    
}




void gpuGridder::CopyVolumeToGPUs()
{
    // Copy the volume to each of the GPUs (the volume is already pinned to CPU memory during SetVolume())

    // Create CUDA streams for asyc memory copying of the gpuVols
    // One stream per GPU for now
    int nStreams = this->GPUs.size();
    cudaStream_t *stream = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nStreams);

    for (int i = 0; i < this->GPUs.size(); i++)
    {
        // Set the current GPU device
        cudaSetDevice(this->GPUs[i]);

        // Sends data to device asynchronously
        cudaMemcpyAsync(d_CASVolume[i], this->CASVolume, VolumeBytes, cudaMemcpyHostToDevice, stream[i]);

    }

}

void gpuGridder::ForwardProject()
{
    std::cout << "ForwardProject()" << '\n';

    // Do we need to convert the volume, copy to the GPUs, etc?
    // Assume for now that we have a new volume for each call to ForwardProject()
    bool newVolumeFlag = 1;

    if (newVolumeFlag == 1)
    {
        // (1): Initilize the needed arrays on each GPU
        InitilizeGPUArrays();

        // (2): Run the volume to CAS volume function
        VolumeToCASVolume();

        // (3): Copy the CASVolume to each of the GPUs
        CopyVolumeToGPUs();
    }

    // Check the error flag to see if we had any issues during the initilization
    if (this->ErrorFlag != 0)
    {
        return;
    }

    // Synchronize all of the CUDA streams before running the kernel
    cudaDeviceSynchronize();

    // Run the forward projection CUDA kernel

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