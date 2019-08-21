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

MultiGPUGridder::MultiGPUGridder()
{
    // Constructor for the MultiGPUGridder class

    // Create a new instance of the MemoryManager class
    this->Mem_obj = new MemoryManager;

    // Detect the number of GPUs and set to be the default value for numGPUs
    int numGPUDetected;
    cudaGetDeviceCount(&numGPUDetected);
    this->numGPUs = numGPUDetected;

    // Set the default number of streams to be the number of GPUs detected times four
    this->nStreams = this->numGPUs * 1;
}

void MultiGPUGridder::Print()
{
    // Output all of the parameters to the console (very useful for debugging purposes)
    std::cout << "Multi GPU Gridder" << '\n';
    std::cout << "Number of GPUs: " << this->numGPUs << '\n';
    std::cout << "Number of CUDA streams: " << this->nStreams << '\n';
    std::cout << "Image size: " << this->imgSize[0] << " " << this->imgSize[1] << " " << this->imgSize[2] << '\n';
    std::cout << "Interpolation factor: " << this->interpFactor << '\n';
    std::cout << "Mask Radius: " << this->maskRadius[0] << '\n';
    std::cout << "Coordinate Axes dimensions: " << this->axesSize[0] << " " << this->axesSize[1] << " " << this->axesSize[2] << '\n';
    std::cout << "Volume size: " << this->volSize[0] << " " << this->volSize[1] << " " << this->volSize[2] << '\n';
    std::cout << "Kaiser Bessel vector size: " << this->kerSize << '\n';
    std::cout << "Kaiser Bessel Width: " << this->kerHWidth << '\n';

    // Display the CUDA GPU memory allocations

    this->Mem_obj->CUDA_disp_mem("all");

    // Display the CPU memory allocations
    this->Mem_obj->disp_mem("all");
}

void MultiGPUGridder::SetNumberGPUs(int numGPUs)
{
    // Set the number of GPUs to use with the CUDA kernels

    // First, check if the computer has the requested number of GPUs
    int numGPUDetected;
    cudaGetDeviceCount(&numGPUDetected);

    // Provide error message if no GPUs are found (i.e. all cards are busy) are an invalid selection is chosen
    if (numGPUDetected == 0)
    {
        std::cerr << "No NVIDIA graphic cards identified on your computer. All cards may be busy and unavailable. Try restarting the program and/or your computer." << '\n';

        this->numGPUs = -1; // Set error value for numGPUs

        return;
    }

    if (numGPUs < 0 || numGPUs >= numGPUDetected + 1)
    {
        std::cerr << "Error in GPU selection. Please provide an integer between 0 and the number of NVIDIA graphic cards on your computer." << '\n';

        this->numGPUs = -1; // Set error value for numGPUs

        return;
    }

    // Save the user requested number of GPUs to use
    this->numGPUs = numGPUs;
}

void MultiGPUGridder::SetNumberStreams(int nStreams)
{
    // Set the number of streams to use with the CUDA kernels
    // Need at least as many streams as numGPUs to use
    if (nStreams < this->numGPUs)
    {
        std::cerr << "Please choose at least as many streams as the number of GPUs to use. Use SetNumberGPUs() first." << '\n';
        return;
    }

    // Save the user requested number of streams to use
    this->nStreams = nStreams;
}

void MultiGPUGridder::SetVolume(float *gpuVol, int *gpuVolSize)
{
    // Set the volume for forward and back projection
    // gpuVol is the CASVol for now (TO DO: do the forward projection and zero padding on the GPU)

    // Pin gpuVol to host (i.e. CPU) memory in order to enable the async CUDA stream copying
    // This will let us copy the volume to all GPUs at the same time
    cudaHostRegister(gpuVol, sizeof(float) * gpuVolSize[0] * gpuVolSize[1] * gpuVolSize[2], 0);

    // Check each GPU to determine if the gpuVol arrays are already allocated
    for (int gpuDevice = 0; gpuDevice < this->numGPUs; gpuDevice++)
    {
        // The name of the gpuVol GPU pointer is gpuVol_0 for GPU 0, gpuVol_1 for GPU 1, etc.
        // Has a gpuVol array already been allocated on this GPU?
        if (Mem_obj->GPUArrayAllocated("gpuVol_" + std::to_string(gpuDevice), gpuDevice) == false)
        {
            // We need to allocate the gpuVol array on this gpuDevice
            Mem_obj->CUDA_alloc("gpuVol_" + std::to_string(gpuDevice), "float", gpuVolSize, gpuDevice);
        }
    }

    // Create CUDA streams for asyc memory copying of the gpuVols
    int nStreams = this->numGPUs;
    cudaStream_t *stream = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nStreams);

    for (int gpuDevice = 0; gpuDevice < this->numGPUs; gpuDevice++)
    {
        // After allocating the gpuVol array on the gpuDevice, lets copy the memory
        // The name of the gpuVol GPU pointer is gpuVol_0 for GPU 0, gpuVol_1 for GPU 1, etc.
        Mem_obj->CUDA_Copy_Asyc("gpuVol_" + std::to_string(gpuDevice), gpuVol, stream[gpuDevice]);
    }

    // Synchronize all of the CUDA streams
    cudaDeviceSynchronize();

    // Unpin gpuVol to host memory (to free pinned memory on the RAM)
    cudaHostUnregister(gpuVol);

    // Save the volume size
    this->volSize = new int(*gpuVolSize);
    this->volSize[0] = gpuVolSize[0];
    this->volSize[1] = gpuVolSize[1];
    this->volSize[2] = gpuVolSize[2];
}

float *MultiGPUGridder::GetVolume()
{
    // Get the volume from all the GPUs and add them together
    // Return a pointer to the volume
    // This is used after the back projection CUDA kernel

    // Create the output array to hold the summed volumes from all of the GPUs
    float *VolSum = new float[this->volSize[0] * this->volSize[1] * this->volSize[2]];
    memset(VolSum, 0, sizeof(float) * this->volSize[0] * this->volSize[1] * this->volSize[2]);

    // Loop through each GPU
    for (int gpuDevice = 0; gpuDevice < this->numGPUs; gpuDevice++)
    {

        // The name of the gpuVol GPU pointer is gpuVol_0 for GPU 0, gpuVol_1 for GPU 1, etc.
        // Does this GPU array exist?
        if (Mem_obj->GPUArrayAllocated("gpuVol_" + std::to_string(gpuDevice), gpuDevice) == true)
        {
            // Get a float pointer to the GPU array after copying back to the host
            float *tempArray = Mem_obj->CUDA_Return("gpuVol_" + std::to_string(gpuDevice));

            // Add the volumes together
            for (int i = 0; i < this->volSize[0] * this->volSize[1] * this->volSize[2]; i++) //
            {
                VolSum[i] = VolSum[i] + tempArray[i];
            }
        }
    }

    // Lastly, divide the sum by the number of GPUs used
    // for (int i = 0; i < this->volSize[0] * this->volSize[1] * this->volSize[2]; i++)
    // {
    //     VolSum[i] = VolSum[i] / this->numGPUs;
    // }

    return VolSum;
}

void MultiGPUGridder::SetVolumeSize(int gpuVolSize)
{
    // The volume needs to be the same dimension in all three dimensions
    this->volSize = new int[3];
    this->volSize[0] = gpuVolSize;
    this->volSize[1] = gpuVolSize;
    this->volSize[2] = gpuVolSize;
    return;
}

void MultiGPUGridder::SetInterpFactor(float interpFactor)
{
    // Set the interpolation factor parameter
    this->interpFactor = interpFactor;
}

void MultiGPUGridder::SetImages(float *newCASImgs)
{
    // Set the CAS Images array to pinned CPU memory

    // Has the image size been defined already?
    if (this->imgSize[0] <= 0)
    {
        std::cerr << "Image size has not been defined yet. Please use SetImgSize() first." << '\n';
        return;
    }

    // Has the output CAS image array been allocated and pinned to the CPU?
    if (Mem_obj->CPUArrayAllocated("CASImgs_CPU_Pinned") == false)
    {
        // Allocate the CAS images array as pinned memory
        Mem_obj->CPU_Pinned_Allocate("CASImgs_CPU_Pinned", "float", this->imgSize);
    }

    // After allocating the CAS images array, lets copy newCASImgs to the pointer
    Mem_obj->mem_Copy("CASImgs_CPU_Pinned", newCASImgs);
}

float *MultiGPUGridder::GetImages()
{
    // Return the CASImgs as a float array
    // The images are store in the following pinned CPU array: "CASImgs_CPU_Pinned"
    float *OutputArray = Mem_obj->ReturnCPUFloatPtr("CASImgs_CPU_Pinned");

    return OutputArray;
}

void MultiGPUGridder::ResetVolume()
{
    // Reset the volume on all GPUs to zeros

    // Loop through all of the GPUs and reset the volume on each GPU
    for (int gpuDevice = 0; gpuDevice < this->numGPUs; gpuDevice++)
    {
        // The name of the gpuVol GPU pointer is gpuVol_0 for GPU 0, gpuVol_1 for GPU 1, etc.
        // Has a gpuVol array already been allocated on this GPU?
        if (Mem_obj->GPUArrayAllocated("gpuVol_" + std::to_string(gpuDevice), gpuDevice) == true)
        {
            cudaSetDevice(gpuDevice);

            // Get the corresponding GPU pointer to the volume on the current GPU
            float *gpuVolPtr = this->Mem_obj->ReturnCUDAFloatPtr("gpuVol_" + std::to_string(gpuDevice));

            // Set all elements in the volume on this GPU to zeros
            cudaMemset(gpuVolPtr, 0, volSize[0] * volSize[1] * volSize[2] * sizeof(float));
        }
        else
        {
            std::cerr << "Volume has not previously allocated on GPU " << gpuDevice << " Please use SetVolume() first." << '\n';
        }
    }
}

void MultiGPUGridder::SetAxes(float *coordAxes, int *axesSize)
{
    // Set the coordinate axes array to pinned CPU memory (this is a row vector)
    this->axesSize = new int(*axesSize);
    this->axesSize[0] = axesSize[0];
    this->axesSize[1] = 1;
    this->axesSize[2] = 1;

    std::cout << "this->axesSize[0]: " << this->axesSize[0] << " " << this->axesSize[1] << " " << this->axesSize[2] << '\n';
    if (coordAxes == NULL)
    {
        return;
    }

    // Has a coordAxes array already been allocated?
    if (Mem_obj->CPUArrayAllocated("coordAxes_CPU_Pinned") == false)
    {
        // Allocate the coordAxes as pinned memory
        Mem_obj->CPU_Pinned_Allocate("coordAxes_CPU_Pinned", "float", this->axesSize);
    }

    // After allocating the coordAxes array on the gpuDevice, lets copy the memory
    Mem_obj->mem_Copy("coordAxes_CPU_Pinned", coordAxes);
}

void MultiGPUGridder::SetKerBesselVector(float *ker_bessel_Vector, int kerSize)
{
    // Set the keiser bessel vector
    this->kerSize = kerSize;

    for (int i = 0; i < kerSize; i++)
    {
        this->ker_bessel_Vector[i] = ker_bessel_Vector[i];
    }
}

void MultiGPUGridder::SetImgSize(int *imgSize)
{
    // Set the output CAS image size parameter
    this->imgSize = new int[3];
    this->imgSize[0] = imgSize[0];
    this->imgSize[1] = imgSize[1];
    this->imgSize[2] = imgSize[2];
}

void MultiGPUGridder::SetMaskRadius(float *maskRadius)
{
    // Set the maskRadius parameter (used in the forward and back projection CUDA kernels)
    this->maskRadius = new float;
    this->maskRadius[0] = maskRadius[0];
}

void MultiGPUGridder::Projection_Initilize()
{
    // Initialize all the needed CPU and GPU pointers for running the CUDA kernels
    // Then check that all the required pointers exist

    if (this->imgSize[0] <= 0 || this->imgSize[0] > 1000000)
    {
        std::cerr << "The image size parameter has not be set. Please use SetImgSize() first." << '\n';
        return;
    }

    // How many coordinate axes where given?
    int nAxes = this->axesSize[0] / 9;

    // Has the output image array been allocated and pinned to the CPU?
    if (Mem_obj->CPUArrayAllocated("CASImgs_CPU_Pinned") == false)
    {
        // Allocate the CAS images array as pinned memory to allow for async CUDA streaming
        Mem_obj->CPU_Pinned_Allocate("CASImgs_CPU_Pinned", "float", this->imgSize);
    }

    // Check each GPU to determine if all the required pointers are already allocated
    for (int gpuDevice = 0; gpuDevice < this->numGPUs; gpuDevice++)
    {
        // One copy of the Kaiser Bessel look up table is needed for each GPU
        // Has the Kaiser bessel vector been allocated and defined?
        // The name of the GPU pointer is ker_bessel_0 for GPU 0, ker_bessel_1 for GPU 1, etc.
        if (Mem_obj->GPUArrayAllocated("ker_bessel_" + std::to_string(gpuDevice), gpuDevice) == false)
        {
            // Set the Kaiser Bessel Function vector to the current gpuDevice
            int arrSize[3];
            arrSize[0] = this->kerSize;
            arrSize[1] = 1;
            arrSize[2] = 1;

            // First allocate the Kaiser Bessel Function vector on the current gpuDevice
            Mem_obj->CUDA_alloc("ker_bessel_" + std::to_string(gpuDevice), "float", arrSize, gpuDevice);

            // After allocating the gpuVol array on the gpuDevice, lets copy the vector to the GPU memory
            Mem_obj->CUDA_Copy("ker_bessel_" + std::to_string(gpuDevice), this->ker_bessel_Vector);
        }

        // One copy of the CASImgs memory allocation is needed for each GPU
        // The name of the GPU pointer is gpuCASImgs_0 for GPU 0, gpuCASImgs_1 for GPU 1, etc.
        if (Mem_obj->GPUArrayAllocated("gpuCASImgs_" + std::to_string(gpuDevice), gpuDevice) == false)
        {
            // What is the array size of the output projection images?
            int *gpuCASImgs_Size = new int[3];
            gpuCASImgs_Size[0] = this->imgSize[0];
            gpuCASImgs_Size[1] = this->imgSize[1];

            // Allocate either the number of coordinate axes given or the maximum number (whichever is smaller)
            gpuCASImgs_Size[2] = std::min(ceil((double)nAxes / (double)this->numGPUs), (double)MaxAxesToAllocate);

            // We need to allocate the gpuCASImgs array on this GPU
            Mem_obj->CUDA_alloc("gpuCASImgs_" + std::to_string(gpuDevice), "float", gpuCASImgs_Size, gpuDevice);
        }

        // One copy of the coordinate axes array needs to be allocate on each GPU
        // The name of the GPU pointer is gpuCoordAxes_0 for GPU 0, gpuCoordAxes_1 for GPU 1, etc.
        if (Mem_obj->GPUArrayAllocated("gpuCoordAxes_" + std::to_string(gpuDevice), gpuDevice) == false)
        {
            // Allocate the gpuCoordAxes on the current gpuDevice
            int *gpuCoordAxes_Size = new int[3];

            // Each coordinate axes has 9 elements so multiply by 9 to get the length
            gpuCoordAxes_Size[0] = std::min(ceil((double)nAxes / (double)this->numGPUs), (double)MaxAxesToAllocate) * 9;
            gpuCoordAxes_Size[1] = this->axesSize[1]; // This should be equal to one
            gpuCoordAxes_Size[2] = this->axesSize[2]; // This should be equal to one

            // Allocate the coordinate axes array on the corresponding GPU
            Mem_obj->CUDA_alloc("gpuCoordAxes_" + std::to_string(gpuDevice), "float", gpuCoordAxes_Size, gpuDevice);
        }
    }
}

void MultiGPUGridder::Forward_Project()
{
    // Run the forward projection CUDA kernel

    // First check to make sure all the needed CPU and GPU required pointers exist
    Projection_Initilize();

    // Create a vector of GPU pointers
    std::vector<float *> gpuVol_Vector;
    std::vector<float *> gpuCASImgs_Vector;
    std::vector<float *> ker_bessel_Vector;
    std::vector<float *> gpuCoordAxes_Vector;

    // Find and add the corresponding GPU pointer to each vector of pointers
    // NOTE: Only need one of these arrays per GPU
    for (int gpuDevice = 0; gpuDevice < this->numGPUs; gpuDevice++)
    {
        // Volume array
        gpuVol_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("gpuVol_" + std::to_string(gpuDevice)));

        // Keiser bessel vector look up table
        ker_bessel_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("ker_bessel_" + std::to_string(gpuDevice)));

        // Output CASImgs array
        gpuCASImgs_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("gpuCASImgs_" + std::to_string(gpuDevice)));

        // Input coordinate axes array
        gpuCoordAxes_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("gpuCoordAxes_" + std::to_string(gpuDevice)));
    }

    // Get the pointers to the pinned CPU input / output arrays
    float *CASImgs_CPU_Pinned = this->Mem_obj->ReturnCPUFloatPtr("CASImgs_CPU_Pinned");
    float *coordAxes_CPU_Pinned = this->Mem_obj->ReturnCPUFloatPtr("coordAxes_CPU_Pinned");

    // Each axes has 9 elements (3 for each x, y, z)
    int nAxes = this->axesSize[0] / 9;

    // NOTE: gridSize times blockSize needs to equal imgSize
    int gridSize = 32;
    int blockSize = this->imgSize[0] / gridSize;

    // Before launching the kernel, first verify that all parameters and inputs are valid
    int parameter_check = ParameterChecking(
        gpuVol_Vector, gpuCASImgs_Vector, gpuCoordAxes_Vector, ker_bessel_Vector,                     // Vector of GPU array pointers
        CASImgs_CPU_Pinned, coordAxes_CPU_Pinned,                                                     // Pointers to pinned CPU arrays for input / output
        this->volSize[0], this->imgSize[0], nAxes, *this->maskRadius, this->kerSize, this->kerHWidth, // kernel Parameters and constants
        numGPUs, this->nStreams, gridSize, blockSize                                                  // Streaming parameters
    );

    // If an error was detected don't start the CUDA kernel
    if (parameter_check != 0)
    {
        std::cerr << "Error detected in input parameters of gpuForwardProjection()." << '\n';
        return;
    }

    // Pass the vector of pointers to the C++ function in gpuForwardProject.cu
    // Which will step up and run the CUDA streams and kernel
    gpuForwardProject(
        gpuVol_Vector, gpuCASImgs_Vector, gpuCoordAxes_Vector, ker_bessel_Vector,                     // Vector of GPU arrays
        CASImgs_CPU_Pinned, coordAxes_CPU_Pinned,                                                     // Pointers to pinned CPU arrays for input / output
        this->volSize[0], this->imgSize[0], nAxes, *this->maskRadius, this->kerSize, this->kerHWidth, // kernel parameters
        numGPUs, this->nStreams, gridSize, blockSize,                                                 // Streaming parameters
        this->MaxAxesToAllocate);

    return;
}

void MultiGPUGridder::Back_Project()
{
    // Run the back projection CUDA kernel

    // First check to make sure all the needed CPU and GPU required pointers exist
    Projection_Initilize();

    // Create a vector of GPU pointers
    std::vector<float *> gpuVol_Vector;
    std::vector<float *> gpuCASImgs_Vector;
    std::vector<float *> ker_bessel_Vector;
    std::vector<float *> gpuCoordAxes_Vector;

    // Find and add the corresponding GPU pointer to each vector of pointers
    // NOTE: Only need one of these GPU arrays per GPU
    for (int gpuDevice = 0; gpuDevice < this->numGPUs; gpuDevice++)
    {
        // Volume array
        gpuVol_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("gpuVol_" + std::to_string(gpuDevice)));

        // Keiser bessel vector look up table
        ker_bessel_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("ker_bessel_" + std::to_string(gpuDevice)));

        // Output CASImgs array
        gpuCASImgs_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("gpuCASImgs_" + std::to_string(gpuDevice)));

        // Input coordinate axes array
        gpuCoordAxes_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("gpuCoordAxes_" + std::to_string(gpuDevice)));
    }

    // Get the pointers to the CPU input / output arrays
    float *CASImgs_CPU_Pinned = this->Mem_obj->ReturnCPUFloatPtr("CASImgs_CPU_Pinned");
    float *coordAxes_CPU_Pinned = this->Mem_obj->ReturnCPUFloatPtr("coordAxes_CPU_Pinned");

    // Each axes has 9 elements (3 for each x, y, z)
    int nAxes = this->axesSize[0] / 9;

    // NOTE: gridSize times blockSize needs to equal imgSize
    int gridSize = this->imgSize[0] / 4;
    int blockSize = 4;

    // Before launching the kernel, first verify that all parameters and inputs are valid
    int parameter_check = ParameterChecking(
        gpuVol_Vector, gpuCASImgs_Vector, gpuCoordAxes_Vector, ker_bessel_Vector,                     // Vector of GPU array pointers
        CASImgs_CPU_Pinned, coordAxes_CPU_Pinned,                                                     // Pointers to pinned CPU arrays for input / output
        this->volSize[0], this->imgSize[0], nAxes, *this->maskRadius, this->kerSize, this->kerHWidth, // kernel Parameters and constants
        numGPUs, this->nStreams, gridSize, blockSize                                                  // Streaming parameters)
    );

    // If an error was detected don't start the CUDA kernel
    if (parameter_check != 0)
    {
        std::cerr << "Error detected in input parameters of gpuBackProjection()." << '\n';
        return;
    }

    // Pass the vector of pointers to the C++ function in gpuBackProject.cu
    // Which will step up and run the CUDA streams and kernel
    gpuBackProject(
        gpuVol_Vector, gpuCASImgs_Vector, gpuCoordAxes_Vector, ker_bessel_Vector,                     // Vector of GPU arrays
        CASImgs_CPU_Pinned, coordAxes_CPU_Pinned,                                                     // Pointers to pinned CPU arrays for input / output
        this->volSize[0], this->imgSize[0], nAxes, *this->maskRadius, this->kerSize, this->kerHWidth, // kernel parameters
        numGPUs, this->nStreams, gridSize, blockSize,                                                 // Streaming parameters
        this->MaxAxesToAllocate);

    return;
}

int MultiGPUGridder::ParameterChecking(
    std::vector<float *> gpuVol_Vector, std::vector<float *> gpuCASImgs_Vector,          // Vector of GPU array pointers
    std::vector<float *> gpuCoordAxes_Vector, std::vector<float *> ker_bessel_Vector,    // Vector of GPU array pointers
    float *CASImgs_CPU_Pinned, float *coordAxes_CPU_Pinned,                              // Pointers to pinned CPU arrays for input / output
    int volSize, int imgSize, int nAxes, float maskRadius, int kerSize, float kerHWidth, // kernel Parameters and constants
    int numGPUs, int nStreams, int gridSize, int blockSize                               // Streaming parameters
)
{
    // Check all the input parameters to verify they are all valid before launching the CUDA kernels

    // Checking parameter: numGPUs
    int numGPUDetected;
    cudaGetDeviceCount(&numGPUDetected);

    if (numGPUs < 0 || numGPUs >= numGPUDetected + 1) //  An invalid numGPUs selection was chosen
    {
        std::cerr << "Error in GPU selection. Please provide an integer between 0 and the number of NVIDIA graphic cards on your computer. Use SetNumberGPUs() function." << '\n';
        return -1;
    }

    if (numGPUDetected == 0) // No GPUs were found (i.e. all cards are busy)
    {
        std::cerr << "No NVIDIA graphic cards identified on your computer. All cards may be busy and unavailable. Try restarting the program and/or your computer." << '\n';
        return -1;
    }

    if (nStreams <= 0 || nStreams < numGPUs) // Number of streams must be greater than or equal to the number of GPUs
    {
        std::cerr << "Invalid number of streams provided. Please use SetNumberStreams() to set number of streams >= number of GPUs to use." << '\n';
        return -1;
    }

    if (volSize <= 0)
    {
        std::cerr << "Invalid volSize parameter. Please use SetVolume() to define the input volume." << '\n';
        return -1;
    }

    if (imgSize <= 0) // Size of the output image array must be non-zero
    {
        std::cerr << "Invalid imgSize parameter. Please use SetImgSize() to set the size of the output image array." << '\n';
        return -1;
    }

    if (nAxes <= 0) // The number of coordinate axes must be non-zero
    {
        std::cerr << "Invalid nAxes parameter. Please use SetAxes() to define the input coordinate axes." << '\n';
        return -1;
    }

    if (maskRadius <= 0) // The kernel mask radius must be non-zero
    {
        std::cerr << "Invalid maskRadius parameter. Please use SetMaskRadius() to set a non-zero integer value." << '\n';
        return -1;
    }

    if (kerSize <= 0) // The legnth of the keiser bessel look up table must be non-zero
    {
        std::cerr << "Invalid kerSize parameter." << '\n';
        return -1;
    }

    if (kerHWidth <= 0) // The keiser bessel width must be non-zero
    {
        std::cerr << "Invalid kerHWidth parameter." << '\n';
        return -1;
    }

    if (gridSize <= 0) // The size of the CUDA kernel grid must be non-zero
    {
        std::cerr << "Invalid gridSize parameter." << '\n';
        return -1;
    }

    // The gridSize times blockSize needs to equal imgSize for both forward and back projectin
    if (blockSize <= 0 || imgSize >= gridSize * blockSize)
    {
        // std::cout << "gridSize: " << gridSize << '\n';
        // std::cout << "blockSize: " << blockSize << '\n';
        // std::cout << "imgSize: " << imgSize << '\n';

        // std::cerr << "Invalid blockSize parameter. gridSize * blockSize must greater than or equal than imgSize" << '\n';
        // return -1;
    }

    // Must be at least one pointer to the GPU array
    if (gpuVol_Vector.size() <= 0 || gpuCASImgs_Vector.size() <= 0 ||
        gpuCoordAxes_Vector.size() <= 0 || ker_bessel_Vector.size() <= 0)
    {
        std::cerr << "gpuForwardProject(): Input GPU pointer vectors are empty. Has SetVolume() and SetAxes() been run?" << '\n';
        return -1;
    }

    // The number of gpuCASImgs and gpuCoordAxes pointers must be less than the number of CUDA streams
    if (gpuCASImgs_Vector.size() != numGPUs || gpuCoordAxes_Vector.size() != numGPUs)
    {
        std::cerr << "gpuForwardProject(): Number of streams is greater than the number of gpu array pointers" << '\n';
        return -1;
    }

    // No errors were detected so return a flag of 0
    return 0;
}

float *MultiGPUGridder::CropVolume(float *inputVol, int inputImgSize, int outputImgSize)
{
    // Crop a volume (of dimensions 3) to the size of the output volume
    // Note: Output volume is smaller than the input volume

    // Check the input parameters
    if(inputImgSize <=0 )
    {
        std::cerr << "CropVolume(): Invalid image size." << '\n';
    }

    // Create the output volume
    float *outputVol = new float[outputImgSize * outputImgSize * outputImgSize];

    std::cout << "Output volume size: " << outputImgSize << '\n';

    // How much to crop on each side?
    int crop = (inputImgSize - outputImgSize) / 2;

    std::cout << "Crop: " << crop << '\n';

    // Iterate over the output image (i.e. the smaller image)
    for (int i = 0; i < outputImgSize; i++)
    {
        for (int j = 0; j < outputImgSize; j++)
        {
            for (int k = 0; k < outputImgSize; k++)
            {

                int output_ndx = i + j*outputImgSize + k*outputImgSize*outputImgSize;

                int input_ndx = (i + crop) + (j+crop)*inputImgSize + (k+crop)*inputImgSize*inputImgSize;

                outputVol[output_ndx] = inputVol[input_ndx];
            }
        }
    }

    return outputVol;
}

float *MultiGPUGridder::PadVolume(float *inputVol, int inputImgSize, int outputImgSize)
{
    // Pad a volume (of dimensions 3) with zeros
    // Note: Output volume is larger than the input volume


    float * outputVol;

    outputVol = gpuFFT::PadVolume(inputVol, inputImgSize, outputImgSize)

    return outputVol;



}




// Define C functions for the C++ class since Python ctypes can only talk to C (not C++)
#define USE_EXTERN_C true
#if USE_EXTERN_C == true

extern "C"
{
    EXPORT MultiGPUGridder *Gridder_new() { return new MultiGPUGridder(); }
    EXPORT void SetNumberGPUs(MultiGPUGridder *gridder, int numGPUs) { gridder->SetNumberGPUs(numGPUs); }
    EXPORT void SetNumberStreams(MultiGPUGridder *gridder, int nStreams) { gridder->SetNumberStreams(nStreams); }
    EXPORT void SetVolume(MultiGPUGridder *gridder, float *gpuVol, int *gpuVolSize) { gridder->SetVolume(gpuVol, gpuVolSize); }
    EXPORT float *GetVolume(MultiGPUGridder *gridder) { return gridder->GetVolume(); }
    EXPORT void ResetVolume(MultiGPUGridder *gridder) { gridder->ResetVolume(); }
    EXPORT void SetImages(MultiGPUGridder *gridder, float *newCASImgs) { gridder->SetImages(newCASImgs); }
    EXPORT float *GetImages(MultiGPUGridder *gridder, float *CASImgs) { return gridder->GetImages(); }
    EXPORT void SetAxes(MultiGPUGridder *gridder, float *coordAxes, int *axesSize) { gridder->SetAxes(coordAxes, axesSize); }
    EXPORT void SetImgSize(MultiGPUGridder *gridder, int *imgSize) { gridder->SetImgSize(imgSize); }
    EXPORT void SetMaskRadius(MultiGPUGridder *gridder, float *maskRadius) { gridder->SetMaskRadius(maskRadius); }
    EXPORT void Forward_Project(MultiGPUGridder *gridder) { gridder->Forward_Project(); }
    EXPORT void Back_Project(MultiGPUGridder *gridder) { gridder->Back_Project(); }
}
#endif