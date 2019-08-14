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

    std::cout << "Number of GPUs: " << this->nStreams << '\n';
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

    std::cout << "Number of streams: " << this->nStreams << '\n';
}

void MultiGPUGridder::SetNumberBatches(int nBatches)
{
    // Set the number of batches to use with the CUDA kernels

    if (nBatches <= 0)
    {
        std::cerr << "Please choose a positive integer value for number of batches." << '\n';
        return;
    }

    // Save the user requested number of batches to use
    this->nBatches = nBatches;

    std::cout << "Number of batches: " << this->nBatches << '\n';
}

void MultiGPUGridder::SetVolume(float *gpuVol, int *gpuVolSize)
{
    // Set the volume for forward and back projection

    // Convert the given volume to a CAS volume
    // Steps are (1) pad with zeros, (2) run forward FFT using CUDA, (3) sum the real and imaginary components
    int array_size = gpuVolSize[0] * gpuVolSize[1] * gpuVolSize[2];
    float *CAS_Vol; // = new float [array_size];
    CAS_Vol = (float *)malloc(sizeof(float) * array_size);

    std::cout << "ThreeD_ArrayToCASArray()..." << '\n';

    CAS_Vol = ThreeD_ArrayToCASArray(gpuVol, gpuVolSize);
    std::cout << "Done with ThreeD_ArrayToCASArray()..." << '\n';

    float *Output_CAS_Vol; // = new float [array_size];
    Output_CAS_Vol = (float *)malloc(sizeof(float) * array_size);

    // for (int x=0; x < array_size; x++)
    // {
    //     Output_CAS_Vol[x] = CAS_Vol[x];
    //     //Output_CAS_Vol[x] = gpuVol[x];
    // }

    // std::cout << "Output_CAS_Vol " <<  << "]: ";
    // for (int x=0; x < 10; x++)
    // {
    //      std::cout << " "   <<  Output_CAS_Vol[y*gpuVolSize[0]+x];
    // }

    // TEST
    std::memcpy(Output_CAS_Vol, CAS_Vol, sizeof(float) * array_size);

    // Show the first slice of the Output_CAS_Vol
    // for (int x=0; x < gpuVolSize[0]; x++) //
    // {
    //     std::cout << "Output_CAS_Vol[" << x << "]: ";
    //     for (int y=0; y < 1; y++)
    //     {
    //         std::cout << " "   <<  Output_CAS_Vol[y*gpuVolSize[0]+x];
    //     }
    //     std::cout << '\n';
    // }

    // for (int x=0; x < gpuVolSize[0]; x++) //
    // {
    //     std::cout << "gpuVol[" << x << "]: ";
    //     for (int y=0; y < 1; y++)
    //     {
    //         std::cout << " "   <<  gpuVol[y*gpuVolSize[0]+x];
    //     }
    //     std::cout << '\n';
    // }
    // END TEST

    std::cout << "GPU Volume Size: " << gpuVolSize[0] << " " << gpuVolSize[1] << " " << gpuVolSize[2] << " " << '\n';

    // Pin gpuVol to host (i.e. CPU) memory in order to enable the async CUDA stream copying
    // This will let us copy the volume to all GPUs at the same time
    cudaHostRegister(Output_CAS_Vol, sizeof(float) * gpuVolSize[0] * gpuVolSize[1] * gpuVolSize[2], 0);

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
    //cudaStream_t stream[nStreams];
    cudaStream_t *stream = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nStreams);

    for (int gpuDevice = 0; gpuDevice < this->numGPUs; gpuDevice++)
    {
        // After allocating the gpuVol array on the gpuDevice, lets copy the memory
        // The name of the gpuVol GPU pointer is gpuVol_0 for GPU 0, gpuVol_1 for GPU 1, etc.
        Mem_obj->CUDA_Copy_Asyc("gpuVol_" + std::to_string(gpuDevice), Output_CAS_Vol, stream[gpuDevice]);
    }

    // Synchronize all of the CUDA streams
    cudaDeviceSynchronize();

    // Unpin gpuVol to host memory (to free pinned memory on the RAM)
    cudaHostUnregister(Output_CAS_Vol);

    std::free(Output_CAS_Vol);

    // Save the volume size
    this->volSize = new int(*gpuVolSize);
    this->volSize[0] = gpuVolSize[0];
    this->volSize[1] = gpuVolSize[1];
    this->volSize[2] = gpuVolSize[2];

    std::cout << "GPU Volume Set" << '\n';
}

void MultiGPUGridder::ConvertVolToCASVol()
{
    // Convert the volume in each GPU to a CASVol using CUDA kernels and cuFFT

    // Create a vector of GPU pointers to the volume on each GPU
    std::vector<float *> gpuVol_Vector;

    // Find and add the corresponding GPU pointer to each vector of pointers
    for (int gpuDevice = 0; gpuDevice < this->numGPUs; gpuDevice++)
    {
        // Volume array
        gpuVol_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("gpuVol_" + std::to_string(gpuDevice)));
    }

    // Since the volume on all GPUs are the same, perform the conversion on one and then copy the result to the rest
    ThreeD_ArrayToCASArray(gpuVol_Vector[0], this->volSize);

    // Since the volume on all GPUs are the same, copy the result to the rest
    for (int i = 0; i < 4; i++)
    {
    }
}

float *MultiGPUGridder::GetVolume()
{
    // Get the volume from all the GPUs and add them together
    // Return a pointer to the volume
    // This is used after the back projection CUDA kernel

    // Create the output array to hold the summed volumes from all of the GPUs
    float *VolSum = new float[this->volSize[0] * this->volSize[1] * this->volSize[2]];

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
    return VolSum;
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
        // Allocate the CAS images array
        Mem_obj->mem_alloc("CASImgs_CPU_Pinned", "float", this->imgSize);

        // Pin the array to allow for async CUDA streaming during kernel execution
        Mem_obj->pin_mem("CASImgs_CPU_Pinned");
    }

    // After allocating the CAS images array, lets copy newCASImgs to the pointer
    Mem_obj->mem_Copy("CASImgs_CPU_Pinned", newCASImgs);
}

float *MultiGPUGridder::GetImages()
{
    // Return the CASImgs as a float array
    // First get the stored size of the corresponding C++ array)
    //int *vol_dims;

    // The images are store in the following pinned CPU array: "CASImgs_CPU_Pinned"
    //vol_dims = Mem_obj->CPU_Get_Array_Size("CASImgs_CPU_Pinned");

    // Call the method
    float *OutputArray = Mem_obj->ReturnCPUFloatPtr("CASImgs_CPU_Pinned");

    // Copy the data to the Matlab array
    //std::memcpy(matlabArrayPtr, OutputArray, sizeof(float) * dims[0] * dims[1] * dims[2]);

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
    // Set the coordinate axes array to pinned CPU memory

    // Has a coordAxes array already been allocated?
    if (Mem_obj->CPUArrayAllocated("coordAxes_CPU_Pinned") == false)
    {
        // We need to allocate the coordAxes array on this axesSize
        Mem_obj->mem_alloc("coordAxes_CPU_Pinned", "float", axesSize);
    }

    // After allocating the coordAxes array on the gpuDevice, lets copy the memory
    Mem_obj->mem_Copy("coordAxes_CPU_Pinned", coordAxes);

    // Lastly, pin the array to allow for async CUDA streaming
    Mem_obj->pin_mem("coordAxes_CPU_Pinned");

    // Remember the axesSize for later
    this->axesSize = new int(*axesSize);
    this->axesSize[0] = axesSize[0];
    this->axesSize[1] = axesSize[1];
    this->axesSize[2] = axesSize[2];

    std::cout << "Coordinate axes size: " << this->axesSize[0] << " " << this->axesSize[1] << " " << this->axesSize[2] << '\n';
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
    this->maskRadius = maskRadius;
}

void MultiGPUGridder::Projection_Initilize()
{
    // Initialize all the needed CPU and GPU pointers for running the CUDA kernels
    // Then check that all the required pointers exist
    std::cout << "this->imgSize[0]: " << this->imgSize[0] << '\n';
    if (this->imgSize[0] <= 0 || this->imgSize[0] > 1000000)
    {
        std::cerr << "The image size parameter has not be set. Please use SetImgSize() first." << '\n';
        return;
    }

    // Has the output image array been allocated and pinned to the CPU?
    if (Mem_obj->CPUArrayAllocated("CASImgs_CPU_Pinned") == false)
    {
        // We need to allocate the coordAxes array on this axesSize
        Mem_obj->mem_alloc("CASImgs_CPU_Pinned", "float", this->imgSize);

        // Lastly, pin the array to allow for async CUDA streaming
        Mem_obj->pin_mem("CASImgs_CPU_Pinned");
    }

    int assigned_nAxes = 0; // Cumulative number of axes which have already been assigned to a CUDA stream


    // Check each GPU to determine if all the required pointers are already allocated
    for (int i = 0; i < this->nStreams; i++)
    {
        int gpuDevice = i % this->numGPUs; // Use the remainder operator to split streams evenly between GPUs

        // What is the array size of the output projection images?
        int *gpuCASImgs_Size = new int[3];
        gpuCASImgs_Size[0] = this->imgSize[0];
        gpuCASImgs_Size[1] = this->imgSize[1];

        // How many images will this stream process?
        int nAxes = this->axesSize[0] / 9;
        int nImgsPerStream = ceil((double)nAxes / (double)this->nStreams / (double)this->nBatches);

        // Check to make sure we don't try to process more coord axes than we have
        if ((assigned_nAxes + nImgsPerStream*this->nBatches) > nAxes) 
        {
            // Process the remaining streams (at least one axes is left)
            nImgsPerStream = nImgsPerStream*this->nBatches - (assigned_nAxes + nImgsPerStream*this->nBatches - nAxes); // Remove the extra axes that are past the maximum nAxes
            nImgsPerStream = ceil((double)nImgsPerStream / (double)this->nBatches);
        }
        
        if (nImgsPerStream < 0)
        {
            nImgsPerStream = 0;
        }

        std::cout << '\n';
        std::cout << "assigned_nAxes: " << assigned_nAxes << '\n';
        std::cout << "nAxes: " << nAxes << '\n';
        std::cout << "nImgsPerStream: " << nImgsPerStream << '\n';
        
        // Update how many axes have been assigned already
        assigned_nAxes = assigned_nAxes + nImgsPerStream*this->nBatches;
     
        // Remember the number of axes (i.e. images) which were assigned to this stream
        // if (nImgsPerStream < 5)
        // {
        //     nImgsPerStream = 0; // TEST TEST TEST
        // }
        numAxesPerStream.push_back(nImgsPerStream);

        gpuCASImgs_Size[2] = std::max(nImgsPerStream, 2); // Must be at least two images (projections are sometimes missing if only 1 image is allocated)

        // Has the output array been allocated and defined already?
        // The name of the GPU pointer is gpuCASImgs_0 for GPU 0, gpuCASImgs_1 for GPU 1, etc.
        if (Mem_obj->GPUArrayAllocated("gpuCASImgs_" + std::to_string(i), gpuDevice) == false)
        {
            // We need to allocate the gpuCASImgs array on this GPU
            Mem_obj->CUDA_alloc("gpuCASImgs_" + std::to_string(i), "float", gpuCASImgs_Size, gpuDevice);
        }

        // Has the coordAxes array been allocated and defined?
        // The name of the GPU pointer is gpuCoordAxes_0 for GPU 0, gpuCoordAxes_1 for GPU 1, etc.
        if (Mem_obj->GPUArrayAllocated("gpuCoordAxes_" + std::to_string(i), gpuDevice) == false)
        {
            // Allocate the gpuCoordAxes on the current gpuDevice
            int *gpuCoordAxes_Size = new int[3];

            // Each GPU only needs to hold a fraction of the total axes vector (based on number of streams and batches)
            int nAxes = this->axesSize[0] / 9;

            // Round up to the number of axes per stream and then multiply by 9 to get the length
            int nAxesPerStream = ceil((double)nAxes / (double)this->nStreams / (double)this->nBatches);
            nAxesPerStream = std::max(nAxesPerStream, 1); // Must be at least one axes

            gpuCoordAxes_Size[0] = nAxesPerStream * 9;
            gpuCoordAxes_Size[1] = this->axesSize[1];
            gpuCoordAxes_Size[2] = this->axesSize[2];

            Mem_obj->CUDA_alloc("gpuCoordAxes_" + std::to_string(i), "float", gpuCoordAxes_Size, gpuDevice);
        }

        // Has the ComplexImgs array been allocated and defined?
        // The name of the GPU pointer is gpuComplexImgs_0 for GPU 0, gpuComplexImgs_1 for GPU 1, etc.
        if (Mem_obj->GPUArrayAllocated("gpuComplexImgs_" + std::to_string(i), gpuDevice) == false)
        {
            // Allocate the ComplexImgs array on the current gpuDevice
            Mem_obj->CUDA_alloc("gpuComplexImgs_" + std::to_string(i), "cufftComplex", gpuCASImgs_Size, gpuDevice);
        }

        // Has the ComplexImgs_Shifted array been allocated and defined?
        // The name of the GPU pointer is gpuComplexImgs_Shifted_0 for GPU 0, gpuComplexImgs_Shifted_1 for GPU 1, etc.
        if (Mem_obj->GPUArrayAllocated("gpuComplexImgs_Shifted_" + std::to_string(i), gpuDevice) == false)
        {
            // Allocate the ComplexImgs_Shifted array on the current gpuDevice
            Mem_obj->CUDA_alloc("gpuComplexImgs_Shifted_" + std::to_string(i), "cufftComplex", gpuCASImgs_Size, gpuDevice);
        }
    }

    // One copy of the Kaiser Bessel look up table is needed for each GPU
    for (int gpuDevice = 0; gpuDevice < this->numGPUs; gpuDevice++)
    {
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

            // After allocating the gpuVol array on the gpuDevice, lets gpuDevicepy the memory
            Mem_obj->CUDA_Copy("ker_bessel_" + std::to_string(gpuDevice), this->ker_bessel_Vector);
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
    std::vector<cufftComplex *> gpuComplexImgs_Vector;
    std::vector<cufftComplex *> gpuComplexImgs_Shifted_Vector;

    // Find and add the corresponding GPU pointer to each vector of pointers
    // NOTE: Only need one of these arrays per GPU
    for (int gpuDevice = 0; gpuDevice < this->numGPUs; gpuDevice++)
    {
        // Volume array
        gpuVol_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("gpuVol_" + std::to_string(gpuDevice)));

        // Keiser bessel vector look up table
        ker_bessel_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("ker_bessel_" + std::to_string(gpuDevice)));
    }

    // Find and add the corresponding GPU pointer for each stream to each vector of pointers
    // NOTE: Need one of these arrays for each of the CUDA streams
    for (int i = 0; i < this->nStreams; i++)
    {
        // Output CASImgs array
        gpuCASImgs_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("gpuCASImgs_" + std::to_string(i)));

        // Input coordinate axes array
        gpuCoordAxes_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("gpuCoordAxes_" + std::to_string(i)));

        // Array to hold the complex version of the CASImgs (intermediate step when doing the FFT)
        gpuComplexImgs_Vector.push_back(this->Mem_obj->ReturnCUDAComplexPtr("gpuComplexImgs_" + std::to_string(i)));

        // Array to hold the FFT shifted complex version of the CASImgs (intermediate step when doing the FFT)
        gpuComplexImgs_Shifted_Vector.push_back(this->Mem_obj->ReturnCUDAComplexPtr("gpuComplexImgs_Shifted_" + std::to_string(i)));
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
        gpuComplexImgs_Vector, gpuComplexImgs_Shifted_Vector,                                         // Vector of GPU array pointers
        CASImgs_CPU_Pinned, coordAxes_CPU_Pinned,                                                     // Pointers to pinned CPU arrays for input / output
        this->volSize[0], this->imgSize[0], nAxes, *this->maskRadius, this->kerSize, this->kerHWidth, // kernel Parameters and constants
        numGPUs, this->nStreams, gridSize, blockSize, this->nBatches,                                  // Streaming parameters
        this->numAxesPerStream
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
        gpuComplexImgs_Vector, gpuComplexImgs_Shifted_Vector,                                         // Vector of GPU array pointers
        CASImgs_CPU_Pinned, coordAxes_CPU_Pinned,                                                     // Pointers to pinned CPU arrays for input / output
        this->volSize[0], this->imgSize[0], nAxes, *this->maskRadius, this->kerSize, this->kerHWidth, // kernel parameters
        numGPUs, this->nStreams, gridSize, blockSize, this->nBatches,                                  // Streaming parameters
        this->numAxesPerStream
    );

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
    std::vector<cufftComplex *> gpuComplexImgs_Vector;
    std::vector<cufftComplex *> gpuComplexImgs_Shifted_Vector;

    // Find and add the corresponding GPU pointer to each vector of pointers
    // NOTE: Only need one of these GPU arrays per GPU
    for (int gpuDevice = 0; gpuDevice < this->numGPUs; gpuDevice++)
    {
        // Volume array
        gpuVol_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("gpuVol_" + std::to_string(gpuDevice)));

        // Keiser bessel vector look up table
        ker_bessel_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("ker_bessel_" + std::to_string(gpuDevice)));
    }

    // Find and add the corresponding GPU pointer for each stream to each vector of pointers
    // NOTE: Need one of these arrays for each of the CUDA streams
    for (int i = 0; i < this->nStreams; i++)
    {
        // Output CASImgs array
        gpuCASImgs_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("gpuCASImgs_" + std::to_string(i)));

        // Input coordinate axes array
        gpuCoordAxes_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("gpuCoordAxes_" + std::to_string(i)));

        // Array to hold the complex version of the CASImgs (intermediate step when doing the FFT)
        // gpuComplexImgs_Vector.push_back(this->Mem_obj->ReturnCUDAComplexPtr("gpuComplexImgs_" + std::to_string(i)));

        // Array to hold the FFT shifted complex version of the CASImgs (intermediate step when doing the FFT)
        // gpuComplexImgs_Shifted_Vector.push_back(this->Mem_obj->ReturnCUDAComplexPtr("gpuComplexImgs_Shifted_" + std::to_string(i)));
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
        gpuComplexImgs_Vector, gpuComplexImgs_Shifted_Vector,                                         // Vector of GPU array pointers
        CASImgs_CPU_Pinned, coordAxes_CPU_Pinned,                                                     // Pointers to pinned CPU arrays for input / output
        this->volSize[0], this->imgSize[0], nAxes, *this->maskRadius, this->kerSize, this->kerHWidth, // kernel Parameters and constants
        numGPUs, this->nStreams, gridSize, blockSize, this->nBatches,                                  // Streaming parameters)
        this->numAxesPerStream
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
        numGPUs, this->nStreams, gridSize, blockSize, this->nBatches                                  // Streaming parameters
    );

    return;
}

int MultiGPUGridder::ParameterChecking(
    std::vector<float *> gpuVol_Vector, std::vector<float *> gpuCASImgs_Vector,          // Vector of GPU array pointers
    std::vector<float *> gpuCoordAxes_Vector, std::vector<float *> ker_bessel_Vector,    // Vector of GPU array pointers
    std::vector<cufftComplex *> gpuComplexImgs_Vector,                                   // Vector of GPU array pointers
    std::vector<cufftComplex *> gpuComplexImgs_Shifted_Vector,                           // Vector of GPU array pointers
    float *CASImgs_CPU_Pinned, float *coordAxes_CPU_Pinned,                              // Pointers to pinned CPU arrays for input / output
    int volSize, int imgSize, int nAxes, float maskRadius, int kerSize, float kerHWidth, // kernel Parameters and constants
    int numGPUs, int nStreams, int gridSize, int blockSize, int nBatches,                 // Streaming parameters
    std::vector<int> numAxesPerStream
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

    if (nBatches <= 0) // Need a non-zero integer value for the number of batches
    {
        std::cerr << "Invalid number of batches provided. Please use SetNumberBatches() to set a non-negative integer number." << '\n';
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
    if (blockSize <= 0 || imgSize != gridSize * blockSize)
    {
        std::cerr << "Invalid blockSize parameter. gridSize * blockSize must equal imgSize" << '\n';
        return -1;
    }

    // Must be at least one pointer to the GPU array
    if (gpuVol_Vector.size() <= 0 || gpuCASImgs_Vector.size() <= 0 || gpuCoordAxes_Vector.size() <= 0 ||
        ker_bessel_Vector.size() <= 0 || gpuComplexImgs_Vector.size() <= 0 || gpuComplexImgs_Shifted_Vector.size() <= 0)
    {
        std::cerr << "gpuForwardProject(): Input GPU pointer vectors are empty. Has SetVolume() and SetAxes() been run?" << '\n';
        return -1;
    }

    // The number of gpuCASImgs and gpuCoordAxes pointers must be less than the number of CUDA streams
    if (gpuCASImgs_Vector.size() != nStreams || gpuCoordAxes_Vector.size() != nStreams || 
    gpuComplexImgs_Vector.size() != nStreams || gpuComplexImgs_Shifted_Vector.size() != nStreams || numAxesPerStream.size() != nStreams)
    {
        std::cerr << "gpuForwardProject(): Number of streams is greater than the number of gpu array pointers" << '\n';
        return -1;
    }

    // No errors were detected so return a flag of 0
    return 0;
}

// Define C functions for the C++ class since Python ctypes can only talk to C (not C++)
#define USE_EXTERN_C true
#if USE_EXTERN_C == true

extern "C"
{
    EXPORT MultiGPUGridder *Gridder_new() { return new MultiGPUGridder(); }
    EXPORT void SetNumberGPUs(MultiGPUGridder *gridder, int numGPUs) { gridder->SetNumberGPUs(numGPUs); }
    EXPORT void SetNumberStreams(MultiGPUGridder *gridder, int nStreams) { gridder->SetNumberStreams(nStreams); }
    EXPORT void SetNumberBatches(MultiGPUGridder *gridder, int nBatches) { gridder->SetNumberBatches(nBatches); }
    EXPORT void SetVolume(MultiGPUGridder *gridder, float *gpuVol, int *gpuVolSize) { gridder->SetVolume(gpuVol, gpuVolSize); }
    EXPORT float *GetVolume(MultiGPUGridder *gridder) { gridder->GetVolume(); }
    EXPORT void ResetVolume(MultiGPUGridder *gridder) { gridder->ResetVolume(); }
    EXPORT void SetImages(MultiGPUGridder *gridder, float *newCASImgs) { gridder->SetImages(newCASImgs); }
    EXPORT void GetImages(MultiGPUGridder *gridder, float *CASImgs) { gridder->GetImages(); }
    EXPORT void SetAxes(MultiGPUGridder *gridder, float *coordAxes, int *axesSize) { gridder->SetAxes(coordAxes, axesSize); }
    EXPORT void SetImgSize(MultiGPUGridder *gridder, int *imgSize) { gridder->SetImgSize(imgSize); }
    EXPORT void SetMaskRadius(MultiGPUGridder *gridder, float *maskRadius) { gridder->SetMaskRadius(maskRadius); }
    EXPORT void Forward_Project(MultiGPUGridder *gridder) { gridder->Forward_Project(); }
    EXPORT void Back_Project(MultiGPUGridder *gridder) { gridder->Back_Project(); }
}
#endif