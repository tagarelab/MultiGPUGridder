#include "CUDA_Gridder.h"

// Constructor
CUDA_Gridder::CUDA_Gridder(){

    // Create a new instance of the CPU_CUDA_Memory class
    this->Mem_obj = new CPU_CUDA_Memory;

    std::cout << "CUDA_Gridder() constructor" << '\n';
}

void CUDA_Gridder::SetNumberGPUs(int numGPUs)
{
    // How many GPUs to use with the CUDA kernels?

    // Does the computer have this many GPUs?

    int  numGPUDetected;
    cudaGetDeviceCount(&numGPUDetected);

    std::cout << "numGPUDetected: " << numGPUDetected << '\n';

    // Provide error message if no GPUs are found (i.e. all cards are busy) are an invalid selection is chosen
    if ( numGPUDetected == 0 )
    {
        std::cerr << "No NVIDIA graphic cards identified on your computer. All cards may be busy and unavailable. Try restarting the program and/or your computer." << '\n';;  
        
        this->numGPUs = -1;

        return;
    }

    if (numGPUs < 0 || numGPUs >= numGPUDetected + 1){

        this->numGPUs = -1;

        std::cerr << "Error in GPU selection. Please provide an integer between 0 and the number of NVIDIA graphic cards on your computer." << '\n';

        return;
    }

    // Save the user requested number of GPUs to use
    this->numGPUs = numGPUs;

    std::cout << "numGPUs: " << numGPUs << '\n';

}

void CUDA_Gridder::SetNumberStreams(int nStreams)
{
    // How many streams to use with the CUDA kernels?
    // Need at least as many streams as numGPUs to use

    if (nStreams < this->numGPUs)
    {
        std::cerr << "Please choose at least as many streams as the number of GPUs to use. Use SetNumberGPUs() first." << '\n';;  
        return;    
    }

    // Save the user requested number of streams to use
    this->nStreams = nStreams;

    std::cout << "nStreams: " << nStreams << '\n';

}

void CUDA_Gridder::SetNumberBatches(int nBatches)
{
    // How many batches to use with the CUDA kernels?

    if (nBatches <= 0)
    {
        std::cerr << "Please choose a positive integer value for number of batches." << '\n';;  
        return;    
    }

    // Save the user requested number of batches to use
    this->nBatches = nBatches;

    std::cout << "this->nBatches: " << this->nBatches << '\n';

}

// Set the GPU Volume
void CUDA_Gridder::SetVolume(float* gpuVol, int* gpuVolSize)
{      
    
    // Pin gpuVol to host memory ( to enable the asyn stream copying)
    cudaHostRegister(gpuVol, sizeof(float)*gpuVolSize[0]*gpuVolSize[1]*gpuVolSize[2], 0);

    // Check each GPU to determine if the gpuVol arrays are already allocated
    for (int gpuDevice = 0; gpuDevice < this->numGPUs; gpuDevice++)
    {
        std::cout << "Setting gpuVol on GPU " << gpuDevice << "..." << '\n';

        // The name of the gpuVol GPU pointer is gpuVol_0 for GPU 0, gpuVol_1 for GPU 1, etc.
        // Has a gpuVol array already been allocated on this GPU?    
        if ( Mem_obj->GPUArrayAllocated("gpuVol_" + std::to_string(gpuDevice), gpuDevice) == false) 
        {
            // We need to allocate the gpuVol array on this gpuDevice
            Mem_obj->CUDA_alloc("gpuVol_" + std::to_string(gpuDevice), "float", gpuVolSize, gpuDevice);
        }
    }

    // Create CUDA streams for asyc memory copying of the gpuVols
    int nStreams = this->numGPUs;
    cudaStream_t stream[nStreams]; 	

    for (int gpuDevice = 0; gpuDevice < this->numGPUs; gpuDevice++)
    {
        // After allocating the gpuVol array on the gpuDevice, lets copy the memory
        Mem_obj->CUDA_Copy_Asyc("gpuVol_" + std::to_string(gpuDevice), gpuVol, stream[gpuDevice]);  
    }

    // Unpin gpuVol to host memory (to free pinned memory on the RAM)
    cudaDeviceSynchronize();
    cudaError_t chk;
    chk = cudaHostUnregister(gpuVol);
    std::cout << "cudaError_t chk: " << chk << '\n';

    // Save the volume size for later
    this->volSize = new int(*gpuVolSize);
    this->volSize[0] = gpuVolSize[0];
    this->volSize[1] = gpuVolSize[1];
    this->volSize[2] = gpuVolSize[2];

}

// Set the coordinate axes Volume
void CUDA_Gridder::SetAxes(float* coordAxes, int* axesSize)
{   
    std::cout << "Setting coordAxes array to pinned CPU memory..." << '\n';

     // Has a coordAxes array already been allocated?    
    if ( Mem_obj->CPUArrayAllocated("coordAxes_CPU_Pinned") == false) 
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

    std::cout << "this->axesSize: " << this->axesSize[0] << " " << this->axesSize[1] << " " << this->axesSize[2] << '\n'; 

}

// Set the output image size parameter
void CUDA_Gridder::SetImgSize(int* imgSize)
{
    this->imgSize = new int[3];
    this->imgSize[0] = imgSize[0];
    this->imgSize[1] = imgSize[1];
    this->imgSize[2] = imgSize[2]; 

    std::cout << "imgSize: " << imgSize[0] << " " << imgSize[1] << " " << imgSize[2] << '\n';
}

// Set the maskRadius parameter
void CUDA_Gridder::SetMaskRadius(float* maskRadius)
{
    this->maskRadius = maskRadius;

    std::cout << "maskRadius: " << maskRadius[0] << '\n';

}

void CUDA_Gridder::Projection_Initilize()
{
    // Initialize all the needed CPU and GPU pointers and check that all the required pointers exist

    // Has the output image array been allocated and pinned to the CPU?
    if ( Mem_obj->CPUArrayAllocated("CASImgs_CPU_Pinned") == false) 
    {
        std::cout << "Allocating CASImgs_CPU_Pinned" << '\n';
        // We need to allocate the coordAxes array on this axesSize
        Mem_obj->mem_alloc("CASImgs_CPU_Pinned", "float", this->imgSize);
        
        std::cout << "Pinning CASImgs_CPU_Pinned" << '\n';

        // Lastly, pin the array to allow for async CUDA streaming
        Mem_obj->pin_mem("CASImgs_CPU_Pinned");
        
    }

    // Check each GPU to determine if all the required pointers are already allocated
    for (int i = 0; i < this->nStreams; i++)
    {
        std::cout << "Projection_Initilize():  Stream number " << i << '\n';

        int gpuDevice = i % this->numGPUs; // Use the remainder operator to split streams evenly between GPUs

        // Has the output array been allocated and defined already?
        // The name of the GPU pointer is gpuCASImgs_0 for GPU 0, gpuCASImgs_1 for GPU 1, etc.
        if ( Mem_obj->GPUArrayAllocated("gpuCASImgs_" + std::to_string(i), gpuDevice) == false) 
        {
            // We need to allocate the gpuCASImgs array on this GPU
            int * gpuCASImgs_Size = new int[3];
            gpuCASImgs_Size[0] = this->imgSize[0];
            gpuCASImgs_Size[1] = this->imgSize[1];
            // gpuCASImgs_Size[2] = this->imgSize[2]; 
   
            // Each GPU only needs to hold a fraction of the total output images (based on number of streams and batches)
            // Should probably be this->numGPUs but am getting error
            gpuCASImgs_Size[2] = ceil(this->imgSize[2] / (this->nStreams));
            gpuCASImgs_Size[2] = ceil(gpuCASImgs_Size[2] / (this->nBatches)) + 1;

            std::cout << "gpuCASImgs_Size: " << gpuCASImgs_Size[0] << " "  << gpuCASImgs_Size[1] << " " << gpuCASImgs_Size[2] << '\n';
            
            Mem_obj->CUDA_alloc("gpuCASImgs_" + std::to_string(i), "float", gpuCASImgs_Size, gpuDevice);        
        }

        // Has the coordAxes array been allocated and defined?
        // The name of the GPU pointer is gpuCoordAxes_0 for GPU 0, gpuCoordAxes_1 for GPU 1, etc.
        if ( Mem_obj->GPUArrayAllocated("gpuCoordAxes_" + std::to_string(i), gpuDevice) == false) 
        {
            // Allocate the gpuCoordAxes on the current gpuDevice
            int * gpuCoordAxes_Size = new int[3];

            // Each GPU only needs to hold a fraction of the total axes vector (based on number of streams and batches)
            // Am getting an error so adding a few bytes to the end
            gpuCoordAxes_Size[0] = ceil(this->axesSize[0] / this->nStreams)  + 10;  /// (this->nBatches)
            // gpuCoordAxes_Size[0] = this->axesSize[0]; 
            gpuCoordAxes_Size[1] = this->axesSize[1];
            gpuCoordAxes_Size[2] = this->axesSize[2];

            std::cout << "gpuCoordAxes_Size[0]: " << gpuCoordAxes_Size[0] << '\n';            

            Mem_obj->CUDA_alloc("gpuCoordAxes_" + std::to_string(i), "float", gpuCoordAxes_Size, gpuDevice);            
        }
    }

    // Only need one per GPU for the Kaiser bessel vector
    for (int gpuDevice = 0; gpuDevice < this->numGPUs; gpuDevice++)
    {
        // Has the Kaiser bessel vector been allocated and defined?      
        // The name of the GPU pointer is ker_bessel_1000 for GPU 0, ker_bessel_1 for GPU 1, etc.
        if ( Mem_obj->GPUArrayAllocated("ker_bessel_" + std::to_string(gpuDevice), gpuDevice) == false) 
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

    std::cout << "Projection_Initilize() finished" << '\n';
}

// Run the Forward Projection CUDA kernel
void CUDA_Gridder::Forward_Project(){
    // Run the forward projection CUDA kernel

    std::cout << "CUDA_Gridder::Forward_Project()" << '\n';
        
    // TO DO: Add more error checking. Are all the parameters valid? e.g. numGPUs > 0, nStreams <= numGPUS, etc.

    // Double check to make sure all the needed CPU and GPU required pointers exist
    Projection_Initilize(); 
    
    // Create a vector of GPU pointers
    std::vector<float*> gpuVol_Vector;
    std::vector<float*> gpuCASImgs_Vector;
    std::vector<float*> ker_bessel_Vector;
    std::vector<float*> gpuCoordAxes_Vector;

    // Find and add the corresponding GPU pointer to each vector of pointers 
    // NOTE: Only need one of these GPU arrays per GPU
    for (int gpuDevice = 0; gpuDevice < this->numGPUs; gpuDevice++)
    {        
        gpuVol_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("gpuVol_" + std::to_string(gpuDevice)));      
        ker_bessel_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("ker_bessel_" + std::to_string(gpuDevice)));        
    }

    // Find and add the corresponding GPU pointer for each stream to each vector of pointers 
    // NOTE: Need one of these arrays for each of the CUDA streams
    for (int i = 0; i < this->nStreams; i++)
    {
        gpuCASImgs_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("gpuCASImgs_" + std::to_string(i)));
        gpuCoordAxes_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("gpuCoordAxes_" + std::to_string(i)));
    }

    // Get the pointers to the CPU input / output arrays
    float * CASImgs_CPU_Pinned   = this->Mem_obj->ReturnCPUFloatPtr("CASImgs_CPU_Pinned");
    float * coordAxes_CPU_Pinned = this->Mem_obj->ReturnCPUFloatPtr("coordAxes_CPU_Pinned");

    // Each axes has 9 elements (3 for each x, y, z)
    int nAxes = this->axesSize[0] / 9; 
    
    // NOTE: gridSize times blockSize needs to equal imgSize
    int gridSize  = 32;// 32  
    int blockSize = this->imgSize[0] / gridSize ; // 4  

    // Verify all parameters and inputs are valid
    int parameter_check = ParameterChecking(    
        gpuVol_Vector, gpuCASImgs_Vector, gpuCoordAxes_Vector, ker_bessel_Vector, // Vector of GPU array pointers
        CASImgs_CPU_Pinned, coordAxes_CPU_Pinned, // Pointers to pinned CPU arrays for input / output
        this->volSize[0], this->imgSize[0], nAxes, *this->maskRadius, this->kerSize, this->kerHWidth, // kernel Parameters and constants
        numGPUs, this->nStreams, gridSize, blockSize, this->nBatches // Streaming parameters)
    );

    // If an error was detected return and don't start the CUDA kernel
    if (parameter_check != 0)
    {
        std::cerr << "Error detected in input parameters. Stopping the gpuForwardProjection now." << '\n';
        return;
    }   

    // Pass the vector of pointers to the C++ function in gpuForwardProject.cu
    // Which will step up and run the CUDA streams
    gpuForwardProject(
        gpuVol_Vector, gpuCASImgs_Vector, gpuCoordAxes_Vector, ker_bessel_Vector, // Vector of GPU arrays
        CASImgs_CPU_Pinned, coordAxes_CPU_Pinned, // Pointers to pinned CPU arrays for input / output
        this->volSize[0], this->imgSize[0], nAxes, *this->maskRadius, this->kerSize, this->kerHWidth, // kernel parameters
        numGPUs, this->nStreams, gridSize, blockSize, this->nBatches // Streaming parameters
        ); 

    return;

}

// Run the Back Projection CUDA kernel
void CUDA_Gridder::Back_Project(){
    // Run the back projection CUDA kernel

    std::cout << "CUDA_Gridder::Back_Project()" << '\n';
        
    // TO DO: Add more error checking. Are all the parameters valid? e.g. numGPUs > 0, nStreams <= numGPUS, etc.

    // Double check to make sure all the needed CPU and GPU required pointers exist
    Projection_Initilize(); 
    
    // Create a vector of GPU pointers
    std::vector<float*> gpuVol_Vector;
    std::vector<float*> gpuCASImgs_Vector;
    std::vector<float*> ker_bessel_Vector;
    std::vector<float*> gpuCoordAxes_Vector;

    // Find and add the corresponding GPU pointer to each vector of pointers 
    // NOTE: Only need one of these GPU arrays per GPU
    for (int gpuDevice = 0; gpuDevice < this->numGPUs; gpuDevice++)
    {        
        gpuVol_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("gpuVol_" + std::to_string(gpuDevice)));      
        ker_bessel_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("ker_bessel_" + std::to_string(gpuDevice)));        
    }

    // Find and add the corresponding GPU pointer for each stream to each vector of pointers 
    // NOTE: Need one of these arrays for each of the CUDA streams
    for (int i = 0; i < this->nStreams; i++)
    {
        gpuCASImgs_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("gpuCASImgs_" + std::to_string(i)));
        gpuCoordAxes_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("gpuCoordAxes_" + std::to_string(i)));
    }

    // Get the pointers to the CPU input / output arrays
    float * CASImgs_CPU_Pinned   = this->Mem_obj->ReturnCPUFloatPtr("CASImgs_CPU_Pinned");
    float * coordAxes_CPU_Pinned = this->Mem_obj->ReturnCPUFloatPtr("coordAxes_CPU_Pinned");

    // Each axes has 9 elements (3 for each x, y, z)
    int nAxes = this->axesSize[0] / 9; 
    
    // NOTE: gridSize times blockSize needs to equal imgSize
    int gridSize  = 32;// 32  
    int blockSize = this->imgSize[0] / gridSize ; // 4  

    // Verify all parameters and inputs are valid
    int parameter_check = ParameterChecking(    
        gpuVol_Vector, gpuCASImgs_Vector, gpuCoordAxes_Vector, ker_bessel_Vector, // Vector of GPU array pointers
        CASImgs_CPU_Pinned, coordAxes_CPU_Pinned, // Pointers to pinned CPU arrays for input / output
        this->volSize[0], this->imgSize[0], nAxes, *this->maskRadius, this->kerSize, this->kerHWidth, // kernel Parameters and constants
        numGPUs, this->nStreams, gridSize, blockSize, this->nBatches // Streaming parameters)
    );

    // If an error was detected return and don't start the CUDA kernel
    if (parameter_check != 0)
    {
        std::cerr << "Error detected in input parameters. Stopping the gpuForwardProjection now." << '\n';
        return;
    }   

    // Pass the vector of pointers to the C++ function in gpuForwardProject.cu
    // Which will step up and run the CUDA streams
    gpuForwardProject(
        gpuVol_Vector, gpuCASImgs_Vector, gpuCoordAxes_Vector, ker_bessel_Vector, // Vector of GPU arrays
        CASImgs_CPU_Pinned, coordAxes_CPU_Pinned, // Pointers to pinned CPU arrays for input / output
        this->volSize[0], this->imgSize[0], nAxes, *this->maskRadius, this->kerSize, this->kerHWidth, // kernel parameters
        numGPUs, this->nStreams, gridSize, blockSize, this->nBatches // Streaming parameters
        ); 

    return;

}



int CUDA_Gridder::ParameterChecking(
    std::vector<float*> gpuVol_Vector, std::vector<float*> gpuCASImgs_Vector,       // Vector of GPU array pointers
    std::vector<float*> gpuCoordAxes_Vector, std::vector<float*> ker_bessel_Vector, // Vector of GPU array pointers
    float * CASImgs_CPU_Pinned, float * coordAxes_CPU_Pinned, // Pointers to pinned CPU arrays for input / output
    int volSize, int imgSize, int nAxes, float maskRadius, int kerSize, float kerHWidth, // kernel Parameters and constants
    int numGPUs, int nStreams, int gridSize, int blockSize, int nBatches // Streaming parameters)
)
{
    // Check all the input parameters to verify they are all valid

    // Checking parameter: numGPUs
    int  numGPUDetected;
    cudaGetDeviceCount(&numGPUDetected);

    if (numGPUs < 0 || numGPUs >= numGPUDetected + 1){ //  An invalid numGPUs selection was chosen
        std::cerr << "Error in GPU selection. Please provide an integer between 0 and the number of NVIDIA graphic cards on your computer. Use SetNumberGPUs() function." << '\n';
        return -1;
    }
    
    if ( numGPUDetected == 0 ) // No GPUs were found (i.e. all cards are busy)
    {
        std::cerr << "No NVIDIA graphic cards identified on your computer. All cards may be busy and unavailable. Try restarting the program and/or your computer." << '\n';;          
        return -1;
    }

    // Checking parameter: nStreams
    if (nStreams <= 0 || nStreams < numGPUs)
    {
        std::cout << "nStreams: " << nStreams << '\n';
        std::cout << "numGPUs: " << numGPUs << '\n';

        std::cerr << "Invalid number of streams provided. Please use SetNumberStreams() to set number of streams >= number of GPUs to use." << '\n';
        return -1;
    }
    
    // Checking parameter: nBatches
    if (nBatches <= 0)
    {
        std::cout << "nBatches: " << nBatches << '\n';

        std::cerr << "Invalid number of batches provided. Please use SetNumberBatches() to set a non-negative integer number." << '\n';
        return -1;
    }
    

    

    // Checking parameter: volSize
    if (volSize <= 0)
    {
        std::cerr << "Invalid volSize parameter. Please use SetVolume() to define the input volume." << '\n';
        return -1;
    }

    // Checking parameter: imgSize
    if (imgSize <= 0)
    {
        std::cout << "imgSize: " << imgSize << '\n';
        std::cerr << "Invalid imgSize parameter." << '\n';
        return -1;
    }
    
    // Checking parameter: nAxes
    if (nAxes <= 0)
    {
        std::cerr << "Invalid nAxes parameter. Please use SetAxes() to define the input coordinate axes." << '\n';
        return -1;
    }
    
    // Checking parameter: maskRadius
    if (maskRadius <= 0)
    {
        std::cerr << "Invalid maskRadius parameter." << '\n';
        return -1;
    }
    
    // Checking parameter: kerSize
    if (kerSize <= 0)
    {
        std::cerr << "Invalid kerSize parameter." << '\n';
        return -1;
    }
    // Checking parameter: kerHWidth
    if (kerHWidth <= 0)
    {
        std::cerr << "Invalid kerHWidth parameter." << '\n';
        return -1;
    }

    // Checking parameter: gridSize
    if (gridSize <= 0)
    {
        std::cerr << "Invalid gridSize parameter." << '\n';
        return -1;
    }

    // Checking parameter: blockSize
    if (blockSize <= 0 || imgSize != gridSize * blockSize) // NOTE: gridSize times blockSize needs to equal imgSize
    {
        std::cerr << "Invalid blockSize parameter. gridSize * blockSize must equal imgSize" << '\n';
        return -1;
    }

    // Checking parameters: gpuVol_Vector, gpuCASImgs_Vector, gpuCoordAxes_Vector, and ker_bessel_Vector
    if (gpuVol_Vector.size() <= 0 || gpuCASImgs_Vector.size() <= 0 || gpuCoordAxes_Vector.size() <= 0 || ker_bessel_Vector.size() <= 0)
    {
        std::cerr << "gpuForwardProject(): Input GPU pointer vectors are empty. Has SetVolume() and SetAxes() been run?" << '\n';
        return -1;
    }

    // No errors were detected so return a flag of 0
    return 0;
}




    // mexCallMATLAB(1, &Uout[1], 2, Uin, "rand");

    // // int mexCallMATLAB(int nlhs, mxArray *plhs[], int nrhs,
    // // mxArray *prhs[], const char *functionName);

    // double* ptr = (double*)mxGetData(Uout[0]);


    // std::cout << "Uout: " << ptr[0] << " " << ptr[1] << '\n';