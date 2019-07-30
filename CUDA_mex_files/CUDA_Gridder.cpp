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
        std::cerr << "No NVIDIA graphic cards located on your computer. All cards may be busy and unavailable. Try restarting the program and/or your computer." << '\n';;  
        
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

// Set the GPU Volume
void CUDA_Gridder::SetVolume(float* gpuVol, int* gpuVolSize)
{      

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

        // After allocating the gpuVol array on the gpuDevice, lets copy the memory
        Mem_obj->CUDA_Copy("gpuVol_" + std::to_string(gpuDevice), gpuVol);    

    }

    // Save the volume size for later
    this->volSize = gpuVolSize;

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
    this->imgSize = imgSize;

    std::cout << "imgSize: " << imgSize[0] << " " << imgSize[1] << " " << imgSize[2] << '\n';
}

// Set the maskRadius parameter
void CUDA_Gridder::SetMaskRadius(float* maskRadius)
{
    this->maskRadius = maskRadius;

    std::cout << "maskRadius: " << maskRadius << '\n';

}

void CUDA_Gridder::Forward_Project_Initilize()
{
    // Initialize all the needed CPU and GPU pointers and check that all the required pointers exist

    // Check each GPU to determine if all the required pointers are already allocated
    for (int gpuDevice = 0; gpuDevice < this->numGPUs; gpuDevice++)
    {
        // Has the output array been allocated and defined already?
        // The name of the GPU pointer is gpuCASImgs_0 for GPU 0, gpuCASImgs_1 for GPU 1, etc.
        if ( Mem_obj->GPUArrayAllocated("gpuCASImgs_" + std::to_string(gpuDevice), gpuDevice) == false) 
        {
            // We need to allocate the gpuCASImgs array on this GPU
            int * gpuCASImgs_Size = new int[3];
            gpuCASImgs_Size[0] = this->imgSize[0];
            gpuCASImgs_Size[1] = this->imgSize[1];
            // gpuCASImgs_Size[2] = this->imgSize[2]; 
   
            //  Each GPU only needs to hold a fraction of the total output images
            // Should probably be this->numGPUs but am getting error
            gpuCASImgs_Size[2] = ceil(this->imgSize[2] / (this->numGPUs)) + 1; 
   
            
            Mem_obj->CUDA_alloc("gpuCASImgs_" + std::to_string(gpuDevice), "float", gpuCASImgs_Size, gpuDevice);        
        }

        // Has the Kaiser bessel vector been allocated and defined?
        // The name of the GPU pointer is ker_bessel_0 for GPU 0, ker_bessel_1 for GPU 1, etc.
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

        // Has the coordAxes array been allocated and defined?
        // The name of the GPU pointer is gpuCoordAxes_0 for GPU 0, gpuCoordAxes_1 for GPU 1, etc.
        if ( Mem_obj->GPUArrayAllocated("gpuCoordAxes_" + std::to_string(gpuDevice), gpuDevice) == false) 
        {
            // Allocate the gpuCoordAxes on the current gpuDevice
            int * gpuCoordAxes_Size = new int[3];

            // Each GPU only needs to hold a fraction of the total axes vector
            // Am getting an error so adding a few bytes to the end
            gpuCoordAxes_Size[0] = ceil(this->axesSize[0] / this->numGPUs) + 10; 
            // gpuCoordAxes_Size[0] = this->axesSize[0]; 
            gpuCoordAxes_Size[1] = this->axesSize[1];
            gpuCoordAxes_Size[2] = this->axesSize[2];

            Mem_obj->CUDA_alloc("gpuCoordAxes_" + std::to_string(gpuDevice), "float", gpuCoordAxes_Size, gpuDevice);            
        }
    }    

    // Has the output image array been allocated and pinned to the CPU?
    if ( Mem_obj->CPUArrayAllocated("CASImgs_CPU_Pinned") == false) 
    {
        // We need to allocate the coordAxes array on this axesSize
        Mem_obj->mem_alloc("CASImgs_CPU_Pinned", "float", this->imgSize);
        
        // Lastly, pin the array to allow for async CUDA streaming
        Mem_obj->pin_mem("CASImgs_CPU_Pinned");
        
    }

}

// Run the Forward Projection CUDA kernel
void CUDA_Gridder::Forward_Project(){
    // Run the forward projection CUDA kernel

    std::cout << "CUDA_Gridder::Forward_Project()" << '\n';
        
    // Initialize all the needed CPU and GPU pointers and check that all the required pointers exist
    Forward_Project_Initilize();

    // TO DO: Check the input variables. Is each one the correct type for the kernel? (i.e. CPU vs GPU, int vs float, etc.)

    // Create a vector of GPU pointers
    std::vector<float*> gpuVol_Vector;
    std::vector<float*> gpuCASImgs_Vector;
    std::vector<float*> ker_bessel_Vector;
    std::vector<float*> gpuCoordAxes_Vector;

    // Find and add the corresponding GPU pointer to each vector of pointers
    for (int gpuDevice = 0; gpuDevice < this->numGPUs; gpuDevice++)
    {        
        gpuVol_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("gpuVol_" + std::to_string(gpuDevice)));
        gpuCASImgs_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("gpuCASImgs_" + std::to_string(gpuDevice)));
        gpuCoordAxes_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("gpuCoordAxes_" + std::to_string(gpuDevice)));
        ker_bessel_Vector.push_back(this->Mem_obj->ReturnCUDAFloatPtr("ker_bessel_" + std::to_string(gpuDevice)));        
    }
    
    // Get the pointers to the CPU input / output arrays
    float * CASImgs_CPU_Pinned   = this->Mem_obj->ReturnCPUFloatPtr("CASImgs_CPU_Pinned");
    float * coordAxes_CPU_Pinned = this->Mem_obj->ReturnCPUFloatPtr("coordAxes_CPU_Pinned");

    // Each axes has 9 elements (3 for each x, y, z)
    int nAxes = this->axesSize[0] / 9; 

    int numGPUs   = 4;
    int nStreams  = 4; // One stream for each GPU for now
    
    // NOTE: gridSize times blockSize needs to equal imgSize
    int gridSize  = 32;// 32  
    int blockSize = 8; // 4    

    int volSize   = 262;//134;
    int imgSize   = 256;//128;

    // Pass the vector of pointers to the C++ function in gpuForwardProject.cu
    // Which will step up and run the CUDA streams
    gpuForwardProject(
        gpuVol_Vector, gpuCASImgs_Vector, gpuCoordAxes_Vector, ker_bessel_Vector, // Vector of GPU arrays
        CASImgs_CPU_Pinned, coordAxes_CPU_Pinned, // Pointers to pinned CPU arrays for input / output
        volSize, imgSize, nAxes, 63, 501, 2, // kernel parameters
        numGPUs, nStreams, gridSize, blockSize// Streaming parameters
        ); //2034

    return;

}



    // mexCallMATLAB(1, &Uout[1], 2, Uin, "rand");

    // // int mexCallMATLAB(int nlhs, mxArray *plhs[], int nrhs,
    // // mxArray *prhs[], const char *functionName);

    // double* ptr = (double*)mxGetData(Uout[0]);


    // std::cout << "Uout: " << ptr[0] << " " << ptr[1] << '\n';