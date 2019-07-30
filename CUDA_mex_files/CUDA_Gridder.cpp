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

    // Provide error message if no GPUs are found (i.e. all cards are busy) are an invalid selection is chosen
    if ( numGPUDetected == 0 )
    {
        std::cerr << "No NVIDIA graphic cards located on your computer. All cards may be busy and unavailable. Try restarting the program and/or your computer." << '\n';;  
    }

    if (numGPUs < 0 || numGPUs >= numGPUDetected){
        std::cerr << "Error in GPU selection. Please provide an integer between 0 and the number of NVIDIA graphic cards on your computer." << '\n';
    }

    // Save the user requested number of GPUs to use
    this->numGPUs = numGPUs;

    std::cout << "numGPUs: " << numGPUs << '\n';

}

// Set the GPU Volume
void CUDA_Gridder::SetVolume(float* gpuVol, int* gpuVolSize)
{   
    std::cout << "Setting gpuVol on GPU 0..." << '\n';

    int gpuDevice = 0; // Just use the first one for now

     // Has a gpuVol array already been allocated?    
    if ( Mem_obj->GPUArrayAllocated("gpuVol", gpuDevice) == false) 
    {
        // We need to allocate the gpuVol array on this gpuDevice
        Mem_obj->CUDA_alloc("gpuVol", "float", gpuVolSize, gpuDevice);
    }

    // After allocating the gpuVol array on the gpuDevice, lets copy the memory
     Mem_obj->CUDA_Copy("gpuVol", gpuVol);    

    // Save the volume size for later
    this->volSize = gpuVolSize;

}

// Set the coordinate axes Volume
void CUDA_Gridder::SetAxes(float* coordAxes, int* axesSize)
{   
    std::cout << "Setting coordAxes array..." << '\n';

    int gpuDevice = 0; // Just use the first GPU for now

     // Has a coordAxes array already been allocated?    
    if ( Mem_obj->GPUArrayAllocated("coordAxes", gpuDevice) == false) 
    {
        // We need to allocate the coordAxes array on this axesSize
        Mem_obj->CUDA_alloc("coordAxes", "float", axesSize, gpuDevice);
    }

    // After allocating the coordAxes array on the gpuDevice, lets copy the memory
     Mem_obj->CUDA_Copy("coordAxes", coordAxes);    

     // Remember the axesSize for later    
     this->axesSize = new int(*axesSize);

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
            Mem_obj->CUDA_alloc("gpuCASImgs_" + std::to_string(gpuDevice), "float", this->imgSize, gpuDevice);        
        }

        // Has the Kaiser bessel vector been allocated and defined?
        // The name of the GPU pointer is ker_bessel_Vector_0 for GPU 0, ker_bessel_Vector_1 for GPU 1, etc.
        if ( Mem_obj->GPUArrayAllocated("ker_bessel_Vector_" + std::to_string(gpuDevice), gpuDevice) == false) 
        {
            // Set the Kaiser Bessel Function vector to the current gpuDevice
            int arrSize[3];
            arrSize[0] = this->kerSize;
            arrSize[1] = 1;
            arrSize[2] = 1;

            // First allocate the Kaiser Bessel Function vector on the current gpuDevice
            Mem_obj->CUDA_alloc("ker_bessel_Vector_" + std::to_string(gpuDevice), "float", arrSize, gpuDevice);            

            // After allocating the gpuVol array on the gpuDevice, lets gpuDevicepy the memory
            Mem_obj->CUDA_Copy("ker_bessel_Vector_" + std::to_string(gpuDevice), this->ker_bessel_Vector);    
        }
    }    




}

// Run the Forward Projection CUDA kernel
void CUDA_Gridder::Forward_Project(){
    // Run the forward projection CUDA kernel

    std::cout << "CUDA_Gridder::Forward_Project()" << '\n';
        
    // Initialize all the needed CPU and GPU pointers and check that all the required pointers exist
    Forward_Project_Initilize();

    return;

    // TO DO: Check the input variables. Is each one the correct type for the kernel? (i.e. CPU vs GPU, int vs float, etc.)
    // Get the pointers to the CUDA GPU arrays first
    float* vol  = this->Mem_obj->ReturnCUDAFloatPtr("gpuVol");
    float* img  = this->Mem_obj->ReturnCUDAFloatPtr("gpuCASImgs");
    float* axes = this->Mem_obj->ReturnCUDAFloatPtr("coordAxes");
    float* ker  = this->Mem_obj->ReturnCUDAFloatPtr("ker");

    int nAxes = this->axesSize[0] / 9; // Each axes has 9 elements (3 for each x, y, z)

    // Run the kernel now   
    gpuForwardProject(vol, img, axes, ker, 134, 128, nAxes, 63, 501, 2 ); //2034

    // Get the pointers to the other parameters (non-GPU) next
    //int* volSize = this->Mem_obj->ReturnCPUIntPtr(Input_Strings[4]);
    //int* imgSize = this->Mem_obj->ReturnCPUIntPtr(Input_Strings[5]);
    //int* nAxes   = this->Mem_obj->ReturnCPUIntPtr(Input_Strings[6]);


    // float* maskRadius = this->Mem_obj->ReturnCPUFloatPtr(Input_Strings[7]);
    // int* kerSize = this->Mem_obj->ReturnCPUIntPtr(Input_Strings[8]);
    // float* kerHWidth = this->Mem_obj->ReturnCPUFloatPtr(Input_Strings[9]);

    //     const float* vol, float* img, float *axes, float* ker, // GPU arrays
    //     int volSize, int imgSize, int nAxes, float maskRadius, int kerSize, float kerHWidth // Parameters
    
    // std::cout << "volSize: " << volSize[0] <<'\n';
    // std::cout << "maskRadius: " << maskRadius[0] <<'\n';
    // gpuForwardProject(vol, img, axes, ker, volSize[0], imgSize[0], nAxes[0], maskRadius[0], kerSize[0], kerHWidth[0] );   

}



    // mexCallMATLAB(1, &Uout[1], 2, Uin, "rand");

    // // int mexCallMATLAB(int nlhs, mxArray *plhs[], int nrhs,
    // // mxArray *prhs[], const char *functionName);

    // double* ptr = (double*)mxGetData(Uout[0]);


    // std::cout << "Uout: " << ptr[0] << " " << ptr[1] << '\n';