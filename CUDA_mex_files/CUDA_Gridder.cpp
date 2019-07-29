#include "CUDA_Gridder.h"

// Constructor
CUDA_Gridder::CUDA_Gridder(){

    // Create a new instance of the CPU_CUDA_Memory class
    this->Mem_obj = new CPU_CUDA_Memory;

    std::cout << "CUDA_Gridder() constructor" << '\n';
}

// Run the Forward Projection CUDA kernel
void CUDA_Gridder::Forward_Project(std::vector<std::string> Input_Strings){
    // Input: Vector of strings in which the strings refer to variable names in the CPU_CUDA_Memory class

    std::cout << "CUDA_Gridder::Forward_Project()" << '\n';
    


    // call: out1 = rand(5,5); out2 = rand(5,5);
    mxArray *Uin[2], *Uout[2];
    Uin[0] = mxCreateDoubleScalar(5);
    Uin[1] = mxCreateDoubleScalar(5);
    mexCallMATLAB(1, &Uout[0], 2, Uin, "rand");
    mexCallMATLAB(1, &Uout[1], 2, Uin, "rand");

    // int mexCallMATLAB(int nlhs, mxArray *plhs[], int nrhs,
    // mxArray *prhs[], const char *functionName);

    double* ptr = (double*)mxGetData(Uout[0]);


    std::cout << "Uout: " << ptr[0] << " " << ptr[1] << '\n';



    // TO DO: Check the input variables. Is each one the correct type for the kernel? (i.e. CPU vs GPU, int vs float, etc.)

    //int * arr_1 = this->Mem_obj->ReturnCPUIntPtr(Input_Strings[0]);
    //std::cout << "arr_1 ptr: " << arr_1 << '\n';

    if (Input_Strings.size() != 10)
    {
        std::cout << "Wrong number of inputs provided. The number should be equal to 10." << '\n';
        return;
    }

    // Get the pointers to the CUDA GPU arrays first
    float* vol = this->Mem_obj->ReturnCUDAFloatPtr(Input_Strings[0]);
    float* img = this->Mem_obj->ReturnCUDAFloatPtr(Input_Strings[1]);
    float* axes = this->Mem_obj->ReturnCUDAFloatPtr(Input_Strings[2]);
    float* ker = this->Mem_obj->ReturnCUDAFloatPtr(Input_Strings[3]);

    // Get the pointers to the other parameters (non-GPU) next
    int* volSize = this->Mem_obj->ReturnCPUIntPtr(Input_Strings[4]);
    int* imgSize = this->Mem_obj->ReturnCPUIntPtr(Input_Strings[5]);
    int* nAxes = this->Mem_obj->ReturnCPUIntPtr(Input_Strings[6]);
    float* maskRadius = this->Mem_obj->ReturnCPUFloatPtr(Input_Strings[7]);
    int* kerSize = this->Mem_obj->ReturnCPUIntPtr(Input_Strings[8]);
    float* kerHWidth = this->Mem_obj->ReturnCPUFloatPtr(Input_Strings[9]);

    //     const float* vol, float* img, float *axes, float* ker, // GPU arrays
    //     int volSize, int imgSize, int nAxes, float maskRadius, int kerSize, float kerHWidth // Parameters
    
    std::cout << "volSize: " << volSize[0] <<'\n';
    std::cout << "maskRadius: " << maskRadius[0] <<'\n';

    // Run the kernel now
    gpuForwardProject(vol, img, axes, ker, volSize[0], imgSize[0], nAxes[0], maskRadius[0], kerSize[0], kerHWidth[0] );
        

}



