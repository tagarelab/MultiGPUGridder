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
    
    // TO DO: Check the input variables. Is each one the correct type for the kernel? (i.e. CPU vs GPU, int vs float, etc.)

    int * arr_1 = this->Mem_obj->ReturnCPUIntPtr(Input_Strings[0]);

    std::cout << "arr_1 ptr: " << arr_1 << '\n';


    gpuForwardProject();

}



