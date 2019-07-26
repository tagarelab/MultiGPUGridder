#include "CUDA_Gridder.h"

// Constructor
CUDA_Gridder::CUDA_Gridder(){

    // Create a new instance of the CPU_CUDA_Memory class
    this->Mem_obj = new CPU_CUDA_Memory;

    std::cout << "CUDA_Gridder() constructor" << '\n';
}




