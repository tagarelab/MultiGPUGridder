#include "gpuGridder.h"

// gpuGridder::gpuGridder()
// {
//     // Constructor
// }

// gpuGridder::~gpuGridder()
// {
//     // Deconstructor
// }

void gpuGridder::ForwardProject()
{
    std::cout << "ForwardProject()" << '\n';

    // Assume that each time we run the forward project that a new volume was previously set
    // Later could add a flag to skip this step
    // First run the volume to CAS volume function
    float* CASVolume = gpuFFT::VolumeToCAS(this->Volume, this->VolumeSize[0], this->interpFactor, this->extraPadding);


    // Note: This modifies the Matlab array in-place

}



 float *gpuGridder::GetVolume()
 {
     std::cout << "Volume: ";
     this->Volume[0] = 12;
     for (int i=0; i<10; i++)
     {
        std::cout << this->Volume[i] << " ";
     }
     std::cout << '\n';
     
    return this->Volume;

 }