#include "AbstractFilter.h"

void AbstractFilter::SetCPUInput(float *input, int *InputSize, int GPU_Device)
{

    this->h_input_struct = new HostMemory<float>(3, InputSize);
    this->h_input_struct->CopyPointer(input);

    cudaSetDevice(GPU_Device);
    this->d_input_struct = new DeviceMemory<float>(3, InputSize, GPU_Device);
    this->d_input_struct->AllocateGPUArray();
    this->d_input_struct->CopyToGPU(this->h_input_struct->GetPointer(), this->h_input_struct->bytes());
}

void AbstractFilter::SetCPUOutput(float *output, int *OutputSize, int GPU_Device)
{
    this->h_output_struct = new HostMemory<float>(3, OutputSize);
    this->h_output_struct->CopyPointer(output);

    this->d_output_struct = new DeviceMemory<float>(3, OutputSize, GPU_Device);    
    this->d_output_struct->AllocateGPUArray();
    this->d_output_struct->Reset();
}

void AbstractFilter::GetCPUOutput(float *output)
{
    this->d_output_struct->CopyFromGPU(output, this->d_output_struct->bytes());
}
