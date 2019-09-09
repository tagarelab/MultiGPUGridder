#ifndef MULTI_GPU_GRIDDER_H // Only define the header once
#define MULTI_GPU_GRIDDER_H

#include "AbstractGridder.h"
#include "gpuGridder.h"

class MultiGPUGridder : public AbstractGridder
{

public:
	// Constructor
	MultiGPUGridder(int VolumeSize, int numCoordAxes, float interpFactor, int Num_GPUs, int *GPU_Devices) : AbstractGridder(VolumeSize, numCoordAxes, interpFactor)
	{
		// Check the input parameters: Number of GPUs
		// Check how many GPUs there are on the computer
		int numGPUDetected;
		cudaGetDeviceCount(&numGPUDetected);
		if (Num_GPUs <= 0 || Num_GPUs > numGPUDetected)
		{
			std::cerr << "Number of GPUs provided " << Num_GPUs << '\n';
			std::cerr << "Error in number of GPUs. Please provide an integer between 0 and the number of NVIDIA graphic cards on your computer. Use SetNumberGPUs() function." << '\n';
			return;
		}

		// Create an intance of the gpuGridder class for each GPU
		for (int i = 0; i < Num_GPUs; i++)
		{
			int GPU_Device = GPU_Devices[i];

			gpuGridder *gpuGridder_obj = new gpuGridder(VolumeSize, numCoordAxes, interpFactor, GPU_Device);

			// Save the new object to the vector of gpuGridder objects
			gpuGridder_vec.push_back(gpuGridder_obj);
		}

		// Save the number of GPUs
		this->Num_GPUs = Num_GPUs;

		// Save the GPU device numbers
		this->GPU_Devices = GPU_Devices;

	}

	// Set the number of CUDA streams to use with each GPU
	void SetNumStreams(int nStreams);

	// Run the forward projection kernel on each gpuGridder object
	void ForwardProject();

private:
	// Vector to hold the gpuGridder objects (one for each GPU)
	std::vector<gpuGridder *> gpuGridder_vec;

	// How many GPUs to use
	int Num_GPUs;

	// GPU device numbers
	int *GPU_Devices;

	// Synchronize all of the GPUs
	void GPU_Sync();
};

#endif