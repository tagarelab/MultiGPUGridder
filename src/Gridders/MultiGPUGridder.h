#ifndef MULTI_GPU_GRIDDER_H // Only define the header once
#define MULTI_GPU_GRIDDER_H

#include "AbstractGridder.h"
#include "gpuGridder.h"
#include "AddVolumeFilter.h"

#include <thread> // For multi-threading on the CPU

class MultiGPUGridder : public AbstractGridder
{

public:
	// Constructor
	MultiGPUGridder(int VolumeSize, int numCoordAxes, float interpFactor, int Num_GPUs, int *GPU_Devices, int RunFFTOnDevice, int NormalizeByDensity) : AbstractGridder(VolumeSize, numCoordAxes, interpFactor)
	{
		std::cout << "RunFFTOnDevice: " << RunFFTOnDevice << '\n';

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
			cudaSetDevice(GPU_Device);

			// Delete any CUDA contexts on the current device (i.e. remove all memory allocations)
			cudaDeviceReset();

			gpuGridder *gpuGridder_obj = new gpuGridder(VolumeSize, numCoordAxes, interpFactor, RunFFTOnDevice, NormalizeByDensity, GPU_Device);

			// Save the new object to the vector of gpuGridder objects
			gpuGridder_vec.push_back(gpuGridder_obj);
		}

		// Save the number of GPUs
		this->Num_GPUs = Num_GPUs;

		// Save the GPU device numbers
		this->GPU_Devices = GPU_Devices;

		// Set the flag to false
		this->ProjectInitializedFlag = false;

		this->RunFFTOnDevice = RunFFTOnDevice;

		this->NormalizeByDensity = NormalizeByDensity;
	}

	// Deconstructor
	~MultiGPUGridder()
	{
		FreeMemory();
	}

	// Set the number of CUDA streams to use with each GPU
	void SetNumStreams(int nStreams);

	// Run the forward projection kernel on each gpuGridder object
	void ForwardProject();

	// Run the back projection kernel on each gpuGridder object
	void BackProject();

	// Combine the CAS volumes on all the GPUs and convert to volume
	void CASVolumeToVolume();

	// Convert CAS volume to volume and normalize by the plane density
	void ReconstructVolume();

	// Get the volumes from each GPU, sum them together, and copy the result back to the host memory
	void SumPlaneDensity();
	void SumVolumes();
	void SumCASVolumes();

	// Sum the CAS volumes on the GPU devices to the given device after running the back projection
	void AddCASVolumes(int GPU_Device);

	// Sum the plane densities on the GPU devices to the given device after running the back projection
	void AddPlaneDensities(int GPU_Device);

private:
	// Plan which GPU will process which coordinate axes
	struct CoordinateAxesPlan
	{
		// Vector to hold the number of axes to assign for each GPU
		std::vector<int> NumAxesPerGPU;

		// Offset index of the starting coordinate axes for each GPU
		int *coordAxesOffset;
	};

	// Function to assign the coordinate axes to each GPU
	CoordinateAxesPlan PlanCoordinateAxes();

	// Vector to hold the gpuGridder objects (one for each GPU)
	std::vector<gpuGridder *> gpuGridder_vec;

	// How many GPUs to use
	int Num_GPUs;

	// GPU device numbers
	int *GPU_Devices;

	// Synchronize all of the GPUs
	void GPU_Sync();

	// Free all of the allocated memory
	void FreeMemory();

	// Flag for remembering if this is the first time running the forward projection
	bool ProjectInitializedFlag;

	// Flag to determine whether we are running the FFT on the GPU or not
	int RunFFTOnDevice;

	// Flag to normalize the back projected volume by the plane density
	int NormalizeByDensity;
};

#endif