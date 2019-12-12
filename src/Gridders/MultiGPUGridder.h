#pragma once

/**
 * @class   MultiGPUGridder
 * @brief   A class for gridding on multiple GPUs
 *
 *
 * This class is used for forward and back projection of a volume on a multiple NVIDIA GPUs. The MultiGPUGridder 
 * inherits from the AbstractClass which is used for setting and getting the host (i.e. CPU) memory pointers to the 
 * volume, coordinate axes vector, etc, and for setting various parameters such as the interpolation factor.
 *
 * The MultiGPUGridder creates a gpuGridder object for each GPU. The gpuGridder then does the processing for forward and 
 * back projection on the GPU it is assigned to. The MultiGPUGridder class simply calculates which coordinate axes each GPU
 * will process. 
 * 
 * After the back projection is completed on all GPUs, the MultiGPUGridder class will combine the results from all the GPUs to the 
 * first GPU for reconstructing the final volume.
 * */

#include "AbstractGridder.h"
#include "gpuGridder.h"
#include "AddVolumeFilter.h"
#include "gpuErrorCheck.h"

#include <thread> // For multi-threading on the CPU

class MultiGPUGridder : public AbstractGridder
{

public:
	/// MultiGPUGridder constructor. The GPU_Devices array is a vector of Num_GPUs size which contains the NVIDIA device number for the GPUs to use which
	/// ranges from 0 to the number of GPUs minus 1 on the current computer. RunFFTOnDevice is a flag to either run the forward and inverse Fourier transforms
	/// on the GPUs (value of 1) or to run them on the host (value of 0) such as within Matlab or Python.
	MultiGPUGridder(int VolumeSize, int numCoordAxes, float interpFactor, int Num_GPUs, int *GPU_Devices, int RunFFTOnDevice, bool verbose = false) : AbstractGridder(VolumeSize, numCoordAxes, interpFactor)
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

		this->verbose = verbose;

		// Create an intance of the gpuGridder class for each GPU
		for (int i = 0; i < Num_GPUs; i++)
		{
			cudaSetDevice(GPU_Devices[i]);

			if (this->verbose == true)
			{
				std::cout << "MultiGPUGridder creating gpuGridder object on GPU " << GPU_Devices[i] << '\n';
			}

			// Delete any CUDA contexts on the current device (i.e. remove all memory allocations)
			cudaDeviceReset();

			gpuGridder *gpuGridder_obj = new gpuGridder(VolumeSize, numCoordAxes, interpFactor, RunFFTOnDevice, GPU_Devices[i], verbose);

			// Save the new object to the vector of gpuGridder objects
			gpuGridder_vec.push_back(gpuGridder_obj);
		}

		// Save the number of GPUs
		this->Num_GPUs = Num_GPUs;

		// Save the GPU device numbers
		this->GPU_Devices = new int[Num_GPUs];
		for (int i = 0; i < Num_GPUs; i++)
		{
			this->GPU_Devices[i] = GPU_Devices[i];
		}

		// Set the flag to false
		this->ProjectInitializedFlag = false;

		this->RunFFTOnDevice = RunFFTOnDevice;
		this->PeerAccessEnabled = false;
	}

	// Deconstructor
	~MultiGPUGridder()
	{
		FreeMemory();
	}

	/// Set the number of CUDA streams to use with each GPU for the forward projection
	void SetNumStreamsFP(int nStreams);

	/// Set the number of CUDA streams to use with each GPU for the back projection
	void SetNumStreamsBP(int nStreams);

	/// Run the forward projection kernel on each gpuGridder object
	void ForwardProject();

	/// Run the back projection kernel on each gpuGridder object
	void BackProject();

	/// Combine the CAS volumes on all the GPUs (the result from the back projection) and convert to volume using the first GPU and gpuGridder.
	void CASVolumeToVolume();

	/// Combine the CAS volumes on all the GPUs (the result from the back projection) and convert to volume using the first GPU and gpuGridder
	/// and normalizing by the plane density array.
	void ReconstructVolume();

	/// Get the plane density arrays from each gpuGridder, sum them together, and copy the result back to the host memory.
	void SumPlaneDensity();

	/// Get the volume arrays from each gpuGridder, sum them together, and copy the result back to the host memory.
	void SumVolumes();

	/// Get the CAS volume arrays from each gpuGridder, sum them together, and copy the result back to the host memory.
	void SumCASVolumes();

	/// Sum the CAS volumes from each gpuGridder to the given device after running the back projection
	void AddCASVolumes(int GPU_For_Reconstruction);

	/// Sum the plane densities on the GPU devices to the given device after running the back projection
	void AddPlaneDensities(int GPU_For_Reconstruction);

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

	/// Allow the first GPU to access the memory of the other GPUs
	/// This is needed for the reconstruct volume function
	// void EnablePeerAccess(int GPU_For_Reconstruction);

	bool PeerAccessEnabled;

	// Should we print status information to the console?
	bool verbose;
};