#include "gpuBackProject.h"
#include <math.h> /* round, floor, ceil, trunc */

__global__ void gpuBackProjectKernel(float *vol, int volSize, float *densityVolume, float *img, int imgSize,
									 float *axes, int nAxes, float maskRadius,
									 const float *ker, int kerSize, float kerHWidth)

{
	float *img_ptr;
	int convW = roundf(kerHWidth);
	int kerIndex, axesIndex;
	int vi, vj, vk, i1, j1;
	float f_imgj, f_imgi, f_imgk;
	int imgi, imgj;
	float imgi1, imgj1, imgk1;
	int volCenter, imgCenter;
	float *nx, *ny, r, *nz;
	float kerCenter = ((float)kerSize - 1) / 2;
	float kerScale = kerCenter / kerHWidth;
	float w, cumSum, cumSumAllAxes, cumDensity, cumSumDensity;

	__shared__ float locKer[1000];

	if (threadIdx.x == 0)
	{
		/* Copy over the kernel */
		for (kerIndex = 0; kerIndex < kerSize; kerIndex++)
		{
			locKer[kerIndex] = *(ker + kerIndex);
		}
	}
	__syncthreads();

	/* Get the volume indices */
	vi = blockDim.x * blockIdx.x + threadIdx.x;
	vj = blockDim.y * blockIdx.y + threadIdx.y;
	vk = blockDim.z * blockIdx.z + threadIdx.z;

	// Are we inside the volume bounds?
	if (vi < 0 || vi > volSize || vj < 0 || vj > volSize || vk < 0 || vk > volSize)
	{
		return;
	}

	volCenter = (int)((float)volSize) / 2;
	imgCenter = (int)((float)imgSize) / 2;

	r = sqrtf((float)(vi - volCenter) * (vi - volCenter) + (vj - volCenter) * (vj - volCenter) + (vk - volCenter) * (vk - volCenter));

	if ((vi < volSize) && (vj < volSize) && (vk < volSize) && (r <= maskRadius))
	{
		cumSumAllAxes = 0;
		cumSumDensity = 0;

		for (axesIndex = 0; axesIndex < nAxes; axesIndex++)
		{

			nx = axes + 9 * axesIndex;
			ny = nx + 3;
			nz = ny + 3;

			/* Calculate coordinates in image frame */
			f_imgi = ((float)vi - volCenter) * (*nx) + ((float)vj - volCenter) * (*(nx + 1)) + ((float)vk - volCenter) * (*(nx + 2)) + imgCenter;
			f_imgj = ((float)vi - volCenter) * (*ny) + ((float)vj - volCenter) * (*(ny + 1)) + ((float)vk - volCenter) * (*(ny + 2)) + imgCenter;
			f_imgk = ((float)vi - volCenter) * (*nz) + ((float)vj - volCenter) * (*(nz + 1)) + ((float)vk - volCenter) * (*(nz + 2));

			if (fabsf(f_imgk) <= kerHWidth)
			{
				imgi = roundf(f_imgi);
				imgj = roundf(f_imgj);

				img_ptr = img + axesIndex * imgSize * imgSize;

				cumSum = 0;
				cumDensity = 0;
				for (j1 = imgj - convW; j1 <= imgj + convW; j1++)
					for (i1 = imgi - convW; i1 <= imgi + convW; i1++)
					{
						imgi1 = (i1 - imgCenter) * (*nx) + (j1 - imgCenter) * (*ny) + volCenter;
						r = (float)imgi1 - vi;
						r = min(max(r, (float)-convW), (float)convW);
						kerIndex = roundf(r * kerScale + kerCenter);
						kerIndex = min(max(kerIndex, 0), kerSize - 1);
						w = *(locKer + kerIndex);

						imgj1 = (i1 - imgCenter) * (*(nx + 1)) + (j1 - imgCenter) * (*(ny + 1)) + volCenter;
						r = (float)imgj1 - vj;
						r = min(max(r, (float)-convW), (float)convW);
						kerIndex = roundf(r * kerScale + kerCenter);
						kerIndex = min(max(kerIndex, 0), kerSize - 1);
						w = w * (*(locKer + kerIndex));

						imgk1 = (i1 - imgCenter) * (*(nx + 2)) + (j1 - imgCenter) * (*(ny + 2)) + volCenter;
						r = (float)imgk1 - vk;
						r = min(max(r, (float)-convW), (float)convW);
						kerIndex = roundf(r * kerScale + kerCenter);
						kerIndex = min(max(kerIndex, 0), kerSize - 1);
						w = w * (*(locKer + kerIndex));

						// Check that this is within the image bounds (not needed when using feval from Matlab it seems)
						if (j1 * imgSize + i1 < imgSize * imgSize && j1 * imgSize + i1 >= 0)
						{
							cumSum += (*(img_ptr + j1 * imgSize + i1)) * w;
							cumDensity += w;
						}

					} //for i1
					  //  atomicAdd((float *)vol+vk*volSize*volSize+vj*volSize+vi,(float)cumSum);
				cumSumAllAxes += cumSum;
				cumSumDensity += cumDensity;
			} // If f_imgk
		}	 // for axesIndex

		/* Add the accumulated All axes sum to the volume */
		atomicAdd((float *)vol + vk * volSize * volSize + vj * volSize + vi, (float)cumSumAllAxes);

		/* Add the plane density to the density volume (for normalizing the CASVolume later) */
		atomicAdd((float *)densityVolume + vk * volSize * volSize + vj * volSize + vi, (float)cumSumDensity);

	} //If vi,vj,vk
}

gpuBackProject::Offsets gpuBackProject::PlanOffsetValues()
{
	// Loop through all of the coordinate axes and calculate the corresponding pointer offset values
	// which are needed for running the CUDA kernels

	// For compactness define the CASImgSize, CASVolSize, and ImgSize here
	int CASImgSize = this->d_CASImgs->GetSize(0);
	int CASVolSize = this->d_CASVolume->GetSize(0);
	int ImgSize = this->d_Imgs->GetSize(0);

	// Create an instance of the Offsets struct
	Offsets Offsets_obj;
	Offsets_obj.num_offsets = 0;

	// Cumulative number of axes which have already been assigned to a CUDA stream
	int processed_nAxes = 0;

	// While we have coordinate axes to process, loop through the GPUs and the streams
	int MaxBatches = 10000; // Maximum iterations in case we get stuck in the while loop for some reason
	int batch = 0;

	// Initilize a variable which will remember how many axes have been assigned to the GPU
	// during the current batch. This is needed for calculating the offset for the stream when
	// the number of streams is greater than the number of GPUs. This resets to zeros after each
	// batch since the same GPU memory is used for the next batch.
	int numAxesGPU_Batch = 0;

	// Estimate how many coordinate axes to assign to each stream
	int EstimatedNumAxesPerStream;
	if (this->nAxes <= this->MaxAxesAllocated)
	{
		// The number of coordinate axes is less than or equal to the total number of axes to process
		EstimatedNumAxesPerStream = ceil((double)this->nAxes / (double)this->nStreams);
	}
	else
	{
		// Several batches will be needed so evenly split the MaxAxesAllocated by the number of streams
		EstimatedNumAxesPerStream = ceil((double)this->MaxAxesAllocated / (double)this->nStreams);
	}

	while (processed_nAxes < this->nAxes && batch < MaxBatches)
	{
		for (int i = 0; i < this->nStreams; i++) // Loop through the streams
		{
			// If we're about to process more than the number of coordinate axes, process the remaining faction of numAxesPerStream
			if (processed_nAxes + EstimatedNumAxesPerStream >= this->nAxes)
			{
				// Process the remaining fraction of EstimatedNumAxesPerStream
				Offsets_obj.numAxesPerStream.push_back(min(EstimatedNumAxesPerStream, nAxes - processed_nAxes));
			}
			else
			{
				// Save the estimated number of axes to the numAxesPerStream
				Offsets_obj.numAxesPerStream.push_back(EstimatedNumAxesPerStream);
			}

			// Calculate the offsets (in bytes) to determine which part of the array to copy for this stream
			// When using multiple GPUs coordAxesOffset will be the number already assigned to other GPUs
			Offsets_obj.CoordAxes_CPU_Offset.push_back((processed_nAxes + coordAxesOffset) * 9); // Each axes has 9 elements (X, Y, Z)
			Offsets_obj.coord_Axes_CPU_streamBytes.push_back(Offsets_obj.numAxesPerStream.back() * 9 * sizeof(float));

			// Use the number of axes already assigned to this GPU since starting the current batch to calculate the currect offset
			Offsets_obj.gpuCASImgs_Offset.push_back(numAxesGPU_Batch * CASImgSize * CASImgSize);
			Offsets_obj.gpuImgs_Offset.push_back(numAxesGPU_Batch * ImgSize * ImgSize);
			Offsets_obj.gpuCoordAxes_Stream_Offset.push_back(numAxesGPU_Batch * 9);

			// Optionally: Copy the resulting CAS images back to the host pinned memory (CPU)
			if (this->CASImgs_CPU_Pinned != NULL)
			{
				// Have to use unsigned long long since the array may be longer than the max value int32 can represent
				// imgSize is the size of the zero padded projection images
				unsigned long long *CASImgs_Offset = new unsigned long long[3];
				CASImgs_Offset[0] = (unsigned long long)(CASImgSize);
				CASImgs_Offset[1] = (unsigned long long)(CASImgSize);
				CASImgs_Offset[2] = (unsigned long long)(processed_nAxes + coordAxesOffset);

				Offsets_obj.CASImgs_CPU_Offset.push_back(CASImgs_Offset[0] * CASImgs_Offset[1] * CASImgs_Offset[2]);

				// How many bytes are the output images?
				Offsets_obj.gpuCASImgs_streamBytes.push_back(CASImgSize * CASImgSize * Offsets_obj.numAxesPerStream.back() * sizeof(float));
			}

			// Have to use unsigned long long since the array may be longer than the max value int32 can represent
			// imgSize is the size of the zero padded projection images
			unsigned long long *Imgs_Offset = new unsigned long long[3];
			Imgs_Offset[0] = (unsigned long long)(ImgSize);
			Imgs_Offset[1] = (unsigned long long)(ImgSize);
			Imgs_Offset[2] = (unsigned long long)(processed_nAxes + coordAxesOffset);

			Offsets_obj.Imgs_CPU_Offset.push_back(Imgs_Offset[0] * Imgs_Offset[1] * Imgs_Offset[2]);

			// How many bytes are the output images?
			Offsets_obj.gpuImgs_streamBytes.push_back(ImgSize * ImgSize * Offsets_obj.numAxesPerStream.back() * sizeof(float));

			// Update the overall number of coordinate axes which have already been assigned to a CUDA stream
			processed_nAxes = processed_nAxes + Offsets_obj.numAxesPerStream.back();

			// Update the number of axes which have been assigned to this GPU during the current batch
			numAxesGPU_Batch = numAxesGPU_Batch + Offsets_obj.numAxesPerStream.back();

			// Add one to the number of offset values
			Offsets_obj.num_offsets++;

			// Remember which stream this is
			Offsets_obj.stream_ID.push_back(i);
		}

		// Increment the batch number
		batch++;

		// Reset the number of axes processed during the current batch variable
		numAxesGPU_Batch = 0;
	}

	return Offsets_obj;
}

void gpuBackProject::Execute()
{
	// For compactness define the CASImgSize, CASVolSize, and ImgSize here
	int ImgSize = this->d_Imgs->GetSize(0);
	int CASImgSize = this->d_CASImgs->GetSize(0);
	int CASVolSize = this->d_CASVolume->GetSize(0);

	this->d_CASVolume->Reset(); // TEST

	// Plan the pointer offset values
	gpuBackProject::Offsets Offsets_obj = PlanOffsetValues();

    // Calculate the block size for running the CUDA kernels
    // NOTE: gridSize times blockSize needs to equal CASimgSize
    // int gridSize = 32; // 32
    // int blockSize = ceil(((double)this->d_Imgs->GetSize(0) * (double)this->interpFactor) / (double)gridSize);


	// Define CUDA kernel dimensions
	int gridSize = ceil(this->d_CASVolume->GetSize(0) / 4);
	int blockSize = 4;

	// Define CUDA kernel dimensions
	dim3 dimGrid(gridSize, gridSize, gridSize);
	dim3 dimBlock(blockSize, blockSize, blockSize);

	for (int i = 0; i < Offsets_obj.num_offsets; i++)
	{
		if (Offsets_obj.numAxesPerStream[i] < 1)
		{
			continue;
		}

		// std::cout << "GPU: " << this->GPU_Device << " stream " << Offsets_obj.stream_ID[i]
		// 		  << " processing " << Offsets_obj.numAxesPerStream[i] << " axes " << '\n';

		// Copy the section of gpuCoordAxes which this stream will process on the current GPU
		cudaMemcpyAsync(
			this->d_CoordAxes->GetPointer(Offsets_obj.gpuCoordAxes_Stream_Offset[i]),
			this->coordAxes_CPU_Pinned->GetPointer(Offsets_obj.CoordAxes_CPU_Offset[i]),
			Offsets_obj.coord_Axes_CPU_streamBytes[i],
			cudaMemcpyHostToDevice, streams[Offsets_obj.stream_ID[i]]);

		bool UseExsistingCASImages = true;

		if (UseExsistingCASImages == true)
		{
			// Copy the CASImages from the pinned CPU array and use instead of the images array
			cudaMemcpyAsync(
				this->d_CASImgs->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
				CASImgs_CPU_Pinned->GetPointer(Offsets_obj.CASImgs_CPU_Offset[i]),
				Offsets_obj.gpuCASImgs_streamBytes[i],
				cudaMemcpyHostToDevice,
				streams[Offsets_obj.stream_ID[i]]);
		}
		else
		{
			// Copy the section of images to the GPU
			// cudaMemcpyAsync(
			// 	this->d_Imgs->GetPointer(Offsets_obj.gpuImgs_Offset[i]),
			// 	Imgs_CPU_Pinned->GetPointer(Offsets_obj.Imgs_CPU_Offset[i]),
			// 	Offsets_obj.gpuImgs_streamBytes[i],
			// 	cudaMemcpyHostToDevice,
			// 	streams[Offsets_obj.stream_ID[i]]);

			// // Run the forward FFT to convert the pinned CPU images to CAS images (CAS type is needed for back projecting into the CAS volume)
			// this->gpuFFT_obj->ImgsToCASImgs(
			// 	streams[Offsets_obj.stream_ID[i]],
			// 	this->d_CASImgs->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
			// 	this->d_Imgs->GetPointer(Offsets_obj.gpuImgs_Offset[i]),
			// 	this->d_CASImgsComplex->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
			// 	Offsets_obj.numAxesPerStream[i]);
		}

		// Optionally: Copy the resulting CAS images back to the host pinned memory (CPU)
		// if (CASImgs_CPU_Pinned != NULL)
		// {
		// 	cudaMemcpyAsync(
		// 		CASImgs_CPU_Pinned->GetPointer(Offsets_obj.CASImgs_CPU_Offset[i]),
		// 		this->d_CASImgs->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
		// 		Offsets_obj.gpuCASImgs_streamBytes[i],
		// 		cudaMemcpyDeviceToHost,
		// 		streams[Offsets_obj.stream_ID[i]]);
		// }

		// cudaDeviceSynchronize();

		// Run the back projection kernel
		gpuBackProjectKernel<<<dimGrid, dimBlock, 0, streams[Offsets_obj.stream_ID[i]]>>>(
			this->d_CASVolume->GetPointer(),
			this->d_CASVolume->GetSize(0),
			this->d_PlaneDensity->GetPointer(),
			this->d_CASImgs->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
			this->d_CASImgs->GetSize(0),
			this->d_CoordAxes->GetPointer(Offsets_obj.gpuCoordAxes_Stream_Offset[i]),
			Offsets_obj.numAxesPerStream[i],
			this->maskRadius,
			this->d_KB_Table->GetPointer(),
			this->d_KB_Table->GetSize(0),
			this->kerHWidth);
	}

	return;
}
