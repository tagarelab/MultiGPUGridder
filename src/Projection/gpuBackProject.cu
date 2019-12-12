#include "gpuBackProject.h"
#include <math.h> /* round, floor, ceil, trunc */

__global__ void gpuBackProjectKernel(float *vol, int volSize, float *img, int imgSize,
									 float *axes, int nAxes, float maskRadius,
									 const float *ker, int kerSize, float kerHWidth, int VolumeOffset)

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
	float w, cumSum, cumSumAllAxes;

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
	vi = blockDim.x * blockIdx.x + threadIdx.x + VolumeOffset;
	vj = blockDim.y * blockIdx.y + threadIdx.y + VolumeOffset;
	vk = blockDim.z * blockIdx.z + threadIdx.z + VolumeOffset;

	// Are we outside the volume bounds?
	if (vi < 0 || vi >= volSize || vj < 0 || vj >= volSize || vk < 0 || vk >= volSize)
	{
		return;
	}

	volCenter = (int)((float)volSize) / 2;
	imgCenter = (int)((float)imgSize) / 2;

	r = sqrtf((float)(vi - volCenter) * (vi - volCenter) + (vj - volCenter) * (vj - volCenter) + (vk - volCenter) * (vk - volCenter));

	if ((vi < volSize) && (vj < volSize) && (vk < volSize) && (r <= maskRadius))
	{
		cumSumAllAxes = 0;

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
						}

					} //for i1
					  //  atomicAdd((float *)vol+vk*volSize*volSize+vj*volSize+vi,(float)cumSum);
				cumSumAllAxes += cumSum;
			} // If f_imgk
		}	 // for axesIndex

		/* Add the accumulated All axes sum to the volume */
		atomicAdd((float *)vol + vk * volSize * volSize + vj * volSize + vi, (float)cumSumAllAxes);
	} //If vi,vj,vk
}

void gpuBackProject::RunKernel(
	float *d_CASVolume,
	float *d_CASImgs,
	float *d_KB_Table,
	float *d_CoordAxes,
	float kerHWidth,
	int nAxes,
	int CASVolSize,
	int CASImgSize,
	int maskRadius,
	int KB_Table_Size,
	int extraPadding,
	cudaStream_t *stream)
{

	// Define CUDA kernel dimensions
	int VolSize = (CASVolSize - extraPadding * 2) / 2;
	int BlockSize = 4;
	int GridSize = ceil((double)VolSize / (double)BlockSize);

	// VolumeOffset is the amount to add to the x,y,z to get the first voxel in the unpadded volume
    // i.e. there is no value in iterating over voxels which will always be zero
	int VolumeOffset = (CASVolSize - VolSize) / 2;

	// Define CUDA kernel dimensions
	dim3 dimGrid(GridSize, GridSize, GridSize);
	dim3 dimBlock(BlockSize, BlockSize, BlockSize);

	// Run the back projection kernel
	if (stream != NULL)
	{
		gpuBackProjectKernel<<<dimGrid, dimBlock, 0, *stream>>>(
			d_CASVolume, CASVolSize, d_CASImgs, CASImgSize, d_CoordAxes,
			nAxes, maskRadius, d_KB_Table, KB_Table_Size, kerHWidth, VolumeOffset);
	}
	else
	{
		gpuBackProjectKernel<<<dimGrid, dimBlock>>>(
			d_CASVolume, CASVolSize, d_CASImgs, CASImgSize, d_CoordAxes,
			nAxes, maskRadius, d_KB_Table, KB_Table_Size, kerHWidth, VolumeOffset);
	}

	gpuErrorCheck(cudaPeekAtLastError());

	return;
}
