#include "gpuBackProject.h"

__global__ void gpuBackProjectKernel(float* vol,int volSize, float* img,int imgSize,
                                     float * axes, int nAxes, float maskRadius,
                                    const float* ker,int kerSize,float kerHWidth)

{
float *img_ptr;
int convW=roundf(kerHWidth);
int kerIndex,axesIndex;
int vi,vj,vk,i1,j1;
float f_imgj,f_imgi,f_imgk;
int imgi,imgj;
float imgi1,imgj1,imgk1;
int volCenter,imgCenter;
float *nx,*ny,r,*nz;
float kerCenter=((float)kerSize-1)/2;
float kerScale=kerCenter/kerHWidth;
float w,cumSum,cumSumAllAxes;

__shared__ float locKer[1000];

    if (threadIdx.x==0)
    {
        /* Copy over the kernel */
        for (kerIndex=0;kerIndex<kerSize;kerIndex++) 
                locKer[kerIndex]=*(ker+kerIndex);
    }
    __syncthreads();

    /* Get the volume indices */
    vi=blockDim.x*blockIdx.x+threadIdx.x;
    vj=blockDim.y*blockIdx.y+threadIdx.y;
    vk=blockDim.z*blockIdx.z+threadIdx.z;

    volCenter=(int)((float)volSize)/2;
    imgCenter=(int)((float)imgSize)/2;

    r=sqrtf((float) (vi-volCenter)*(vi-volCenter)+(vj-volCenter)*(vj-volCenter)+(vk-volCenter)*(vk-volCenter));
   
    if ( (vi<volSize)&&(vj<volSize)&&(vk<volSize) && (r<=maskRadius) )
    {
            cumSumAllAxes=0;

            for (axesIndex=0;axesIndex<nAxes;axesIndex++)
                {

                nx=axes+9*axesIndex;  
                ny=nx+3;
                nz=ny+3;

                /* Calculate coordinates in image frame */
                f_imgi= ((float)vi-volCenter)*(*nx)+((float)vj-volCenter)*(*(nx+1))+((float)vk-volCenter)*(*(nx+2))+imgCenter;
                f_imgj= ((float)vi-volCenter)*(*ny)+((float)vj-volCenter)*(*(ny+1))+((float)vk-volCenter)*(*(ny+2))+imgCenter;
                f_imgk= ((float)vi-volCenter)*(*nz)+((float)vj-volCenter)*(*(nz+1))+((float)vk-volCenter)*(*(nz+2));  
                
                if (fabsf(f_imgk)<=kerHWidth)
                {
                        imgi=roundf(f_imgi);
                        imgj=roundf(f_imgj);

                        img_ptr=img+axesIndex*imgSize*imgSize;
                        
                   cumSum=0;
                    for (j1=imgj-convW;j1<=imgj+convW;j1++)
                        for (i1=imgi-convW;i1<=imgi+convW;i1++)
                        {
                            imgi1= (i1-imgCenter)*(*nx) + (j1-imgCenter)*(*ny)+volCenter;
                                        r= (float)imgi1-vi;
                                        r=min(max(r,(float)-convW),(float)convW);
                                        kerIndex=roundf( r*kerScale+kerCenter);
                                        kerIndex=min(max(kerIndex,0),kerSize-1);
                                        w=*(locKer+kerIndex);
                                        
                            imgj1= (i1-imgCenter)*(*(nx+1)) + (j1-imgCenter)*(*(ny+1))+volCenter;
                                        r= (float)imgj1-vj;
                                        r=min(max(r,(float)-convW),(float)convW);
                                        kerIndex=roundf( r*kerScale+kerCenter);
                                        kerIndex=min(max(kerIndex,0),kerSize-1);
                                        w=w*(*(locKer+kerIndex));

                            imgk1= (i1-imgCenter)*(*(nx+2)) + (j1-imgCenter)*(*(ny+2))+volCenter;
                                        r= (float)imgk1-vk;
                                        r=min(max(r,(float)-convW),(float)convW);
                                        kerIndex=roundf( r*kerScale+kerCenter);
                                        kerIndex=min(max(kerIndex,0),kerSize-1);
                                        w=w*(*(locKer+kerIndex));
                                        
                            // Check that this is within the image bounds (not needed when using feval from Matlab it seems)
                            if ( j1*imgSize+i1 < imgSize * imgSize && j1*imgSize+i1 >= 0)
                            {
                                cumSum += (*(img_ptr+j1*imgSize+i1))*w; 
                            }
                            
                        } //for i1
                  //  atomicAdd((float *)vol+vk*volSize*volSize+vj*volSize+vi,(float)cumSum);
                        cumSumAllAxes += cumSum;
                    }// If f_imgk                   
            }// for axesIndex
            /* Add the accumulated All axes sum to the volume */            
        atomicAdd((float *)vol+vk*volSize*volSize+vj*volSize+vi,(float)cumSumAllAxes);
        //atomicAdd((float *)vol + volIndex, cumSumAllAxes);
    } //If vi,vj,vk
}

void gpuBackProject(
    std::vector<float*> gpuVol_Vector, std::vector<float*> gpuCASImgs_Vector,       // Vector of GPU array pointers
    std::vector<float*> gpuCoordAxes_Vector, std::vector<float*> ker_bessel_Vector, // Vector of GPU array pointers
	// std::vector<cufftComplex *> gpuComplexImgs_Vector,                                   // Vector of GPU array pointers
	// std::vector<cufftComplex *> gpuComplexImgs_Shifted_Vector,                           // Vector of GPU array pointers
    float * CASImgs_CPU_Pinned, float * coordAxes_CPU_Pinned, // Pointers to pinned CPU arrays for input / output
    int volSize, int imgSize, int nAxes, float maskRadius, int kerSize, float kerHWidth, // kernel Parameters and constants
    int numGPUs, int nStreams, int gridSize, int blockSize, int nBatches, // Streaming parameters
	int MaxAxesAllocated
	)
{   
    // Define CUDA kernel dimensions
    dim3 dimGrid(gridSize, gridSize, gridSize);
    dim3 dimBlock(blockSize, blockSize, blockSize);
    
    // Create the CUDA streams
	cudaStream_t *stream = (cudaStream_t *)malloc(sizeof(cudaStream_t)*nStreams);

    for (int i = 0; i < nStreams; i++) // Loop through the streams
    { 
        int curr_GPU = i % numGPUs; // Use the remainder operator to split evenly between GPUs
        cudaSetDevice(curr_GPU);      
        cudaStreamCreate(&stream[i]);
    }

	// Initilize a vector which will remember how many axes have been assigned to each GPU
	// during the current batch. This is needed for calculating the offset for the stream when 
	// the number of streams is greater than the number of GPUs. This resets to zeros after each
	// batch since the same GPU memory is used for the next batch.
	std::vector<int> numAxesGPU_Batch;
	for (int i = 0; i < numGPUs; i++)
	{
		numAxesGPU_Batch.push_back(0);
	}

	// How many coordinate axes to assign to each stream?
	int numAxesPerStream;
	if (nAxes <= MaxAxesAllocated)
	{
		// The number of coordinate axes is less than or equal to the total number of axes to process
		numAxesPerStream = (double)nAxes / (double)nStreams;
	}
	else
	{
		// Several batches will be needed so evenly split the MaxAxesAllocated by the number of streams
		numAxesPerStream = (double)MaxAxesAllocated / (double)nStreams;
	}

	int processed_nAxes = 0; // Cumulative number of axes which have already been assigned to a CUDA stream

	// While we have coordinate axes to process, loop through the GPUs and the streams
	int MaxBatches = 50; // Maximum iterations in case we get stuck in the while loop
	int batch = 0;
	while (processed_nAxes < nAxes && batch < MaxBatches)
	{
		for (int i = 0; i < nStreams; i++) // Loop through the streams   
		{
			std::cout << "Running stream " << i << " and batch " << batch << '\n';

			int curr_GPU = i % numGPUs; // Use the remainder operator to split evenly between GPUs        
			cudaSetDevice(curr_GPU);

			// If we're passed the number of axes, process the remaining faction of numAxesPerStream
			if (processed_nAxes + numAxesPerStream >= nAxes)
			{
				// Process the remaining fraction of numAxesPerStream
				numAxesPerStream = min(numAxesPerStream, nAxes - processed_nAxes);
			}

			// Check to make sure we don't try to process more coord axes than we have and that we have at least one axes to process
			if (processed_nAxes + numAxesPerStream > nAxes || numAxesPerStream < 1)
			{
				std::cout << "Done with gpuBackProject()..." << '\n';
				return;
			}

			std::cout << "Running stream " << i << " and batch " << batch << '\n';
			std::cout << "numAxesPerStream: " << numAxesPerStream << '\n';

			// Calculate the offsets (in bytes) to determine which part of the array to copy for this stream
			int gpuCoordAxes_Offset = processed_nAxes * 9;  // Each axes has 9 elements (X, Y, Z)
			int coord_Axes_streamBytes = numAxesPerStream * 9 * sizeof(float);

			// How many bytes are the output images?
			int gpuCASImgs_streamBytes = imgSize * imgSize * numAxesPerStream * sizeof(float);

			// Use the number of axes already assigned to this GPU since starting the current batch to calculate the currect offset			
			int gpuCASImgs_Offset = numAxesGPU_Batch[curr_GPU] * imgSize * imgSize;
			int gpuCoordAxes_Stream_Offset = numAxesGPU_Batch[curr_GPU] * 9;


			// Use unsigned long long int type to allow for array length larger than maximum int32 value 
			// Number of bytes of already processed images
			// Have to use unsigned long long since the array may be longer than the max value int32 can represent
			unsigned long long *CASImgs_CPU_Offset = new  unsigned long long[3];
			CASImgs_CPU_Offset[0] = (unsigned long long)(imgSize);
			CASImgs_CPU_Offset[1] = (unsigned long long)(imgSize);
			CASImgs_CPU_Offset[2] = (unsigned long long)(processed_nAxes);

			// Copy the section of gpuCoordAxes which this stream will process on the current GPU
			cudaMemcpyAsync(
				&gpuCoordAxes_Vector[curr_GPU][gpuCoordAxes_Stream_Offset],
				&coordAxes_CPU_Pinned[gpuCoordAxes_Offset],
				coord_Axes_streamBytes,
				cudaMemcpyHostToDevice,
				stream[i]);

			// Copy CAS image to the GPU
			cudaMemcpyAsync(
				&gpuCASImgs_Vector[curr_GPU][gpuCASImgs_Offset],
				&CASImgs_CPU_Pinned[CASImgs_CPU_Offset[0] * CASImgs_CPU_Offset[1] * CASImgs_CPU_Offset[2]],
				gpuCASImgs_streamBytes,
				cudaMemcpyHostToDevice,
				stream[i]);

			// Run the back projection kernel
			gpuBackProjectKernel <<< dimGrid, dimBlock, 0, stream[i] >> > (
				gpuVol_Vector[curr_GPU], volSize, &gpuCASImgs_Vector[curr_GPU][gpuCASImgs_Offset],
				imgSize, &gpuCoordAxes_Vector[curr_GPU][gpuCoordAxes_Stream_Offset], numAxesPerStream,
				maskRadius, ker_bessel_Vector[curr_GPU], 501, kerHWidth);

			// Update the number of coordinate axes which have already been assigned to a CUDA stream
			processed_nAxes = processed_nAxes + numAxesPerStream;

			std::cout << "processed_nAxes: " << processed_nAxes << '\n';
			std::cout << "Axes remaining: " << nAxes - processed_nAxes << '\n';

		}

		// Increment the batch number
		batch++;

		// Synchronize before running the next batch
		// TO DO: Is this needed?
		cudaDeviceSynchronize();
	}
	

	std::cout << "Done with gpuBackProject()..." << '\n';

    return; 

}

