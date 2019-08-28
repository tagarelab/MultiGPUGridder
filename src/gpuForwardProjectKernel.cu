#include "gpuForwardProject.h"
#include <math.h>       /* round, floor, ceil, trunc */
 
__global__ void gpuForwardProjectKernel(const float* vol, int volSize, float* img,int imgSize, float *axes, int nAxes,float maskRadius,
    float* ker, int kerSize, float kerHWidth)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;
    int volCenter= volSize/2;
    int imgCenter=imgSize/2;
    float f_vol_i,f_vol_j,f_vol_k;
    int img_i;
    float *img_ptr;
    int int_vol_i,int_vol_j,int_vol_k;
    int i1,j1,k1;//,kerIndex;
    float r=sqrtf( (float) (i-imgCenter)*(i-imgCenter)+(j-imgCenter)*(j-imgCenter));
    float *nx,*ny;
    int convW=roundf(kerHWidth);
    float ri,rj,rk,w;
    //float sigma=0.33*convW;
    float wi,wj,wk;
    float kerCenter=((float)kerSize-1)/2;
    float kerScale=kerCenter/kerHWidth;
    int kerIndex;   

    __shared__ float locKer[1000];

    if (threadIdx.x==0)
    {
        /* Copy over the kernel */
        for (kerIndex=0;kerIndex<kerSize;kerIndex++) 
        locKer[kerIndex]=*(ker+kerIndex);
    }
    __syncthreads();      

    // Are we inside the image bounds?
    if ( i < 0 || i > volSize || j < 0 || j > volSize)
    {
        return;
    }

    for(img_i=0;img_i<nAxes;img_i++)
    {
        img_ptr=img+img_i*imgSize*imgSize;

        if (r<=maskRadius)
        {
            nx=axes+9*img_i;
            ny=nx+3;

            f_vol_i= (*(nx))*((float)(i-imgCenter))+(*(ny))*((float)(j-imgCenter))+(float)volCenter;
            f_vol_j= (*(nx+1))*((float)(i-imgCenter))+(*(ny+1))*((float)(j-imgCenter))+(float)volCenter;
            f_vol_k= (*(nx+2))*((float)(i-imgCenter))+(*(ny+2))*((float)(j-imgCenter))+(float)volCenter;

            int_vol_i= roundf(f_vol_i);
            int_vol_j= roundf(f_vol_j);
            int_vol_k= roundf(f_vol_k);

            *(img_ptr+j*imgSize+i)=0;
            
            for (i1=int_vol_i-convW;i1<=int_vol_i+convW;i1++)
            {
                ri= (float)i1-f_vol_i;
                ri=min(max(ri,(float)-convW),(float)convW);
                kerIndex=roundf( ri*kerScale+kerCenter);
                kerIndex=min(max(kerIndex,0),kerSize-1);
                //  wi=*(ker+kerIndex);
                wi=*(locKer+kerIndex);

                for (j1=int_vol_j-convW;j1<=int_vol_j+convW;j1++)
                {

                    rj= (float)j1-f_vol_j;
                    rj=min(max(rj,(float)-convW),(float)convW);
                    kerIndex=roundf( rj*kerScale+kerCenter);
                    kerIndex=min(max(kerIndex,0),kerSize-1);
                //  wj=*(ker+kerIndex);
                    wj=*(locKer+kerIndex);

                    for (k1=int_vol_k-convW;k1<=int_vol_k+convW;k1++)
                    {
                        rk= (float)k1-f_vol_k;
                        rk=min(max(rk,(float)-convW),(float)convW);
                        kerIndex=roundf( rk*kerScale+kerCenter);
                        kerIndex=min(max(kerIndex,0),kerSize-1);
                    //   wk=*(ker+kerIndex);
                        wk=*(locKer+kerIndex);
                        w=wi*wj*wk;

                        //w=expf(-(ri*ri+rj*rj+rk*rk)/(2*sigma*sigma));  

                        *(img_ptr+j*imgSize+i)=*(img_ptr+j*imgSize+i)+//w;
                                w*( *(vol+k1*volSize*volSize+j1*volSize+i1));

                        // }
                    } //End k1
                }//End j1   
            }//End i1
        }//End if r
    }//End img_i
}

void gpuForwardProjectLaunch(gpuGridder * gridder)
{ 
    std::cout << "Running gpuForwardProject()..." << '\n';
	
    // Copy the relevent parameters to a local variable to make the code cleaner
    int gridSize         = gridder->GetGridSize();
    int blockSize        = gridder->GetBlockSize();
    int nAxes            = gridder->GetNumAxes();
    int MaxAxesAllocated = gridder->GetMaxAxesAllocated();
    int nStreams         = gridder->GetNumStreams();
    int GPU_Device       = gridder->GetGPUDevice();
    int* ImgSizePtr      = gridder->GetImgSize();    
    int* volSizePtr      = gridder->GetVolumeSize();
    float maskRadius     = gridder->GetMaskRadius();

    int ImgSize  = ImgSizePtr[0]; // The volume must be a square for now so just use the first dimension
    int volSize  = volSizePtr[0]; // The volume must be a square for now so just use the first dimension

    int* CASImgSize  = gridder->GetCASImagesSize(); // The volume must be a square for now so just use the first dimension
    int CASVolSize  = gridder->GetCASVolumeSize(); // The volume must be a square for now so just use the first dimension

    Log2("gridSize", gridSize);
    Log2("blockSize", blockSize);
    Log2("nAxes", nAxes);
    Log2("MaxAxesAllocated", MaxAxesAllocated);
    Log2("nStreams", nStreams);
    Log2("nStreams", nStreams);
    Log2("GPU_Device", GPU_Device);
    Log2("maskRadius", maskRadius);
    Log2("ImgSize", ImgSize);
    Log2("volSize", volSize);
    Log2("CASImgSize[0]", CASImgSize[0]);
    Log2("CASVolSize", CASVolSize);

    // Pointers to memory already allocated on the GPU (i.e. the device)
    float * d_CASVolume = gridder->GetCASVolumePtr_Device();
    float * d_CASImgs   = gridder->GetCASImgsPtr_Device();
    float * d_CoordAxes = gridder->GetCoordAxesPtr_Device();
    float * d_KB_Table  = gridder->GetKBTablePtr_Device();
    float * d_Imgs      = gridder->GetImgsPtr_Device();
    cufftComplex * d_CASImgsComplex = gridder->GetComplexCASImgsPtr_Device();

    // Pointers to pinned CPU memory
    float * coordAxes_CPU_Pinned = gridder->GetCoordAxesPtr_CPU();
    float * CASImgs_CPU_Pinned   = gridder->GetCASImgsPtr_CPU();
    float * Imgs_CPU_Pinned      = gridder->GetImgsPtr_CPU();

    // CUDA streams
    cudaStream_t *streams = gridder->GetStreamsPtr();
    cudaSetDevice(GPU_Device);     
    // cudaStream_t * streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * nStreams);
    
    // for (int i = 0; i < nStreams; i++) // Loop through the streams
    // {             
    //     cudaStreamCreate(&streams[i]);
    // }



    // Set the current GPU device to run the kernel
    cudaSetDevice(GPU_Device);    

    // Define CUDA kernel dimensions
    dim3 dimGrid(gridSize, gridSize, 1);
    dim3 dimBlock(blockSize, blockSize, 1);

    // Initilize a variable which will remember how many axes have been assigned to the GPU
	// during the current batch. This is needed for calculating the offset for the stream when 
	// the number of streams is greater than the number of GPUs. This resets to zeros after each
	// batch since the same GPU memory is used for the next batch.
	int numAxesGPU_Batch = 0;
    
	// How many coordinate axes to assign to each stream?
	int numAxesPerStream;
	if (nAxes <= MaxAxesAllocated)
	{
		// The number of coordinate axes is less than or equal to the total number of axes to process
		numAxesPerStream = ceil((double)nAxes / (double)nStreams);
	}
	else
	{
		// Several batches will be needed so evenly split the MaxAxesAllocated by the number of streams
		numAxesPerStream = ceil((double)MaxAxesAllocated / (double)nStreams);
	}	

    int processed_nAxes = 0; // Cumulative number of axes which have already been assigned to a CUDA stream

	// While we have coordinate axes to process, loop through the GPUs and the streams
	int MaxBatches = 1000; // Maximum iterations in case we get stuck in the while loop for some reason
	int batch = 0;

	while (processed_nAxes < nAxes && batch < MaxBatches)
	{
		for (int i = 0; i < nStreams; i++) // Loop through the streams 
		{   

            // If we're about to process more than the number of coordinate axes, process the remaining faction of numAxesPerStream
			if (processed_nAxes + numAxesPerStream >= nAxes)
			{
				// Process the remaining fraction of numAxesPerStream
				numAxesPerStream = min(numAxesPerStream, nAxes - processed_nAxes);
			}

			// Check to make sure we don't try to process more coordinate axes than we have and that we have at least one axes to process
			if (processed_nAxes + numAxesPerStream > nAxes || numAxesPerStream < 1)
			{
				return;
            }

            Log2("processed_nAxes", processed_nAxes);
            Log2("numAxesPerStream", numAxesPerStream);
            Log2("numAxesGPU_Batch", numAxesGPU_Batch);

            // return;
						
			// Calculate the offsets (in bytes) to determine which part of the array to copy for this stream
			int CoordAxes_CPU_Offset = processed_nAxes * 9;  // Each axes has 9 elements (X, Y, Z)
			int coord_Axes_CPU_streamBytes = numAxesPerStream * 9 * sizeof(float);

			// Use the number of axes already assigned to this GPU since starting the current batch to calculate the currect offset			
            int gpuCASImgs_Offset          = numAxesGPU_Batch * CASImgSize[0] * CASImgSize[1];
            int gpuImgs_Offset             = numAxesGPU_Batch * ImgSize * ImgSize;
			int gpuCoordAxes_Stream_Offset = numAxesGPU_Batch * 9;

            Log2("CoordAxes_CPU_Offset", CoordAxes_CPU_Offset);
            Log2("coord_Axes_CPU_streamBytes", coord_Axes_CPU_streamBytes);
            Log2("gpuCASImgs_Offset", gpuCASImgs_Offset);
            Log2("gpuCoordAxes_Stream_Offset", gpuCoordAxes_Stream_Offset);

            Log2("cudaMemcpyAsync", i);
        	// Copy the section of gpuCoordAxes which this stream will process on the current GPU
            cudaMemcpyAsync(
                &d_CoordAxes[gpuCoordAxes_Stream_Offset], 
                &coordAxes_CPU_Pinned[CoordAxes_CPU_Offset],
                coord_Axes_CPU_streamBytes,
                cudaMemcpyHostToDevice, streams[i]);                
            
            Log2("gpuForwardProjectKernel", i);
            // Run the forward projection kernel     
			gpuForwardProjectKernel <<< dimGrid, dimBlock, 0, streams[i] >> > (
				d_CASVolume, CASVolSize, &d_CASImgs[gpuCASImgs_Offset],
				CASImgSize[0], &d_CoordAxes[gpuCoordAxes_Stream_Offset], numAxesPerStream,
				maskRadius, d_KB_Table, 501, 2);

            // Have to use unsigned long long since the array may be longer than the max value int32 can represent
			// imgSize is the size of the zero padded projection images
			unsigned long long *CASImgs_CPU_Offset = new  unsigned long long[3];
			CASImgs_CPU_Offset[0] = (unsigned long long)(CASImgSize[0]);
			CASImgs_CPU_Offset[1] = (unsigned long long)(CASImgSize[1]);
			CASImgs_CPU_Offset[2] = (unsigned long long)(processed_nAxes);
            
            Log2("CASImgs_CPU_Offset[0]", CASImgs_CPU_Offset[0]);
            Log2("CASImgs_CPU_Offset[1]", CASImgs_CPU_Offset[1]);
            Log2("CASImgs_CPU_Offset[2]", CASImgs_CPU_Offset[2]);

			// How many bytes are the output images?
			int gpuCASImgs_streamBytes = CASImgSize[0] * CASImgSize[1] * numAxesPerStream * sizeof(float);
            
            Log2("gpuCASImgs_streamBytes", gpuCASImgs_streamBytes);

            Log2("cudaMemcpyAsync", i);

			// Lastly, copy the resulting projection images back to the host pinned memory (CPU)
			cudaMemcpyAsync(
				&CASImgs_CPU_Pinned[CASImgs_CPU_Offset[0] * CASImgs_CPU_Offset[1] * CASImgs_CPU_Offset[2]],
			 	&d_CASImgs[gpuCASImgs_Offset], gpuCASImgs_streamBytes, cudaMemcpyDeviceToHost, streams[i]);

            cudaDeviceSynchronize();

            // Convert the CAS projection images back to images using an inverse FFT and cropping out the zero padding
            gpuFFT::CASImgsToImgs(
                streams[i], gridSize, blockSize, CASImgSize[0],
                ImgSize, &d_CASImgs[gpuCASImgs_Offset], &d_Imgs[gpuImgs_Offset], &d_CASImgsComplex[gpuCASImgs_Offset],
                numAxesPerStream);

                // cudaMemcpyAsync(
                //     &CASImgs_CPU_Pinned[CASImgs_CPU_Offset[0] * CASImgs_CPU_Offset[1] * CASImgs_CPU_Offset[2]],
                //      &d_CASImgs[gpuCASImgs_Offset], gpuCASImgs_streamBytes, cudaMemcpyDeviceToHost, streams[i]);
    

            // Have to use unsigned long long since the array may be longer than the max value int32 can represent
			// imgSize is the size of the zero padded projection images
			unsigned long long *Imgs_CPU_Offset = new  unsigned long long[3];
			Imgs_CPU_Offset[0] = (unsigned long long)(ImgSize);
			Imgs_CPU_Offset[1] = (unsigned long long)(ImgSize);
			Imgs_CPU_Offset[2] = (unsigned long long)(processed_nAxes);
            
            Log2("Imgs_CPU_Offset[0]", Imgs_CPU_Offset[0]);
            Log2("Imgs_CPU_Offset[1]", Imgs_CPU_Offset[1]);
            Log2("Imgs_CPU_Offset[2]", Imgs_CPU_Offset[2]);

			// How many bytes are the output images?
			int gpuImgs_streamBytes = ImgSize * ImgSize * numAxesPerStream * sizeof(float);
            
            Log2("gpuImgs_streamBytes", gpuImgs_streamBytes);

            Log2("cudaMemcpyAsync", i);

			// Lastly, copy the resulting cropped projection images back to the host pinned memory (CPU)
			cudaMemcpyAsync(
				&Imgs_CPU_Pinned[Imgs_CPU_Offset[0] * Imgs_CPU_Offset[1] * Imgs_CPU_Offset[2]],
                &d_Imgs[gpuImgs_Offset], gpuImgs_streamBytes, cudaMemcpyDeviceToHost, streams[i]);
                
			// Update the overall number of coordinate axes which have already been assigned to a CUDA stream
            processed_nAxes = processed_nAxes + numAxesPerStream;    
            cudaDeviceSynchronize();

            // Update the number of axes which have been assigned to this GPU during the current batch
            numAxesGPU_Batch = numAxesGPU_Batch + numAxesPerStream;

            // TEST
            // CASImgs_CPU_Pinned[0] = 14;
            // END TEST
        
            return; //debug

        }

		// Increment the batch number
		batch++;

        // Reset the number of axes processed during the current batch variable
        numAxesGPU_Batch = 0;
        

		// Synchronize before running the next batch
		// TO DO: Consider replacing with cudaStreamWaitEvent or similar to prevent blocking of the CPU
        cudaDeviceSynchronize();
        
        return; //debug
	}

    return; 
}




