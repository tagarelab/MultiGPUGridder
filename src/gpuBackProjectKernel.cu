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
   
    if ( (vi<volSize)&&(vj<volSize)&&(vk<volSize) &&(r<=maskRadius) )
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


// https://github.com/marwan-abdellah/cufftShift/blob/master/Src/CUDA/Kernels/out-of-place/cufftShift_3D_OP.cu
__global__ void cufftShift_3D_slice_kernel_2(cufftComplex* input, cufftComplex* output, int N, int nSlices)
{
	// 3D Volume & 2D Slice & 1D Line
	int sLine = N;
	int sSlice = N * N;
	int sVolume = N * N * N;

	// Transformations Equations
	int sEq1 = (sVolume + sSlice + sLine) / 2;
	int sEq2 = (sVolume + sSlice - sLine) / 2;
	int sEq3 = (sVolume - sSlice + sLine) / 2;
	int sEq4 = (sVolume - sSlice - sLine) / 2;

	// Thread
	int xThreadIdx = threadIdx.x;
	int yThreadIdx = threadIdx.y;

	// Block Width & Height
	int blockWidth = blockDim.x;
	int blockHeight = blockDim.y;

	// Thread Index 2D
	int xIndex = blockIdx.x * blockWidth + xThreadIdx;
	int yIndex = blockIdx.y * blockHeight + yThreadIdx;

	// Each thread will do all the slices for some X, Y position in the 3D matrix

	for (int zIndex = 0; zIndex < nSlices; zIndex++)
	{


		// Thread Index Converted into 1D Index
		int index = (zIndex * sSlice) + (yIndex * sLine) + xIndex;

		if (zIndex < N / 2)
		{
			if (xIndex < N / 2)
			{
				if (yIndex < N / 2)
				{
					// First Quad
					output[index].x = input[index + sEq1].x;
					output[index].y = input[index + sEq1].y;
				}
				else
				{
					// Third Quad
					output[index].x = input[index + sEq3].x;
					output[index].y = input[index + sEq3].y;
				}
			}
			else
			{
				if (yIndex < N / 2)
				{
					// Second Quad
					output[index].x = input[index + sEq2].x;
					output[index].y = input[index + sEq2].y;
				}
				else
				{
					// Fourth Quad
					output[index].x = input[index + sEq4].x;
					output[index].y = input[index + sEq4].y;
				}
			}
		}

		else
		{
			if (xIndex < N / 2)
			{
				if (yIndex < N / 2)
				{
					// First Quad
					output[index].x = input[index - sEq4].x;
					output[index].y = input[index - sEq4].y;
				}
				else
				{
					// Third Quad
					output[index].x = input[index - sEq2].x;
					output[index].y = input[index - sEq2].y;
				}
			}
			else
			{
				if (yIndex < N / 2)
				{
					// Second Quad
					output[index].x = input[index - sEq3].x;
					output[index].y = input[index - sEq3].y;
				}
				else
				{
					// Fourth Quad
					output[index].x = input[index - sEq1].x;
					output[index].y = input[index - sEq1].y;
				}
			}
		}
	}
}

__global__ void ComplexImgsToCASImgs_2(float* CASimgs, cufftComplex* imgs, int imgSize)
{
	// CUDA kernel for converting the CASImgs to imgs
	int i = blockIdx.x * blockDim.x + threadIdx.x; // Column
	int j = blockIdx.y * blockDim.y + threadIdx.y; // Row
	//int k = blockIdx.z * blockDim.z + threadIdx.z; // Which image?

	// Are we outside the bounds of the image?
	if (i >= imgSize || i < 0 || j >= imgSize || j < 0) {
		return;
	}

	// Each thread will do all the slices for position X and Y
	for (int k = 0; k < imgSize; k++)
	{
		// CASimgs is the same dimensions as imgs
		int ndx = i + j * imgSize + k * imgSize * imgSize;

		// Summation of the real and imaginary components
		CASimgs[ndx] = imgs[ndx].x + imgs[ndx].y;
	}

	return;
}



void gpuBackProject(
    std::vector<float*> gpuVol_Vector, std::vector<float*> gpuCASImgs_Vector,       // Vector of GPU array pointers
    std::vector<float*> gpuCoordAxes_Vector, std::vector<float*> ker_bessel_Vector, // Vector of GPU array pointers
	std::vector<cufftComplex *> gpuComplexImgs_Vector,                                   // Vector of GPU array pointers
	std::vector<cufftComplex *> gpuComplexImgs_Shifted_Vector,                           // Vector of GPU array pointers
    float * CASImgs_CPU_Pinned, float * coordAxes_CPU_Pinned, // Pointers to pinned CPU arrays for input / output
    int volSize, int imgSize, int nAxes, float maskRadius, int kerSize, float kerHWidth, // kernel Parameters and constants
    int numGPUs, int nStreams, int gridSize, int blockSize, int nBatches, // Streaming parameters
	std::vector<int> numAxesPerStream
)
{   
	std::cout << "Running gpuBackProject()..." << '\n';

    // Define CUDA kernel dimensions
    dim3 dimGrid(gridSize, gridSize, gridSize);
    dim3 dimBlock(blockSize, blockSize, blockSize);
    
    // Create the CUDA streams
	cudaStream_t *stream = (cudaStream_t *)malloc(sizeof(cudaStream_t)*nStreams);
    //cudaStream_t stream[nStreams]; 		

    for (int i = 0; i < nStreams; i++) // Loop through the streams
    { 
        int curr_GPU = i % numGPUs; // Use the remainder operator to split evenly between GPUs
        cudaSetDevice(curr_GPU);      
        cudaStreamCreate(&stream[i]);
    }

    int processed_nAxes = 0; // Cumulative number of axes which have already been assigned to a CUDA stream

    for (int currBatch = 0; currBatch < nBatches; currBatch++) // Loop through the batches
    {   
        for (int i = 0; i < nStreams; i++) // Loop through the streams 
        {             
            int curr_GPU = i % numGPUs; // Use the remainder operator to split evenly between GPUs
            
            cudaSetDevice(curr_GPU);   

			// Check to make sure we don't try to process more coord axes than we have
			if (processed_nAxes + numAxesPerStream[i] >= nAxes)
			{
				numAxesPerStream[i] = min(numAxesPerStream[i], nAxes - processed_nAxes);
			}
            

			// Is there at least one coordinate axes to process for this stream?
			if (numAxesPerStream[i] >= 1) // TO DO: Fix this && processed_nAxes < nAxes
			{

				std::cout << "Number of axes in this stream " << numAxesPerStream[i] << '\n';

				// This seems to be needed
				cudaMemset(gpuComplexImgs_Vector[i], 0, imgSize * imgSize * numAxesPerStream[i] * sizeof(cufftComplex));
				cudaMemset(gpuComplexImgs_Shifted_Vector[i], 0, imgSize * imgSize * numAxesPerStream[i] * sizeof(cufftComplex));
				cudaMemset(gpuCASImgs_Vector[i], 0, imgSize * imgSize * numAxesPerStream[i] * sizeof(float));

				// Calculate the offsets (in bytes) to determine which part of the array to copy for this stream
				int gpuCoordAxes_Offset = processed_nAxes * 9 * 1;                    // Each axes has 9 elements (X, Y, Z)
				int coord_Axes_streamBytes = numAxesPerStream[i] * 9 * sizeof(float); // Copy the entire vector for now

				// Use unsigned long long int type to allow for array length larger than maximum int32 value 
				// Number of bytes of already processed images
				// Have to use unsigned long long since the array may be longer than the max value int32 can represent
				unsigned long long *CASImgs_CPU_Offset = new  unsigned long long[3];
				CASImgs_CPU_Offset[0] = (unsigned long long)(imgSize);
				CASImgs_CPU_Offset[1] = (unsigned long long)(imgSize);
				CASImgs_CPU_Offset[2] = (unsigned long long)(processed_nAxes);

				// How many bytes are the output images?
				int gpuCASImgs_streamBytes = imgSize * imgSize * numAxesPerStream[i] * sizeof(float);

				// Copy the section of gpuCoordAxes which this stream will process on the current GPU
				cudaMemcpyAsync(
					gpuCoordAxes_Vector[i],
					&coordAxes_CPU_Pinned[gpuCoordAxes_Offset],
					coord_Axes_streamBytes,
					cudaMemcpyHostToDevice,
					stream[i]);
				
				// gpuCASImgs_Vector is actually in real space (not CAS) and needs to be converted to the frequency domain
				int * volSizeCAS = new int[3];
				volSizeCAS[0] = imgSize;
				volSizeCAS[1] = imgSize;
				volSizeCAS[2] = numAxesPerStream[i];

				// To DO: Make this function support GPU arrays too
				float *h_CAS_Vol;
				h_CAS_Vol = (float *)malloc(sizeof(float) * volSizeCAS[0] * volSizeCAS[1] * volSizeCAS[2]);
				h_CAS_Vol = ThreeD_ArrayToCASArray(&CASImgs_CPU_Pinned[CASImgs_CPU_Offset[0] * CASImgs_CPU_Offset[1] * CASImgs_CPU_Offset[2]], volSizeCAS);

				// Convert the CAS volume to cufftComplex
				cufftComplex *h_complex_array;
				h_complex_array = (cufftComplex *)malloc(sizeof(cufftComplex) * volSizeCAS[0] * volSizeCAS[1] * volSizeCAS[2]);

				for (int k = 0; k < volSizeCAS[0] * volSizeCAS[1] * volSizeCAS[2]; k++) {
					h_complex_array[k].x = h_CAS_Vol[k]; // Real component
					h_complex_array[k].y = 0;            // Imaginary component
				}

				// Copy from host to device
				cudaMemcpy(gpuComplexImgs_Vector[i], h_complex_array, volSizeCAS[0] * volSizeCAS[1] * volSizeCAS[2] * sizeof(cufftComplex), cudaMemcpyHostToDevice);

				// Run FFTShift on gpuComplexImgs_Vector[i] (can't reuse the same input as the output for the FFT shift kernel)
				cufftShift_3D_slice_kernel_2 <<< dimGrid, dimBlock, 0, stream[i] >>> (gpuComplexImgs_Vector[i], gpuComplexImgs_Shifted_Vector[i], imgSize, numAxesPerStream[i]);

				// Create a plan for taking the inverse of the CAS imgs
				cufftHandle forwardFFTPlan;
				int nRows = imgSize;
				int nCols = imgSize;
				int batch = numAxesPerStream[i];       // --- Number of batched executions
				int rank = 2;                   // --- 2D FFTs
				int n[2] = { nRows, nCols };      // --- Size of the Fourier transform
				int idist = nRows * nCols;        // --- Distance between batches
				int odist = nRows * nCols;        // --- Distance between batches

				int inembed[] = { nRows, nCols }; // --- Input size with pitch
				int onembed[] = { nRows, nCols }; // --- Output size with pitch

				int istride = 1;                // --- Distance between two successive input/output elements
				int ostride = 1;                // --- Distance between two successive input/output elements

				cufftPlanMany(&forwardFFTPlan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);
				cufftSetStream(forwardFFTPlan, stream[i]); // Set the FFT plan to the current stream to process

				// Forward FFT
				cufftExecC2C(forwardFFTPlan, (cufftComplex *)gpuComplexImgs_Shifted_Vector[i], (cufftComplex *)gpuComplexImgs_Shifted_Vector[i], CUFFT_FORWARD);

				// Run FFTShift on gpuComplexImgs_Vector[i] (can't reuse the same input as the output for the FFT shift kernel)
				cufftShift_3D_slice_kernel_2 <<< dimGrid, dimBlock, 0, stream[i] >>> (gpuComplexImgs_Shifted_Vector[i], gpuComplexImgs_Vector[i], imgSize, numAxesPerStream[i]);

				// Convert the complex result of the forward FFT to a CAS img type
				ComplexImgsToCASImgs_2 <<< dimGrid, dimBlock, 0, stream[i] >>> (
					gpuCASImgs_Vector[i], gpuComplexImgs_Vector[i], imgSize // Assume the volume is a square for now
					);


				// Copy CAS image to the GPU
				//cudaMemcpyAsync(
				//	gpuCASImgs_Vector[i],
				//	&h_CAS_Vol,
				//	gpuCASImgs_streamBytes,
				//	cudaMemcpyHostToDevice,
				//	stream[i]);

				// Copy the section of gpuCASImgs which this stream will process on the current GPU
				//cudaMemcpyAsync(
				//	gpuCASImgs_Vector[i],
				//	&CASImgs_CPU_Pinned[CASImgs_CPU_Offset[0] * CASImgs_CPU_Offset[1] * CASImgs_CPU_Offset[2]],
				//	gpuCASImgs_streamBytes,
				//	cudaMemcpyHostToDevice,
				//	stream[i]);

				// Run the back projection kernel
				gpuBackProjectKernel <<< dimGrid, dimBlock, 0, stream[i] >> > (
					gpuVol_Vector[curr_GPU], volSize, gpuCASImgs_Vector[i],
					imgSize, gpuCoordAxes_Vector[i], numAxesPerStream[i],
					maskRadius, ker_bessel_Vector[curr_GPU], 501, 2);


				// Update the number of axes which have already been assigned to a CUDA stream
				processed_nAxes = processed_nAxes + numAxesPerStream[i];
			}
        }
    }


    cudaDeviceSynchronize();


	std::cout << "Done with gpuBackProject()..." << '\n';

    return; 

}

