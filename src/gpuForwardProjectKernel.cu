#include "gpuForwardProject.h"

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
                    } //End k1
                }//End j1   
            }//End i1
        }//End if r
    }//End img_i
}

__global__ void CASImgsToImgs(float* CASimgs, cufftComplex* imgs, int imgSize)
{
    // CUDA kernel for converting the CASImgs to imgs
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row
    int k = blockIdx.z * blockDim.z + threadIdx.z; // Which image?

    // For now, CASimgs is the same dimensions as imgs
    int ndx_1 = i + j * imgSize + k * imgSize * imgSize;
    
    // Skip the first row and first column
    if (i == 0 || j == 0)
    {
        // Real component
        imgs[ndx_1].x = 0;

        // Imaginary component
        imgs[ndx_1].y = 0;

        return;
    }

    // Are we outside the bounds of the image?
    if (i >= imgSize || i < 0 || j >= imgSize || j < 0){
        return;
    }

    // Offset to skip the first row then subtract from the end of the matrix and add the offset where the particular image starts in CASimgs
    int ndx_2 = imgSize + imgSize * imgSize - ndx_1 + k * imgSize * imgSize;

    // Real component
    imgs[ndx_1].x = 0.5*(CASimgs[ndx_1] + CASimgs[ndx_2]);

    // Imaginary component
    imgs[ndx_1].y = 0.5*(CASimgs[ndx_1] - CASimgs[ndx_2]);

    return;
}




void gpuForwardProject(
    std::vector<float*> gpuVol_Vector, std::vector<float*> gpuCASImgs_Vector,       // Vector of GPU array pointers
    std::vector<float*> gpuCoordAxes_Vector, std::vector<float*> ker_bessel_Vector, // Vector of GPU array pointers
    float * CASImgs_CPU_Pinned, float * coordAxes_CPU_Pinned, // Pointers to pinned CPU arrays for input / output
    int volSize, int imgSize, int nAxes, float maskRadius, int kerSize, float kerHWidth, // kernel Parameters and constants
    int numGPUs, int nStreams, int gridSize, int blockSize, int nBatches // Streaming parameters
)
{
    std::cout << "Running gpuForwardProject()..." << '\n';
    
    // Define CUDA kernel dimensions
    dim3 dimGrid(gridSize, gridSize, 1);
    dim3 dimBlock(blockSize, blockSize, 1);

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

            // How many coordinate axes to assign to this CUDA stream? 
            int nAxes_Stream = ceil((double)nAxes / (nBatches * nStreams)); // Ceil needed if nStreams is not a multiple of numGPUs            

            // Check to make sure we don't try to process more coord axes than we have
            if (processed_nAxes + nAxes_Stream > nAxes) 
            {
                // Process the remaining streams (at least one axes is left)
                nAxes_Stream = nAxes_Stream - (processed_nAxes + nAxes_Stream - nAxes); // Remove the extra axes that are past the maximum nAxes
            }
            
            // Is there at least one coordinate axes to process for this stream?
            if (nAxes_Stream < 1)
            {
                continue; // Otherwise skip this stream
            }  
                    
            // Calculate the offsets (in bytes) to determine which part of the array to copy for this stream
            int gpuCoordAxes_Offset    = processed_nAxes * 9 * 1;          // Each axes has 9 elements (X, Y, Z)
            int coord_Axes_streamBytes = nAxes_Stream * 9 * sizeof(float); // Copy the entire vector for now

            // Use unsigned long long int type to allow for array length larger than maximum int32 value 
            // Number of bytes of already processed images
            // Have to use unsigned long long since the array may be longer than the max value int32 can represent
            unsigned long long *CASImgs_CPU_Offset = new  unsigned long long[3];
            CASImgs_CPU_Offset[0] = (unsigned long long)(imgSize);
            CASImgs_CPU_Offset[1] = (unsigned long long)(imgSize);
            CASImgs_CPU_Offset[2] = (unsigned long long)(processed_nAxes);
            
            // How many bytes are the output images?
            int gpuCASImgs_streamBytes = imgSize * imgSize * nAxes_Stream * sizeof(float);          
            
            // Copy the section of gpuCoordAxes which this stream will process on the current GPU
            cudaMemcpyAsync(gpuCoordAxes_Vector[i], &coordAxes_CPU_Pinned[gpuCoordAxes_Offset], coord_Axes_streamBytes, cudaMemcpyHostToDevice, stream[i]);
            
              // Define CUDA kernel dimensions for converting CASImgs to imgs
            // dim3 dimGrid_CAS_to_Imgs(gridSize, gridSize, 1);
            // dim3 dimBlock_CAS_to_Imgs(blockSize, blockSize, nAxes_Stream);
            
            // // Run the CUDA kernel for transforming the CASImgs to complex imgs (in order to apply the inverse FFT)
            // CASImgsToImgs<<< dimGrid_CAS_to_Imgs, dimBlock_CAS_to_Imgs, 0, stream[i] >>>(
            //     gpuCASImgs_Vector[i], gpuImgs_Vector[i], imgSize
            // );
            
                /*
            // Transform the CASImgs to complex float2 type
            int size = 100;
            cufftComplex *h_complex_array, *h_imgs, *d_imgs;
            float * d_CASImgs_test;
            float * h_CASImgs_test;


            h_CASImgs_test = (float *) malloc(sizeof(float) * size);
            cudaMalloc(&d_CASImgs_test, sizeof(float) * size);

            for (int k = 0; k < size; k++) {
                h_CASImgs_test[k] = k;
            }

            cudaMalloc(&d_imgs, sizeof(cufftComplex) * size);


            h_complex_array = (cufftComplex *) malloc(sizeof(cufftComplex) * size);
            h_imgs = (cufftComplex *) malloc(sizeof(cufftComplex) * size);

            for (int k = 0; k < size; k++) {
                h_complex_array[k].x = k;//rand() / (float) RAND_MAX;
                h_complex_array[k].y = 0;
              }
 
            // Example output array (cufftReal)
            cufftComplex *output_test = (cufftComplex*)malloc(size*sizeof(cufftComplex));


            
            // Run the forward projection kernel
            // NOTE: Only need one gpuVol_Vector and one ker_bessel_Vector per GPU
            // NOTE: Each stream needs its own gpuCASImgs_Vector and gpuCoordAxes_Vector
            gpuForwardProjectKernel<<< dimGrid, dimBlock, 0, stream[i] >>>(
                gpuVol_Vector[curr_GPU], volSize, gpuCASImgs_Vector[i],
                imgSize, gpuCoordAxes_Vector[i], nAxes_Stream,
                maskRadius, ker_bessel_Vector[curr_GPU], 501, 2);        

            // Copy the resulting gpuCASImgs to the host (CPU)
            cudaMemcpyAsync(
                &CASImgs_CPU_Pinned[CASImgs_CPU_Offset[0] * CASImgs_CPU_Offset[1] * CASImgs_CPU_Offset[2]],
                gpuCASImgs_Vector[i], gpuCASImgs_streamBytes, cudaMemcpyDeviceToHost, stream[i]);

            // Update the number of coordinate axes which have already been assigned to a CUDA stream
            processed_nAxes = processed_nAxes + nAxes_Stream;
        } 

        // cudaDeviceSynchronize();
    }

    cudaDeviceSynchronize();

    std::cout << "Done with gpuForwardProject()..." << '\n';

    return; 
}
