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


void gpuBackProject(
    std::vector<float*> gpuVol_Vector, std::vector<float*> gpuCASImgs_Vector,       // Vector of GPU array pointers
    std::vector<float*> gpuCoordAxes_Vector, std::vector<float*> ker_bessel_Vector, // Vector of GPU array pointers
    float * CASImgs_CPU_Pinned, float * coordAxes_CPU_Pinned, // Pointers to pinned CPU arrays for input / output
    int volSize, int imgSize, int nAxes, float maskRadius, int kerSize, float kerHWidth, // kernel Parameters and constants
    int numGPUs, int nStreams, int gridSize, int blockSize, int nBatches // Streaming parameters
)
{   
    
    // Define CUDA kernel dimensions
    dim3 dimGrid(gridSize, gridSize, gridSize);
    dim3 dimBlock(blockSize, blockSize, blockSize);
    
    // Create the CUDA streams
    cudaStream_t stream[nStreams]; 		

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

            // Check to make sure we dont try to process more coord axes than we have
            if (processed_nAxes + nAxes_Stream > nAxes) 
            {
                // Process the remaining streams (at least one axes is left)
                nAxes_Stream = nAxes_Stream - (processed_nAxes + nAxes_Stream - nAxes); // Remove the extra axes that are past the maximum nAxes
            }
            
            // Is there at least one axes to process for this stream?
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
            cudaMemcpyAsync(
                gpuCoordAxes_Vector[i],
                &coordAxes_CPU_Pinned[gpuCoordAxes_Offset],
                coord_Axes_streamBytes,
                cudaMemcpyHostToDevice,
                stream[i]);
                
            // Copy the section of gpuCASImgs which this stream will process on the current GPU
            cudaMemcpyAsync(
                gpuCASImgs_Vector[i],
                &CASImgs_CPU_Pinned[CASImgs_CPU_Offset[0] * CASImgs_CPU_Offset[1] * CASImgs_CPU_Offset[2]],
                gpuCASImgs_streamBytes,
                cudaMemcpyHostToDevice,
                stream[i]);
                
            // Run the back projection kernel
            gpuBackProjectKernel<<< dimGrid, dimBlock, 0, stream[i] >>>(
                gpuVol_Vector[curr_GPU], volSize, gpuCASImgs_Vector[i],
                imgSize, gpuCoordAxes_Vector[i], nAxes_Stream,
                maskRadius, ker_bessel_Vector[curr_GPU], 501, 2);        
                
            // Update the number of axes which have already been assigned to a CUDA stream
            processed_nAxes = processed_nAxes + nAxes_Stream;
        }
    }


    cudaDeviceSynchronize();

    return; 

}

