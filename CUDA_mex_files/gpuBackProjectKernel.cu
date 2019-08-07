#include "gpuBackProject.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void gpuBackProjectKernel(float* vol,int volSize, float* img,int imgSize,
                                     float * axes, int nAxes, float maskRadius,
                                    const float* ker,int kerSize,float kerHWidth)

{
float *img_ptr;
// kerHWidth = 2;

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

    // int volIndex = vk*volSize*volSize+vj*volSize+vi;
    // vol[volIndex] = volIndex;
    

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
    
    std::cout << "nStreams: " << nStreams << '\n';

    // Define CUDA kernel dimensions
    dim3 dimGrid(gridSize, gridSize, gridSize);
    dim3 dimBlock(blockSize, blockSize, blockSize);
    
    // Create the CUDA streams
    cudaStream_t stream[nStreams]; 		

    for (int i = 0; i < nStreams; i++) // Loop through the streams
    { 
        int curr_GPU = i % numGPUs; // Use the remainder operator to split evenly between GPUs

        cudaSetDevice(curr_GPU); // TO DO: Is this needed?        
        gpuErrchk( cudaStreamCreate(&stream[i]) );

    }

    int processed_nAxes = 0; // Cumulative number of axes which have already been assigned to a CUDA stream

    // Loop through the batches
    for (int currBatch = 0; currBatch < nBatches; currBatch++)
    {   
        for (int i = 0; i < nStreams; i++) // Loop through the streams 
        {             
            int curr_GPU = i % numGPUs; // Use the remainder operator to split evenly between GPUs
            
            std::cout << "curr_GPU: " << curr_GPU << '\n';

            cudaSetDevice(curr_GPU); // TO DO: Is this needed?        

            if (curr_GPU < 0 || curr_GPU > 3)
            {
                std::cerr << "Error in curr_GPU" << '\n';
                return;
            }
            
            // How many coordinate axes to assign to this CUDA stream? 
            int nAxes_Stream = ceil((double)nAxes / (nBatches * nStreams)); // Ceil needed if nStreams is not a multiple of numGPUs            

            std::cout << "nAxes: " << nAxes << '\n';
            std::cout << "nStreams: " << nStreams << '\n';
            std::cout << "nBatches: " << nBatches << '\n';
            std::cout << "nAxes_Stream: " << nAxes_Stream << '\n';

            // Check to make sure we dont try to process more coord axes than we have
            if (processed_nAxes + nAxes_Stream > nAxes) 
            {
                // Process the remaining streams (at least one axes is left)
                nAxes_Stream = nAxes_Stream - (processed_nAxes + nAxes_Stream - nAxes); // Remove the extra axes that are past the maximum nAxes
            }
            
            // Is there at least one axes to process for this stream?
            if (nAxes_Stream < 1)
            {
                std::cerr << "nAxes_Stream < 1. Skipping this stream." << '\n';
                continue; // Skip this stream
            }  
                    
            if (gpuCASImgs_Vector.size() <= i || gpuCASImgs_Vector.size() == 0 || gpuCoordAxes_Vector.size() <= i || gpuCoordAxes_Vector.size() == 0)
            {
                std::cout << "gpuCASImgs_Vector.size(): " << gpuCASImgs_Vector.size() << '\n';
                std::cout << "gpuCoordAxes_Vector.size(): " << gpuCoordAxes_Vector.size() << '\n';
                std::cerr << "Number of streams is greater than the number of gpu array pointers." << '\n';
                return;
            }        

            // Calculate the offsets (in bytes) to determine which part of the array to copy for this stream
            int gpuCoordAxes_Offset    = processed_nAxes * 9 * 1;          // Each axes has 9 elements (X, Y, Z)
            int coord_Axes_streamBytes = nAxes_Stream * 9 * sizeof(float); // Copy the entire vector for now

            // Use unsigned long long int type to allow for array length larger than maximum int32 value 
            unsigned long long *CASImgs_CPU_Offset = new  unsigned long long[3];
            CASImgs_CPU_Offset[0] = (unsigned long long)(imgSize);
            CASImgs_CPU_Offset[1] = (unsigned long long)(imgSize);
            CASImgs_CPU_Offset[2] = (unsigned long long)(processed_nAxes);
            
            // int CASImgs_CPU_Offset     = imgSize * imgSize * processed_nAxes;              // Number of bytes of already processed images
            int gpuCASImgs_streamBytes = imgSize * imgSize * nAxes_Stream * sizeof(float); // Copy the images which were processed

            std::cout << "nAxes_Stream: " << nAxes_Stream << '\n';
            std::cout << "gpuCoordAxes_Offset: " << gpuCoordAxes_Offset << '\n';
            std::cout << "coord_Axes_streamBytes: " << coord_Axes_streamBytes << '\n';
            
            std::cout << "CASImgs_CPU_Offset: " << CASImgs_CPU_Offset[0] * CASImgs_CPU_Offset[1] * CASImgs_CPU_Offset[2] << '\n';
            std::cout << "gpuCASImgs_streamBytes: " << gpuCASImgs_streamBytes << '\n';

            // Check to make sure the coord axes array has the requested memory
            if ( (gpuCoordAxes_Offset + coord_Axes_streamBytes) >= nAxes * 9 * sizeof(float))
            {
                std::cerr << "(gpuCoordAxes_Offset + coord_Axes_streamBytes) >= nAxes * 9 * sizeof(float). Skipping this stream." << '\n';
                continue; // Skip this stream
            }          
            
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
                
              // cudaMemcpyAsync(
            //     &CASImgs_CPU_Pinned[CASImgs_CPU_Offset[0] * CASImgs_CPU_Offset[1] * CASImgs_CPU_Offset[2]],
            //     gpuCASImgs_Vector[i], gpuCASImgs_streamBytes, cudaMemcpyDeviceToHost, stream[i]);


            // Run the back projection kernel
            // NOTE: Only need one gpuVol_Vector and one ker_bessel_Vector per GPU
            // NOTE: Each stream needs its own gpuCASImgs_Vector and gpuCoordAxes_Vector
            gpuBackProjectKernel<<< dimGrid, dimBlock, 0, stream[i] >>>(
                gpuVol_Vector[curr_GPU], volSize, gpuCASImgs_Vector[i],
                imgSize, gpuCoordAxes_Vector[i], nAxes_Stream,
                maskRadius, ker_bessel_Vector[curr_GPU], 501, 2);        
               
            gpuErrchk( cudaPeekAtLastError() );

            std::cout << "cudaDeviceSynchronize()" << '\n';

            gpuErrchk( cudaDeviceSynchronize() ); // Synchronize all the streams before reusing them (if number of batches > 1)

            // return;
            
            // float * h_Vol;
            // h_Vol = (float *) malloc(sizeof(float) * volSize * volSize * volSize);

            // cudaMemcpy(h_Vol, gpuVol_Vector[curr_GPU], sizeof(float) * volSize * volSize * volSize, cudaMemcpyDeviceToHost);

            // for (int z = 0; z < 5000; z ++)
            // {
            //     std::cout << "h_Vol[" << z << "]: " << h_Vol[z] << '\n';
            // }


            // Copy the resulting gpuCASImgs to the host (CPU)
            // cudaMemcpyAsync(
            //     &CASImgs_CPU_Pinned[CASImgs_CPU_Offset[0] * CASImgs_CPU_Offset[1] * CASImgs_CPU_Offset[2]],
            //     gpuCASImgs_Vector[i], gpuCASImgs_streamBytes, cudaMemcpyDeviceToHost, stream[i]);

            gpuErrchk( cudaPeekAtLastError() );
                
            // Update the number of axes which have already been assigned to a CUDA stream
            processed_nAxes = processed_nAxes + nAxes_Stream;
        }

        std::cout << "cudaDeviceSynchronize()" << '\n';

        gpuErrchk( cudaDeviceSynchronize() ); // Synchronize all the streams before reusing them (if number of batches > 1)

    }

    std::cout << "Done with gpuBackProjectKernel" << '\n';

    return; 

}

