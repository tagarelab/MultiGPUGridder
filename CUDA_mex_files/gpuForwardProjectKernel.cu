#include "gpuForwardProject.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



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


void gpuForwardProject(
    std::vector<float*> gpuVol_Vector, std::vector<float*> gpuCASImgs_Vector,       // Vector of GPU array pointers
    std::vector<float*> gpuCoordAxes_Vector, std::vector<float*> ker_bessel_Vector, // Vector of GPU array pointers
    float * CASImgs_CPU_Pinned, float * coordAxes_CPU_Pinned, // Pointers to pinned CPU arrays for input / output
    int volSize, int imgSize, int nAxes, float maskRadius, int kerSize, float kerHWidth, // kernel Parameters and constants
    int numGPUs, int nStreams, int gridSize, int blockSize // Streaming parameters
)
{
      
    // Check the input vector sizes first
    if (gpuVol_Vector.size() != nStreams || gpuCASImgs_Vector.size() != nStreams || gpuCoordAxes_Vector.size() != nStreams || ker_bessel_Vector.size() != nStreams)
    {
        std::cerr << "gpuForwardProject(): Input GPU pointer sizes is not equal to the number of CUDA streams." << '\n';
        return;
    }

    // What are the dimensions of each kernel?
    dim3 dimGrid(gridSize, gridSize, 1);
    dim3 dimBlock(blockSize, blockSize, 1);

    // Create some CUDA streams
    cudaStream_t stream[nStreams]; 		

    int processed_nAxes = 0; // Cumulative number of axes which have already been assigned to a CUDA stream

    for (int i = 0; i < numGPUs; i++) //numGPUs
    { 

        cudaSetDevice(i); // TO DO: Is this needed?        
        gpuErrchk( cudaStreamCreate(&stream[i]) );

        // How many axes to assign to this CUDA stream? Need if nAxes is not a multiple of numGPUs
        int nAxes_Stream = ceil((double)nAxes / nStreams);

        if (nAxes_Stream * (i + 1) > nAxes)
        {
            nAxes_Stream = nAxes_Stream - (nAxes_Stream * (i + 1) - nAxes); // Remove the extra axes that are past the maximum nAxes
        }
    
        // Calculate the offsets (in bytes) to determine which part of the array to copy for this stream
        int gpuCoordAxes_Offset = processed_nAxes * 9 * 1; // Each axes has 9 elements (X, Y, Z)
        int coord_Axes_streamBytes = nAxes_Stream * 9 * sizeof(float); // Copy the entire vector for now

        int CASImgs_CPU_Offset     = imgSize * imgSize * processed_nAxes; // Number of bytes of already processed images
        int gpuCASImgs_streamBytes = imgSize * imgSize * nAxes_Stream * sizeof(float); // Copy the images which were processed

        // Copy the section of gpuCoordAxes which this stream will process on the current GPU
        cudaMemcpyAsync(gpuCoordAxes_Vector[i], &coordAxes_CPU_Pinned[gpuCoordAxes_Offset], coord_Axes_streamBytes, cudaMemcpyHostToDevice, stream[i]);
           
        // Run the forward projection kernel
        gpuForwardProjectKernel<<< dimGrid, dimBlock, 0, stream[i] >>>(
            gpuVol_Vector[i], volSize, gpuCASImgs_Vector[i],
            128, gpuCoordAxes_Vector[i], nAxes_Stream,
            63, ker_bessel_Vector[i], 501, 2);        
  
        gpuErrchk( cudaPeekAtLastError() );

        // Copy the resulting gpuCASImgs to the host (CPU)
        cudaMemcpyAsync(&CASImgs_CPU_Pinned[CASImgs_CPU_Offset], gpuCASImgs_Vector[i], gpuCASImgs_streamBytes, cudaMemcpyDeviceToHost, stream[i]);
        gpuErrchk( cudaPeekAtLastError() );

        // Update the number of axes which have already been assigned to a CUDA stream
        processed_nAxes = processed_nAxes + nAxes_Stream;

    }

    gpuErrchk( cudaDeviceSynchronize() );

    std::cout << "Done with gpuForwardProjectKernel" << '\n';

    return; 

}