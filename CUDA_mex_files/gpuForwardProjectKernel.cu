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
    int volSize, int imgSize, int nAxes, float maskRadius, int kerSize, float kerHWidth // Parameters and constants
)
{
  
    int nStreams = 1; // One stream for each GPU for now
    int numGPUs  = 4;  // TO DO: make this an input variable

    // Check the input vector sizes first
    if (gpuVol_Vector.size() != nStreams || gpuCASImgs_Vector.size() != nStreams || gpuCoordAxes_Vector.size() != nStreams || ker_bessel_Vector.size() != nStreams)
    {
        // std::cerr << "gpuForwardProject(): Input GPU pointer sizes is not equal to the number of CUDA streams." << '\n';
        // return;
    }

    std::cout << "gpuVol_Vector.size(): " << gpuVol_Vector.size() << '\n';
    std::cout << "gpuCASImgs_Vector.size(): " << gpuCASImgs_Vector.size() << '\n';
    std::cout << "gpuCoordAxes_Vector.size(): " << gpuCoordAxes_Vector.size() << '\n';
    std::cout << "ker_bessel_Vector.size(): " << ker_bessel_Vector.size() << '\n';

    // Try without the streams first 

    int i = 0;


    // How many bytes is each array
    int coord_Axes_streamBytes = 2034 * sizeof(float);//nAxes * 9 * sizeof(float); // Copy the entire vector for now
    int gpuCASImgs_streamBytes = 128* 128 * 226 * sizeof(float); // Copy the entire array for now

    std::cout << "coord_Axes_streamBytes: " << coord_Axes_streamBytes << '\n';
    std::cout << "gpuCASImgs_streamBytes: " << gpuCASImgs_streamBytes << '\n';

    std::cout << "coordAxes_CPU_Pinned: " << coordAxes_CPU_Pinned << '\n';
    std::cout << "gpuVol_Vector[i]: " << gpuVol_Vector[i] << '\n';
    std::cout << "gpuCASImgs_Vector[i]: " << gpuCASImgs_Vector[i] << '\n';
    std::cout << "gpuCoordAxes_Vector[i]: " << gpuCoordAxes_Vector[i] << '\n';
    std::cout << "ker_bessel_Vector[i]: " << ker_bessel_Vector[i] << '\n';

  

    cudaMemcpy(gpuCoordAxes_Vector[i], coordAxes_CPU_Pinned, coord_Axes_streamBytes, cudaMemcpyHostToDevice);
    gpuErrchk( cudaPeekAtLastError() );

    gpuErrchk( cudaDeviceSynchronize() );

    // Run the forward projection kernel
    dim3 dimGrid(32, 32, 1);
    dim3 dimBlock(4, 4, 1);

    cudaSetDevice(0);
    gpuForwardProjectKernel<<< dimGrid, dimBlock >>>(
        gpuVol_Vector[i], 134, gpuCASImgs_Vector[i],
         128, gpuCoordAxes_Vector[i], nAxes,
         63, ker_bessel_Vector[i], 501, 2);

    
    gpuErrchk( cudaDeviceSynchronize() );

    // gpuForwardProjectKernel<<< dimGrid, dimBlock >>>(vol, volSize, img, imgSize, axes, nAxes, maskRadius,ker, kerSize, kerHWidth);

    gpuErrchk( cudaPeekAtLastError() );

    // Copy the resulting gpuCASImgs to the host (CPU)
    // TO DO: Only copy a subset of this
    cudaMemcpy(CASImgs_CPU_Pinned, gpuCASImgs_Vector[i], gpuCASImgs_streamBytes, cudaMemcpyDeviceToHost);
    gpuErrchk( cudaPeekAtLastError() );


    gpuErrchk( cudaDeviceSynchronize() );


    std::cout << "Done with gpuForwardProjectKernel" << '\n';

    return; 









   	// // Create some CUDA streams
    // cudaStream_t stream[nStreams];
    // for (int i = 0; i < nStreams; i++){

    //     // Split streams by GPU
    //     int curr_GPU = i % numGPUs; // Use the remainder operator to split evenly between GPUs

    //     if (curr_GPU <= numGPUs)
    //     {
    //         cudaSetDevice(curr_GPU);
    //         cudaStreamCreate(&stream[i]);
    //     } else 
    //     {
    //         std::cerr << "gpuForwardProject(): Failed to create CUDA stream." << '\n';
    //         return;
    //     }        
    // }       

    // // How many bytes is each async streaming?
    // int coord_Axes_streamBytes = nAxes * 9 * sizeof(float); // Copy the entire vector for now
    // int gpuCASImgs_streamBytes = 128 * 128 * nAxes * 9 * sizeof(float); // Copy the entire array for now

    // std::cout << "coord_Axes_streamBytes: " << coord_Axes_streamBytes << '\n';
    // std::cout << "gpuCASImgs_streamBytes: " << gpuCASImgs_streamBytes << '\n';


    // gpuErrchk( cudaDeviceSynchronize() ); // Probably not needed

    // // Setup the CUDA streams now
    // for (int i = 0; i < nStreams; ++i){
        
    //     // TO DO: Is it necessary to use cudaSetDevice() here?
    //     int curr_GPU = i % numGPUs; // Use the remainder operator to split evenly between GPUs

    //     curr_GPU = 0; // For now
    //     cudaSetDevice(curr_GPU);
        
    //     // Get the GPU pointers for this strea
    //     // float *devPtr_gpuVol     = gpuVol_V    // std::cout << "Done with gpuForwardProjectKernel" << '\n';ctor[i];
    //     // float *devPtr_gpuCASImgs = gpuCASImgs_Vector[i];
    //     // float *devPtr_Coord_Axes = gpuCoordAxes_Vector[i];
    //     // float *devPtr_ker_bessel = ker_bessel_Vector[i];

    //     // TO DO: Only copy a subset of the array

    //     // Copy coord axes from pinned host (CPU) to device (GPU)
    //     cudaMemcpyAsync(&gpuCoordAxes_Vector[i], coordAxes_CPU_Pinned, coord_Axes_streamBytes, cudaMemcpyHostToDevice, stream[i]);
    //     gpuErrchk( cudaPeekAtLastError() );

    //     // Run the forward projection kernel
    //     dim3 dimGrid(32, 32, 1);
    //     dim3 dimBlock(4, 4, 1);

    //     gpuForwardProjectKernel<<< dimGrid, dimBlock >>>(
    //         gpuVol_Vector[i], 134, gpuCASImgs_Vector[i],
    //          128, gpuCoordAxes_Vector[i], nAxes,
    //          63, ker_bessel_Vector[i], 501, 2);

    //     // gpuForwardProjectKernel<<< dimGrid, dimBlock >>>(vol, volSize, img, imgSize, axes, nAxes, maskRadius,ker, kerSize, kerHWidth);

    //     gpuErrchk( cudaPeekAtLastError() );

    //     // Copy the resulting gpuCASImgs to the host (CPU)
    //     // TO DO: Only copy a subset of this
    //     cudaMemcpyAsync(CASImgs_CPU_Pinned, &gpuCASImgs_Vector[i], gpuCASImgs_streamBytes, cudaMemcpyDeviceToHost, stream[i]);
    //     gpuErrchk( cudaPeekAtLastError() );

    // }

    // gpuErrchk( cudaDeviceSynchronize() );

    // std::cout << "Done with gpuForwardProjectKernel" << '\n';

}






   // float *d_test;
    // cudaMalloc((void **) &d_test, 3702784*sizeof(float));

    //float *h_test = new float[3702784];

    // float *h_test;
    // h_test = (float *)malloc(3702784*sizeof(float));

        



// // Which memory location should we start the transfer on?
// int streamOffset = i * streamSize;

// int grid_dim = ceil(size/nStreams/blockSize);



// // Run the addition kernel
// AddVectorsMask<<<grid_dim, blockSize, 0, stream[i]>>>(devPtrA, devPtrB, devPtrC, size, streamOffset);

// // Run the square elements kernel
// SquareElements<<<grid_dim, blockSize, 0, stream[i]>>>(devPtrC, size, streamOffset);

// // Run the cosine of elements kernel
// CosineElements<<<grid_dim, blockSize, 0, stream[i]>>>(devPtrC, size, streamOffset);

// // Run the square elements kernel
// SquareElements<<<grid_dim, blockSize, 0, stream[i]>>>(devPtrC, size, streamOffset);

// // Copy the result back to the host
// cudaMemcpyAsync(&C[streamOffset], &devPtrC[streamOffset], streamBytes, cudaMemcpyDeviceToHost, stream[i]);