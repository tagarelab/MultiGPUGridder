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

    
    // Verify all parameters and inputs are valid
    int parameter_check = ParameterChecking(    
        gpuVol_Vector, gpuCASImgs_Vector, gpuCoordAxes_Vector, ker_bessel_Vector, // Vector of GPU array pointers
        CASImgs_CPU_Pinned, coordAxes_CPU_Pinned, // Pointers to pinned CPU arrays for input / output
        volSize, imgSize, nAxes, maskRadius, kerSize, kerHWidth, // kernel Parameters and constants
        numGPUs, nStreams, gridSize, blockSize // Streaming parameters)
    );

    // If an error was detected return
    if (parameter_check != 0)
    {
        std::cerr << "Error detected in input parameters. Stopping the gpuForwardProjection now." << '\n';
        return;
    }    
    
    std::cout << "nStreams: " << nStreams << '\n';

    // Define CUDA kernel dimensions
    dim3 dimGrid(gridSize, gridSize, 1);
    dim3 dimBlock(blockSize, blockSize, 1);

    // Create the CUDA streams
    cudaStream_t stream[nStreams]; 		

    int processed_nAxes = 0; // Cumulative number of axes which have already been assigned to a CUDA stream

    for (int i = 0; i < nStreams; i++) 
    { 

        int curr_GPU = i % numGPUs; // Use the remainder operator to split evenly between GPUs

        cudaSetDevice(curr_GPU); // TO DO: Is this needed?        
        gpuErrchk( cudaStreamCreate(&stream[i]) );

        if (curr_GPU < 0 || curr_GPU > 3)
        {
            std::cerr << "Error in curr_GPU" << '\n';
            return;
        }

        // How many coordinate axes to assign to this CUDA stream? 
        int nAxes_Stream = ceil((double)nAxes / nStreams); // Ceil needed if nStreams is not a multiple of numGPUs

        if (nAxes_Stream * (i + 1) > nAxes) 
        {
            nAxes_Stream = nAxes_Stream - (nAxes_Stream * (i + 1) - nAxes); // Remove the extra axes that are past the maximum nAxes
        }
        
        // Is there at least one axes to process for this stream?
        if (nAxes_Stream < 1)
        {
            std::cerr << "nAxes_Stream < 1. Skipping this stream." << '\n';
            continue; // Skip this stream
        }

        std::cout << "nAxes_Stream: " << nAxes_Stream << '\n';
        
        if (gpuCASImgs_Vector.size() <= i || gpuCoordAxes_Vector.size() <= i)
        {
            std::cout << "gpuCASImgs_Vector.size(): " << gpuCASImgs_Vector.size() << '\n';
            std::cout << "gpuCoordAxes_Vector.size(): " << gpuCoordAxes_Vector.size() << '\n';
            std::cerr << "Number of streams is greater than the number of gpu array pointers." << '\n';
            return;
        }

        // Calculate the offsets (in bytes) to determine which part of the array to copy for this stream
        int gpuCoordAxes_Offset    = processed_nAxes * 9 * 1;          // Each axes has 9 elements (X, Y, Z)
        int coord_Axes_streamBytes = nAxes_Stream * 9 * sizeof(float); // Copy the entire vector for now

        int CASImgs_CPU_Offset     = imgSize * imgSize * processed_nAxes;              // Number of bytes of already processed images
        int gpuCASImgs_streamBytes = imgSize * imgSize * nAxes_Stream * sizeof(float); // Copy the images which were processed

        // Copy the section of gpuCoordAxes which this stream will process on the current GPU
        cudaMemcpyAsync(gpuCoordAxes_Vector[i], &coordAxes_CPU_Pinned[gpuCoordAxes_Offset], coord_Axes_streamBytes, cudaMemcpyHostToDevice, stream[i]);
           
        // Run the forward projection kernel
        // NOTE: Only need one gpuVol_Vector and one ker_bessel_Vector per GPU
        // NOTE: Each stream needs its own gpuCASImgs_Vector and gpuCoordAxes_Vector
        gpuForwardProjectKernel<<< dimGrid, dimBlock, 0, stream[i] >>>(
            gpuVol_Vector[curr_GPU], volSize, gpuCASImgs_Vector[i],
            imgSize, gpuCoordAxes_Vector[i], nAxes_Stream,
            63, ker_bessel_Vector[curr_GPU], 501, 2);        
  
        gpuErrchk( cudaPeekAtLastError() );

        // Copy the resulting gpuCASImgs to the host (CPU)
        cudaMemcpyAsync(&CASImgs_CPU_Pinned[CASImgs_CPU_Offset], gpuCASImgs_Vector[i], gpuCASImgs_streamBytes, cudaMemcpyDeviceToHost, stream[i]);
        gpuErrchk( cudaPeekAtLastError() );

        // Update the number of axes which have already been assigned to a CUDA stream
        processed_nAxes = processed_nAxes + nAxes_Stream;

    }

    // TO DO: Add batching if the device memory is not large enough
    // TO DO: Perhaps add a flag if batching is needed


    gpuErrchk( cudaDeviceSynchronize() );

    std::cout << "Done with gpuForwardProjectKernel" << '\n';

    return; 

}


int ParameterChecking(
    std::vector<float*> gpuVol_Vector, std::vector<float*> gpuCASImgs_Vector,       // Vector of GPU array pointers
    std::vector<float*> gpuCoordAxes_Vector, std::vector<float*> ker_bessel_Vector, // Vector of GPU array pointers
    float * CASImgs_CPU_Pinned, float * coordAxes_CPU_Pinned, // Pointers to pinned CPU arrays for input / output
    int volSize, int imgSize, int nAxes, float maskRadius, int kerSize, float kerHWidth, // kernel Parameters and constants
    int numGPUs, int nStreams, int gridSize, int blockSize // Streaming parameters)
)
{
    // Check all the input parameters to verify they are all valid

    // Checking parameter: numGPUs
    int  numGPUDetected;
    cudaGetDeviceCount(&numGPUDetected);

    if (numGPUs < 0 || numGPUs >= numGPUDetected + 1){ //  An invalid numGPUs selection was chosen
        std::cerr << "Error in GPU selection. Please provide an integer between 0 and the number of NVIDIA graphic cards on your computer. Use SetNumberGPUs() function." << '\n';
        return -1;
    }
    
    if ( numGPUDetected == 0 ) // No GPUs were found (i.e. all cards are busy)
    {
        std::cerr << "No NVIDIA graphic cards identified on your computer. All cards may be busy and unavailable. Try restarting the program and/or your computer." << '\n';;          
        return -1;
    }

    // Checking parameter: nStreams
    if (nStreams <= 0 || nStreams < numGPUs)
    {
        std::cout << "nStreams: " << nStreams << '\n';
        std::cout << "numGPUs: " << numGPUs << '\n';

        std::cerr << "Invalid number of streams provided. Please use SetNumberStreams() to set number of streams >= number of GPUs to use." << '\n';
        return -1;
    }
    
    // Checking parameter: volSize
    if (volSize <= 0)
    {
        std::cerr << "Invalid volSize parameter. Please use SetVolume() to define the input volume." << '\n';
        return -1;
    }

    // Checking parameter: imgSize
    if (imgSize <= 0)
    {
        std::cerr << "Invalid imgSize parameter." << '\n';
        return -1;
    }
    
    // Checking parameter: nAxes
    if (nAxes <= 0)
    {
        std::cerr << "Invalid nAxes parameter. Please use SetAxes() to define the input coordinate axes." << '\n';
        return -1;
    }
    
    // Checking parameter: maskRadius
    if (maskRadius <= 0)
    {
        std::cerr << "Invalid maskRadius parameter." << '\n';
        return -1;
    }
    
    // Checking parameter: kerSize
    if (kerSize <= 0)
    {
        std::cerr << "Invalid kerSize parameter." << '\n';
        return -1;
    }
    // Checking parameter: kerHWidth
    if (kerHWidth <= 0)
    {
        std::cerr << "Invalid kerHWidth parameter." << '\n';
        return -1;
    }

    // Checking parameter: gridSize
    if (gridSize <= 0)
    {
        std::cerr << "Invalid gridSize parameter." << '\n';
        return -1;
    }

    // Checking parameter: blockSize
    if (blockSize <= 0 || imgSize != gridSize * blockSize) // NOTE: gridSize times blockSize needs to equal imgSize
    {
        std::cerr << "Invalid blockSize parameter. gridSize * blockSize must equal imgSize" << '\n';
        return -1;
    }

    // Checking parameters: gpuVol_Vector, gpuCASImgs_Vector, gpuCoordAxes_Vector, and ker_bessel_Vector
    if (gpuVol_Vector.size() <= 0 || gpuCASImgs_Vector.size() <= 0 || gpuCoordAxes_Vector.size() <= 0 || ker_bessel_Vector.size() <= 0)
    {
        std::cerr << "gpuForwardProject(): Input GPU pointer vectors are empty. Has SetVolume() and SetAxes() been run?" << '\n';
        return -1;
    }

    // No errors were detected so return a flag of 0
    return 0;
}
