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

void gpuForwardProject(
    std::vector<float *> gpuVol_Vector, std::vector<float *> gpuCASImgs_Vector,          // Vector of GPU array pointers
    std::vector<float *> gpuCoordAxes_Vector, std::vector<float *> ker_bessel_Vector,    // Vector of GPU array pointers
    // std::vector<cufftComplex *> gpuComplexImgs_Vector,                                   // Vector of GPU array pointers
    // std::vector<cufftComplex *> gpuComplexImgs_Shifted_Vector,                           // Vector of GPU array pointers
    float *CASImgs_CPU_Pinned, float *coordAxes_CPU_Pinned,                              // Pointers to pinned CPU arrays for input / output
    int volSize, int imgSize, int nAxes, float maskRadius, int kerSize, float kerHWidth, // kernel Parameters and constants
    int numGPUs, int nStreams, int gridSize, int blockSize, int nBatches,                 // Streaming parameters
    std::vector<int> numAxesPerStream
)
{ 
    std::cout << "Running gpuForwardProject()..." << '\n';

    // Define CUDA kernel dimensions
    dim3 dimGrid(gridSize, gridSize, 1);
    dim3 dimBlock(blockSize, blockSize, 1);

    // Create the CUDA streams
	cudaStream_t *stream = (cudaStream_t *)malloc(sizeof(cudaStream_t)*nStreams);

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
            std::cout << "Running stream " << i << " and batch " << currBatch << '\n';

            int curr_GPU = i % numGPUs; // Use the remainder operator to split evenly between GPUs        
            cudaSetDevice(curr_GPU);  
            
            // Check to make sure we don't try to process more coord axes than we have
            if (processed_nAxes + numAxesPerStream[i] >= nAxes) 
            {
                numAxesPerStream[i] = min(numAxesPerStream[i], nAxes - processed_nAxes);
            }

            // Is there at least one coordinate axes to process for this stream?
            if (numAxesPerStream[i] <= 1 ) // TO DO: Fix this && processed_nAxes < nAxes
            {
                continue;
            }
                            
            // Calculate the offsets (in bytes) to determine which part of the array to copy for this stream
            int gpuCoordAxes_Offset    = processed_nAxes * 9 * 1;          // Each axes has 9 elements (X, Y, Z)
            int coord_Axes_streamBytes = numAxesPerStream[i] * 9 * sizeof(float); // Copy the entire vector for now
                
            // Copy the section of gpuCoordAxes which this stream will process on the current GPU
            cudaMemcpyAsync(gpuCoordAxes_Vector[i], &coordAxes_CPU_Pinned[gpuCoordAxes_Offset], coord_Axes_streamBytes, cudaMemcpyHostToDevice, stream[i]);            

            // Run the forward projection kernel
            // NOTE: Only need one gpuVol_Vector and one ker_bessel_Vector per GPU
            // NOTE: Each stream needs its own gpuCASImgs_Vector and gpuCoordAxes_Vector
            gpuForwardProjectKernel<<< dimGrid, dimBlock, 0, stream[i] >>>(
                gpuVol_Vector[curr_GPU], volSize, gpuCASImgs_Vector[i],
                imgSize, gpuCoordAxes_Vector[i], numAxesPerStream[i],
                maskRadius, ker_bessel_Vector[curr_GPU], 501, 2);    

            // Have to use unsigned long long since the array may be longer than the max value int32 can represent
            // imgSize is the size of the zero padded projection images
            unsigned long long *CASImgs_CPU_Offset = new  unsigned long long[3];
            CASImgs_CPU_Offset[0] = (unsigned long long)(imgSize);
            CASImgs_CPU_Offset[1] = (unsigned long long)(imgSize);
            CASImgs_CPU_Offset[2] = (unsigned long long)(processed_nAxes);
            
            // How many bytes are the output images?
            int gpuCASImgs_streamBytes = imgSize * imgSize * numAxesPerStream[i] * sizeof(float);    

            // Lastly, copy the resulting cropped projection images back to the host pinned memory (CPU)
            cudaMemcpyAsync(
                &CASImgs_CPU_Pinned[CASImgs_CPU_Offset[0] * CASImgs_CPU_Offset[1] * CASImgs_CPU_Offset[2]],
                gpuCASImgs_Vector[i], gpuCASImgs_streamBytes, cudaMemcpyDeviceToHost, stream[i]);

            // Update the number of coordinate axes which have already been assigned to a CUDA stream
            processed_nAxes = processed_nAxes + numAxesPerStream[i];
            
            std::cout << "processed_nAxes: " << processed_nAxes << '\n';
            std::cout << "Axes remaining: " << nAxes - processed_nAxes << '\n';
        } 
    }

    cudaDeviceSynchronize();

    std::cout << "Done with gpuForwardProject()..." << '\n';

    return; 
}













/*                 
            dim3 dimGrid_CAS_to_Imgs(32, 32, nAxes_Stream);
            dim3 dimBlock_CAS_to_Imgs(imgSize/32,imgSize/32,1); 

            std::cout << "gpuImgs_Vector.size(): " << gpuImgs_Vector.size() << '\n';
            std::cout << "stream " << i << '\n';
            
            // cufftComplex *d_imgs; // TEST
            // cudaMalloc(&d_imgs, sizeof(cufftComplex) * imgSize * imgSize * nAxes_Stream); // TEST

            float * d_CASImgs_test;
            cudaMalloc(&d_CASImgs_test, sizeof(float) * imgSize * imgSize * nAxes_Stream);

            // Run the CUDA kernel for transforming the CASImgs to complex imgs (in order to apply the inverse FFT)
            CASImgsToImgs<<< dimGrid_CAS_to_Imgs, dimBlock_CAS_to_Imgs, 0, stream[i] >>>(
                gpuCASImgs_Vector[i], gpuImgs_Vector[i], imgSize
            );
     */

            // Plan the inverse FFT operation (for transforming the CASImgs back to imgs)
            // https://arcb.csc.ncsu.edu/~mueller/cluster/nvidia/0.8/NVIDIA_CUFFT_Library_0.8.pdf
            // https://docs.nvidia.com/cuda/cufft/index.html

       




            // TO DO: Need to apply fftshift before the inverse FFT https://github.com/OrangeOwlSolutions/FFT/wiki/The-fftshift-in-CUDA
            // http://www.orangeowlsolutions.com/archives/251

            // Execute the inverse FFT on each 2D slice of the gpuCASImgs
            //cufftExecC2C(plan, (cufftComplex *) gpuImgs_Vector[i], (cufftComplex *) gpuImgs_Vector[i], CUFFT_INVERSE);


            //cudaMemcpy(h_imgs, d_imgs, sizeof(cufftComplex) * size, cudaMemcpyDeviceToHost);
            
            //for (int z = 0; z < size; z ++)
            //{
            //    std::cout << "cufftComplex h_imgs.x[" << z << "]: " << h_imgs[z].x << '\n';
            //    std::cout << "cufftComplex h_imgs.y[" << z << "]: " << h_imgs[z].y << '\n';
   
           // }









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

            */








    // // imgSize = 5;

    // int nRows = 5;
    // int nCols = 5;
    // // int n[2] = {nRows, nCols};
    // // int howMany = 1; //nAxes_Stream

    // int IMAGE_DIM = 5;
    // int NUM_IMGS = 2;

    // int num_real_elements = NUM_IMGS * IMAGE_DIM * IMAGE_DIM; 
    
    // int batch = NUM_IMGS;           // --- Number of batched executions
    // int rank = 2;                   // --- 2D FFTs
    // int n[2] = {nRows, nCols};      // --- Size of the Fourier transform
    // int idist = nRows*nCols;        // --- Distance between batches
    // int odist = nRows*nCols;        // --- Distance between batches

    // int inembed[] = {nRows, nCols}; // --- Input size with pitch
    // int onembed[] = {nRows, nCols}; // --- Output size with pitch

    // int istride = 1;                // --- Distance between two successive input/output elements
    // int ostride = 1;                // --- Distance between two successive input/output elements

    // cufftPlanMany(&forwardFFTPlan,  rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);

    // // ALLOCATE HOST MEMORY
    // float *h_img;
    // cufftComplex* h_complex_img;
    // h_complex_img = (cufftComplex*)malloc(num_real_elements * sizeof(cufftComplex));
    // std::cout << "INPUT" << '\n';
    // for (int x=0; x < IMAGE_DIM; x++)
    // {
    //     for (int y=0; y < IMAGE_DIM; y++)
    //     {
    //         h_complex_img[y*IMAGE_DIM+x].x = x * IMAGE_DIM + y;
    //         std::cout << "h_complex_img[" << x << "][" << y << "].x: " <<  h_complex_img[y*IMAGE_DIM+x].x << '\n';
    //     }
    // }

    // for (int x=0; x < IMAGE_DIM; x++)
    // {
    //     int temp_x = x + IMAGE_DIM*IMAGE_DIM; // offset for image two
    //     for (int y=0; y < IMAGE_DIM; y++)
    //     {
    //         h_complex_img[y*IMAGE_DIM+temp_x].x = x * IMAGE_DIM + y ;
    //         std::cout << "h_complex_img[" << temp_x << "][" << y << "].x: " <<  h_complex_img[y*IMAGE_DIM+temp_x].x << '\n';
    //     }
    // }

    // // DEVICE MEMORY
    // float *d_img;
    // cufftComplex *d_complex_imgSpec, *d_output;

    // // ALLOCATE DEVICE MEMORY
    // cudaMalloc( (void**) &d_complex_imgSpec, num_real_elements * sizeof(cufftComplex));	
    // cudaMalloc( (void**) &d_output, num_real_elements * sizeof(cufftComplex));

    // // copy host memory to device (input image)
    // cudaMemcpy( d_complex_imgSpec, h_complex_img, num_real_elements * sizeof(cufftComplex), cudaMemcpyHostToDevice);        

    // // now run the forward FFT on the device (real to complex)
    // cufftExecC2C(forwardFFTPlan, (cufftComplex *) d_complex_imgSpec, (cufftComplex *) d_output, CUFFT_FORWARD);

    // cudaDeviceSynchronize();
    // // cufftExecR2C(forwardFFTPlan, d_img, d_complex_imgSpec, CUFFT_FORWARD);

    // // copy the DEVICE complex data to the HOST
    // // NOTE: we are only doing this so that you can see the data -- in general you want
    // // to do your computation on the GPU without wasting the time of copying it back to the host
    // cudaMemcpy( h_complex_img, d_output, num_real_elements * sizeof(cufftComplex), cudaMemcpyDeviceToHost) ;
    // cudaDeviceSynchronize();
    
    // std::cout << "" << '\n';
    // std::cout << "" << '\n';
    // std::cout << "" << '\n';
    // std::cout << "OUTPUT" << '\n';
    // std::cout << "IMAGE ONE" << '\n';
    // for (int x=0; x < (IMAGE_DIM); x++)
    // {
    //     std::cout << "h_complex_img[" << x << "]: ";
    //     for (int y=0; y < IMAGE_DIM; y++)
    //     {
    //         if ((h_complex_img[y*IMAGE_DIM+x].x*h_complex_img[y*IMAGE_DIM+x].x) < 0.001)
    //         {
    //             std::cout << " "   <<  0  << " + " << h_complex_img[y*IMAGE_DIM+x].y << "i   ";
    //         } else
    //         {
    //             std::cout << " "   <<  h_complex_img[y*IMAGE_DIM+x].x  << " + " << h_complex_img[y*IMAGE_DIM+x].y << "i   ";
    //         }
          
    //     }
    //     std::cout << '\n';
    // }

    // std::cout << '\n';
    // std::cout << '\n';
    // std::cout << '\n';
    // std::cout << "IMAGE TWO" << '\n';
    // for (int x=0; x < (IMAGE_DIM); x++)
    // {
    //     // Offset is IMAGE_DIM * IMAGE_DIM since we are on image two now
    //     int temp_x = x + IMAGE_DIM * IMAGE_DIM;
        
    //     std::cout << "h_complex_img[" << x << "]: ";
    //     for (int y=0; y < IMAGE_DIM; y++)
    //     {     
    //         //std::cout << "y*IMAGE_DIM+temp_x: " << y*IMAGE_DIM+temp_x << '\n';

            
    //         if ((h_complex_img[y*IMAGE_DIM+temp_x].x*h_complex_img[y*IMAGE_DIM+temp_x].x) < 0.001)
    //         {
    //             std::cout << " "   <<  0  << " + " << h_complex_img[y*IMAGE_DIM+temp_x].y << "i   ";
    //         } else
    //         {
    //             std::cout << " "   <<  h_complex_img[y*IMAGE_DIM+temp_x].x  << " + " << h_complex_img[y*IMAGE_DIM+temp_x].y << "i   ";
    //         }
          
    //     }
    //     std::cout << '\n';
    // }

