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

                        // if (k1*volSize*volSize+j1*volSize+i1 < volSize*volSize*volSize) // test
                        // {
                        *(img_ptr+j*imgSize+i)=*(img_ptr+j*imgSize+i)+//w;
                                w*( *(vol+k1*volSize*volSize+j1*volSize+i1));
                        // }
                    } //End k1
                }//End j1   
            }//End i1
        }//End if r
    }//End img_i
}

__global__ void CASImgsToComplexImgs(float* CASimgs, cufftComplex* imgs, int imgSize, int nSlices)
{
    // CUDA kernel for converting the CASImgs to imgs
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row
    // int k = blockIdx.z * blockDim.z + threadIdx.z; // Which image?

    // Are we outside the bounds of the image?
    if (i >= imgSize || i < 0 || j >= imgSize || j < 0){
        return;
    }
    
    // Each thread will do all the slices for position X and Y
    for (int k=0; k < nSlices; k++)  {        

        // CASimgs is the same dimensions as imgs
        int ndx_1 = i + j * imgSize + k * imgSize * imgSize;
        
        // Skip the first row and first column
        if (i == 0 || j == 0)
        {
            // Real component
            imgs[ndx_1].x = 0;

            // Imaginary component
            imgs[ndx_1].y = 0;

            
        } else 
        {

            // Offset to skip the first row then subtract from the end of the matrix and add the offset where the particular image starts in CASimgs
            int ndx_2 = imgSize + imgSize * imgSize - (i + j * imgSize) + k * imgSize * imgSize;

            // Real component
            imgs[ndx_1].x = 0.5*(CASimgs[ndx_1] + CASimgs[ndx_2]);

            // Imaginary component
            imgs[ndx_1].y = 0.5*(CASimgs[ndx_1] - CASimgs[ndx_2]);

        }
    }

    return;
}


__global__ void ComplexToReal(cufftComplex* ComplexImg, float* RealImg, int imgSize, int nSlices)
{
    // Extract the real component of a cufftComplex and save to a float array

    // CUDA kernel for converting the CASImgs to imgs
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row
    //int k = blockIdx.z * blockDim.z + threadIdx.z; // Which image?

    // Are we outside the bounds of the image?
    if (i >= imgSize || i < 0 || j >= imgSize || j < 0){
        return;
    }
    
    // Each thread will do all the slices for position X and Y
    for (int k=0; k < nSlices; k++)
    {         
        // Get the linear index of the current position
        int ndx = i + j * imgSize + k * imgSize * imgSize;       

        RealImg[ndx] = ComplexImg[ndx].x;
    }

}


__global__ void CropImgs(float* input, float* output, int imgSize, int nSlices)
{
    // Given the final projection images, crop out the zero padding to reduce memory size and transfer speed back to the CPU
    // imgSize is the size of the ouput (smaller) image

    // Index of the output (smaller) image
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row

    int interpFactor = 2;

    // Are we outside the bounds of the image?
    if (i >= imgSize || i < 0 || j >= imgSize || j < 0){
        return;
    }

    for (int k = 0; k < 1; k++){

        // Get the linear index of the output (smaller) image
        int ndx_1 = i + j * imgSize + k * imgSize * imgSize;   

        // Get the linear index of the input (larger) image
        // NOTE: The z axis is not affected since this is a stack of 2D images
        // + imgSize/interpFactor
        

        int ndx_2 = 
        (i + imgSize/interpFactor) + 
        (j + imgSize/interpFactor) * (imgSize * interpFactor) +
        k * imgSize *  imgSize;  

        output[ndx_1] = input[ndx_2];

    }
}



__global__ void ComplexImgsToCASImgs(float* CASimgs, cufftComplex* imgs, int imgSize)
{
    // CUDA kernel for converting the CASImgs to imgs
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Column
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row
    //int k = blockIdx.z * blockDim.z + threadIdx.z; // Which image?

    // Are we outside the bounds of the image?
    if (i >= imgSize || i < 0 || j >= imgSize || j < 0){
        return;
    }
    
    // Each thread will do all the slices for position X and Y
    for (int k=0; k< imgSize; k++)
    {    
        // CASimgs is the same dimensions as imgs
        int ndx = i + j * imgSize + k * imgSize * imgSize;       
        
        // Summation of the real and imaginary components
        CASimgs[ndx] = imgs[ndx].x + imgs[ndx].y;
    }

    return;
}


// https://github.com/marwan-abdellah/cufftShift/blob/master/Src/CUDA/Kernels/out-of-place/cufftShift_3D_OP.cu
__global__ void cufftShift_3D_slice_kernel(cufftComplex* input, cufftComplex* output, int N, int nSlices)
{
    // 3D Volume & 2D Slice & 1D Line
    int sLine   = N;
    int sSlice  = N * N;
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

    for (int zIndex = 0; zIndex < nSlices; zIndex ++)
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


float* ThreeD_ArrayToCASArray(float* gpuVol, int* volSize)
{
    // Convert a CUDA array to CAS array
    // Step 1: TO DO: Pad with zeros
    // Step 2: Take discrete Fourier transform using cuFFT
    // Step 3: TO DO: fftshift?
    // Step 4: TO DO: Convert to CAS volume using CUDA kernel

    // https://devtalk.nvidia.com/default/topic/410009/cufftexecr2c-only-gives-half-the-answer-33-/

    cufftHandle forwardFFTPlan;           
    cufftPlan3d(&forwardFFTPlan, volSize[0], volSize[1], volSize[2], CUFFT_C2C);

    int array_size = volSize[0] * volSize[1] * volSize[2];
    
    // Allocate memory for the resulting CAS volume
    float * d_CAS_Vol, *h_CAS_Vol;
    cudaMalloc(&d_CAS_Vol, sizeof(float) * array_size);
    h_CAS_Vol = (float *) malloc(sizeof(float) * array_size);

    // Create temporary arrays to hold the cufftComplex array        
    cufftComplex *h_complex_array, *d_complex_array, *d_complex_output_array;

    cudaMalloc(&d_complex_array, sizeof(cufftComplex) * array_size);
    cudaMalloc(&d_complex_output_array, sizeof(cufftComplex) * array_size);

    h_complex_array = (cufftComplex *) malloc(sizeof(cufftComplex) * array_size);

    // Convert the GPU volume to a cufftComplex array
    // TO DO: Replace this with the CUDA kernel
    for (int k = 0; k < array_size; k++) {
        h_complex_array[k].x = gpuVol[k]; // Real component
        h_complex_array[k].y = 0;         // Imaginary component
    }

    // Copy the complex version of the GPU volume to the first GPU
    cudaMemcpy( d_complex_array, h_complex_array, array_size * sizeof(cufftComplex), cudaMemcpyHostToDevice);        

    int gridSize  = ceil(volSize[0] / 32);
    int blockSize = 32; // 10*10*10 threads

    // Define CUDA kernel dimensions for converting the complex volume to a CAS volume
    dim3 dimGrid(gridSize, gridSize, 1);
    dim3 dimBlock(blockSize, blockSize, 1);

    // Apply a 3D FFT Shift
    cufftShift_3D_slice_kernel <<< dimGrid, dimBlock >>> (d_complex_array, d_complex_output_array, volSize[0], volSize[0]);
            
    // Execute the forward FFT on the 3D array
    cufftExecC2C(forwardFFTPlan, (cufftComplex *) d_complex_output_array, (cufftComplex *) d_complex_output_array, CUFFT_FORWARD);

    // Apply a second 3D FFT Shift
    cufftShift_3D_slice_kernel <<< dimGrid, dimBlock >>> (d_complex_output_array, d_complex_array, volSize[0], volSize[0]);

    // Convert the complex result of the forward FFT to a CAS img type
    ComplexImgsToCASImgs<<< dimGrid, dimBlock >>>(
        d_CAS_Vol, d_complex_array, volSize[0] // Assume the volume is a square for now
    );
    
    // Copy the resulting CAS volume back to the host
    cudaMemcpy( h_CAS_Vol, d_CAS_Vol, array_size * sizeof(float), cudaMemcpyDeviceToHost);        

    cudaDeviceSynchronize();
    
    // Free the temporary memory
    cudaFree(d_complex_array);
    cudaFree(d_complex_output_array);
    cudaFree(d_CAS_Vol);

    // Return the resulting CAS volume
    return h_CAS_Vol;

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
    std::cout << "maskRadius: " << maskRadius << '\n';

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
        std::cout << "Current batch : " << currBatch << " out of " << nBatches << '\n';

        for (int i = 0; i < nStreams; i++) // Loop through the streams   
        {             
            int curr_GPU = i % numGPUs; // Use the remainder operator to split evenly between GPUs
        
            cudaSetDevice(curr_GPU);     

            std::cout << "Running stream " << i << '\n';

            // How many coordinate axes to assign to this CUDA stream? 
            std::cout <<  '\n';
            
            // Check to make sure we don't try to process more coord axes than we have
            if (processed_nAxes + numAxesPerStream[i] >= nAxes) 
            {
                numAxesPerStream[i] = min(numAxesPerStream[i], nAxes - processed_nAxes);
            }

            // Is there at least one coordinate axes to process for this stream?
            if (numAxesPerStream[i] >= 1 ) // TO DO: Fix this && processed_nAxes < nAxes
            {
              
                // if (numAxesPerStream[i] + processed_nAxes >= nAxes)
                // {
                //     numAxesPerStream[i] = nAxes - processed_nAxes;
                // }

                std::cout << "Number of axes in this stream " << numAxesPerStream[i] << '\n';
                
                // int interpFactor = 2;

                // This seems to be needed
                // cudaMemset(gpuComplexImgs_Vector[i], 0,  imgSize * imgSize * numAxesPerStream[i] * sizeof(cufftComplex));
                // cudaMemset(gpuComplexImgs_Shifted_Vector[i], 0,  imgSize * imgSize * numAxesPerStream[i] * sizeof(cufftComplex));
                cudaMemset(gpuCASImgs_Vector[i], 0, imgSize * imgSize * numAxesPerStream[i] * sizeof(float));                

                
                // Calculate the offsets (in bytes) to determine which part of the array to copy for this stream
                int gpuCoordAxes_Offset    = processed_nAxes * 9 * 1;          // Each axes has 9 elements (X, Y, Z)
                int coord_Axes_streamBytes = numAxesPerStream[i] * 9 * sizeof(float); // Copy the entire vector for now
                   
                // Copy the section of gpuCoordAxes which this stream will process on the current GPU
                cudaMemcpyAsync(gpuCoordAxes_Vector[i], &coordAxes_CPU_Pinned[gpuCoordAxes_Offset], coord_Axes_streamBytes, cudaMemcpyHostToDevice, stream[i]);            

                // Temporary array to hold the gpuCAS images 
                // TO DO: allocate this in the gridder class
                // float *d_CAS_imgs; // imgSize is the size of the zero padded projection images
                // cudaMalloc(&d_CAS_imgs, sizeof(float) * imgSize * imgSize * numAxesPerStream[i]); 
                

                // Run the forward projection kernel
                // NOTE: Only need one gpuVol_Vector and one ker_bessel_Vector per GPU
                // NOTE: Each stream needs its own gpuCASImgs_Vector and gpuCoordAxes_Vector
                gpuForwardProjectKernel<<< dimGrid, dimBlock, 0, stream[i] >>>(
                    gpuVol_Vector[curr_GPU], volSize, gpuCASImgs_Vector[i],
                    imgSize, gpuCoordAxes_Vector[i], numAxesPerStream[i],
                    maskRadius, ker_bessel_Vector[curr_GPU], 501, 2);    

                // gpuForwardProjectKernel<<< dimGrid, dimBlock, 0, stream[i] >>>(
                //     gpuVol_Vector[curr_GPU], volSize, gpuCASImgs_Vector[i],
                //     imgSize, gpuCoordAxes_Vector[i], numAxesPerStream[i],
                //     maskRadius, ker_bessel_Vector[curr_GPU], 501, 2);        
                
                // bool Use_CUDA_FFT = false;

                // if (Use_CUDA_FFT == true)
                // {

                //     // Convert the CASImgs to complex cufft type
                //     CASImgsToComplexImgs<<< dimGrid, dimBlock, 0, stream[i] >>>(gpuCASImgs_Vector[i], gpuComplexImgs_Vector[i], imgSize, numAxesPerStream[i]);
                    
                //     // Run FFTShift on gpuComplexImgs_Vector[i] (can't reuse the same input as the output for the FFT shift kernel)
                //     cufftShift_3D_slice_kernel <<< dimGrid, dimBlock, 0, stream[i] >>> (gpuComplexImgs_Vector[i], gpuComplexImgs_Shifted_Vector[i], imgSize, numAxesPerStream[i]);
                
                //     // Create a plan for taking the inverse of the CAS imgs
                //     cufftHandle inverseFFTPlan;   
                //     int nRows = imgSize;
                //     int nCols = imgSize;
                //     int batch = numAxesPerStream[i];       // --- Number of batched executions
                //     int rank = 2;                   // --- 2D FFTs
                //     int n[2] = {nRows, nCols};      // --- Size of the Fourier transform
                //     int idist = nRows*nCols;        // --- Distance between batches
                //     int odist = nRows*nCols;        // --- Distance between batches

                //     int inembed[] = {nRows, nCols}; // --- Input size with pitch
                //     int onembed[] = {nRows, nCols}; // --- Output size with pitch

                //     int istride = 1;                // --- Distance between two successive input/output elements
                //     int ostride = 1;                // --- Distance between two successive input/output elements

                //     cufftPlanMany(&inverseFFTPlan,  rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);            
                //     cufftSetStream(inverseFFTPlan, stream[i]); // Set the FFT plan to the current stream to process

                //     // Inverse FFT
                //     cufftExecC2C(inverseFFTPlan, (cufftComplex *) gpuComplexImgs_Shifted_Vector[i], (cufftComplex *) gpuComplexImgs_Shifted_Vector[i], CUFFT_INVERSE);
                    
                //     // FFTShift again (can't reuse the same input as the output for the FFT shift kernel)
                //     cufftShift_3D_slice_kernel <<< dimGrid, dimBlock, 0, stream[i]>>> (gpuComplexImgs_Shifted_Vector[i], gpuComplexImgs_Vector[i], imgSize, numAxesPerStream[i]);
                
                //     // Convert from the complex images to the real (resued the d_CAS_imgs GPU array)
                //     ComplexToReal<<< dimGrid, dimBlock, 0, stream[i] >>>(gpuComplexImgs_Vector[i], gpuCASImgs_Vector[i], imgSize, numAxesPerStream[i]);            
                // }
                
                // Perform the cropping now

                // Run kernel to crop the projection images (to remove the zero padding)
                // int gridSize = 32;
                // int blockSize = imgSize / gridSize;

                // dim3 dimGridCrop(gridSize, gridSize, gridSize);
                // dim3 dimBlockCrop(blockSize, blockSize, blockSize);

                
                // CropImgs<<< dimGridCrop, dimBlockCrop, 0 , stream[i]>>>(d_CAS_imgs, gpuCASImgs_Vector[i], imgSize/interpFactor, numAxesPerStream[i]);
                

                // Use unsigned long long int type to allow for array length larger than maximum int32 value 
                // Number of bytes of already processed images
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

                // Lastly, copy the resulting gpuCASImgs to the host pinned memory (CPU)
                // cudaMemcpyAsync(
                //     &CASImgs_CPU_Pinned[CASImgs_CPU_Offset[0] * CASImgs_CPU_Offset[1] * CASImgs_CPU_Offset[2]],
                //     gpuCASImgs_Vector[i], gpuCASImgs_streamBytes, cudaMemcpyDeviceToHost, stream[i]);
                
                //cudaDeviceSynchronize();
				
                //cudaStreamSynchronize(stream[i]);


                // Update the number of coordinate axes which have already been assigned to a CUDA stream
                processed_nAxes = processed_nAxes + numAxesPerStream[i];

            }


            
            std::cout << "processed_nAxes: " << processed_nAxes << '\n';
            std::cout << "Axes remaining: " << nAxes - processed_nAxes << '\n';
            cudaStreamSynchronize(stream[i]);
            //cudaDeviceSynchronize();

        } 

           //cudaDeviceSynchronize();
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

