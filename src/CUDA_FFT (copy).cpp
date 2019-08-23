








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

     
		
			// This seems to be needed
			// cudaMemset(gpuComplexImgs_Vector[i], 0, imgSize * imgSize * numAxesPerStream[i] * sizeof(cufftComplex));
			// cudaMemset(gpuComplexImgs_Shifted_Vector[i], 0, imgSize * imgSize * numAxesPerStream[i] * sizeof(cufftComplex));
			cudaMemset(gpuCASImgs_Vector[i], 0, imgSize * imgSize * numAxesPerStream[i] * sizeof(float));
			cudaMemset(gpuCoordAxes_Vector[i], 0, numAxesPerStream[i] * 9 * sizeof(float));
            		
			// // gpuCASImgs_Vector is actually in real space (not CAS) and needs to be converted to the frequency domain
			// int * volSizeCAS = new int[3];
			// volSizeCAS[0] = imgSize;
			// volSizeCAS[1] = imgSize;
			// volSizeCAS[2] = numAxesPerStream[i];

			// // To DO: Make this function support GPU arrays too
			// float *h_CAS_Vol;
			// h_CAS_Vol = (float *)malloc(sizeof(float) * volSizeCAS[0] * volSizeCAS[1] * volSizeCAS[2]);
			// h_CAS_Vol = ThreeD_ArrayToCASArray(&CASImgs_CPU_Pinned[CASImgs_CPU_Offset[0] * CASImgs_CPU_Offset[1] * CASImgs_CPU_Offset[2]], volSizeCAS);

			// // Convert the CAS volume to cufftComplex
			// cufftComplex *h_complex_array;
			// h_complex_array = (cufftComplex *)malloc(sizeof(cufftComplex) * volSizeCAS[0] * volSizeCAS[1] * volSizeCAS[2]);

			// for (int k = 0; k < volSizeCAS[0] * volSizeCAS[1] * volSizeCAS[2]; k++) {
			// 	h_complex_array[k].x = h_CAS_Vol[k]; // Real component
			// 	h_complex_array[k].y = 0;            // Imaginary component
			// }

			// // Copy from host to device
			// cudaMemcpy(gpuComplexImgs_Vector[i], h_complex_array, volSizeCAS[0] * volSizeCAS[1] * volSizeCAS[2] * sizeof(cufftComplex), cudaMemcpyHostToDevice);

			// // Run FFTShift on gpuComplexImgs_Vector[i] (can't reuse the same input as the output for the FFT shift kernel)
			// cufftShift_3D_slice_kernel_2 <<< dimGrid, dimBlock, 0, stream[i] >>> (gpuComplexImgs_Vector[i], gpuComplexImgs_Shifted_Vector[i], imgSize, numAxesPerStream[i]);

			// // Create a plan for taking the inverse of the CAS imgs
			// cufftHandle forwardFFTPlan;
			// int nRows = imgSize;
			// int nCols = imgSize;
			// int batch = numAxesPerStream[i];       // --- Number of batched executions
			// int rank = 2;                   // --- 2D FFTs
			// int n[2] = { nRows, nCols };      // --- Size of the Fourier transform
			// int idist = nRows * nCols;        // --- Distance between batches
			// int odist = nRows * nCols;        // --- Distance between batches

			// int inembed[] = { nRows, nCols }; // --- Input size with pitch
			// int onembed[] = { nRows, nCols }; // --- Output size with pitch

			// int istride = 1;                // --- Distance between two successive input/output elements
			// int ostride = 1;                // --- Distance between two successive input/output elements

			// cufftPlanMany(&forwardFFTPlan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, batch);
			// cufftSetStream(forwardFFTPlan, stream[i]); // Set the FFT plan to the current stream to process

			// // Forward FFT
			// cufftExecC2C(forwardFFTPlan, (cufftComplex *)gpuComplexImgs_Shifted_Vector[i], (cufftComplex *)gpuComplexImgs_Shifted_Vector[i], CUFFT_FORWARD);

			// // Run FFTShift on gpuComplexImgs_Vector[i] (can't reuse the same input as the output for the FFT shift kernel)
			// cufftShift_3D_slice_kernel_2 <<< dimGrid, dimBlock, 0, stream[i] >>> (gpuComplexImgs_Shifted_Vector[i], gpuComplexImgs_Vector[i], imgSize, numAxesPerStream[i]);

			// // Convert the complex result of the forward FFT to a CAS img type
			// ComplexImgsToCASImgs_2 <<< dimGrid, dimBlock, 0, stream[i] >>> (
			// 	gpuCASImgs_Vector[i], gpuComplexImgs_Vector[i], imgSize // Assume the volume is a square for now
			// 	);





                // This seems to be needed
                // cudaMemset(gpuComplexImgs_Vector[i], 0,  imgSize * imgSize * numAxesPerStream[i] * sizeof(cufftComplex));
                // cudaMemset(gpuComplexImgs_Shifted_Vector[i], 0,  imgSize * imgSize * numAxesPerStream[i] * sizeof(cufftComplex));
                // cudaMemset(gpuCASImgs_Vector[i], 0, imgSize * imgSize * numAxesPerStream[i] * sizeof(float));       

   // // Has the ComplexImgs array been allocated and defined?
        // // The name of the GPU pointer is gpuComplexImgs_0 for GPU 0, gpuComplexImgs_1 for GPU 1, etc.
        // if (Mem_obj->GPUArrayAllocated("gpuComplexImgs_" + std::to_string(i), gpuDevice) == false)
        // {
        //     // Allocate the ComplexImgs array on the current gpuDevice
        //     Mem_obj->CUDA_alloc("gpuComplexImgs_" + std::to_string(i), "cufftComplex", gpuCASImgs_Size, gpuDevice);
        // }

        // // Has the ComplexImgs_Shifted array been allocated and defined?
        // // The name of the GPU pointer is gpuComplexImgs_Shifted_0 for GPU 0, gpuComplexImgs_Shifted_1 for GPU 1, etc.
        // if (Mem_obj->GPUArrayAllocated("gpuComplexImgs_Shifted_" + std::to_string(i), gpuDevice) == false)
        // {
        //     // Allocate the ComplexImgs_Shifted array on the current gpuDevice
        //     Mem_obj->CUDA_alloc("gpuComplexImgs_Shifted_" + std::to_string(i), "cufftComplex", gpuCASImgs_Size, gpuDevice);
        // }

            // std::vector<cufftComplex *> gpuComplexImgs_Vector;
    // std::vector<cufftComplex *> gpuComplexImgs_Shifted_Vector;



    
        // Array to hold the complex version of the CASImgs (intermediate step when doing the FFT)
        // gpuComplexImgs_Vector.push_back(this->Mem_obj->ReturnCUDAComplexPtr("gpuComplexImgs_" + std::to_string(i)));

        // // Array to hold the FFT shifted complex version of the CASImgs (intermediate step when doing the FFT)
        // gpuComplexImgs_Shifted_Vector.push_back(this->Mem_obj->ReturnCUDAComplexPtr("gpuComplexImgs_Shifted_" + std::to_string(i)));


                // gpuComplexImgs_Vector, gpuComplexImgs_Shifted_Vector,                                         // Vector of GPU array pointers


    // Convert the given volume to a CAS volume
    // Steps are (1) pad with zeros, (2) run forward FFT using CUDA, (3) sum the real and imaginary components
    int array_size = gpuVolSize[0] * gpuVolSize[1] * gpuVolSize[2];
    float *CAS_Vol; // = new float [array_size];
    CAS_Vol = (float *)malloc(sizeof(float) * array_size);

    std::cout << "ThreeD_ArrayToCASArray()..." << '\n';

    // CAS_Vol = ThreeD_ArrayToCASArray(gpuVol, gpuVolSize);
    std::cout << "Done with ThreeD_ArrayToCASArray()..." << '\n';

    float *Output_CAS_Vol; // = new float [array_size];
    Output_CAS_Vol = (float *)malloc(sizeof(float) * array_size);

    std::memcpy(Output_CAS_Vol, gpuVol, sizeof(float) * array_size);




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



// Temporary array to hold the gpuCAS images 
// TO DO: allocate this in the gridder class
// float *d_CAS_imgs; // imgSize is the size of the zero padded projection images
// cudaMalloc(&d_CAS_imgs, sizeof(float) * imgSize * imgSize * numAxesPerStream[i]); 


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


