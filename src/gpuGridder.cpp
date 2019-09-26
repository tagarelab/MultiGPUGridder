#include "gpuGridder.h"

#define Log(x)                  \
    {                           \
        std::cout << x << '\n'; \
    }

int gpuGridder::EstimateMaxAxesToAllocate(int VolumeSize, int interpFactor)
{
    // Estimate the maximum number of coordinate axes to allocate on the GPU
    cudaSetDevice(this->GPU_Device);
    size_t mem_tot = 0;
    size_t mem_free = 0;
    cudaMemGetInfo(&mem_free, &mem_tot);

    // Throw error if mem_free is zero
    if (mem_free <= 0)
    {
        std::cerr << "No free memory on GPU " << this->GPU_Device << '\n';
        this->ErrorFlag = 1;
        return -1;
    }

    // Estimate how many bytes of memory is needed to process each coordinate axe
    int CASImg_Length = (VolumeSize * interpFactor + this->extraPadding * 2) * (VolumeSize * interpFactor + this->extraPadding * 2);
    int Img_Length = (VolumeSize * interpFactor) * (VolumeSize * interpFactor);
    int Bytes_per_Img = Img_Length * sizeof(float);
    int Bytes_per_CASImg = CASImg_Length * sizeof(float);
    int Bytes_per_ComplexCASImg = CASImg_Length * sizeof(cufftComplex);
    int Bytes_for_CASVolume = pow((VolumeSize * interpFactor + this->extraPadding * 2), 3) * sizeof(float);
    int Bytes_for_CoordAxes = 9 * sizeof(float); // 9 elements per axes

    // How many coordinate axes would fit in the remaining free GPU memory?
    int EstimatedMaxAxes = (mem_free - Bytes_for_CASVolume) / (Bytes_per_Img + Bytes_per_CASImg + Bytes_per_ComplexCASImg + Bytes_for_CoordAxes);

    // Leave room on the GPU to run the FFTs and CUDA kernels so only use 30% of the maximum possible
    EstimatedMaxAxes = floor(EstimatedMaxAxes * 0.3);

    return EstimatedMaxAxes;
}

float *gpuGridder::GetCASVolumePtr()
{
    // Return the pointer to the CAS volume (needed for adding the volumes from all GPUs together for the volume reconstruction)
    return this->d_CASVolume->GetPointer();
}

float *gpuGridder::GetPlaneDensityPtr()
{
    // Return the pointer to the plane density volume (needed for adding the volumes from all GPUs together for the volume reconstruction)
    return this->d_PlaneDensity->GetPointer();
}

void gpuGridder::CopyVolumeToHost()
{
    // Copy the volume from the GPU to pinned CPU memory
    this->d_Volume->CopyFromGPU(this->h_Volume->GetPointer(), this->h_Volume->bytes());
}

void gpuGridder::CopyCASVolumeToHost()
{
    // Copy the CAS volume from the GPU to pinned CPU memory
    this->d_CASVolume->CopyFromGPU(this->h_CASVolume->GetPointer(), this->h_CASVolume->bytes());
}

void gpuGridder::CopyPlaneDensityToHost()
{
    // Copy the plane density volume from the GPU to pinned CPU memory
    this->d_PlaneDensity->CopyFromGPU(this->h_PlaneDensity->GetPointer(), this->h_PlaneDensity->bytes());
}

float *gpuGridder::GetVolumeFromDevice()
{
    float *Volume = new float[this->d_Volume->length()];
    this->d_Volume->CopyFromGPU(Volume, this->d_Volume->bytes());

    return Volume;
}

float *gpuGridder::GetCASVolumeFromDevice()
{
    float *CASVolume = new float[this->d_CASVolume->length()];
    this->d_CASVolume->CopyFromGPU(CASVolume, this->d_CASVolume->bytes());

    return CASVolume;
}

float *gpuGridder::GetPlaneDensityFromDevice()
{
    float *PlaneDensity = new float[this->d_PlaneDensity->length()];
    this->d_PlaneDensity->CopyFromGPU(PlaneDensity, this->d_PlaneDensity->bytes());

    return PlaneDensity;
}

void gpuGridder::VolumeToCASVolume()
{
    cudaSetDevice(this->GPU_Device);

    // Convert the volume to CAS volume
    // gpuFFT::VolumeToCAS(
    //     this->h_Volume->GetPointer(),
    //     this->h_Volume->GetSize(0),
    //     this->h_CASVolume->GetPointer(),
    //     this->interpFactor,
    //     this->extraPadding);

    // Convert a GPU volume to CAS volume
    // Note: The volume must be square (i.e. have the same dimensions for the X, Y, and Z)
    // Step 1: Pad the input volume with zeros and convert to cufftComplex type
    // Step 2: fftshift
    // Step 3: Take discrete Fourier transform using cuFFT
    // Step 4: fftshift
    // Step 5: Convert to CAS volume using CUDA kernel
    // Step 6: Apply extra zero padding

    // Note: d_CAS_Vol has size of d_Vol * interpFactor + extraPadding * 2
    // Note: d_CASVol_Interp has size of d_Vol * interpFactor

    std::cout << "GPUVolumeToCAS: " << '\n';

    // Example: input size = 128; interpFactor = 2 -> paddedVolSize = 256
    int PaddedVolumeSize = this->VolumeSize * this->interpFactor;

    // Example: input size = 128; interpFactor = 2; extra padding = 3; -> paddedVolSize_Extra = 262
    int PaddedVolumeSize_Extra = PaddedVolumeSize + this->extraPadding * 2;

    // Allocate memory for the intermediate steps
    // float *d_CASVol_Interp;
    // cufftComplex *d_complex_array;

    // CAS Volume interpolated using interpFactor
    // cudaError_t status = cudaMalloc(&d_CASVol_Interp, sizeof(float) * pow(inputVolSize * interpFactor, 3));
    // std::cerr << "cudaMalloc: " << cudaGetErrorString(status) << '\n';

    // status = cudaMalloc(&d_complex_array, sizeof(cufftComplex) * paddedVolSize * paddedVolSize * paddedVolSize);
    // std::cerr << "cudaMalloc: " << cudaGetErrorString(status) << '\n';

    // STEP 1: Pad the input volume with zeros
    // gpuFFT::PadVolume(d_Vol, d_PaddedVolume, inputVolSize, paddedVolSize);

    PadVolumeFilter *PadFilter = new PadVolumeFilter();
    PadFilter->SetInput(this->d_Volume->GetPointer());
    PadFilter->SetOutput(this->d_PaddedVolume->GetPointer());
    PadFilter->SetInputSize(this->VolumeSize);
    PadFilter->SetPaddingX((PaddedVolumeSize - this->VolumeSize) / 2);
    PadFilter->SetPaddingY((PaddedVolumeSize - this->VolumeSize) / 2);
    PadFilter->SetPaddingZ((PaddedVolumeSize - this->VolumeSize) / 2);
    // PadFilter->SetNumberOfSlices(numImgs);
    PadFilter->Update();
    // cudaDeviceSynchronize();

    // Convert the d_CASVol_Interp to complex type (need cufftComplex type for the forward FFT)
    // gpuFFT::RealToComplex(d_PaddedVolume, d_PaddedVolumeComplex, paddedVolSize, paddedVolSize);

    RealToComplexFilter *RealToComplex = new RealToComplexFilter();
    RealToComplex->SetRealInput(this->d_PaddedVolume->GetPointer());
    RealToComplex->SetComplexOutput(this->d_PaddedVolumeComplex->GetPointer());
    RealToComplex->SetVolumeSize(PaddedVolumeSize);
    // CASFilter->SetNumberOfSlices(CroppedCASVolumeSize);
    RealToComplex->Update();

    // STEP 2: Apply an in place 3D FFT Shift
    // gpuFFT::cufftShift_3D(d_PaddedVolumeComplex, paddedVolSize, paddedVolSize);
    FFTShift3DFilter *FFTShiftFilter = new FFTShift3DFilter();
    FFTShiftFilter->SetInput(this->d_PaddedVolumeComplex->GetPointer());
    FFTShiftFilter->SetVolumeSize(PaddedVolumeSize);
    FFTShiftFilter->Update();

    // STEP 3: Execute the forward FFT on the 3D array
    // gpuFFT::FowardFFT(d_PaddedVolumeComplex, PaddedVolumeSize);

    // Plan the forward FFT
    cufftHandle forwardFFTPlan;
    cufftPlan3d(&forwardFFTPlan, PaddedVolumeSize, PaddedVolumeSize, PaddedVolumeSize, CUFFT_C2C);
    cufftExecC2C(
        forwardFFTPlan,
        (cufftComplex *)this->d_PaddedVolumeComplex->GetPointer(),
        (cufftComplex *)this->d_PaddedVolumeComplex->GetPointer(),
        CUFFT_FORWARD);

    // Free the plan
    cufftDestroy(forwardFFTPlan);

    // cudaDeviceSynchronize();
    // STEP 4: Apply a second in place 3D FFT Shift
    // gpuFFT::cufftShift_3D(d_PaddedVolumeComplex, paddedVolSize, paddedVolSize);

    FFTShiftFilter->SetInput(this->d_PaddedVolumeComplex->GetPointer());
    FFTShiftFilter->SetVolumeSize(PaddedVolumeSize);
    FFTShiftFilter->Update();

    // STEP 5: Convert the complex result of the forward FFT to a CAS img type
    // gpuFFT::ComplexImgsToCAS(d_PaddedVolumeComplex, d_PaddedVolume, paddedVolSize, paddedVolSize);

    ComplexToCASFilter *CASFilter = new ComplexToCASFilter();
    CASFilter->SetComplexInput(this->d_PaddedVolumeComplex->GetPointer());
    CASFilter->SetCASVolumeOutput(this->d_PaddedVolume->GetPointer());
    CASFilter->SetVolumeSize(PaddedVolumeSize);
    // CASFilter->SetNumberOfSlices(CroppedCASVolumeSize);
    CASFilter->Update();

    // STEP 6: Pad the result with the additional padding
    // gpuFFT::PadVolume(d_PaddedVolume, d_CAS_Vol, paddedVolSize, paddedVolSize_Extra);
    // PadVolumeFilter *PadFilter = new PadVolumeFilter();
    PadFilter->SetInput(this->d_PaddedVolume->GetPointer());
    PadFilter->SetOutput(this->d_CASVolume->GetPointer());
    PadFilter->SetInputSize(PaddedVolumeSize);
    PadFilter->SetPaddingX((PaddedVolumeSize_Extra - PaddedVolumeSize) / 2);
    PadFilter->SetPaddingY((PaddedVolumeSize_Extra - PaddedVolumeSize) / 2);
    PadFilter->SetPaddingZ((PaddedVolumeSize_Extra - PaddedVolumeSize) / 2);
    // PadFilter->SetNumberOfSlices(numImgs);
    PadFilter->Update();
    cudaDeviceSynchronize();

    // Free the temporary memory allocated
}

void gpuGridder::CopyCASVolumeToGPUAsyc()
{
    // Copy the CAS volume to the GPU asynchronously
    this->d_CASVolume->CopyToGPUAsyc(this->h_CASVolume->GetPointer(), this->h_CASVolume->bytes());
}

void gpuGridder::SetGPU(int GPU_Device)
{
    // Set which GPU to use

    // Check how many GPUs there are on the computer
    int numGPUDetected;
    cudaGetDeviceCount(&numGPUDetected);

    // Check wether the given GPU_Device value is valid
    if (GPU_Device < 0 || GPU_Device >= numGPUDetected) //  An invalid numGPUs selection was chosen
    {
        std::cerr << "GPU_Device number provided " << GPU_Device << '\n';
        std::cerr << "Error in GPU selection. Please provide an integer between 0 and the number of NVIDIA graphic cards on your computer. Use SetNumberGPUs() function." << '\n';
        this->ErrorFlag = 1;
        return;
    }

    this->GPU_Device = GPU_Device;
}

void gpuGridder::InitilizeGPUArrays()
{
    // Initilize the GPU arrays and allocate the needed memory on the GPU
    cudaSetDevice(this->GPU_Device);
    if (this->MaxAxesToAllocate == 0)
    {
        std::cerr << "MaxAxesToAllocate must be a positive integer. Please run EstimateMaxAxesToAllocate() first. " << '\n';
        this->ErrorFlag = 1;
        return;
    }

    int PaddedVolumeSize = this->VolumeSize * this->interpFactor;

    // Allocate the volume
    this->d_Volume = new MemoryStructGPU<float>(this->h_Volume->GetDim(), this->h_Volume->GetSize(), this->GPU_Device);
    this->d_Volume->AllocateGPUArray();

    // Allocate padded volume (used for converting the volume to CAS volume)
    this->d_PaddedVolume = new MemoryStructGPU<float>(3, PaddedVolumeSize, PaddedVolumeSize, PaddedVolumeSize, this->GPU_Device);
    this->d_PaddedVolume->AllocateGPUArray();

    this->d_PaddedVolumeComplex = new MemoryStructGPU<cufftComplex>(3, PaddedVolumeSize, PaddedVolumeSize, PaddedVolumeSize, this->GPU_Device);
    this->d_PaddedVolumeComplex->AllocateGPUArray();

    // status = cudaMalloc(&d_complex_array, sizeof(cufftComplex) * paddedVolSize * paddedVolSize * paddedVolSize);
    // std::cerr << "cudaMalloc: " << cudaGetErrorString(status) << '\n';

    // Allocate the CAS volume
    this->d_CASVolume = new MemoryStructGPU<float>(this->h_CASVolume->GetDim(), this->h_CASVolume->GetSize(), this->GPU_Device);
    this->d_CASVolume->AllocateGPUArray();

    // Allocate the plane density array (for the back projection)
    this->d_PlaneDensity = new MemoryStructGPU<float>(this->h_CASVolume->GetDim(), this->h_CASVolume->GetSize(), this->GPU_Device);
    this->d_PlaneDensity->AllocateGPUArray();

    // Allocate the CAS images
    if (this->h_CASImgs != nullptr)
    {
        // The pinned CASImgs was previously created so use its deminsions (i.e. creating CASImgs is optional)
        this->d_CASImgs = new MemoryStructGPU<float>(this->h_CASImgs->GetDim(), this->h_CASImgs->GetSize(), this->GPU_Device);
        this->d_CASImgs->AllocateGPUArray();

        // Allocate the complex CAS images array
        this->d_CASImgsComplex = new MemoryStructGPU<cufftComplex>(this->h_CASImgs->GetDim(), this->h_CASImgs->GetSize(), this->GPU_Device);
        this->d_CASImgsComplex->AllocateGPUArray();
    }
    else
    {
        // First, create a dims array of the correct size of d_CASImgs
        int *size = new int[3];
        size[0] = this->h_Imgs->GetSize(0) * this->interpFactor;
        size[1] = this->h_Imgs->GetSize(1) * this->interpFactor;
        size[2] = std::min(this->GetNumAxes(), this->MaxAxesToAllocate);

        this->d_CASImgs = new MemoryStructGPU<float>(3, size, this->GPU_Device);
        this->d_CASImgs->AllocateGPUArray();
        delete[] size;

        // Allocate the complex CAS images array
        this->d_CASImgsComplex = new MemoryStructGPU<cufftComplex>(this->h_CASImgs->GetDim(), size, this->GPU_Device);
        this->d_CASImgsComplex->AllocateGPUArray();
    }

    // Limit the number of axes to allocate to be MaxAxesToAllocate
    int *imgs_size = new int[3];
    imgs_size[0] = this->h_Imgs->GetSize(0);
    imgs_size[1] = this->h_Imgs->GetSize(1);
    imgs_size[2] = std::min(this->GetNumAxes(), this->MaxAxesToAllocate);

    // Allocate the images
    this->d_Imgs = new MemoryStructGPU<float>(this->h_Imgs->GetDim(), imgs_size, this->GPU_Device);
    this->d_Imgs->AllocateGPUArray();
    delete[] imgs_size;

    // Allocate the coordinate axes array
    int *axes_size = new int[1];
    axes_size[0] = std::min(this->GetNumAxes(), this->MaxAxesToAllocate);
    axes_size[0] = axes_size[0] * 9; // 9 elements per coordinate axes

    this->d_CoordAxes = new MemoryStructGPU<float>(this->h_CoordAxes->GetDim(), this->h_CoordAxes->GetSize(), this->GPU_Device);
    this->d_CoordAxes->AllocateGPUArray();
    delete[] axes_size;

    // Allocate the Kaiser bessel lookup table
    this->d_KB_Table = new MemoryStructGPU<float>(this->h_KB_Table->GetDim(), this->h_KB_Table->GetSize(), this->GPU_Device);
    this->d_KB_Table->AllocateGPUArray();
    this->d_KB_Table->CopyToGPUAsyc(this->h_KB_Table->GetPointer(), this->h_KB_Table->bytes());
}

void gpuGridder::InitilizeCUDAStreams()
{
    cudaSetDevice(this->GPU_Device);

    // Create the CUDA streams
    this->streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * this->nStreams);

    for (int i = 0; i < this->nStreams; i++) // Loop through the streams
    {
        cudaStreamCreate(&this->streams[i]);
    }
}

void gpuGridder::Allocate()
{
    // Allocate the needed GPU memory

    // Have the GPU arrays already been allocated?
    if (this->GPUArraysAllocatedFlag == false)
    {
        // Estimate the maximum number of coordinate axes to allocate per stream
        this->MaxAxesToAllocate = EstimateMaxAxesToAllocate(this->h_Volume->GetSize(0), this->interpFactor);

        // Initilize the needed arrays on the GPU
        InitilizeGPUArrays();

        // Initilize the CUDA streams
        InitilizeCUDAStreams();

        this->GPUArraysAllocatedFlag = true;
    }
}

void gpuGridder::PrintMemoryAvailable()
{
    // Check to make sure the GPU has enough available memory left
    size_t mem_tot_0 = 0;
    size_t mem_free_0 = 0;
    cudaMemGetInfo(&mem_free_0, &mem_tot_0);
    std::cout << "Memory remaining on GPU " << this->GPU_Device << " " << mem_free_0 << " out of " << mem_tot_0 << '\n';
}

gpuGridder::Offsets gpuGridder::PlanOffsetValues(int coordAxesOffset, int nAxes)
{
    // Loop through all of the coordinate axes and calculate the corresponding pointer offset values
    // which are needed for running the CUDA kernels

    // For compactness define the CASImgSize, CASVolSize, and ImgSize here
    int CASImgSize = this->d_CASImgs->GetSize(0);
    int CASVolSize = this->d_CASVolume->GetSize(0);
    int ImgSize = this->d_Imgs->GetSize(0);

    // Create an instance of the Offsets struct
    gpuGridder::Offsets Offsets_obj;
    Offsets_obj.num_offsets = 0;

    // Cumulative number of axes which have already been assigned to a CUDA stream
    int processed_nAxes = 0;

    // While we have coordinate axes to process, loop through the GPUs and the streams
    int MaxBatches = 10000; // Maximum iterations in case we get stuck in the while loop for some reason
    int batch = 0;

    // Initilize a variable which will remember how many axes have been assigned to the GPU
    // during the current batch. This is needed for calculating the offset for the stream when
    // the number of streams is greater than the number of GPUs. This resets to zeros after each
    // batch since the same GPU memory is used for the next batch.
    int numAxesGPU_Batch = 0;

    // Estimate how many coordinate axes to assign to each stream
    int EstimatedNumAxesPerStream;
    if (nAxes <= this->MaxAxesToAllocate)
    {
        // The number of coordinate axes is less than or equal to the total number of axes to process
        EstimatedNumAxesPerStream = ceil((double)nAxes / (double)this->nStreams);
    }
    else
    {
        // Several batches will be needed so evenly split the MaxAxesToAllocate by the number of streams
        EstimatedNumAxesPerStream = ceil((double)this->MaxAxesToAllocate / (double)this->nStreams);
    }

    while (processed_nAxes < nAxes && batch < MaxBatches)
    {
        for (int i = 0; i < this->nStreams; i++) // Loop through the streams
        {
            // If we're about to process more than the number of coordinate axes, process the remaining faction of numAxesPerStream
            if (processed_nAxes + EstimatedNumAxesPerStream >= nAxes)
            {
                // Process the remaining fraction of EstimatedNumAxesPerStream
                Offsets_obj.numAxesPerStream.push_back(std::min(EstimatedNumAxesPerStream, nAxes - processed_nAxes));
            }
            else
            {
                // Save the estimated number of axes to the numAxesPerStream
                Offsets_obj.numAxesPerStream.push_back(EstimatedNumAxesPerStream);
            }

            // Calculate the offsets (in bytes) to determine which part of the array to copy for this stream
            // When using multiple GPUs coordAxesOffset will be the number already assigned to other GPUs
            Offsets_obj.CoordAxes_CPU_Offset.push_back((processed_nAxes + coordAxesOffset) * 9); // Each axes has 9 elements (X, Y, Z)
            Offsets_obj.coord_Axes_CPU_streamBytes.push_back(Offsets_obj.numAxesPerStream.back() * 9 * sizeof(float));

            // Use the number of axes already assigned to this GPU since starting the current batch to calculate the currect offset
            Offsets_obj.gpuCASImgs_Offset.push_back(numAxesGPU_Batch * CASImgSize * CASImgSize);
            Offsets_obj.gpuImgs_Offset.push_back(numAxesGPU_Batch * ImgSize * ImgSize);
            Offsets_obj.gpuCoordAxes_Stream_Offset.push_back(numAxesGPU_Batch * 9);

            // Optionally: Copy the resulting CAS images back to the host pinned memory (CPU)
            if (this->h_CASImgs != NULL)
            {
                // Have to use unsigned long long since the array may be longer than the max value int32 can represent
                // imgSize is the size of the zero padded projection images
                unsigned long long *CASImgs_Offset = new unsigned long long[3];
                CASImgs_Offset[0] = (unsigned long long)(CASImgSize);
                CASImgs_Offset[1] = (unsigned long long)(CASImgSize);
                CASImgs_Offset[2] = (unsigned long long)(processed_nAxes + coordAxesOffset);

                Offsets_obj.CASImgs_CPU_Offset.push_back(CASImgs_Offset[0] * CASImgs_Offset[1] * CASImgs_Offset[2]);

                // How many bytes are the output images?
                Offsets_obj.gpuCASImgs_streamBytes.push_back(CASImgSize * CASImgSize * Offsets_obj.numAxesPerStream.back() * sizeof(float));
            }

            // Have to use unsigned long long since the array may be longer than the max value int32 can represent
            // imgSize is the size of the zero padded projection images
            unsigned long long *Imgs_Offset = new unsigned long long[3];
            Imgs_Offset[0] = (unsigned long long)(ImgSize);
            Imgs_Offset[1] = (unsigned long long)(ImgSize);
            Imgs_Offset[2] = (unsigned long long)(processed_nAxes + coordAxesOffset);

            Offsets_obj.Imgs_CPU_Offset.push_back(Imgs_Offset[0] * Imgs_Offset[1] * Imgs_Offset[2]);

            // How many bytes are the output images?
            Offsets_obj.gpuImgs_streamBytes.push_back(ImgSize * ImgSize * Offsets_obj.numAxesPerStream.back() * sizeof(float));

            // Update the overall number of coordinate axes which have already been assigned to a CUDA stream
            processed_nAxes = processed_nAxes + Offsets_obj.numAxesPerStream.back();

            // Update the number of axes which have been assigned to this GPU during the current batch
            numAxesGPU_Batch = numAxesGPU_Batch + Offsets_obj.numAxesPerStream.back();

            // Add one to the number of offset values
            Offsets_obj.num_offsets++;

            // Remember which stream this is
            Offsets_obj.stream_ID.push_back(i);

            // Have all the axes been processed?
            if (processed_nAxes == nAxes)
            {
                break;
            }
        }

        // Increment the batch number
        batch++;

        // Reset the number of axes processed during the current batch variable
        numAxesGPU_Batch = 0;
    }

    return Offsets_obj;
}

void gpuGridder::ForwardProject(int AxesOffset, int nAxesToProcess)
{
    // Run the forward projection on some subset of the coordinate axes (needed when using multiple GPUs)
    cudaSetDevice(this->GPU_Device);
    cudaDeviceSynchronize();

    // Have the GPU arrays been allocated?
    if (this->GPUArraysAllocatedFlag == false)
    {
        Allocate();

        this->GPUArraysAllocatedFlag = true;
    }

    // PrintMemoryAvailable();

    // Convert the volume to CAS volume if we are running the FFT on the device
    if (this->RunFFTOnDevice == true)
    {
        // Copy the volume to the corresponding GPU array
        this->d_Volume->CopyToGPU(this->h_Volume->GetPointer(), this->h_Volume->bytes());

        // Run the volume to CAS volume function
        VolumeToCASVolume();
    }
    else
    {
        // Copy the CAS volume to the corresponding GPU array
        this->d_CASVolume->CopyToGPU(this->h_CASVolume->GetPointer(), this->h_CASVolume->bytes());
    }

    // Check the error flags to see if we had any issues during the initilization
    if (this->ErrorFlag == 1 ||
        this->d_CASVolume->GetErrorFlag() == 1 ||
        this->d_CASImgs->GetErrorFlag() == 1 ||
        this->d_Imgs->GetErrorFlag() == 1 ||
        this->d_CoordAxes->GetErrorFlag() == 1 ||
        this->d_KB_Table->GetErrorFlag() == 1)
    {
        std::cerr << "Error during intilization." << '\n';
        return; // Don't run the kernel and return
    }

    // Run the forward projection CUDA kernel
    cudaSetDevice(this->GPU_Device);

    // For compactness define the CASImgSize, CASVolSize, and ImgSize here
    int ImgSize = this->d_Imgs->GetSize(0);
    int CASImgSize = this->d_CASImgs->GetSize(0);
    int CASVolSize = this->d_CASVolume->GetSize(0);

    // Plan the pointer offset values
    gpuGridder::Offsets Offsets_obj = PlanOffsetValues(AxesOffset, nAxesToProcess);

    // Calculate the block size for running the CUDA kernels
    // NOTE: gridSize times blockSize needs to equal CASimgSize
    int gridSize = 32; // 32
    int blockSize = ceil(((double)this->d_Imgs->GetSize(0) * (double)this->interpFactor) / (double)gridSize);

    for (int i = 0; i < Offsets_obj.num_offsets; i++)
    {
        if (Offsets_obj.numAxesPerStream[i] < 1)
        {
            continue;
        }

        std::cout << "GPU: " << this->GPU_Device << " forward projection stream " << Offsets_obj.stream_ID[i]
                  << " processing " << Offsets_obj.numAxesPerStream[i] << " axes " << '\n';

        // Copy the section of gpuCoordAxes which this stream will process on the current GPU
        cudaMemcpyAsync(
            this->d_CoordAxes->GetPointer(Offsets_obj.gpuCoordAxes_Stream_Offset[i]),
            this->h_CoordAxes->GetPointer(Offsets_obj.CoordAxes_CPU_Offset[i]),
            Offsets_obj.coord_Axes_CPU_streamBytes[i],
            cudaMemcpyHostToDevice, streams[Offsets_obj.stream_ID[i]]);

        // Run the forward projection kernel
        gpuForwardProject::RunKernel(
            this->d_CASVolume->GetPointer(),
            this->d_CASImgs->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
            this->d_KB_Table->GetPointer(),
            this->d_CoordAxes->GetPointer(Offsets_obj.gpuCoordAxes_Stream_Offset[i]),
            this->kerHWidth,
            Offsets_obj.numAxesPerStream[i],
            gridSize,
            blockSize,
            CASVolSize,
            CASImgSize,
            this->maskRadius,
            this->d_KB_Table->GetSize(0),
            &streams[Offsets_obj.stream_ID[i]]);

        // Optionally: Copy the resulting CAS images back to the host pinned memory (CPU)
        if (this->h_CASImgs != NULL)
        {
            cudaMemcpyAsync(
                this->h_CASImgs->GetPointer(Offsets_obj.CASImgs_CPU_Offset[i]),
                this->d_CASImgs->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
                Offsets_obj.gpuCASImgs_streamBytes[i],
                cudaMemcpyDeviceToHost,
                streams[Offsets_obj.stream_ID[i]]);
        }

        // If running the inverse FFT on the device
        if (this->RunFFTOnDevice == true)
        {
            // Convert the CAS projection images back to images using an inverse FFT and cropping out the zero padding
            this->gpuFFT_obj->CASImgsToImgs(
                streams[Offsets_obj.stream_ID[i]],
                CASImgSize,
                ImgSize,
                this->d_CASImgs->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
                this->d_Imgs->GetPointer(Offsets_obj.gpuImgs_Offset[i]),
                this->d_CASImgsComplex->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
                Offsets_obj.numAxesPerStream[i]);

            // Lastly, copy the resulting cropped projection images back to the host pinned memory (CPU)
            cudaMemcpyAsync(
                this->h_Imgs->GetPointer(Offsets_obj.Imgs_CPU_Offset[i]),
                this->d_Imgs->GetPointer(Offsets_obj.gpuImgs_Offset[i]),
                Offsets_obj.gpuImgs_streamBytes[i],
                cudaMemcpyDeviceToHost,
                streams[Offsets_obj.stream_ID[i]]);
        }
    }

    cudaDeviceSynchronize();
}

void gpuGridder::BackProject(int AxesOffset, int nAxesToProcess)
{
    // Run the forward projection on some subset of the coordinate axes (needed when using multiple GPUs)
    cudaSetDevice(this->GPU_Device);
    cudaDeviceSynchronize();

    // For compactness define the CASImgSize, CASVolSize, and ImgSize here
    int ImgSize = this->d_Imgs->GetSize(0);
    int CASImgSize = this->d_CASImgs->GetSize(0);
    int CASVolSize = this->d_CASVolume->GetSize(0);

    // Plan the pointer offset values
    gpuGridder::Offsets Offsets_obj = PlanOffsetValues(AxesOffset, nAxesToProcess);

    // cufftComplex *test = new cufftComplex[this->d_CASImgsComplex->length()];

    // for (int i = 0; i < this->d_CASImgsComplex->length(); i++)
    // {
    //     test[i].x=0;
    //     test[i].y=0;
    // }

    // this->d_CASImgsComplex->CopyToGPU(test, this->d_CASImgsComplex->bytes());

    this->d_CASVolume->Reset();
    this->d_CASImgs->Reset();
    // this->d_Imgs->Reset();
    // this->d_CoordAxes->Reset();
    // this->d_PlaneDensity->Reset();

    // cudaDeviceSynchronize();

    // Copy the CAS volume to the corresponding GPU array
    // this->d_CASVolume->CopyToGPU(this->h_CASVolume->GetPointer(), this->h_CASVolume->bytes());

    // Define CUDA kernel dimensions
    int gridSize = ceil(this->d_CASVolume->GetSize(0) / 4);
    int blockSize = 4;

    for (int i = 0; i < Offsets_obj.num_offsets; i++)
    {
        if (Offsets_obj.numAxesPerStream[i] < 1)
        {
            continue;
        }

        // std::cout << "GPU: " << this->GPU_Device << " back projection stream " << Offsets_obj.stream_ID[i]
        //           << " processing " << Offsets_obj.numAxesPerStream[i] << " axes " << '\n';

        // Copy the section of gpuCoordAxes which this stream will process on the current GPU
        cudaMemcpyAsync(
            this->d_CoordAxes->GetPointer(Offsets_obj.gpuCoordAxes_Stream_Offset[i]),
            this->h_CoordAxes->GetPointer(Offsets_obj.CoordAxes_CPU_Offset[i]),
            Offsets_obj.coord_Axes_CPU_streamBytes[i],
            cudaMemcpyHostToDevice, streams[Offsets_obj.stream_ID[i]]);

        if (this->RunFFTOnDevice == false)
        {
            // Copy the CASImages from the pinned CPU array and use instead of the images array
            cudaMemcpyAsync(
                this->d_CASImgs->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
                this->h_CASImgs->GetPointer(Offsets_obj.CASImgs_CPU_Offset[i]),
                Offsets_obj.gpuCASImgs_streamBytes[i],
                cudaMemcpyHostToDevice,
                streams[Offsets_obj.stream_ID[i]]);
        }
        else
        {
            // Copy the section of images to the GPU
            cudaMemcpyAsync(
                this->d_Imgs->GetPointer(Offsets_obj.gpuImgs_Offset[i]),
                this->h_Imgs->GetPointer(Offsets_obj.Imgs_CPU_Offset[i]),
                Offsets_obj.gpuImgs_streamBytes[i],
                cudaMemcpyHostToDevice,
                streams[Offsets_obj.stream_ID[i]]);

            // Run the forward FFT to convert the pinned CPU images to CAS images (CAS type is needed for back projecting into the CAS volume)
            this->ImgsToCASImgs(
                streams[Offsets_obj.stream_ID[i]],
                this->d_CASImgs->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
                this->d_Imgs->GetPointer(Offsets_obj.gpuImgs_Offset[i]),
                this->d_CASImgsComplex->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
                Offsets_obj.numAxesPerStream[i]);
        }

        // Run the back projection kernel
        gpuBackProject::RunKernel(
            this->d_CASVolume->GetPointer(),
            this->d_CASImgs->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
            this->d_KB_Table->GetPointer(),
            this->d_CoordAxes->GetPointer(Offsets_obj.gpuCoordAxes_Stream_Offset[i]),
            this->d_PlaneDensity->GetPointer(),
            this->kerHWidth,
            Offsets_obj.numAxesPerStream[i],
            gridSize,
            blockSize,
            CASVolSize,
            CASImgSize,
            this->maskRadius,
            this->d_KB_Table->GetSize(0),
            &streams[Offsets_obj.stream_ID[i]]);
    }

    cudaDeviceSynchronize();
}

void gpuGridder::ImgsToCASImgs(cudaStream_t &stream, float *CASImgs, float *Imgs, cufftComplex *CASImgsComplex, int numImgs)
{
    // Convert projection images to CAS images by running a forward FFT
    // CASImgs, Imgs, and CASImgsComplex, are the device allocated arrays (e.g. d_CASImgs) at some offset from the beginning of the array

    int CASImgSize = this->d_CASImgs->GetSize(0);
    int ImgSize = this->d_Imgs->GetSize(0);

    // Has the forward FFT been planned? If not create one now
    if (this->forwardFFTPlannedFlag == false)
    {
        std::cout << "gpuFFT::ImgsToCASImgs(): Planning forward FFT" << '\n';
        cufftPlan2d(&this->forwardFFTPlan, CASImgSize, CASImgSize, CUFFT_C2C);

        this->forwardFFTPlannedFlag = true;
    }

    // First pad the Imgs with zeros to be the same size as CASImgs
    PadVolumeFilter *PadFilter = new PadVolumeFilter();
    PadFilter->SetInput(Imgs);
    PadFilter->SetOutput(CASImgs);
    PadFilter->SetInputSize(ImgSize);
    PadFilter->SetPaddingX((CASImgSize - ImgSize) / 2);
    PadFilter->SetPaddingY((CASImgSize - ImgSize) / 2);
    PadFilter->SetPaddingZ(0);
    PadFilter->SetNumberOfSlices(numImgs);
    PadFilter->Update(&stream);

    // Convert the images to complex cufft type
    RealToComplexFilter *RealFilter = new RealToComplexFilter();
    RealFilter->SetRealInput(CASImgs);
    RealFilter->SetComplexOutput(CASImgsComplex);
    RealFilter->SetVolumeSize(CASImgSize);
    RealFilter->SetNumberOfSlices(numImgs);
    RealFilter->Update(&stream);

    // Run FFTShift on each 2D slice
    FFTShift2DFilter *FFTShiftFilter = new FFTShift2DFilter();
    FFTShiftFilter->SetInput(CASImgsComplex);
    FFTShiftFilter->SetImageSize(CASImgSize);
    FFTShiftFilter->SetNumberOfSlices(numImgs);
    FFTShiftFilter->Update(&stream);

    // Set the FFT plan to the current stream to process
    cufftSetStream(this->forwardFFTPlan, stream);

    // Execute the forward FFT on each 2D array
    // cufftPlanMany is not feasible since the number of images changes and
    // cufftDestroy is blocks the CPU and causes memory leaks if not called
    // FFT on each 2D slice has similar computation speed as cufftPlanMany
    for (int i = 0; i < numImgs; i++)
    {
        cufftExecC2C(this->forwardFFTPlan,
                     (cufftComplex *)&CASImgsComplex[i * CASImgSize * CASImgSize],
                     (cufftComplex *)&CASImgsComplex[i * CASImgSize * CASImgSize],
                     CUFFT_FORWARD);
    }

    // Run the 2D FFTShift again
    FFTShiftFilter->SetInput(CASImgsComplex);
    FFTShiftFilter->SetImageSize(CASImgSize);
    FFTShiftFilter->SetNumberOfSlices(numImgs);
    FFTShiftFilter->Update(&stream);

    // Convert the complex result of the forward FFT to a CAS img type
    ComplexToCASFilter *ComplexToCAS = new ComplexToCASFilter();
    ComplexToCAS->SetComplexInput(CASImgsComplex);
    ComplexToCAS->SetCASVolumeOutput(CASImgs);
    ComplexToCAS->SetVolumeSize(CASImgSize);
    ComplexToCAS->SetNumberOfSlices(numImgs);
    ComplexToCAS->Update(&stream);

    return;
}

void gpuGridder::FreeMemory()
{
    // Free all of the allocated memory
    cudaSetDevice(this->GPU_Device);
    cudaDeviceReset(); // This deletes the CUDA context (i.e. deallocates all memory)
}

void gpuGridder::ReconstructVolume()
{
    // Convert the CAS volume on this GPU to to volume

    std::cout << "gpuGridder::ReconstructVolume()" << '\n';

    int CASVolumeSize = this->d_CASVolume->GetSize(0);
    int CASImgSize = this->d_CASImgs->GetSize(0);
    int ImgSize = this->d_Imgs->GetSize(0);

    // First, copy the Kaiser Bessel precompensation filter to the GPU
    // Size is volume times interp factor
    int *KB_PreComp_size = new int[3];
    KB_PreComp_size[0] = this->h_Volume->GetSize(0) * this->interpFactor;
    KB_PreComp_size[1] = this->h_Volume->GetSize(1) * this->interpFactor;
    KB_PreComp_size[2] = this->h_Volume->GetSize(2) * this->interpFactor;

    cudaDeviceSynchronize(); // test

    // MemoryStructGPU<float> *d_KBPreComp = new MemoryStructGPU<float>(this->d_CASVolume->GetDim(), KB_PreComp_size, this->GPU_Device);
    // d_KBPreComp->AllocateGPUArray();

    std::cout << "   d_KBPreComp->CopyToGPU(this->h_KBPreComp->GetPointer(), d_KBPreComp->bytes());" << '\n';

    // d_KBPreComp->CopyToGPU(this->h_KBPreComp->GetPointer(), d_KBPreComp->bytes());

    cudaDeviceSynchronize(); // test

    delete[] KB_PreComp_size;
    // Convert a GPU CAS volume to volume
    // Note: The volume must be square (i.e. have the same dimensions for the X, Y, and Z)
    // Step 1: Pad the input volume with zeros and convert to cufftComplex type
    // Step 2: fftshift
    // Step 3: Take discrete Fourier transform using cuFFT
    // Step 4: fftshift
    // Step 5: Convert to CAS volume using CUDA kernel
    // Step 6: Apply extra zero padding

    // Note: d_CAS_Vol has size of d_Vol * interpFactor + extraPadding * 2
    // Note: d_CASVol_Interp has size of d_Vol * interpFactor

    int CroppedCASVolumeSize = CASVolumeSize - extraPadding * 2;
    int VolumeSize = (CASVolumeSize - extraPadding * 2) / interpFactor;

    if (VolumeSize <= 0)
    {
        std::cerr << "Error GPU_CASVolumeToVolume(): VolumeSize must be a positive integer value. " << VolumeSize << " was calculated." << '\n';
        return;
    }

    std::cout << "GPU_CASVolumeToVolume: " << '\n';
    std::cout << "VolumeSize: " << VolumeSize << '\n';
    std::cout << "CASVolumeSize: " << CASVolumeSize << '\n';
    std::cout << "CroppedCASVolumeSize: " << CroppedCASVolumeSize << '\n';
    std::cout << "interpFactor: " << interpFactor << '\n';
    std::cout << "extraPadding: " << extraPadding << '\n';

    // Allocate GPU memory for CAS volume without the extra padding
    float *d_CASVolume_Cropped;
    cudaMalloc(&d_CASVolume_Cropped, sizeof(float) * CroppedCASVolumeSize * CroppedCASVolumeSize * CroppedCASVolumeSize);

    // Allocate GPU memory for cufftComplex type of the cropped CAS volume (i.e. d_CASVolume_Cropped)
    cufftComplex *d_CASVolume_Cropped_Complex;
    cudaMalloc(&d_CASVolume_Cropped_Complex, sizeof(cufftComplex) * CroppedCASVolumeSize * CroppedCASVolumeSize * CroppedCASVolumeSize);

    // Reset the volume to all zeros
    // cudaMemset(d_Volume, 0, sizeof(float) * VolumeSize * VolumeSize * VolumeSize);
    // cudaMemset(d_CASVolume_Cropped_Complex, 0, sizeof(cufftComplex) * CroppedCASVolumeSize * CroppedCASVolumeSize * CroppedCASVolumeSize);
    // cudaMemset(d_CASVolume_Cropped, 0, sizeof(float) * CroppedCASVolumeSize * CroppedCASVolumeSize * CroppedCASVolumeSize);

    // cudaDeviceSynchronize(); // test

    // Divide the CASVolume by the plane density to normalize
    // gpuFFT::DivideVolumes(d_CAS_Volume, d_PlaneDensity, CASVolumeSize);

    DivideVolumeFilter *DivideFilter = new DivideVolumeFilter();
    DivideFilter->SetVolumeSize(CASVolumeSize);
    // DivideFilter->SetNumberOfSlices(CASVolumeSize);
    DivideFilter->SetVolumeOne(this->d_CASVolume->GetPointer());
    DivideFilter->SetVolumeTwo(this->d_PlaneDensity->GetPointer());
    DivideFilter->Update();

    // gpuFFT::CropVolume(this->d_CASVolume->GetPointer(), d_CASVolume_Cropped, CASVolumeSize, CroppedCASVolumeSize);

    // Remove the extraPadding from the CAS volume
    CropVolumeFilter *CropFilter = new CropVolumeFilter();
    CropFilter->SetInput(this->d_CASVolume->GetPointer());
    CropFilter->SetInputSize(CASVolumeSize);
    CropFilter->SetOutput(d_CASVolume_Cropped);
    CropFilter->SetCropX((CASVolumeSize - CroppedCASVolumeSize) / 2);
    CropFilter->SetCropY((CASVolumeSize - CroppedCASVolumeSize) / 2);
    CropFilter->SetCropZ((CASVolumeSize - CroppedCASVolumeSize) / 2);
    CropFilter->SetNumberOfSlices(CroppedCASVolumeSize);
    CropFilter->Update();

    // TEST: Apply a spherical mask to set all area outside the radius to be zero
    // gpuFFT::MaskVolume(d_CASVolume_Cropped, CroppedCASVolumeSize, maskRadius);

    // Convert the CASImgs to complex cufft type
    // gpuFFT::CASImgsToComplex(d_CASVolume_Cropped, d_CASVolume_Cropped_Complex, CroppedCASVolumeSize, CroppedCASVolumeSize);
    // gpuFFT::CASVolumeToComplex(d_CASVolume_Cropped, d_CASVolume_Cropped_Complex, CroppedCASVolumeSize);

    CASToComplexFilter *CASFilter = new CASToComplexFilter();
    CASFilter->SetCASVolume(d_CASVolume_Cropped);
    CASFilter->SetComplexOutput(d_CASVolume_Cropped_Complex);
    CASFilter->SetVolumeSize(CroppedCASVolumeSize);
    // CASFilter->SetNumberOfSlices(CroppedCASVolumeSize);
    CASFilter->Update();

    // Run FFTShift on the 3D volume
    // gpuFFT::cufftShift_3D(d_CASVolume_Cropped_Complex, CroppedCASVolumeSize, CroppedCASVolumeSize);
    FFTShift3DFilter *FFTShiftFilter = new FFTShift3DFilter();
    FFTShiftFilter->SetInput(d_CASVolume_Cropped_Complex);
    FFTShiftFilter->SetVolumeSize(CroppedCASVolumeSize);
    FFTShiftFilter->Update();

    //  Execute the forward FFT on the 3D array
    // gpuFFT::InverseFFT(d_CASVolume_Cropped_Complex, CroppedCASVolumeSize);

    cudaDeviceSynchronize();

    // Plan the inverse FFT
    cufftHandle inverseFFTPlan;
    cufftPlan3d(&inverseFFTPlan, CroppedCASVolumeSize, CroppedCASVolumeSize, CroppedCASVolumeSize, CUFFT_C2C);
    cufftExecC2C(inverseFFTPlan, (cufftComplex *)d_CASVolume_Cropped_Complex, (cufftComplex *)d_CASVolume_Cropped_Complex, CUFFT_INVERSE);

    cudaDeviceSynchronize();

    // Free the plan
    cufftDestroy(inverseFFTPlan);

    // Apply a second in place 3D FFT Shift
    // gpuFFT::cufftShift_3D(d_CASVolume_Cropped_Complex, CroppedCASVolumeSize, CroppedCASVolumeSize);
    FFTShiftFilter->SetInput(d_CASVolume_Cropped_Complex);
    FFTShiftFilter->SetVolumeSize(CroppedCASVolumeSize);
    FFTShiftFilter->Update();

    // // Multiply by the Kaiser Bessel precompensation array
    // // gpuFFT::MultiplyVolumes(d_CASVolume_Cropped_Complex, d_KBPreComp, CroppedCASVolumeSize);
    // MultiplyVolumeFilter *MultiplyFilter = new MultiplyVolumeFilter();
    // MultiplyFilter->SetVolumeSize(CroppedCASVolumeSize);
    // // DivideFilter->SetNumberOfSlices(CASVolumeSize);
    // MultiplyFilter->SetVolumeOne(d_CASVolume_Cropped_Complex);
    // MultiplyFilter->SetVolumeTwo(d_KBPreComp->GetPointer());
    // MultiplyFilter->Update();

    // Run kernel to crop the d_CASVolume_Cropped_Complex (to remove the zero padding), extract the real value,
    // and normalize the scaling introduced during the FFT
    int normalizationFactor = CroppedCASVolumeSize * CroppedCASVolumeSize * CroppedCASVolumeSize;
    // gpuFFT::ComplexToCroppedNormalizedVol(d_CASVolume_Cropped_Complex, d_Volume->GetPointer(), CroppedCASVolumeSize, VolumeSize, normalizationFactor);

    ComplexToRealFilter *ComplexToReal = new ComplexToRealFilter();
    ComplexToReal->SetComplexInput(d_CASVolume_Cropped_Complex);
    ComplexToReal->SetRealOutput(d_CASVolume_Cropped);
    ComplexToReal->SetVolumeSize(CroppedCASVolumeSize);
    ComplexToReal->Update();

    CropFilter->SetInput(d_CASVolume_Cropped);
    CropFilter->SetInputSize(CroppedCASVolumeSize);
    CropFilter->SetOutput(this->d_Volume->GetPointer());
    CropFilter->SetCropX((CroppedCASVolumeSize - VolumeSize) / 2);
    CropFilter->SetCropY((CroppedCASVolumeSize - VolumeSize) / 2);
    CropFilter->SetCropZ((CroppedCASVolumeSize - VolumeSize) / 2);
    // CropFilter->SetNumberOfSlices(CroppedCASVolumeSize);

    // CropFilter->SetInputSize(CroppedCASVolumeSize);
    // CropFilter->SetOutputSize(VolumeSize);
    CropFilter->Update();

    DivideScalarFilter *DivideScalar = new DivideScalarFilter();
    DivideScalar->SetInput(this->d_Volume->GetPointer());
    DivideScalar->SetScalar(float(normalizationFactor));
    DivideScalar->SetVolumeSize(VolumeSize);
    DivideScalar->Update();

    // Lastly, apply a spherical mask to set all area outside the radius to be zero
    // gpuFFT::MaskVolume(d_Volume, VolumeSize, maskRadius);

    cudaDeviceSynchronize(); // test

    // Free the temporary variables
    cudaFree(d_CASVolume_Cropped);
    cudaFree(d_CASVolume_Cropped_Complex);
}