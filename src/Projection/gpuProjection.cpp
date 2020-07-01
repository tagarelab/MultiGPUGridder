#include "gpuProjection.h"

void gpuProjection::Allocate()
{
    // Allocate the needed GPU memory
    // Have the GPU arrays already been allocated?
    if (this->GPUArraysAllocatedFlag == false)
    {
        // Estimate the maximum number of coordinate axes to allocate per stream
        this->MaxAxesToAllocate = EstimateMaxAxesToAllocate(this->VolumeSize, this->interpFactor);

        // Initialize the needed arrays on the GPU
        InitializeGPUArrays();

        // Initialize the CUDA streams
        InitializeCUDAStreams();

        this->GPUArraysAllocatedFlag = true;
    }
}

int gpuProjection::EstimateMaxAxesToAllocate(int VolumeSize, int interpFactor)
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
    int Bytes_for_Volume = pow(VolumeSize, 3) * sizeof(float);
    int Bytes_for_Padded_Volume = pow((VolumeSize * interpFactor), 3) * sizeof(float);
    int Bytes_for_Padded_Volume_Complex = Bytes_for_Padded_Volume * 2;
    int Bytes_for_CASVolume = pow((VolumeSize * interpFactor + this->extraPadding * 2), 3) * sizeof(float);
    int Bytes_for_Plane_Density = Bytes_for_CASVolume;
    int Bytes_for_CoordAxes = 9 * sizeof(float); // 9 elements per axes

    // How many coordinate axes would fit in the remaining free GPU memory?

    int EstimatedMaxAxes;

    if (this->RunFFTOnDevice == 1)
    {
        // If running the FFT on the device we need to allocate intermediate arrays
        EstimatedMaxAxes =
            (mem_free - Bytes_for_Volume - Bytes_for_Padded_Volume - Bytes_for_Padded_Volume_Complex - Bytes_for_Plane_Density) / (Bytes_per_Img + Bytes_per_CASImg + Bytes_per_ComplexCASImg + Bytes_for_CoordAxes);
    }
    else
    {
        EstimatedMaxAxes = (mem_free - Bytes_for_Volume - Bytes_for_CASVolume - Bytes_for_Plane_Density) / (Bytes_per_Img + Bytes_per_CASImg + Bytes_for_CoordAxes);
    }

    // Leave room on the GPU to run the FFTs and CUDA kernels so only use 30% of the maximum possible
    EstimatedMaxAxes = floor(EstimatedMaxAxes * 0.3);

    // Set a maximum value for this or else for small volume size this because too large and can cause errors
    EstimatedMaxAxes = std::min(EstimatedMaxAxes, 5000);

    if (this->verbose == true)
    {
        std::cout << "gpuProjection::EstimateMaxAxesToAllocate() on GPU " << this->GPU_Device << " estimated maximum axes to allocate = " << EstimatedMaxAxes << '\n';
    }

    return EstimatedMaxAxes;
}

void gpuProjection::InitializeCUDAStreams()
{
    cudaSetDevice(this->GPU_Device);

    // Create the CUDA streams
    this->FP_streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * this->nStreamsFP);

    for (int i = 0; i < this->nStreamsFP; i++) // Loop through the streams
    {
        gpuErrorCheck(cudaStreamCreate(&this->FP_streams[i]));
    }

    // Create the CUDA streams for the back projection
    this->BP_streams = (cudaStream_t *)malloc(sizeof(cudaStream_t) * this->nStreamsBP);

    for (int i = 0; i < this->nStreamsBP; i++) // Loop through the streams
    {
        gpuErrorCheck(cudaStreamCreate(&this->BP_streams[i]));
    }
}

void gpuProjection::InitializeGPUArrays()
{
    // Initialize the GPU arrays and allocate the needed memory on the GPU

    if (this->verbose == true)
    {
        std::cout << "gpuProjection::InitializeGPUArrays() on GPU " << '\n';
        PrintMemoryAvailable();
    }

    cudaSetDevice(this->GPU_Device);
    if (this->MaxAxesToAllocate == 0)
    {
        std::cerr << "MaxAxesToAllocate must be a positive integer. Please run EstimateMaxAxesToAllocate() first. " << '\n';
        this->ErrorFlag = 1;
        return;
    }

    int PaddedVolumeSize = this->VolumeSize * this->interpFactor;
    int CASVolumeSize = this->VolumeSize * this->interpFactor + this->extraPadding * 2;

    // First, create a dims array of the correct size of d_CASImgs
    // Limit the number of axes to allocate to be MaxAxesToAllocate
    int *imgs_size = new int[3];
    imgs_size[0] = this->VolumeSize;
    imgs_size[1] = this->VolumeSize;
    imgs_size[2] = std::min(this->numCoordAxes, this->MaxAxesToAllocate);

    int *CASimgs_size = new int[3];
    CASimgs_size[0] = PaddedVolumeSize;
    CASimgs_size[1] = PaddedVolumeSize;
    CASimgs_size[2] = std::min(this->numCoordAxes, this->MaxAxesToAllocate);

    int *axes_size = new int[1];
    axes_size[0] = std::min(this->numCoordAxes, this->MaxAxesToAllocate);
    axes_size[0] = axes_size[0] * 9; // 9 elements per coordinate axes

    // If running the FFT on the device allocate the required intermediate array
    if (this->RunFFTOnDevice == 1)
    {
        // Allocate padded volume (used for converting the volume to CAS volume)
        this->d_PaddedVolume = new DeviceMemory<float>(3, PaddedVolumeSize, PaddedVolumeSize, PaddedVolumeSize, this->GPU_Device);
        this->d_PaddedVolume->AllocateGPUArray();

        //      int CASVolumeSize = this->d_CASVolume->GetSize(0);
        // int CASImgSize = this->d_CASImgs->GetSize(0);
        // int ImgSize = this->d_Imgs->GetSize(0);

        // First, copy the Kaiser Bessel precompensation filter to the GPU
        // Size is volume times interp factor
        int *KB_PreComp_size = new int[3];
        KB_PreComp_size[0] = this->VolumeSize * this->interpFactor;
        KB_PreComp_size[1] = this->VolumeSize * this->interpFactor;
        KB_PreComp_size[2] = this->VolumeSize * this->interpFactor;

        // float *d_KBPreComp;
        // gpuErrorCheck(cudaMalloc(&d_KBPreComp, sizeof(float) * KB_PreComp_size[0] * KB_PreComp_size[1] * KB_PreComp_size[2]));
        // gpuErrorCheck(cudaMemcpy(d_KBPreComp, this->h_KBPreComp->GetPointer(), sizeof(float) * KB_PreComp_size[0] * KB_PreComp_size[1] * KB_PreComp_size[2], cudaMemcpyHostToDevice));

        //  float *d_KBPreComp;
        // gpuErrorCheck(cudaMalloc(&d_KBPreComp, sizeof(float) * KB_PreComp_size[0] * KB_PreComp_size[1] * KB_PreComp_size[2]));

        this->d_KBPreComp = new DeviceMemory<float>(3, KB_PreComp_size, this->GPU_Device);
        this->d_KBPreComp->AllocateGPUArray();

        // delete[] KB_PreComp_size;

        int CroppedCASVolumeSize = CASVolumeSize - extraPadding * 2;
        // int VolumeSize = (CASVolumeSize - extraPadding * 2) / interpFactor;

        // Allocate GPU memory for CAS volume without the extra padding
        // float *d_CASVolume_Cropped;
        // gpuErrorCheck(cudaMalloc(&d_CASVolume_Cropped, sizeof(float) * CroppedCASVolumeSize * CroppedCASVolumeSize * CroppedCASVolumeSize));

        this->d_CASVolume_Cropped = new DeviceMemory<float>(3, CroppedCASVolumeSize, CroppedCASVolumeSize, CroppedCASVolumeSize, this->GPU_Device);
        this->d_CASVolume_Cropped->AllocateGPUArray();

        // Allocate a complex version of the padded volume (needed for the forward and inverse FFT)
        this->d_PaddedVolumeComplex = new DeviceMemory<cufftComplex>(3, PaddedVolumeSize, PaddedVolumeSize, PaddedVolumeSize, this->GPU_Device);
        this->d_PaddedVolumeComplex->AllocateGPUArray();

        this->d_CASVolume_Cropped_Complex = new DeviceMemory<cufftComplex>(3, CroppedCASVolumeSize, CroppedCASVolumeSize, CroppedCASVolumeSize, this->GPU_Device);
        this->d_CASVolume_Cropped_Complex->AllocateGPUArray();

        //cufftComplex *d_CASVolume_Cropped_Complex;
        //gpuErrorCheck(cudaMalloc(&d_CASVolume_Cropped_Complex, sizeof(cufftComplex) * CroppedCASVolumeSize * CroppedCASVolumeSize * CroppedCASVolumeSize));

        // // Allocate the complex CAS images array
        // DeviceMemory<cufftComplex> * CASImgsComplex = new DeviceMemory<cufftComplex>(3, this->d_CASImgs->GetSize(0), this->d_CASImgs->GetSize(1), numImgs, this->GPU_Device);
        // CASImgsComplex->AllocateGPUArray();

        // Allocate the complex CAS images array
        this->d_CASImgsComplex = new DeviceMemory<cufftComplex>(3, CASimgs_size, this->GPU_Device);
        this->d_CASImgsComplex->AllocateGPUArray();
    }

    // Allocate the CAS images
    this->d_CASImgs = new DeviceMemory<float>(3, CASimgs_size, this->GPU_Device);
    this->d_CASImgs->AllocateGPUArray();

    // Allocate the images
    this->d_Imgs = new DeviceMemory<float>(3, imgs_size, this->GPU_Device);
    this->d_Imgs->AllocateGPUArray();

    // Allocate the coordinate axes array
    this->d_CoordAxes = new DeviceMemory<float>(1, this->numCoordAxes * 9, this->GPU_Device);
    this->d_CoordAxes->AllocateGPUArray();

    // Allocate the Kaiser bessel lookup table
    this->d_KB_Table = new DeviceMemory<float>(1, this->KB_Table_Size, this->GPU_Device);
    this->d_KB_Table->AllocateGPUArray();

    delete[] CASimgs_size;
    delete[] imgs_size;
    delete[] axes_size;
}

gpuProjection::Offsets gpuProjection::PlanOffsetValues(int coordAxesOffset, int nAxes, int numStreams)
{
    // Loop through all of the coordinate axes and calculate the corresponding pointer offset values
    // which are needed for running the CUDA kernels

    if (this->verbose == true)
    {
        std::cout << "gpuProjection::PlanOffsetValues()" << '\n';
    }

    // For compactness define the CASImgSize, CASVolSize, and ImgSize here
    int CASImgSize = this->d_CASImgs->GetSize(0);
    int CASVolSize = this->d_CASVolume->GetSize(0);
    int ImgSize = this->d_Imgs->GetSize(0);

    // Create an instance of the Offsets struct
    gpuProjection::Offsets Offsets_obj;
    Offsets_obj.num_offsets = 0;

    // Cumulative number of axes which have already been assigned to a CUDA stream
    int processed_nAxes = 0;

    // While we have coordinate axes to process, loop through the GPUs and the streams
    int MaxBatches = 500; // Maximum iterations in case we get stuck in the while loop for some reason
    int batch = 0;

    // Initialize a variable which will remember how many axes have been assigned to the GPU
    // during the current batch. This is needed for calculating the offset for the stream when
    // the number of streams is greater than the number of GPUs. This resets to zeros after each
    // batch since the same GPU memory is used for the next batch.
    int numAxesGPU_Batch = 0;

    // Estimate how many coordinate axes to assign to each stream
    int EstimatedNumAxesPerStream;
    if (nAxes <= this->MaxAxesToAllocate)
    {
        // The number of coordinate axes is less than or equal to the total number of axes to process
        EstimatedNumAxesPerStream = ceil((double)nAxes / (double)numStreams);
    }
    else
    {
        // Several batches will be needed so evenly split the MaxAxesToAllocate by the number of streams
        EstimatedNumAxesPerStream = ceil((double)this->MaxAxesToAllocate / (double)numStreams);
    }

    if (this->verbose == true)
    {
        std::cout << "gpuProjection::PlanOffsetValues() "
                  << "estimated number of axes to process on each stream = " << EstimatedNumAxesPerStream << '\n';

        std::cout << "gpuProjection::PlanOffsetValues() "
                  << " number of axes to process on this GPU  " << nAxes << '\n';

        std::cout << "gpuProjection::PlanOffsetValues() "
                  << "number of streams for this GPU " << numStreams << '\n';
    }

    while (processed_nAxes < nAxes && batch < MaxBatches)
    {
        for (int i = 0; i < numStreams; i++) // Loop through the streams
        {
            // Make sure we have enough memory allocated to process this stream on the current batch
            if (numAxesGPU_Batch + EstimatedNumAxesPerStream > this->MaxAxesToAllocate)
            {
                continue;
            }

std::cout << "processed_nAxes: " << processed_nAxes << '\n';
std::cout << "nAxes: " << nAxes << '\n';

            // Have all the axes been processed?
            if (processed_nAxes >= nAxes)
            {
                Offsets_obj.numAxesPerStream.push_back(0);

                std::cout << "Offsets_obj.numAxesPerStream.push_back(0)" << '\n';
                continue;
            }
            
            // Save the current batch number
            Offsets_obj.currBatch.push_back(batch);

            if (batch > Offsets_obj.Batches)
            {
                Offsets_obj.Batches = batch;
            }

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
            // Have to use unsigned long long since the array may be longer than the max value int32 can represent
            // imgSize is the size of the zero padded projection images
            unsigned long long *CASImgs_Offset = new unsigned long long[3];
            CASImgs_Offset[0] = (unsigned long long)(CASImgSize);
            CASImgs_Offset[1] = (unsigned long long)(CASImgSize);
            CASImgs_Offset[2] = (unsigned long long)(processed_nAxes + coordAxesOffset);

            Offsets_obj.CASImgs_CPU_Offset.push_back(CASImgs_Offset[0] * CASImgs_Offset[1] * CASImgs_Offset[2]);

            // How many bytes are the output images?
            Offsets_obj.gpuCASImgs_streamBytes.push_back(CASImgSize * CASImgSize * Offsets_obj.numAxesPerStream.back() * sizeof(float));

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
        }

        // Increment the batch number
        batch++;

        // Reset the number of axes processed during the current batch variable
        numAxesGPU_Batch = 0;
    }

    return Offsets_obj;
}

void gpuProjection::VolumeToCASVolume()
{
    // Convert a GPU volume to CAS volume
    // Note: The volume must be square (i.e. have the same dimensions for the X, Y, and Z)
    // Step 1: Pad the input volume with zeros and convert to cufftComplex type
    // Step 2: fftshift
    // Step 3: Take discrete Fourier transform using cuFFT
    // Step 4: fftshift
    // Step 5: Convert to CAS volume using CUDA kernel
    // Step 6: Apply extra zero padding

    cudaSetDevice(this->GPU_Device);

    if (this->verbose == true)
    {
        std::cout << "gpuProjection::VolumeToCASVolume() on GPU " << '\n';
        PrintMemoryAvailable();
    }

    // Example: input size = 128; interpFactor = 2 -> paddedVolSize = 256
    int PaddedVolumeSize = this->VolumeSize * this->interpFactor;

    // Example: input size = 128; interpFactor = 2; extra padding = 3; -> paddedVolSize_Extra = 262
    int PaddedVolumeSize_Extra = PaddedVolumeSize + this->extraPadding * 2;

    this->d_Volume->Reset();
    this->d_CASVolume->Reset();
    this->d_PaddedVolume->Reset();

    // First, copy the Kaiser Bessel precompensation filter to the GPU
    this->d_KBPreComp->CopyToGPU(this->h_KBPreComp->GetPointer());

    // Copy the volume to the corresponding GPU array
    this->d_Volume->CopyToGPU(this->h_Volume->GetPointer(), this->h_Volume->bytes());

    // STEP 1: Pad the input volume with zeros
    std::unique_ptr<PadVolumeFilter> PadFilter(new PadVolumeFilter());
    PadFilter->SetInput(this->d_Volume->GetPointer());
    PadFilter->SetOutput(this->d_PaddedVolume->GetPointer());
    PadFilter->SetInputSize(this->VolumeSize);
    PadFilter->SetPaddingX((PaddedVolumeSize - this->VolumeSize) / 2);
    PadFilter->SetPaddingY((PaddedVolumeSize - this->VolumeSize) / 2);
    PadFilter->SetPaddingZ((PaddedVolumeSize - this->VolumeSize) / 2);
    PadFilter->Update();

    // Multiply by the Kaiser Bessel precompensation array
    std::unique_ptr<MultiplyVolumeFilter<float>> MultiplyFilter(new MultiplyVolumeFilter<float>());
    MultiplyFilter->SetVolumeSize(this->d_PaddedVolume->GetSize(1));
    MultiplyFilter->SetVolumeOne(this->d_PaddedVolume->GetPointer());
    MultiplyFilter->SetVolumeTwo(this->d_KBPreComp->GetPointer());
    MultiplyFilter->Update();

    // Convert the padded volume to complex type (need cufftComplex type for the forward FFT)
    std::unique_ptr<RealToComplexFilter> RealToComplex(new RealToComplexFilter());
    RealToComplex->SetRealInput(this->d_PaddedVolume->GetPointer());
    RealToComplex->SetComplexOutput(this->d_PaddedVolumeComplex->GetPointer());
    RealToComplex->SetVolumeSize(PaddedVolumeSize);
    RealToComplex->Update();

    // STEP 2: Apply an in place 3D FFT Shift
    std::unique_ptr<FFTShift3DFilter<cufftComplex>> FFTShiftFilter(new FFTShift3DFilter<cufftComplex>());
    FFTShiftFilter->SetInput(this->d_PaddedVolumeComplex->GetPointer());
    FFTShiftFilter->SetVolumeSize(PaddedVolumeSize);
    FFTShiftFilter->Update();

    // STEP 3: Execute the forward FFT on the 3D array
    // Plan the forward FFT if there is one not available
    if (this->forwardFFTVolumePlannedFlag == false)
    {
        cufftPlan3d(&this->forwardFFTVolume, PaddedVolumeSize, PaddedVolumeSize, PaddedVolumeSize, CUFFT_C2C);

        this->forwardFFTVolumePlannedFlag = true;
    }

    cufftExecC2C(
        this->forwardFFTVolume,
        (cufftComplex *)this->d_PaddedVolumeComplex->GetPointer(),
        (cufftComplex *)this->d_PaddedVolumeComplex->GetPointer(),
        CUFFT_FORWARD);

    // STEP 4: Apply a second in place 3D FFT Shift
    FFTShiftFilter->SetInput(this->d_PaddedVolumeComplex->GetPointer());
    FFTShiftFilter->SetVolumeSize(PaddedVolumeSize);
    FFTShiftFilter->Update();

    // STEP 5: Convert the complex result of the forward FFT to a CAS img type
    std::unique_ptr<ComplexToCASFilter> CASFilter(new ComplexToCASFilter());
    CASFilter->SetComplexInput(this->d_PaddedVolumeComplex->GetPointer());
    CASFilter->SetCASVolumeOutput(this->d_PaddedVolume->GetPointer());
    CASFilter->SetVolumeSize(PaddedVolumeSize);
    CASFilter->Update();

    // STEP 6: Pad the result with the additional padding
    PadFilter->SetInput(this->d_PaddedVolume->GetPointer());
    PadFilter->SetOutput(this->d_CASVolume->GetPointer());
    PadFilter->SetInputSize(PaddedVolumeSize);
    PadFilter->SetPaddingX((PaddedVolumeSize_Extra - PaddedVolumeSize) / 2);
    PadFilter->SetPaddingY((PaddedVolumeSize_Extra - PaddedVolumeSize) / 2);
    PadFilter->SetPaddingZ((PaddedVolumeSize_Extra - PaddedVolumeSize) / 2);
    PadFilter->Update();
}

void gpuProjection::CASVolumeToVolume()
{
    gpuErrorCheck(cudaDeviceSynchronize());

    if (this->verbose == true)
    {
        std::cout << "gpuProjection::CASVolumeToVolume() on GPU " << this->GPU_Device << '\n';
        PrintMemoryAvailable();
    }

    // Convert a GPU CAS volume to volume
    // Note: The volume must be square (i.e. have the same dimensions for the X, Y, and Z)
    // Step 1: Pad the input volume with zeros and convert to cufftComplex type
    // Step 2: fftshift
    // Step 3: Take discrete Fourier transform using cuFFT
    // Step 4: fftshift
    // Step 5: Convert to CAS volume using CUDA kernel
    // Step 6: Apply extra zero padding

    int CASVolumeSize = this->d_CASVolume->GetSize(0);
    int CASImgSize = this->d_CASImgs->GetSize(0);
    int ImgSize = this->d_Imgs->GetSize(0);

    this->d_KBPreComp->CopyToGPU(this->h_KBPreComp->GetPointer());

    this->d_Volume->Reset();
    this->d_CASVolume_Cropped->Reset();
    this->d_CASVolume_Cropped_Complex->Reset();

    int CroppedCASVolumeSize = CASVolumeSize - extraPadding * 2;
    int VolumeSize = this->d_Volume->GetSize(1);

    // Remove the extraPadding from the CAS volume
    // std::unique_ptr<CropVolumeFilter> CropFilter(new CropVolumeFilter());
    CropVolumeFilter *CropFilter = new CropVolumeFilter();
    CropFilter->SetInput(this->d_CASVolume->GetPointer());
    CropFilter->SetInputSize(CASVolumeSize);
    CropFilter->SetOutput(this->d_CASVolume_Cropped->GetPointer());
    CropFilter->SetCropX((CASVolumeSize - CroppedCASVolumeSize) / 2);
    CropFilter->SetCropY((CASVolumeSize - CroppedCASVolumeSize) / 2);
    CropFilter->SetCropZ((CASVolumeSize - CroppedCASVolumeSize) / 2);
    CropFilter->SetNumberOfSlices(CroppedCASVolumeSize);
    CropFilter->Update();

    // Convert the CAS volume to complex cufft type
    // std::unique_ptr<CASToComplexFilter> CASFilter(new CASToComplexFilter());
    CASToComplexFilter *CASFilter = new CASToComplexFilter();
    CASFilter->SetCASVolume(this->d_CASVolume_Cropped->GetPointer());
    CASFilter->SetComplexOutput(this->d_CASVolume_Cropped_Complex->GetPointer());
    CASFilter->SetVolumeSize(CroppedCASVolumeSize);
    CASFilter->Update();

    // Run FFTShift on the 3D volume
    //std::unique_ptr<FFTShift3DFilter<cufftComplex>> FFTShiftFilter(new FFTShift3DFilter<cufftComplex>());
    FFTShift3DFilter<cufftComplex> *FFTShiftFilter = new FFTShift3DFilter<cufftComplex>();
    FFTShiftFilter->SetInput(this->d_CASVolume_Cropped_Complex->GetPointer());
    FFTShiftFilter->SetVolumeSize(CroppedCASVolumeSize);
    FFTShiftFilter->Update();

    // Plane and execute the inverse FFT on the 3D array
    cufftHandle inverseFFTPlan;
    cufftPlan3d(&inverseFFTPlan, CroppedCASVolumeSize, CroppedCASVolumeSize, CroppedCASVolumeSize, CUFFT_C2C);
    cufftExecC2C(inverseFFTPlan, (cufftComplex *)this->d_CASVolume_Cropped_Complex->GetPointer(), (cufftComplex *)this->d_CASVolume_Cropped_Complex->GetPointer(), CUFFT_INVERSE);

    gpuErrorCheck(cudaDeviceSynchronize());

    cufftDestroy(inverseFFTPlan);

    // Apply a second in place 3D FFT Shift
    FFTShiftFilter->SetInput(this->d_CASVolume_Cropped_Complex->GetPointer());
    FFTShiftFilter->SetVolumeSize(CroppedCASVolumeSize);
    FFTShiftFilter->Update();

    // Run kernel to crop the d_CASVolume_Cropped_Complex (to remove the zero padding), extract the real value,
    // and normalize the scaling introduced during the FFT
    // std::unique_ptr<ComplexToRealFilter> ComplexToReal(new ComplexToRealFilter());
    ComplexToRealFilter *ComplexToReal = new ComplexToRealFilter();
    ComplexToReal->SetComplexInput(this->d_CASVolume_Cropped_Complex->GetPointer());
    ComplexToReal->SetRealOutput(this->d_CASVolume_Cropped->GetPointer());
    ComplexToReal->SetVolumeSize(CroppedCASVolumeSize);
    ComplexToReal->Update();

    // Multiply by the Kaiser Bessel precompensation array
    // std::unique_ptr<MultiplyVolumeFilter<float>> MultiplyFilter(new MultiplyVolumeFilter<float>());
    MultiplyVolumeFilter<float> *MultiplyFilter = new MultiplyVolumeFilter<float>();
    MultiplyFilter->SetVolumeSize(this->d_CASVolume_Cropped->GetSize(1));
    MultiplyFilter->SetVolumeOne(this->d_CASVolume_Cropped->GetPointer());
    MultiplyFilter->SetVolumeTwo(this->d_KBPreComp->GetPointer());
    MultiplyFilter->Update();

    CropFilter->SetInput(this->d_CASVolume_Cropped->GetPointer());
    CropFilter->SetInputSize(CroppedCASVolumeSize);
    CropFilter->SetOutput(this->d_Volume->GetPointer());
    CropFilter->SetCropX((CroppedCASVolumeSize - VolumeSize) / 2);
    CropFilter->SetCropY((CroppedCASVolumeSize - VolumeSize) / 2);
    CropFilter->SetCropZ((CroppedCASVolumeSize - VolumeSize) / 2);
    CropFilter->Update();

    float normalizationFactor = this->d_Volume->GetSize(1) * interpFactor;
    normalizationFactor = normalizationFactor * normalizationFactor * normalizationFactor;

    // std::unique_ptr<DivideScalarFilter> DivideScalar(new DivideScalarFilter());
    DivideScalarFilter *DivideScalar = new DivideScalarFilter();
    DivideScalar->SetInput(this->d_Volume->GetPointer());
    DivideScalar->SetScalar(float(normalizationFactor));
    DivideScalar->SetVolumeSize(VolumeSize);
    DivideScalar->Update();

    delete DivideScalar;
    delete MultiplyFilter;
    delete ComplexToReal;
    delete FFTShiftFilter;
    delete CASFilter;
    delete CropFilter;

    gpuErrorCheck(cudaDeviceSynchronize());
}

void gpuProjection::CASImgsToImgs(cudaStream_t &stream, float *CASImgs, float *Imgs, int numImgs, cufftComplex *CASImgsComplex)
{
    // Convert CAS images to images using an inverse FFT
    // CASImgs, Imgs, and CASImgsComplex, are the device allocated arrays (e.g. d_CASImgs) at some offset from the beginning of the array

    if (this->verbose == true)
    {
        std::cout << "gpuProjection::CASImgsToImgs() on GPU " << this->GPU_Device << '\n';
        PrintMemoryAvailable();
    }

    int CASImgSize = this->d_CASImgs->GetSize(0);
    int ImgSize = this->d_Imgs->GetSize(0);

    // Has the inverse FFT been planned? If not create one now
    if (this->inverseFFTImagesFlag == false)
    {
        cufftPlan2d(&this->inverseFFTImages, CASImgSize, CASImgSize, CUFFT_C2C);

        this->inverseFFTImagesFlag = true;
    }

    // Convert the CASImgs to complex cufft type
    std::unique_ptr<CASToComplexFilter> CASFilter(new CASToComplexFilter());
    CASFilter->SetCASVolume(CASImgs);
    CASFilter->SetComplexOutput(CASImgsComplex);
    CASFilter->SetVolumeSize(CASImgSize);
    CASFilter->SetNumberOfSlices(numImgs);
    CASFilter->Update(&stream);

    // Run a FFTShift on each 2D slice
    std::unique_ptr<FFTShift2DFilter<cufftComplex>> FFTShiftFilter(new FFTShift2DFilter<cufftComplex>());
    FFTShiftFilter->SetInput(CASImgsComplex);
    FFTShiftFilter->SetImageSize(CASImgSize);
    FFTShiftFilter->SetNumberOfSlices(numImgs);
    FFTShiftFilter->Update(&stream);

    // Set the FFT plan to the current stream to process
    cufftSetStream(this->inverseFFTImages, stream);

    // Execute the inverse FFT on each 2D array
    // cufftPlanMany is not feasible since the number of images changes and
    // cufftDestroy is blocks the CPU and causes memory leaks if not called
    // FFT on each 2D slice has similar computation speed as cufftPlanMany
    for (int i = 0; i < numImgs; i++)
    {
        cufftExecC2C(this->inverseFFTImages,
                     &CASImgsComplex[i * CASImgSize * CASImgSize],
                     &CASImgsComplex[i * CASImgSize * CASImgSize],
                     CUFFT_INVERSE);
    }

    // Run a 2D FFTShift again
    FFTShiftFilter->SetInput(CASImgsComplex);
    FFTShiftFilter->SetImageSize(CASImgSize);
    FFTShiftFilter->SetNumberOfSlices(numImgs);
    FFTShiftFilter->Update(&stream);

    // Extract the real component of the complex images
    std::unique_ptr<ComplexToRealFilter> ComplexToReal(new ComplexToRealFilter());
    ComplexToReal->SetComplexInput(CASImgsComplex);
    ComplexToReal->SetRealOutput(CASImgs);
    ComplexToReal->SetVolumeSize(CASImgSize);
    ComplexToReal->SetNumberOfSlices(numImgs);
    ComplexToReal->Update(&stream);

    // Crop the images to remove the zero padding
    std::unique_ptr<CropVolumeFilter> CropFilter(new CropVolumeFilter());
    CropFilter->SetInput(CASImgs);
    CropFilter->SetInputSize(CASImgSize);
    CropFilter->SetNumberOfSlices(numImgs);
    CropFilter->SetOutput(Imgs);
    CropFilter->SetCropX((CASImgSize - ImgSize) / 2);
    CropFilter->SetCropY((CASImgSize - ImgSize) / 2);
    CropFilter->SetCropZ(0);
    CropFilter->Update(&stream);

    // Normalize for the scaling introduced during the FFT
    float normalizationFactor = ImgSize * ImgSize * interpFactor * interpFactor;

    std::unique_ptr<DivideScalarFilter> DivideScalar(new DivideScalarFilter());
    DivideScalar->SetInput(Imgs);
    DivideScalar->SetScalar(float(normalizationFactor));
    DivideScalar->SetVolumeSize(ImgSize);
    DivideScalar->SetNumberOfSlices(numImgs);
    DivideScalar->Update(&stream);
}

void gpuProjection::ImgsToCASImgs(cudaStream_t &stream, float *CASImgs, float *Imgs, int numImgs)
{
    // Convert projection images to CAS images by running a forward FFT
    // CASImgs, Imgs, and CASImgsComplex, are the device allocated arrays (e.g. d_CASImgs) at some offset from the beginning of the array

    if (this->verbose == true)
    {
        std::cout << "gpuProjection::ImgsToCASImgs() on GPU " << this->GPU_Device << '\n';
        PrintMemoryAvailable();
    }

    int CASImgSize = this->d_CASImgs->GetSize(0);
    int ImgSize = this->d_Imgs->GetSize(0);

    this->d_CASImgsComplex->Reset();

    // Has the forward FFT been planned? If not create one now
    if (this->forwardFFTImagesFlag == false)
    {
        cufftPlan2d(&this->forwardFFTImages, CASImgSize, CASImgSize, CUFFT_C2C);

        this->forwardFFTImagesFlag = true;
    }

    // First pad the Imgs with zeros to be the same size as CASImgs
    std::unique_ptr<PadVolumeFilter> PadFilter(new PadVolumeFilter());
    PadFilter->SetInput(Imgs);
    PadFilter->SetOutput(CASImgs);
    PadFilter->SetInputSize(ImgSize);
    PadFilter->SetPaddingX((CASImgSize - ImgSize) / 2);
    PadFilter->SetPaddingY((CASImgSize - ImgSize) / 2);
    PadFilter->SetPaddingZ(0);
    PadFilter->SetNumberOfSlices(numImgs);
    PadFilter->Update(&stream);

    // Convert the images to complex cufft type
    std::unique_ptr<RealToComplexFilter> RealFilter(new RealToComplexFilter());
    RealFilter->SetRealInput(CASImgs);
    RealFilter->SetComplexOutput(this->d_CASImgsComplex->GetPointer());
    RealFilter->SetVolumeSize(CASImgSize);
    RealFilter->SetNumberOfSlices(numImgs);
    RealFilter->Update(&stream);

    // Run FFTShift on each 2D slice
    std::unique_ptr<FFTShift2DFilter<cufftComplex>> FFTShiftFilter(new FFTShift2DFilter<cufftComplex>());
    FFTShiftFilter->SetInput(this->d_CASImgsComplex->GetPointer());
    FFTShiftFilter->SetImageSize(CASImgSize);
    FFTShiftFilter->SetNumberOfSlices(numImgs);
    FFTShiftFilter->Update(&stream);

    // Set the FFT plan to the current stream to process
    cufftSetStream(this->forwardFFTImages, stream);

    // Execute the forward FFT on each 2D array
    // cufftPlanMany is not feasible since the number of images changes and
    // cufftDestroy is blocks the CPU and causes memory leaks if not called
    // FFT on each 2D slice has similar computation speed as cufftPlanMany
    for (int i = 0; i < numImgs; i++)
    {
        cufftExecC2C(this->forwardFFTImages,
                     this->d_CASImgsComplex->GetPointer(i * CASImgSize * CASImgSize),
                     this->d_CASImgsComplex->GetPointer(i * CASImgSize * CASImgSize),
                     CUFFT_FORWARD);
    }

    // Run the 2D FFTShift again
    FFTShiftFilter->SetInput(this->d_CASImgsComplex->GetPointer());
    FFTShiftFilter->SetImageSize(CASImgSize);
    FFTShiftFilter->SetNumberOfSlices(numImgs);
    FFTShiftFilter->Update(&stream);

    // Convert the complex result of the forward FFT to a CAS img type
    std::unique_ptr<ComplexToCASFilter> ComplexToCAS(new ComplexToCASFilter());
    ComplexToCAS->SetComplexInput(this->d_CASImgsComplex->GetPointer());
    ComplexToCAS->SetCASVolumeOutput(CASImgs);
    ComplexToCAS->SetVolumeSize(CASImgSize);
    ComplexToCAS->SetNumberOfSlices(numImgs);
    ComplexToCAS->Update(&stream);
}

void gpuProjection::ForwardProject(int AxesOffset, int nAxesToProcess)
{
    // Run the forward projection on some subset of the coordinate axes (needed when using multiple GPUs)
    cudaSetDevice(this->GPU_Device);
    if (this->verbose == true)
    {
        std::cout << "gpuProjection::ForwardProject() on GPU " << this->GPU_Device << '\n';
        PrintMemoryAvailable();
    }

    this->d_KB_Table->CopyToGPU(this->h_KB_Table->GetPointer(), this->h_KB_Table->bytes());

    // Convert and copy the volume to CAS volume if we are running the FFT on the device
    if (this->RunFFTOnDevice == 1)
    {
        // Run the volume to CAS volume function
        VolumeToCASVolume();
    }
    else
    {
        // Copy the CAS volume to the corresponding GPU array
        this->d_CASVolume->Reset();
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
        std::cerr << "Error during initialization." << '\n';
        return; // Don't run the kernel and return
    }

    // Run the forward projection CUDA kernel
    cudaSetDevice(this->GPU_Device);

    // For compactness define the CASImgSize, CASVolSize, and ImgSize here
    int ImgSize = this->d_Imgs->GetSize(0);
    int CASImgSize = this->d_CASImgs->GetSize(0);
    int CASVolSize = this->d_CASVolume->GetSize(0);

    // Plan the pointer offset values
    gpuProjection::Offsets Offsets_obj = PlanOffsetValues(AxesOffset, nAxesToProcess, this->nStreamsFP);

    for (int i = 0; i < Offsets_obj.num_offsets; i++)
    {
        if (this->verbose == true)
        {
            std::cout << '\n';
            std::cout << '\n';
            std::cout << "GPU: " << this->GPU_Device << " forward projection stream " << Offsets_obj.stream_ID[i]
                      << " batch " << Offsets_obj.currBatch[i]
                      << " processing " << Offsets_obj.numAxesPerStream[i] << " axes "
                      << " running FFT on device " << this->RunFFTOnDevice << '\n';
        }

        if (Offsets_obj.numAxesPerStream[i] < 1)
        {
            continue;
        }

        gpuErrorCheck(cudaMemsetAsync(
            this->d_CASImgs->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
            0,
            Offsets_obj.gpuCASImgs_streamBytes[i],
            FP_streams[Offsets_obj.stream_ID[i]]));

        gpuErrorCheck(cudaMemsetAsync(
            this->d_Imgs->GetPointer(Offsets_obj.gpuImgs_Offset[i]),
            0,
            Offsets_obj.gpuImgs_streamBytes[i],
            FP_streams[Offsets_obj.stream_ID[i]]));

        // Copy the section of gpuCoordAxes which this stream will process on the current GPU
        gpuErrorCheck(cudaMemcpyAsync(
            this->d_CoordAxes->GetPointer(Offsets_obj.gpuCoordAxes_Stream_Offset[i]),
            this->h_CoordAxes->GetPointer(Offsets_obj.CoordAxes_CPU_Offset[i]),
            Offsets_obj.coord_Axes_CPU_streamBytes[i],
            cudaMemcpyHostToDevice, FP_streams[Offsets_obj.stream_ID[i]]));

        // Run the forward projection kernel
        gpuForwardProject::RunKernel(
            this->d_CASVolume->GetPointer(),
            this->d_CASImgs->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
            this->d_KB_Table->GetPointer(),
            this->d_CoordAxes->GetPointer(Offsets_obj.gpuCoordAxes_Stream_Offset[i]),
            this->kerHWidth,
            Offsets_obj.numAxesPerStream[i],
            CASVolSize,
            CASImgSize,
            this->extraPadding,
            this->maskRadius,
            this->d_KB_Table->GetSize(0),
            &FP_streams[Offsets_obj.stream_ID[i]]);

        if (this->RunFFTOnDevice == 0 || this->verbose == true)
        {
            // Copy the resulting CAS images back to the host pinned memory (CPU)
            gpuErrorCheck(cudaMemcpyAsync(
                this->h_CASImgs->GetPointer(Offsets_obj.CASImgs_CPU_Offset[i]),
                this->d_CASImgs->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
                Offsets_obj.gpuCASImgs_streamBytes[i],
                cudaMemcpyDeviceToHost,
                FP_streams[Offsets_obj.stream_ID[i]]));
        }

        // If running the inverse FFT on the device
        if (this->RunFFTOnDevice == 1)
        {
            gpuErrorCheck(cudaMemsetAsync(
                this->d_CASImgsComplex->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
                0,
                2 * Offsets_obj.gpuCASImgs_streamBytes[i], // cufftComplex type so multiply the bytes by 2
                FP_streams[Offsets_obj.stream_ID[i]]));

            // Convert the CAS projection images back to images using an inverse FFT and cropping out the zero padding
            this->CASImgsToImgs(
                FP_streams[Offsets_obj.stream_ID[i]],
                this->d_CASImgs->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
                this->d_Imgs->GetPointer(Offsets_obj.gpuImgs_Offset[i]),
                Offsets_obj.numAxesPerStream[i],
                this->d_CASImgsComplex->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]));

            // Lastly, copy the resulting cropped projection images back to the host pinned memory (CPU)
            gpuErrorCheck(cudaMemcpyAsync(
                this->h_Imgs->GetPointer(Offsets_obj.Imgs_CPU_Offset[i]),
                this->d_Imgs->GetPointer(Offsets_obj.gpuImgs_Offset[i]),
                Offsets_obj.gpuImgs_streamBytes[i],
                cudaMemcpyDeviceToHost,
                FP_streams[Offsets_obj.stream_ID[i]]));
        }

        if (this->verbose == true)
        {
            std::cout << "Stream completed" << '\n';
        }
    }
}

void gpuProjection::BackProject(int AxesOffset, int nAxesToProcess)
{
    // Run the forward projection on some subset of the coordinate axes (needed when using multiple GPUs)
    gpuErrorCheck(cudaSetDevice(this->GPU_Device));

    if (this->verbose == true)
    {
        std::cout << "gpuProjection::BackProject() on GPU " << this->GPU_Device << '\n';
        PrintMemoryAvailable();
    }

    this->d_KB_Table->CopyToGPU(this->h_KB_Table->GetPointer(), this->h_KB_Table->bytes());

    // For compactness define the CASImgSize, CASVolSize, and ImgSize here
    int ImgSize = this->d_Imgs->GetSize(0);
    int CASImgSize = this->d_CASImgs->GetSize(0);
    int CASVolSize = this->d_CASVolume->GetSize(0);

    // Plan the pointer offset values
    gpuProjection::Offsets Offsets_obj = PlanOffsetValues(AxesOffset, nAxesToProcess, this->nStreamsBP);

    // Reset the CAS volume on the device to all zeros before the back projection
    this->d_Imgs->Reset();
    this->d_CASImgs->Reset();
    this->d_CASVolume->Reset(); // needed?
    this->d_CoordAxes->Reset();
    this->d_Volume->Reset();

    if (this->d_PlaneDensity != NULL)
    {
        this->d_PlaneDensity->Reset();
    }

    // Convert and copy the volume to CAS volume if we are running the FFT on the device
    // if (this->RunFFTOnDevice == 1)
    // {
    //     // Run the volume to CAS volume function
    //     VolumeToCASVolume();
    // }
    // else
    // {
    //     this->d_CASVolume->Reset();

    //     // Copy the CAS volume to the corresponding GPU array
    //     this->d_CASVolume->CopyToGPU(this->h_CASVolume->GetPointer(), this->h_CASVolume->bytes());
    // }

    gpuErrorCheck(cudaDeviceSynchronize());

    for (int i = 0; i < Offsets_obj.num_offsets; i++)
    {
        if (Offsets_obj.numAxesPerStream[i] < 1)
        {
            continue;
        }

        if (this->verbose == true)
        {
            std::cout << '\n';
            std::cout << '\n';
            std::cout << "GPU: " << this->GPU_Device << " back projection stream " << Offsets_obj.stream_ID[i]
                      << " batch " << Offsets_obj.currBatch[i]
                      << " processing " << Offsets_obj.numAxesPerStream[i] << " axes " << '\n';
        }

        gpuErrorCheck(cudaMemsetAsync(
            this->d_CASImgs->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
            0,
            Offsets_obj.gpuCASImgs_streamBytes[i],
            BP_streams[Offsets_obj.stream_ID[i]]));

        // Copy the section of gpuCoordAxes which this stream will process on the current GPU
        gpuErrorCheck(cudaMemcpyAsync(
            this->d_CoordAxes->GetPointer(Offsets_obj.gpuCoordAxes_Stream_Offset[i]),
            this->h_CoordAxes->GetPointer(Offsets_obj.CoordAxes_CPU_Offset[i]),
            Offsets_obj.coord_Axes_CPU_streamBytes[i],
            cudaMemcpyHostToDevice, BP_streams[Offsets_obj.stream_ID[i]]));

        if (this->RunFFTOnDevice == 0)
        {
            // Copy the CASImages from the pinned CPU array and use instead of the images array
            gpuErrorCheck(cudaMemcpyAsync(
                this->d_CASImgs->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
                this->h_CASImgs->GetPointer(Offsets_obj.CASImgs_CPU_Offset[i]),
                Offsets_obj.gpuCASImgs_streamBytes[i],
                cudaMemcpyHostToDevice,
                BP_streams[Offsets_obj.stream_ID[i]]));
        }
        else
        {
            gpuErrorCheck(cudaMemsetAsync(
                this->d_Imgs->GetPointer(Offsets_obj.gpuImgs_Offset[i]),
                0,
                Offsets_obj.gpuImgs_streamBytes[i],
                BP_streams[Offsets_obj.stream_ID[i]]));

            gpuErrorCheck(cudaMemsetAsync(
                this->d_CASImgsComplex->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
                0,
                2 * Offsets_obj.gpuCASImgs_streamBytes[i],
                BP_streams[Offsets_obj.stream_ID[i]]));

            // Copy the section of images to the GPU
            gpuErrorCheck(cudaMemcpyAsync(
                this->d_Imgs->GetPointer(Offsets_obj.gpuImgs_Offset[i]),
                this->h_Imgs->GetPointer(Offsets_obj.Imgs_CPU_Offset[i]),
                Offsets_obj.gpuImgs_streamBytes[i],
                cudaMemcpyHostToDevice,
                BP_streams[Offsets_obj.stream_ID[i]]));

            // Run the forward FFT to convert the pinned CPU images to CAS images (CAS type is needed for back projecting into the CAS volume)
            this->ImgsToCASImgs(
                BP_streams[Offsets_obj.stream_ID[i]],
                this->d_CASImgs->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
                this->d_Imgs->GetPointer(Offsets_obj.gpuImgs_Offset[i]),
                Offsets_obj.numAxesPerStream[i]);
        }
        this->d_CASVolume->Reset(); // needed?
        // Run the back projection kernel
        gpuBackProject::RunKernel(
            this->d_CASVolume->GetPointer(),
            this->d_CASImgs->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
            this->d_KB_Table->GetPointer(),
            this->d_CoordAxes->GetPointer(Offsets_obj.gpuCoordAxes_Stream_Offset[i]),
            this->kerHWidth,
            Offsets_obj.numAxesPerStream[i],
            CASVolSize,
            CASImgSize,
            this->maskRadius,
            this->d_KB_Table->GetSize(0),
            this->extraPadding,
            &BP_streams[Offsets_obj.stream_ID[i]]);
    }
}

void gpuProjection::CalculatePlaneDensity(int AxesOffset, int nAxesToProcess)
{
    // Calculate the plane density by running the back projection kernel with CASimages equal to one
    gpuErrorCheck(cudaSetDevice(this->GPU_Device));

    if (this->verbose == true)
    {
        std::cout << "gpuProjection::CalculatePlaneDensity() on GPU " << this->GPU_Device << '\n';
        PrintMemoryAvailable();
    }

    // For compactness define the CASImgSize, CASVolSize, and ImgSize here
    int ImgSize = this->d_Imgs->GetSize(0);
    int CASImgSize = this->d_CASImgs->GetSize(0);
    int CASVolSize = this->d_CASVolume->GetSize(0);

    // Plan the pointer offset values
    gpuProjection::Offsets Offsets_obj = PlanOffsetValues(AxesOffset, nAxesToProcess, this->nStreamsBP);

    // Reset the CAS volume on the device to all zeros before the back projection
    this->d_Imgs->Reset();
    this->d_CASImgs->Reset();
    this->d_CoordAxes->Reset();
    this->d_PlaneDensity->Reset();

    // Set the CAS images to a value of all ones
    float *CASImgsOnes = new float[this->d_CASImgs->length()];
    for (int i = 0; i < this->d_CASImgs->length(); i++)
    {
        CASImgsOnes[i] = 1;
    }
    this->d_CASImgs->CopyToGPU(CASImgsOnes, this->d_CASImgs->bytes());

    delete[] CASImgsOnes;

    gpuErrorCheck(cudaDeviceSynchronize());

    for (int i = 0; i < Offsets_obj.num_offsets; i++)
    {
        if (Offsets_obj.numAxesPerStream[i] < 1)
        {
            continue;
        }

        if (this->verbose == true)
        {
            std::cout << '\n';
            std::cout << '\n';
            std::cout << "GPU: " << this->GPU_Device << " plane density stream " << Offsets_obj.stream_ID[i]
                      << " batch " << Offsets_obj.currBatch[i]
                      << " processing " << Offsets_obj.numAxesPerStream[i] << " axes " << '\n';
            PrintMemoryAvailable();
        }

        // Copy the section of gpuCoordAxes which this stream will process on the current GPU
        gpuErrorCheck(cudaMemcpyAsync(
            this->d_CoordAxes->GetPointer(Offsets_obj.gpuCoordAxes_Stream_Offset[i]),
            this->h_CoordAxes->GetPointer(Offsets_obj.CoordAxes_CPU_Offset[i]),
            Offsets_obj.coord_Axes_CPU_streamBytes[i],
            cudaMemcpyHostToDevice, BP_streams[Offsets_obj.stream_ID[i]]));

        // Run the back projection kernel
        gpuBackProject::RunKernel(
            this->d_PlaneDensity->GetPointer(),
            this->d_CASImgs->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
            this->d_KB_Table->GetPointer(),
            this->d_CoordAxes->GetPointer(Offsets_obj.gpuCoordAxes_Stream_Offset[i]),
            this->kerHWidth,
            Offsets_obj.numAxesPerStream[i],
            CASVolSize,
            CASImgSize,
            this->maskRadius,
            this->d_KB_Table->GetSize(0),
            this->extraPadding,
            &BP_streams[Offsets_obj.stream_ID[i]]);
    }
}

void gpuProjection::PrintMemoryAvailable()
{
    // Check to make sure the GPU has enough available memory left
    size_t mem_tot_0 = 0;
    size_t mem_free_0 = 0;
    gpuErrorCheck(cudaMemGetInfo(&mem_free_0, &mem_tot_0));
    std::cout << "Memory remaining on GPU " << this->GPU_Device << " " << mem_free_0 << " out of " << mem_tot_0 << '\n';
}

void gpuProjection::FreeMemory()
{

    if (this->verbose == true)
    {
        std::cout << "gpuProjection::FreeMemory()" << '\n';
    }

    // If running the FFT on the device deallocate the arrays
    if (this->RunFFTOnDevice == 1)
    {
        delete d_PaddedVolume;
        delete d_KBPreComp;
        delete d_CASVolume_Cropped;
        delete d_PaddedVolumeComplex;
        delete d_CASVolume_Cropped_Complex;
        delete d_CASImgsComplex;
    }

    delete d_CASVolume;
    delete d_CASImgs;
    delete d_Imgs;
    delete d_KB_Table;
    delete d_CoordAxes;
    delete d_PlaneDensity;
    delete d_Volume;

    delete FP_streams;

    delete BP_streams;
}