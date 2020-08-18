#include "gpuGridder.h"

#define Log(x)                  \
    {                           \
        std::cout << x << '\n'; \
    }

float *gpuGridder::GetCASVolumePtr()
{
    // Return the pointer to the CAS volume (needed for adding the volumes from all GPUs together for the volume reconstruction)
    return this->d_CASVolume->GetPointer();
}

void gpuGridder::CopyCASVolumeToHost()
{
    // Copy the CAS volume from the GPU to pinned CPU memory
    this->d_CASVolume->CopyFromGPU(this->h_CASVolume->GetPointer(), this->h_CASVolume->bytes());
}

float *gpuGridder::GetCASVolumeFromDevice()
{
    float *CASVolume = new float[this->d_CASVolume->length()];
    this->d_CASVolume->CopyFromGPU(CASVolume, this->d_CASVolume->bytes());

    return CASVolume;
}

void gpuGridder::SetGPU(int GPU_Device)
{
    // Set which GPU to use

    // Check how many GPUs there are on the computer
    int numGPUDetected;
    gpuErrorCheck(cudaGetDeviceCount(&numGPUDetected));

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

void gpuGridder::InitializeGPUArrays()
{
    // Initialize the GPU arrays and allocate the needed memory on the GPU

    if (this->verbose == true)
    {
        std::cout << "gpuGridder::InitializeGPUArrays() on GPU " << '\n';
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

    // Allocate the CAS volume
    this->d_CASVolume = new DeviceMemory<float>(3, CASVolumeSize, CASVolumeSize, CASVolumeSize, this->GPU_Device);
    this->d_CASVolume->AllocateGPUArray();

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

    cudaDeviceSynchronize();
}

void gpuGridder::Allocate()
{
    // Allocate the needed GPU memory
    // Have the GPU arrays already been allocated?
    if (this->GPUArraysAllocatedFlag == false)
    {
        // Initialize the needed arrays for the gpuGridder on the GPU
        InitializeGPUArrays();

        // Initialize the CUDA streams
        InitializeCUDAStreams();

        this->GPUArraysAllocatedFlag = true;
    }
}

void gpuGridder::InitializeCUDAStreams()
{
    cudaSetDevice(this->GPU_Device);
    if (this->GPUArraysAllocatedFlag == true)
    {
        return;
    }
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

void gpuGridder::PrintMemoryAvailable()
{
    // Check to make sure the GPU has enough available memory left
    size_t mem_tot_0 = 0;
    size_t mem_free_0 = 0;
    gpuErrorCheck(cudaMemGetInfo(&mem_free_0, &mem_tot_0));
    std::cout << "Memory remaining on GPU " << this->GPU_Device << " " << mem_free_0 << " out of " << mem_tot_0 << '\n';
}

void gpuGridder::ForwardProject(int AxesOffset, int nAxesToProcess)
{
    // Run the forward projection on some subset of the coordinate axes (needed when using multiple GPUs)
    cudaSetDevice(this->GPU_Device);

    if (this->verbose == true)
    {
        std::cout << "gpuGridder::ForwardProject() on GPU " << this->GPU_Device << '\n';
        PrintMemoryAvailable();
    }

    this->d_KB_Table->CopyToGPU(this->h_KB_Table->GetPointer(), this->h_KB_Table->bytes());

    // Copy the CAS volume to the corresponding GPU array
    this->d_CASVolume->CopyToGPU(this->h_CASVolume->GetPointer(), this->h_CASVolume->bytes());

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

    // For compactness define the CASImgSize, CASVolSize, and ImgSize here
    int ImgSize = this->d_Imgs->GetSize(0);
    int CASImgSize = this->d_CASImgs->GetSize(0);
    int CASVolSize = this->d_CASVolume->GetSize(0);

    // Plan the pointer offset values
    gpuGridder::Offsets Offsets_obj = PlanOffsetValues(AxesOffset, nAxesToProcess, this->nStreamsFP);

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

        // Reset the d_CASImgs and d_Imgs arrays back to zeros
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

            // Reset the d_CASImgsComplex array back to zeros
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
    }
}

void gpuGridder::BackProject(int AxesOffset, int nAxesToProcess)
{
    // Run the back projection on some subset of the coordinate axes (needed when using multiple GPUs)
    gpuErrorCheck(cudaSetDevice(this->GPU_Device));

    if (this->verbose == true)
    {
        std::cout << "gpuGridder::BackProject() on GPU " << this->GPU_Device << '\n';
        PrintMemoryAvailable();
    }

    // Copy the Kaiser-Bessel vector to the GPU
    this->d_KB_Table->CopyToGPU(this->h_KB_Table->GetPointer(), this->h_KB_Table->bytes());

    // For compactness define the CASImgSize, CASVolSize, and ImgSize here
    int ImgSize = this->d_Imgs->GetSize(0);
    int CASImgSize = this->d_CASImgs->GetSize(0);
    int CASVolSize = this->d_CASVolume->GetSize(0);

    // Plan the pointer offset values
    gpuGridder::Offsets Offsets_obj = PlanOffsetValues(AxesOffset, nAxesToProcess, this->nStreamsBP);

    // Reset the arrays on the device to all zeros before the back projection
    this->d_Imgs->Reset();
    this->d_CASImgs->Reset();
    this->d_CASVolume->Reset();
    this->d_CoordAxes->Reset();

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
            // Reset the d_Imgs, d_CASImgs, and d_CASImgsComplex arrays back to all zeros
            gpuErrorCheck(cudaMemsetAsync(
                this->d_Imgs->GetPointer(Offsets_obj.gpuImgs_Offset[i]),
                0,
                Offsets_obj.gpuImgs_streamBytes[i],
                BP_streams[Offsets_obj.stream_ID[i]]));

            gpuErrorCheck(cudaMemsetAsync(
                this->d_CASImgs->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
                0,
                Offsets_obj.gpuCASImgs_streamBytes[i],
                BP_streams[Offsets_obj.stream_ID[i]]));

            gpuErrorCheck(cudaMemsetAsync(
                this->d_CASImgsComplex->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
                0,
                2 * Offsets_obj.gpuCASImgs_streamBytes[i],
                BP_streams[Offsets_obj.stream_ID[i]]));

            // Copy the section of host images to the GPU
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
                this->d_CASImgsComplex->GetPointer(Offsets_obj.gpuCASImgs_Offset[i]),
                this->d_Imgs->GetPointer(Offsets_obj.gpuImgs_Offset[i]),
                NULL,
                NULL,
                Offsets_obj.numAxesPerStream[i]);
        }

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

void gpuGridder::CASImgsToImgs(cudaStream_t &stream, float *CASImgs, float *Imgs, int numImgs, cufftComplex *CASImgsComplex)
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


void gpuGridder::ImgsToCASImgs(cudaStream_t &stream, float *CASImgs, cufftComplex *CASImgsComplex, float *Imgs, float *CTFs, float *CTFsPadded, int numImgs)
{
    // Convert projection images to CAS images by running a forward FFT
    // CASImgs, Imgs, and CASImgsComplex, are the device allocated arrays (e.g. d_CASImgs) at some offset from the beginning of the array

    if (this->verbose == true)
    {
        std::cout << "gpuGridder::ImgsToCASImgs() on GPU " << this->GPU_Device << '\n';
        PrintMemoryAvailable();
    }

    int CASImgSize = this->d_CASImgs->GetSize(0);
    int ImgSize = this->d_Imgs->GetSize(0);

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
    RealFilter->SetComplexOutput(CASImgsComplex);
    RealFilter->SetVolumeSize(CASImgSize);
    RealFilter->SetNumberOfSlices(numImgs);
    RealFilter->Update(&stream);

    // Run FFTShift on each 2D slice
    std::unique_ptr<FFTShift2DFilter<cufftComplex>> FFTShiftFilter(new FFTShift2DFilter<cufftComplex>());
    FFTShiftFilter->SetInput(CASImgsComplex);
    FFTShiftFilter->SetImageSize(CASImgSize);
    FFTShiftFilter->SetNumberOfSlices(numImgs);
    FFTShiftFilter->Update(&stream);

    // Set the FFT plan to the current stream to process
    cufftSetStream(this->forwardFFTImages, stream);

    // Execute the forward FFT on each 2D array
    // cufftPlanMany is not feasible since the number of images changes and
    // cufftDestroy blocks the CPU and causes memory leaks if not called
    // FFT on each 2D slice has similar computation speed as cufftPlanMany
    for (int i = 0; i < numImgs; i++)
    {
        cufftExecC2C(this->forwardFFTImages,
                     &CASImgsComplex[i * CASImgSize * CASImgSize],
                     &CASImgsComplex[i * CASImgSize * CASImgSize],
                     CUFFT_FORWARD);
    }

    // Run the 2D FFTShift again
    FFTShiftFilter->SetInput(CASImgsComplex);
    FFTShiftFilter->SetImageSize(CASImgSize);
    FFTShiftFilter->SetNumberOfSlices(numImgs);
    FFTShiftFilter->Update(&stream);

    if (this->ApplyCTFs == true)
    {
    // FFTShift the CTFs
    std::unique_ptr<FFTShift2DFilter<float>> FFTShiftFilterCTF(new FFTShift2DFilter<float>());
    FFTShiftFilterCTF->SetInput(CTFs);
    FFTShiftFilterCTF->SetImageSize(ImgSize);
    FFTShiftFilterCTF->SetNumberOfSlices(numImgs);
    FFTShiftFilterCTF->Update(&stream);

    // First pad the CTFs with zeros to be the same size as CASImgs
    std::unique_ptr<PadVolumeFilter> CTFPadFilter(new PadVolumeFilter());
    CTFPadFilter->SetInput(CTFs);
    CTFPadFilter->SetOutput(CTFsPadded);
    CTFPadFilter->SetInputSize(ImgSize); // CTFs are the same size as the images
    CTFPadFilter->SetPaddingX((CASImgSize - ImgSize) / 2);
    CTFPadFilter->SetPaddingY((CASImgSize - ImgSize) / 2);
    CTFPadFilter->SetPaddingZ(0);
    CTFPadFilter->SetNumberOfSlices(numImgs);
    CTFPadFilter->Update(&stream);

    // Multiply the CASImgsComplex with the CTFs
    std::unique_ptr<MultiplyVolumeFilter<cufftComplex>> MultiplyFilter(new MultiplyVolumeFilter<cufftComplex>());
    MultiplyFilter->SetVolumeSize(CASImgSize);
    MultiplyFilter->SetVolumeOne(CASImgsComplex);
    MultiplyFilter->SetVolumeTwo(CTFsPadded);
    MultiplyFilter->SetNumberOfSlices(numImgs);
    MultiplyFilter->Update(&stream);

    }

    // Convert the complex result of the forward FFT to a CAS img type
    std::unique_ptr<ComplexToCASFilter> ComplexToCAS(new ComplexToCASFilter());
    ComplexToCAS->SetComplexInput(CASImgsComplex);
    ComplexToCAS->SetCASVolumeOutput(CASImgs);
    ComplexToCAS->SetVolumeSize(CASImgSize);
    ComplexToCAS->SetNumberOfSlices(numImgs);
    ComplexToCAS->Update(&stream);
}


gpuGridder::Offsets gpuGridder::PlanOffsetValues(int coordAxesOffset, int nAxes, int numStreams)
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
    gpuGridder::Offsets Offsets_obj;
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
        std::cout << "gpuGridder::PlanOffsetValues() "
                  << "estimated number of axes to process on each stream = " << EstimatedNumAxesPerStream << '\n';

        std::cout << "gpuGridder::PlanOffsetValues() "
                  << " number of axes to process on this GPU  " << nAxes << '\n';

        std::cout << "gpuGridder::PlanOffsetValues() "
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


void gpuGridder::FreeMemory()
{
    // Free all of the allocated memory
    gpuErrorCheck(cudaSetDevice(this->GPU_Device));
    gpuErrorCheck(cudaDeviceReset()); // This deletes the CUDA context (i.e. deallocates all memory)

    if (this->verbose == true)
    {
        std::cout << "gpuGridder::FreeMemory()" << '\n';
    }

}
