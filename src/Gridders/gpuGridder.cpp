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

float *gpuGridder::GetPlaneDensityPtr()
{
    // Return the pointer to the plane density volume (needed for adding the volumes from all GPUs together for the volume reconstruction)
    if (this->d_PlaneDensity != NULL)
    {
        return this->d_PlaneDensity->GetPointer();
    }
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
    if (this->d_PlaneDensity != NULL)
    {
        this->d_PlaneDensity->CopyFromGPU(this->h_PlaneDensity->GetPointer(), this->h_PlaneDensity->bytes());
    }
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
    if (this->d_PlaneDensity != NULL)
    {
        float *PlaneDensity = new float[this->d_PlaneDensity->length()];
        this->d_PlaneDensity->CopyFromGPU(PlaneDensity, this->d_PlaneDensity->bytes());
        return PlaneDensity;
    }
    else
    {
        return NULL;
    }
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

    // Allocate the volume
    this->d_Volume = new DeviceMemory<float>(3, this->VolumeSize, this->VolumeSize, this->VolumeSize, this->GPU_Device);
    this->d_Volume->AllocateGPUArray();

    // Allocate the plane density array (for the back projection)
    this->d_PlaneDensity = new DeviceMemory<float>(3, CASVolumeSize, CASVolumeSize, CASVolumeSize, this->GPU_Device);
    this->d_PlaneDensity->AllocateGPUArray();

    // Allocate the CAS volume
    this->d_CASVolume = new DeviceMemory<float>(3, CASVolumeSize, CASVolumeSize, CASVolumeSize, this->GPU_Device);
    this->d_CASVolume->AllocateGPUArray();
}

void gpuGridder::Allocate()
{
    // Allocate the needed GPU memory
    // Have the GPU arrays already been allocated?
    if (this->GPUArraysAllocatedFlag == false)
    {
        // Initialize the needed arrays for the gpuGridder on the GPU
        InitializeGPUArrays();

        // Create a new projection object for running the forward and back projection on the GPU
        this->gpuProjection_Obj = new gpuProjection(
            GPU_Device, nStreamsFP, nStreamsBP, VolumeSize, interpFactor, numCoordAxes,
            extraPadding, RunFFTOnDevice, verbose);

        this->GPUArraysAllocatedFlag = true;
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

    // Pass the host pointers to the projection object
    this->gpuProjection_Obj->SetHostVolume(this->h_Volume);
    this->gpuProjection_Obj->SetHostCoordinateAxes(this->h_CoordAxes);
    this->gpuProjection_Obj->SetHostKBTable(this->h_KB_Table);
    this->gpuProjection_Obj->SetHostImages(this->h_Imgs);
    this->gpuProjection_Obj->SetHostKBPreCompArray(this->h_KBPreComp);

    if (this->RunFFTOnDevice == false || this->verbose == true)
    {
        this->gpuProjection_Obj->SetHostCASVolume(this->h_CASVolume);
        this->gpuProjection_Obj->SetHostCASImages(this->h_CASImgs);
    }

    // Pass the device pointers to the projection object
    this->gpuProjection_Obj->SetDeviceVolume(this->d_Volume);
    this->gpuProjection_Obj->SetDeviceCASVolume(this->d_CASVolume);
    this->gpuProjection_Obj->SetDevicePlaneDensity(this->d_PlaneDensity);

    // Set other needed parameters
    this->gpuProjection_Obj->SetMaskRadius(this->maskRadius);

    // Run the forward projection
    this->gpuProjection_Obj->ForwardProject(AxesOffset, nAxesToProcess);
}

void gpuGridder::BackProject(int AxesOffset, int nAxesToProcess)
{
    // Run the forward projection on some subset of the coordinate axes (needed when using multiple GPUs)
    gpuErrorCheck(cudaSetDevice(this->GPU_Device));

    // Pass the host pointers to the projection object
    this->gpuProjection_Obj->SetHostVolume(this->h_Volume);
    this->gpuProjection_Obj->SetHostCoordinateAxes(this->h_CoordAxes);
    this->gpuProjection_Obj->SetHostKBTable(this->h_KB_Table);
    this->gpuProjection_Obj->SetHostImages(this->h_Imgs);
    this->gpuProjection_Obj->SetHostKBPreCompArray(this->h_KBPreComp);

    if (this->ApplyCTFs == true)
    {
        this->gpuProjection_Obj->SetHostCTFs(this->h_CTFs);
    }
    
    if (this->RunFFTOnDevice == false || this->verbose == true)
    {
        this->gpuProjection_Obj->SetHostCASVolume(this->h_CASVolume);
        this->gpuProjection_Obj->SetHostCASImages(this->h_CASImgs);
    }

    // Pass the device pointers to the projection object
    this->gpuProjection_Obj->SetDeviceVolume(this->d_Volume);
    this->gpuProjection_Obj->SetDeviceCASVolume(this->d_CASVolume);
    this->gpuProjection_Obj->SetDevicePlaneDensity(this->d_PlaneDensity);

    // Set other needed parameters
    this->gpuProjection_Obj->SetMaskRadius(this->maskRadius);

    // Run the back projection
    this->gpuProjection_Obj->BackProject(AxesOffset, nAxesToProcess);
}

void gpuGridder::CalculatePlaneDensity(int AxesOffset, int nAxesToProcess)
{
    // Calculate the plane density by running the back projection kernel with CASimages equal to one
    gpuErrorCheck(cudaSetDevice(this->GPU_Device));

    // Pass the host pointers to the projection object
    this->gpuProjection_Obj->SetHostVolume(this->h_Volume);
    this->gpuProjection_Obj->SetHostCoordinateAxes(this->h_CoordAxes);
    this->gpuProjection_Obj->SetHostKBTable(this->h_KB_Table);
    this->gpuProjection_Obj->SetHostImages(this->h_Imgs);
    this->gpuProjection_Obj->SetHostKBPreCompArray(this->h_KBPreComp);

    if (this->RunFFTOnDevice == false)
    {
        this->gpuProjection_Obj->SetHostCASVolume(this->h_CASVolume);
        this->gpuProjection_Obj->SetHostCASImages(this->h_CASImgs);
    }

    // Pass the device pointers to the projection object
    this->gpuProjection_Obj->SetDeviceVolume(this->d_Volume);
    this->gpuProjection_Obj->SetDeviceCASVolume(this->d_CASVolume);
    this->gpuProjection_Obj->SetDevicePlaneDensity(this->d_PlaneDensity);

    // Set other needed parameters
    this->gpuProjection_Obj->SetMaskRadius(this->maskRadius);

    // Run the back projection
    this->gpuProjection_Obj->CalculatePlaneDensity(AxesOffset, nAxesToProcess);
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

    delete gpuProjection_Obj;
}

void gpuGridder::ReconstructVolume()
{
    // Convert a GPU CAS volume to volume
    // Note: The volume must be square (i.e. have the same dimensions for the X, Y, and Z)
    // Step 1: Pad the input volume with zeros and convert to cufftComplex type
    // Step 2: fftshift
    // Step 3: Take discrete Fourier transform using cuFFT
    // Step 4: fftshift
    // Step 5: Convert to CAS volume using CUDA kernel
    // Step 6: Apply extra zero padding

    if (this->verbose == true)
    {
        std::cout << "gpuGridder::ReconstructVolume() on GPU " << this->GPU_Device << '\n';
        PrintMemoryAvailable();
    }

    int CASVolumeSize = this->d_CASVolume->GetSize(0);
    int CASImgSize = this->VolumeSize * this->interpFactor + this->extraPadding * 2;
    int ImgSize = this->VolumeSize;

    // First, copy the Kaiser Bessel precompensation filter to the GPU
    // Size is volume times interp factor
    int *KB_PreComp_size = new int[3];
    KB_PreComp_size[0] = this->h_Volume->GetSize(0) * this->interpFactor;
    KB_PreComp_size[1] = this->h_Volume->GetSize(1) * this->interpFactor;
    KB_PreComp_size[2] = this->h_Volume->GetSize(2) * this->interpFactor;

    DeviceMemory<float> *d_KBPreComp = new DeviceMemory<float>(this->d_CASVolume->GetDim(), KB_PreComp_size, this->GPU_Device);
    d_KBPreComp->AllocateGPUArray();
    d_KBPreComp->CopyToGPU(this->h_KBPreComp->GetPointer(), d_KBPreComp->bytes());

    delete[] KB_PreComp_size;

    int CroppedCASVolumeSize = CASVolumeSize - extraPadding * 2;
    int VolumeSize = (CASVolumeSize - extraPadding * 2) / interpFactor;

    // Allocate GPU memory for CAS volume without the extra padding
    float *d_CASVolume_Cropped;
    gpuErrorCheck(cudaMalloc(&d_CASVolume_Cropped, sizeof(float) * CroppedCASVolumeSize * CroppedCASVolumeSize * CroppedCASVolumeSize));

    // Allocate GPU memory for cufftComplex type of the cropped CAS volume (i.e. d_CASVolume_Cropped)
    cufftComplex *d_CASVolume_Cropped_Complex;
    gpuErrorCheck(cudaMalloc(&d_CASVolume_Cropped_Complex, sizeof(cufftComplex) * CroppedCASVolumeSize * CroppedCASVolumeSize * CroppedCASVolumeSize));

    // Divide the CASVolume by the plane density to normalize
    DivideVolumeFilter *DivideFilter = new DivideVolumeFilter();
    DivideFilter->SetVolumeSize(CASVolumeSize);
    DivideFilter->SetVolumeOne(this->d_CASVolume->GetPointer());
    DivideFilter->SetVolumeTwo(this->d_PlaneDensity->GetPointer());
    DivideFilter->Update();

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

    // Convert the CASImgs to complex cufft type
    CASToComplexFilter *CASFilter = new CASToComplexFilter();
    CASFilter->SetCASVolume(d_CASVolume_Cropped);
    CASFilter->SetComplexOutput(d_CASVolume_Cropped_Complex);
    CASFilter->SetVolumeSize(CroppedCASVolumeSize);
    CASFilter->Update();

    // Run FFTShift on the 3D volume
    FFTShift3DFilter<cufftComplex> *FFTShiftFilter = new FFTShift3DFilter<cufftComplex>();
    FFTShiftFilter->SetInput(d_CASVolume_Cropped_Complex);
    FFTShiftFilter->SetVolumeSize(CroppedCASVolumeSize);
    FFTShiftFilter->Update();

    //  Plane and execute the forward FFT on the 3D array
    cufftHandle inverseFFTPlan;
    cufftPlan3d(&inverseFFTPlan, CroppedCASVolumeSize, CroppedCASVolumeSize, CroppedCASVolumeSize, CUFFT_C2C);
    cufftExecC2C(inverseFFTPlan, (cufftComplex *)d_CASVolume_Cropped_Complex, (cufftComplex *)d_CASVolume_Cropped_Complex, CUFFT_INVERSE);
    cufftDestroy(inverseFFTPlan);

    // Apply a second in place 3D FFT Shift
    FFTShiftFilter->SetInput(d_CASVolume_Cropped_Complex);
    FFTShiftFilter->SetVolumeSize(CroppedCASVolumeSize);
    FFTShiftFilter->Update();

    // Multiply by the Kaiser Bessel precompensation array
    MultiplyVolumeFilter<cufftComplex> *MultiplyFilter = new MultiplyVolumeFilter<cufftComplex>();
    MultiplyFilter->SetVolumeSize(CroppedCASVolumeSize);
    MultiplyFilter->SetVolumeOne(d_CASVolume_Cropped_Complex);
    MultiplyFilter->SetVolumeTwo(d_KBPreComp->GetPointer());
    MultiplyFilter->Update();

    // Run kernel to crop the d_CASVolume_Cropped_Complex (to remove the zero padding), extract the real value,
    // and normalize the scaling introduced during the FFT
    int normalizationFactor = CroppedCASVolumeSize * CroppedCASVolumeSize * CroppedCASVolumeSize;

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
    CropFilter->Update();

    DivideScalarFilter *DivideScalar = new DivideScalarFilter();
    DivideScalar->SetInput(this->d_Volume->GetPointer());
    DivideScalar->SetScalar(float(normalizationFactor));
    DivideScalar->SetVolumeSize(VolumeSize);
    DivideScalar->Update();

    // Free the temporary variables
    gpuErrorCheck(cudaFree(d_CASVolume_Cropped));
    gpuErrorCheck(cudaFree(d_CASVolume_Cropped_Complex));
}

void gpuGridder::CASVolumeToVolume()
{
    this->gpuProjection_Obj->CASVolumeToVolume();
}