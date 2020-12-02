\mainpage Multi-GPU Gridding Index Page

\section Introduction

For many applications, it is needed to perform many iterations of forward and back projection in the Fourier domain.
Here, we provide a class for fast forward and back projection which utilizes multiple NVIDIA GPUs, CUDA, and C++ along with a 
wrapper for calling all the functions from within Matlab or from within Python.

\section Dependencies 

| Library      | Version   | Usage                                                                                       |
|--------------|-----------|---------------------------------------------------------------------------------------------|
| **CUDA**         | >= 10.0   | Used for calling the GPU functions and processing data on the GPU.|
| **NVCC**         | >= 10.0   | NVIDIA CUDA compiler for compiling the GPU code. This should be included with the CUDA download.|
| **C++ Compiler** | >= C++11  | A C++ compiler is needed for compiling the C++ code.|
| **Nvidia GPU**    |  >=3.0  Compute capability       | Between 1 and 12 NVIDIA GPUs are required.|
| **Matlab**       | >= R2018a | Optional: If compiling the MATLAB wrappers for calling the class.|
| **Python**       | >= 3.0    | Optional: If compiling the Python wrappers for calling the class.|

\section install_sec Installation

### Step 1: Clone or download the GitHub repository to your computer.

### Step 2: Open the CMake GUI with the source code path to be the "src" folder and the binaries path to be a new folder name "bin".

### Step 3: Within CMake, click on Configure

### Step 4: Check or uncheck the optional settings (such as BUILD_TESTS and WITH_MATLAB).

### Step 5: Click configure again and then click on Generate.

### Step 6 (Windows): Click on Open Project which should open Visual Studio. Then right click on ALL_BUILD and click on build. Decide between Debug or Release (see the drop down on the top center within Visual Studio).

### Step 6 (Linux): Close CMake and open a terminal within the bin folder (which was created by CMake). Within the terminal type "make" which will then compile the code.

### Step 7: Optionally, run the units tests within Matlab and / or Python to check everything is functioning correctly.

\section Video Tutorial

[Please see the video tutorial for further details and examples of the software](https://youtu.be/gO2kiizHO4g)

\section Example

Here is a simple example on running the Matlab wrapper:

    % Add the required Matlab file paths
    mfilepath=fileparts(which('MultiGPUGridder_Matlab_Class.m'));
    addpath(genpath(fullfile(mfilepath)));

    % Parameters for creating the volume and coordinate axes
    VolumeSize = 128;
    interpFactor = 2;
    n1_axes = 100;
    n2_axes = 100;

    % Create the volume
    load mri;
    MRI_volume = squeeze(D);
    MRI_volume = imresize3(MRI_volume,[VolumeSize, VolumeSize, VolumeSize]);

    % Define the projection directions
    coordAxes = create_uniform_axes(n1_axes,n2_axes,0,10);

    % Create the gridder object
    gridder = MultiGPUGridder_Matlab_Class(VolumeSize, n1_axes * n2_axes, interpFactor);

    % Set the volume
    gridder.setVolume(MRI_volume);

    % Run the forward projection
    images = gridder.forwardProject(coordAxes);    
    easyMontage(images, 1)

    % Run the back projection
    gridder.resetVolume();
    gridder.backProject(gridder.Images, coordAxes)

    vol=gridder.getVol();
    easyMontage(vol, 2)

    % Reconstruct the volume
    reconstructVol = gridder.reconstructVol();
    easyMontage(reconstructVol, 3)

![Matlab_Example](../../Documentation/Images/Matlab_Example.png)

\section matlab_unit_tests Matlab - Unit Tests

The package also provides units tests to run within Matlab. The tests verify that each CUDA kernel returns the expected output. Additionally, there are unit tests for the forward and back projection kernels which test the output from each GPU, a combintation of GPUs, and varied parameters such as the volume size, number of projection directions, and the number of CUDA streams. In order to run the unit tests, go to the /src/unit_tests folder. Then within Matlab, run the Run_Unit_Tests.m script. 

For changing the testing parameters (such as if your computer has a different number of GPUs), modify the GPU_Device parameter at the top of FilterTest.m, ForwardProjectTests.m, and BackProjectTests.m. Also feel free to modify the other testing parameters as well.

\section cuda CUDA - Asynchronous memory transfers

For large datasets, there is significant copying of data to/from the GPUs. CUDA allows for overlapping of memory transfers and kernel executation (see [NVIDIA documentation for further information](https://devblogs.nvidia.com/how-overlap-data-transfers-cuda-cc/). This greatly lowers the computation time for large datasets. CUDA streams allow us to perform this.

![CUDA_Streaming_Overview](../../Documentation/Images/CUDA_Streaming_Overview.png)

For example, in the gpuGridder::ForwardProject function, we perform asynchronous (async) memory transfers from the host (i.e.the CPU) to the device (i.e.the GPU). The following simplified code illustrates this

    cudaMemcpyAsync(
        d_CoordAxes->GetPointer(),
        h_CoordAxes->GetPointer(),
        bytes,
        cudaMemcpyHostToDevice,
        CUDA_Stream);

A **bytes** length of memory from the host memory pointer **h_CoordAxes->GetPointer()** to the device memory pointer **d_CoordAxes->GetPointer()** is copied using the **CUDA_Stream**. A corresponding call can be used to copy device memory back to the host.

        

\section cuda CUDA - Asynchronous kernel calls

Similarly to the memory transfers, we also use CUDA streams for calling the GPU kernels. This lets us create a queue of work for each GPU to execute simultaneously. The included filters also use CUDA streams such as the FFTShift2DFilter as shown below

    // Run a FFTShift on each 2D slice
    FFTShift2DFilter<cufftComplex> *FFTShiftFilter = new FFTShift2DFilter<cufftComplex>();
    FFTShiftFilter->SetInput(Images);
    FFTShiftFilter->SetImageSize(ImageSize);
    FFTShiftFilter->SetNumberOfSlices(nSlices);
    FFTShiftFilter->Update(&stream);

The CUDA stream is assigned by simply passing the reference to the stream (&stream) to the FFTShift2DFilter::Update function. The other filters work in the same way.

\section matlab_wrapper Matlab Wrapper - Memory Persistance

First, we need to have the host memory persistant when going back to Matlab in order to only need to allocate both the host and device memory once.

We achieved this though the mexFunctionWrapper.h. This header file and associated functions are used for keeping both the host (i.e. CPU) and the device (i.e. GPU) memory persisant when going back and forth to Matlab. Essentially, when the MultiGPUGridder object is created, the corresponding memory pointer is converted to a **real uint64 scalar** and returned to Matlab. This scalar is remembered within the Matlab class (by the **objectHandle** member). Then, after the object is created, when a mex function is called the objectHandle (i.e.the **real uint64 scalar**) is passed from Matlab to the C++ code and then recasted back into the MultiGPUGridder C++ object. This allows us to maintain the memory between Matlab mex function calls.

Therefore, each of the mex functions has this line of code:

    MultiGPUGridder *MultiGPUGridderObj = convertMat2Ptr<MultiGPUGridder>(prhs[0]);

The prhs[0] refers to the first variable sent from Matlab (see the Matlab documentation on the prhs). The prhs stands for right hand side (i.e.the inputs) while the plhs stands for the left hand side (i.e.the outputs), similarly to Matlab functions.

The

    convertMat2Ptr<>()

function takes the **real uint64 scalar** (which was passed from the Matlab class to the mex function) and casts the pointer to the MultiGPUGridder class to get the object back. The convertPtr2Mat<>() function does that opposite (convert the memory pointer to a real uint64 scalar) to return back to Matlab.

\section matlab_in_place Matlab Wrapper - In-place computation

In order to avoid unnecessary copying to and from Matlab as well as to avoid unneeded memory allocation, we perform the calculations in place. The steps were then

1. Matlab allocated the arrays (within Matlab) the standard way. For example, X = zeros(64, 64, 64).
2. Using the mexSetVariables wrapper, the mex function mxGetData() was used to get the memory pointer to the Matlab allocated array.
3. The pointer was then passed to the MultiGPUGridder object.
4. The memory was directly read using the pointer any output was copied to the corresponding memory location.
5. After returning to Matlab, the associated changes to the array are displayed within Matlab (without the need to copy to/from Matlab).

For a specific example, when setting the array which corresponds to the volumethe mexSetVariables was called with the following inputs:
(1) a string 'SetVolume' to specify which array we're passing, (2) objectHandle (i.e.the real uint64 scalar which corresponds to the MultiGPUGridder object created in mexCreateGridder), (3) the volume, and (4) a vector with the volume dimenstions. Within mexSetVariables, the function call was the following:

    // Pointer to the volume array and the dimensions of the array
    MultiGPUGridderObj->SetVolume((float *)mxGetData(prhs[2]), (int *)mxGetData(prhs[3]));

Then within the AbstractGridder (which is the parent class of the MultiGPUGridder class) the SetVolume function is the following

    void AbstractGridder::SetVolume(float *Volume, int *ArraySize)
    {
        // First save the given pointer
        if (this->VolumeInitialized == false)
        {
            this->h_Volume = new HostMemory<float>(3, ArraySize);
            this->h_Volume->CopyPointer(Volume);
            this->h_Volume->PinArray();

            this->VolumeInitialized = true;
        }
        else
        {
            // Just copy the pointer
            this->h_Volume->CopyPointer(Volume);
        }
    }

Here, we create a new HostMemory object and copy the pointer of the Matlab allocated array. We lastly pin the memory to allow for
asynchronous memory transfer to and from the GPUs. Please see the [NVIDIA documentation for further information](https://devblogs.nvidia.com/how-overlap-data-transfers-cuda-cc/).

\section matlab_mex Matlab Wrapper - Mex Functions

| Mex Wrapper File Name        | Purpose                                                                            |
|------------------------------|------------------------------------------------------------------------------------|
| mexCreateGridder             | Creates the MultiGPUGridder object                                                 |
| mexDeleteGridder             | Deletes and deallocates the MultiGPUGridder object                                 |
| mexSetVariables              | Passes the pointers from the Matlab allocated arrays to the MultiGPUGridder object |
| mexMultiGPUForwardProject    | Calls the forward projection function                                              |
| mexMultiGPUBackProject       | Call the back projection function                                                  |
| mexMultiGPUGetVolume         | Runs an inverse FFT to get the volume                                              |
| mexMultiGPUReconstructVolume | Normalizes by the plane density and runs an inverse FFT to get the volume          |


The mexCreateGridder creates an instance of the MultiGPUGridder class. The MultiGPUGridder class then creates an instance of the gpuGridder class with one gpuGridder object per GPU. Then in the mexMultiGPUForwardProject function for example, the MultiGPUGridder object simply iterates over the gpuGridder objects and calls the gpuGridder::ForwardProject function. See the figure below for a graphical representation of this.


![Multi_GPU_Gridder_Overview](../../Documentation/Images/Multi_GPU_Gridder_Overview.png)

